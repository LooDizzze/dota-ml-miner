"""
train.py v4 — Wald-отбор TOP-15 + CatBoost Small Data

Пайплайн:
  1. Загрузка и хронологическая сортировка датасета
  2. Wald feature selection: для каждой фичи фитируем univariate logit,
     W = (beta/SE)^2 ~ chi2(1), берём TOP_N_FEATURES по убыванию W
     (подтверждено 4-тестовым анализом: MW + Chi2 + LRT + Wald)
  3. TimeSeriesSplit(4) кросс-валидация на отобранных признаках
  4. CatBoost с агрессивной регуляризацией (depth=3, lr=0.01, l2=15)
  5. Финальная модель на всём датасете → dota_model.cbm
"""

import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from catboost import CatBoostClassifier
from scipy.stats import chi2 as chi2dist
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CSV_PATH   = Path(__file__).parent / "dota_ml_features_final.csv"
MODEL_PATH = Path(__file__).parent / "dota_model.cbm"

TARGET_COL     = "radiant_win"
DROP_COLS      = ["match_id", "start_time", "patch_version"]
N_SPLITS       = 4
TOP_N_FEATURES = 15   # подтверждено sweep: best AUC=0.630, best Acc=58.5%

CATBOOST_PARAMS = dict(
    iterations=1000,
    learning_rate=0.01,
    depth=3,
    l2_leaf_reg=15,
    loss_function="Logloss",
    eval_metric="AUC",
    early_stopping_rounds=100,
    random_seed=42,
    verbose=False,
)


# ---------------------------------------------------------------------------
# Шаг 1: Загрузка
# ---------------------------------------------------------------------------

def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    log.info("Loading dataset: %s", csv_path)
    df = pd.read_csv(csv_path)
    log.info("Rows: %d, columns: %d", len(df), len(df.columns))

    df = df.sort_values("start_time").reset_index(drop=True)

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    null_count = df.isnull().sum().sum()
    if null_count > 0:
        df = df.fillna(df.median(numeric_only=True))
        log.info("Filled %d NaN cells with median.", null_count)

    return df


# ---------------------------------------------------------------------------
# Шаг 2: Wald feature selection
# ---------------------------------------------------------------------------

def select_features_wald(X: pd.DataFrame, y: pd.Series, top_n: int) -> list[str]:
    """
    Ранжирует признаки по Wald W = (beta/SE)^2 из univariate logistic regression.

    Почему Wald лучше PCA/SVD для этой задачи:
      - PCA/SVD ранжируют по дисперсии в X (unsupervised) — находят
        самые «разнообразные» признаки, но не обязательно предсказательные.
      - Wald напрямую измеряет связь каждой фичи с целевой переменной.
      - Подтверждено 4-тестовым анализом: MW + Chi2 + LRT + Wald дали
        согласованный топ (radiant_winrate_advantage, radiant_recent_winrate,
        radiant_basic_dispel_count как наиболее значимые).
      - Sweep (TOP-10/15/20/32/ALL) показал TOP-15 = best AUC и best Acc.
    """
    log.info("Wald feature selection (top-%d from %d features)...", top_n, X.shape[1])

    rows = []
    for col in X.columns:
        x_sc = (X[col] - X[col].mean()) / (X[col].std() + 1e-10)
        Xc = sm.add_constant(x_sc)
        try:
            res = sm.Logit(y, Xc).fit(disp=False, maxiter=200)
            beta = res.params[col]
            se   = res.bse[col]
            W    = (beta / se) ** 2
            p    = chi2dist.sf(W, df=1)
            OR   = np.exp(beta)
        except Exception:
            W, p, OR = 0.0, 1.0, 1.0
        rows.append(dict(feature=col, W=W, p_wald=p, OR=OR))

    ranking = (pd.DataFrame(rows)
               .sort_values("W", ascending=False)
               .reset_index(drop=True))

    top = ranking.head(top_n)
    log.info("Top-%d features by Wald W:", top_n)
    log.info("  %-3s  %-42s  %-8s  %-8s  %-6s", "#", "Feature", "W", "p_wald", "OR")
    log.info("  " + "-" * 72)
    for i, row in top.iterrows():
        star = " *" if row["p_wald"] < 0.05 else "  "
        log.info("  %-3d  %-42s  %-8.3f  %-8.4f  %-6.3f%s",
                 i + 1, row["feature"], row["W"], row["p_wald"], row["OR"], star)
    log.info("  (* p < 0.05 uncorrected)")
    log.info("")

    return top["feature"].tolist()


# ---------------------------------------------------------------------------
# Шаг 3: TimeSeriesSplit CV
# ---------------------------------------------------------------------------

def run_timeseries_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
) -> tuple[list[float], list[float], pd.Series]:
    tscv = TimeSeriesSplit(n_splits=n_splits)

    auc_scores:  list[float] = []
    acc_scores:  list[float] = []
    importances: list[pd.Series] = []

    log.info("TimeSeriesSplit CV (%d folds, %d features)...", n_splits, X.shape[1])
    log.info("-" * 60)

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        log.info(
            "Fold %d/%d | train: %d rows (%d..%d) | test: %d rows (%d..%d)",
            fold_idx, n_splits,
            len(X_train), train_idx[0], train_idx[-1],
            len(X_test),  test_idx[0],  test_idx[-1],
        )

        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

        y_proba  = model.predict_proba(X_test)[:, 1]
        auc      = roc_auc_score(y_test, y_proba)
        acc      = accuracy_score(y_test, (y_proba > 0.5).astype(int))

        auc_scores.append(auc)
        acc_scores.append(acc)
        importances.append(pd.Series(model.get_feature_importance(), index=X.columns))

        log.info("  -> ROC-AUC: %.4f | Accuracy: %.4f | best_iter: %d",
                 auc, acc, model.best_iteration_)

    log.info("-" * 60)
    mean_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return auc_scores, acc_scores, mean_importance


# ---------------------------------------------------------------------------
# Шаг 4: Финальная модель
# ---------------------------------------------------------------------------

def train_final_model(X: pd.DataFrame, y: pd.Series) -> CatBoostClassifier:
    """Обучает на ВСЁМ датасете без early stopping (нет eval_set)."""
    params = {**CATBOOST_PARAMS, "early_stopping_rounds": None, "verbose": 100}
    model  = CatBoostClassifier(**params)
    log.info("Training final model on full dataset (%d rows, %d features)...",
             len(X), X.shape[1])
    model.fit(X, y)
    log.info("Done.")
    return model


# ---------------------------------------------------------------------------
# Шаг 5: Вывод итогов
# ---------------------------------------------------------------------------

def print_cv_summary(
    auc_scores: list[float],
    acc_scores: list[float],
    mean_importance: pd.Series,
) -> None:
    log.info("=" * 62)
    log.info("CV RESULTS (%d folds) | Coverage=100%% | threshold=0.50", len(auc_scores))
    log.info("")
    log.info("  %-6s  %-12s  %-12s", "Fold", "ROC-AUC", "Accuracy")
    log.info("  " + "-" * 34)
    for i, (auc, acc) in enumerate(zip(auc_scores, acc_scores), start=1):
        log.info("  %-6d  %-12.4f  %-12.4f", i, auc, acc)
    log.info("")
    log.info("  MEAN:  ROC-AUC=%.4f +/- %.4f", np.mean(auc_scores), np.std(auc_scores))
    log.info("         Accuracy=%.4f +/- %.4f", np.mean(acc_scores), np.std(acc_scores))
    log.info("")
    log.info("  EV at 1.85/1.85 odds: %.2f%% per bet",
             (1.85 * np.mean(acc_scores) - 1) * 100)
    log.info("=" * 62)
    log.info("")
    log.info("Feature importance (avg across folds):")
    log.info("\n%s", mean_importance.to_string())
    log.info("")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not CSV_PATH.exists():
        log.error("Dataset not found: %s — run processor.py first", CSV_PATH)
        return

    df = load_and_prepare(CSV_PATH)
    X  = df.drop(columns=[TARGET_COL])
    y  = df[TARGET_COL]

    # Wald-отбор признаков
    top_features = select_features_wald(X, y, TOP_N_FEATURES)
    X_top = X[top_features]

    # CV на отобранных признаках
    auc_scores, acc_scores, mean_importance = run_timeseries_cv(X_top, y)
    print_cv_summary(auc_scores, acc_scores, mean_importance)

    # Финальная модель → сохраняем
    final_model = train_final_model(X_top, y)
    final_model.save_model(str(MODEL_PATH))
    log.info("Model saved: %s", MODEL_PATH)
    log.info("Features used (%d): %s", len(top_features), top_features)


if __name__ == "__main__":
    main()
