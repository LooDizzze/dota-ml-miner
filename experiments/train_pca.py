"""
train_pca.py — PCA-отбор топ-15 признаков + CatBoost

Отличие от SVD:
  PCA работает с ковариационной матрицей центрированных данных.
  Компоненты ортогональны и упорядочены по убыванию объясняемой дисперсии.
  Для ранжирования признаков используем explained_variance_ratio_ как вес:
    score[feature] = sum_k( |loading[k, feature]| * variance_ratio[k] )
  Это даёт "процент общей дисперсии, который несёт данный признак".
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

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
TARGET_COL = "radiant_win"
DROP_COLS  = ["match_id", "start_time", "patch_version"]

N_SPLITS       = 4
TOP_N_FEATURES = 15
PCA_COMPONENTS = 10

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
    log.info("Загружаем датасет: %s", csv_path)
    df = pd.read_csv(csv_path)
    log.info("Загружено строк: %d, колонок: %d", len(df), len(df.columns))

    df = df.sort_values("start_time").reset_index(drop=True)

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    null_count = df.isnull().sum().sum()
    if null_count > 0:
        df = df.fillna(df.median(numeric_only=True))
        log.info("Заполнено пропусков медианой: %d ячеек.", null_count)

    return df


# ---------------------------------------------------------------------------
# Шаг 2: PCA-ранжирование признаков
# ---------------------------------------------------------------------------

def select_features_via_pca(
    X: pd.DataFrame,
    n_components: int = PCA_COMPONENTS,
    top_n: int = TOP_N_FEATURES,
) -> list[str]:
    """
    Ранжирует признаки по вкладу в главные компоненты PCA.

    Алгоритм:
      1. StandardScaler — PCA требует нормализации
      2. PCA(n_components) — разложение ковариационной матрицы
      3. score[j] = sum_k( |components_[k,j]| * explained_variance_ratio_[k] )
         Это доля дисперсии, объясняемая признаком j через компоненту k,
         взвешенная по значимости самой компоненты.
      4. top_n признаков с наибольшим суммарным вкладом.

    Ключевое отличие от SVD: веса — explained_variance_ratio_ (сумма = 1.0),
    а не сырые сингулярные значения. Это делает score интерпретируемым:
    score[j] ≈ "доля общей дисперсии датасета, приходящаяся на признак j".
    """
    log.info("Запускаем PCA (%d компонент) для отбора топ-%d признаков...", n_components, top_n)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled)

    # components_ shape: (n_components, n_features)
    # explained_variance_ratio_ shape: (n_components,)
    weighted = np.abs(pca.components_) * pca.explained_variance_ratio_[:, np.newaxis]
    feature_scores = weighted.sum(axis=0)

    ranking = pd.Series(feature_scores, index=X.columns).sort_values(ascending=False)

    cumvar = pca.explained_variance_ratio_.cumsum()
    log.info(
        "PCA: топ-10 компонент объясняют %.1f%% дисперсии",
        pca.explained_variance_ratio_.sum() * 100,
    )
    log.info("  Распределение по компонентам: %s",
             " | ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(pca.explained_variance_ratio_)))
    log.info("")
    log.info("Топ-%d признаков по PCA-score (доля дисперсии):", top_n)
    for i, (feat, score) in enumerate(ranking.head(top_n).items(), start=1):
        log.info("  %2d. %-42s  %.4f", i, feat, score)
    log.info("")

    # Показываем что НЕ попало в топ-15 но было важно у CatBoost
    catboost_top = [
        "radiant_winrate_advantage", "radiant_recent_winrate",
        "dire_strong_dispel_count", "d5_hero_winrate", "radiant_meta_score"
    ]
    missed = [f for f in catboost_top if f not in ranking.head(top_n).index]
    if missed:
        log.info("  [!] Признаки из CatBoost топ-5, не вошедшие в PCA-15: %s", missed)
        for f in missed:
            log.info("      %-42s  PCA-score=%.4f  (ранг #%d)",
                     f, ranking[f], ranking.index.get_loc(f) + 1)
    log.info("")

    return ranking.head(top_n).index.tolist()


# ---------------------------------------------------------------------------
# Шаг 3: TimeSeriesSplit CV
# ---------------------------------------------------------------------------

def run_timeseries_cv(
    X: pd.DataFrame,
    y: pd.Series,
    label: str = "",
) -> tuple[list[float], list[float], pd.Series]:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    auc_scores: list[float] = []
    acc_scores: list[float] = []
    importances: list[pd.Series] = []

    log.info("CV (%s) — %d фолдов, %d признаков", label, N_SPLITS, X.shape[1])
    log.info("-" * 60)

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, (y_proba > 0.5).astype(int))

        auc_scores.append(auc)
        acc_scores.append(acc)
        importances.append(pd.Series(model.get_feature_importance(), index=X.columns))

        log.info("  Фолд %d | ROC-AUC: %.4f | Accuracy: %.4f | best_iter: %d",
                 fold_idx, auc, acc, model.best_iteration_)

    log.info("-" * 60)
    mean_imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return auc_scores, acc_scores, mean_imp


# ---------------------------------------------------------------------------
# Шаг 4: Сравнение
# ---------------------------------------------------------------------------

def print_comparison(
    label_a: str, auc_a: list, acc_a: list,
    label_b: str, auc_b: list, acc_b: list,
) -> None:
    log.info("=" * 65)
    log.info("СРАВНЕНИЕ: %-20s  vs  %-20s", label_a, label_b)
    log.info("=" * 65)
    log.info("  %-6s  %-12s %-12s  |  %-12s %-12s",
             "Фолд", f"AUC[{label_a}]", f"Acc[{label_a}]",
                     f"AUC[{label_b}]", f"Acc[{label_b}]")
    log.info("  " + "-" * 58)
    for i, (a1, c1, a2, c2) in enumerate(zip(auc_a, acc_a, auc_b, acc_b), start=1):
        log.info("  %-6d  %-12.4f %-12.4f  |  %-12.4f %-12.4f", i, a1, c1, a2, c2)
    log.info("")
    log.info("  СРЕДНЕЕ  %-12.4f %-12.4f  |  %-12.4f %-12.4f",
             np.mean(auc_a), np.mean(acc_a), np.mean(auc_b), np.mean(acc_b))
    d_auc = np.mean(auc_b) - np.mean(auc_a)
    d_acc = np.mean(acc_b) - np.mean(acc_a)
    sign_auc = "+" if d_auc >= 0 else ""
    sign_acc = "+" if d_acc >= 0 else ""
    log.info("  ДЕЛЬТА   %-24s  %s%.4f      %s%.4f",
             "", sign_auc, d_auc, sign_acc, d_acc)
    log.info("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not CSV_PATH.exists():
        log.error("Датасет не найден: %s", CSV_PATH)
        return

    df     = load_and_prepare(CSV_PATH)
    X_full = df.drop(columns=[TARGET_COL])
    y      = df[TARGET_COL]

    # PCA отбор
    top_features = select_features_via_pca(X_full)
    X_top15      = X_full[top_features]

    # Baseline (все признаки)
    log.info("=== BASELINE: все %d признаков ===", X_full.shape[1])
    auc_full, acc_full, _ = run_timeseries_cv(X_full, y, label="ALL")

    log.info("")
    log.info("=== PCA TOP-15 ===")
    auc_pca, acc_pca, imp_pca = run_timeseries_cv(X_top15, y, label="PCA-15")

    log.info("")
    print_comparison(
        "ALL-64", auc_full, acc_full,
        "PCA-15", auc_pca,  acc_pca,
    )

    log.info("")
    log.info("Feature Importance CatBoost (PCA-15 модель, усреднено по фолдам):")
    log.info("\n%s", imp_pca.to_string())


if __name__ == "__main__":
    main()
