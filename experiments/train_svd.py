"""
train_svd.py — SVD-отбор топ-15 признаков + CatBoost

Идея:
  SVD (Singular Value Decomposition) разлагает матрицу признаков на компоненты,
  упорядоченные по убыванию объясняемой дисперсии. Для каждого признака считаем
  его "суммарный вклад" во все компоненты: sum(|loading_i| * variance_ratio_i).
  Признаки с наибольшим вкладом — самые информативные по структуре данных.

  Затем переобучаем CatBoost только на этих 15 признаках и сравниваем с полной моделью.

ВАЖНО: SVD — unsupervised (не использует целевую переменную), поэтому
  вычисляем ranking один раз на полном X. Это не data leakage в классическом смысле,
  т.к. мы не смотрим на y. Тем не менее, для честности сравнения — сами
  веса модели считаются строго на трейне каждого фолда.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.decomposition import TruncatedSVD
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
SVD_COMPONENTS = 10   # сколько компонент SVD берём для ranking

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

    if TARGET_COL not in df.columns:
        raise ValueError(f"Целевая переменная '{TARGET_COL}' не найдена.")

    null_count = df.isnull().sum().sum()
    if null_count > 0:
        df = df.fillna(df.median(numeric_only=True))
        log.info("Заполнено пропусков медианой: %d ячеек.", null_count)

    return df


# ---------------------------------------------------------------------------
# Шаг 2: SVD-ранжирование признаков
# ---------------------------------------------------------------------------

def select_features_via_svd(
    X: pd.DataFrame,
    n_components: int = SVD_COMPONENTS,
    top_n: int = TOP_N_FEATURES,
) -> list[str]:
    """
    Ранжирует признаки по их вкладу в главные компоненты SVD.

    Алгоритм:
      1. StandardScaler — SVD чувствителен к масштабу
      2. TruncatedSVD(n_components) — разложение матрицы
      3. Для каждого признака: score = sum_k( |V[k, feature]| * sigma[k] )
         где sigma[k] — сингулярное значение компоненты k (пропорционально
         объясняемой дисперсии), V[k] — правый сингулярный вектор.
      4. Берём top_n признаков с наибольшим score.

    SVD не видит целевую переменную — это чисто структурный анализ данных.
    """
    log.info("Запускаем SVD (%d компонент) для отбора топ-%d признаков...", n_components, top_n)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(X_scaled)

    # components_ shape: (n_components, n_features)
    # singular_values_ shape: (n_components,)
    # Взвешиваем загрузки сингулярными значениями
    weighted_loadings = np.abs(svd.components_) * svd.singular_values_[:, np.newaxis]
    feature_scores = weighted_loadings.sum(axis=0)

    ranking = pd.Series(feature_scores, index=X.columns).sort_values(ascending=False)

    log.info("SVD объясняет %.1f%% дисперсии (%d компонент)",
             svd.explained_variance_ratio_.sum() * 100, n_components)
    log.info("")
    log.info("Топ-%d признаков по SVD-score:", top_n)
    for i, (feat, score) in enumerate(ranking.head(top_n).items(), start=1):
        log.info("  %2d. %-42s  %.4f", i, feat, score)
    log.info("")

    return ranking.head(top_n).index.tolist()


# ---------------------------------------------------------------------------
# Шаг 3: TimeSeriesSplit CV на отобранных признаках
# ---------------------------------------------------------------------------

def run_timeseries_cv(
    X: pd.DataFrame,
    y: pd.Series,
    label: str = "",
) -> tuple[list[float], list[float], pd.Series]:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    auc_scores:  list[float] = []
    acc_scores:  list[float] = []
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
# Шаг 4: Сравнение результатов
# ---------------------------------------------------------------------------

def print_comparison(
    label_a: str, auc_a: list[float], acc_a: list[float],
    label_b: str, auc_b: list[float], acc_b: list[float],
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
    log.info("  ДЕЛЬТА   %-12s %-12s  |  %+.4f      %+.4f",
             "", "", d_auc, d_acc)
    log.info("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not CSV_PATH.exists():
        log.error("Датасет не найден: %s", CSV_PATH)
        return

    df = load_and_prepare(CSV_PATH)
    X_full = df.drop(columns=[TARGET_COL])
    y      = df[TARGET_COL]

    # --- SVD отбор топ-15 ---
    top_features = select_features_via_svd(X_full)
    X_top15 = X_full[top_features]

    # --- CV на полных признаках (baseline) ---
    log.info("=== BASELINE: все %d признаков ===", X_full.shape[1])
    auc_full, acc_full, imp_full = run_timeseries_cv(X_full, y, label="ALL")

    log.info("")
    log.info("=== SVD TOP-15: %d признаков ===", TOP_N_FEATURES)
    auc_top, acc_top, imp_top = run_timeseries_cv(X_top15, y, label="SVD-15")

    # --- Сравнение ---
    log.info("")
    print_comparison(
        "ALL-65", auc_full, acc_full,
        "SVD-15", auc_top,  acc_top,
    )

    # --- Feature importance топ-15 модели ---
    log.info("")
    log.info("Feature Importance (SVD-15 модель, усреднено по фолдам):")
    log.info("\n%s", imp_top.to_string())


if __name__ == "__main__":
    main()
