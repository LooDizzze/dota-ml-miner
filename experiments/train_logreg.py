"""
train_logreg.py — Логистическая Регрессия на предматчевых признаках Dota 2

Пайплайн:
  1. Загрузка и сортировка датасета по времени
  2. TimeSeriesSplit кросс-валидация (4 фолда)
  3. StandardScaler на трейне → применяем к тесту (обязательно для LogReg)
  4. LogisticRegression(C=0.1, class_weight='balanced')
  5. Метрики по фолдам: ROC-AUC, Accuracy (порог 0.5, Coverage=100%)
  6. Финальная модель на всём датасете → Топ-15 признаков по |coef_|
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
N_SPLITS   = 4


# ---------------------------------------------------------------------------
# Шаг 1: Загрузка и подготовка
# ---------------------------------------------------------------------------

def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    log.info("Загружаем датасет: %s", csv_path)
    df = pd.read_csv(csv_path)
    log.info("Загружено строк: %d, колонок: %d", len(df), len(df.columns))

    df = df.sort_values("start_time").reset_index(drop=True)
    log.info("Датафрейм отсортирован по start_time.")

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    log.info("Удалены служебные колонки: %s", cols_to_drop)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Целевая переменная '{TARGET_COL}' не найдена.")

    # Для LogReg заполняем NaN нулями (в отличие от CatBoost, который терпит NaN)
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        df = df.fillna(0)
        log.info("Заполнено NaN нулями: %d ячеек.", null_count)

    return df


# ---------------------------------------------------------------------------
# Шаг 2: TimeSeriesSplit кросс-валидация
# ---------------------------------------------------------------------------

def run_timeseries_cv(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[list[float], list[float]]:
    """
    Для каждого фолда:
      1. Обучаем StandardScaler ТОЛЬКО на трейне
      2. Трансформируем трейн и тест
      3. Обучаем LogisticRegression
      4. Считаем ROC-AUC и Accuracy на тесте

    Важно: скейлер никогда не видит тестовые данные при fit() —
    это была бы утечка статистики из будущего.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    auc_scores: list[float] = []
    acc_scores: list[float] = []

    log.info("Запускаем TimeSeriesSplit CV (%d фолдов)...", N_SPLITS)
    log.info("-" * 60)

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx].values, X.iloc[test_idx].values
        y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values

        log.info(
            "Фолд %d/%d | train: %d строк (idx %d..%d) | test: %d строк (idx %d..%d)",
            fold_idx, N_SPLITS,
            len(X_train), train_idx[0], train_idx[-1],
            len(X_test),  test_idx[0],  test_idx[-1],
        )

        # Масштабирование: fit только на трейне
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        model = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )
        model.fit(X_train_scaled, y_train)

        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred  = (y_proba > 0.5).astype(int)

        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        auc_scores.append(auc)
        acc_scores.append(acc)

        log.info(
            "  -> ROC-AUC: %.4f | Accuracy: %.4f | Coverage: 100%%",
            auc, acc,
        )

    log.info("-" * 60)
    return auc_scores, acc_scores


# ---------------------------------------------------------------------------
# Шаг 3: Финальная модель + топ признаков
# ---------------------------------------------------------------------------

def train_final_and_show_features(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Обучает финальную модель на всём датасете и выводит топ-15 признаков
    по абсолютному значению коэффициента.

    Знак коэффициента:
      + означает: рост признака → выше вероятность победы Radiant
      - означает: рост признака → ниже вероятность победы Radiant (победа Dire)
    """
    log.info("Обучаем финальную модель на всём датасете (%d строк)...", len(X))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    model = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    model.fit(X_scaled, y.values)

    coefs = pd.Series(model.coef_[0], index=X.columns)
    top15 = coefs.abs().sort_values(ascending=False).head(15)

    log.info("Топ-15 признаков по |коэффициенту|:")
    log.info("")
    for feat, abs_val in top15.items():
        raw = coefs[feat]
        sign = "+" if raw >= 0 else "-"
        log.info("  %s  %-40s  %.4f", sign, feat, abs_val)
    log.info("")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not CSV_PATH.exists():
        log.error("Датасет не найден: %s", CSV_PATH)
        return

    df = load_and_prepare(CSV_PATH)
    X  = df.drop(columns=[TARGET_COL])
    y  = df[TARGET_COL]

    auc_scores, acc_scores = run_timeseries_cv(X, y)

    log.info("=" * 60)
    log.info("ИТОГИ КРОСС-ВАЛИДАЦИИ (LogReg, %d фолдов) | Coverage = 100%%", N_SPLITS)
    log.info("")
    log.info("  %-6s  %-12s  %-12s", "Фолд", "ROC-AUC", "Accuracy")
    log.info("  " + "-" * 34)
    for i, (auc, acc) in enumerate(zip(auc_scores, acc_scores), start=1):
        log.info("  %-6d  %-12.4f  %-12.4f", i, auc, acc)
    log.info("")
    log.info("  СРЕДНЕЕ:  ROC-AUC=%.4f ± %.4f", np.mean(auc_scores), np.std(auc_scores))
    log.info("            Accuracy=%.4f ± %.4f", np.mean(acc_scores), np.std(acc_scores))
    log.info("=" * 60)
    log.info("")

    train_final_and_show_features(X, y)


if __name__ == "__main__":
    main()
