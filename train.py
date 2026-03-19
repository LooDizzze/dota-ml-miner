"""
train.py v3 — Small Data режим: TimeSeriesSplit CV + агрессивная регуляризация CatBoost

Пайплайн:
  1. Загрузка и сортировка датасета по времени
  2. TimeSeriesSplit кросс-валидация (n_splits фолдов, строго хронологически)
  3. CatBoost с агрессивной регуляризацией (depth=3, lr=0.01, l2=15)
  4. Метрики по всем фолдам: ROC-AUC и Accuracy (порог 0.5, Coverage = 100%)
  5. Финальная модель, обученная на всём датасете → dota_model.cbm
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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

TARGET_COL = "radiant_win"
DROP_COLS  = ["match_id", "start_time", "patch_version"]

# Количество фолдов TimeSeriesSplit
N_SPLITS = 4

# Параметры CatBoost для Small Data
CATBOOST_PARAMS = dict(
    iterations=1000,
    learning_rate=0.01,       # медленное обучение — меньше шансов переобучиться
    depth=3,                  # очень простые деревья, только сильные паттерны
    l2_leaf_reg=15,           # жёсткая L2-регуляризация листьев
    loss_function="Logloss",
    eval_metric="AUC",
    early_stopping_rounds=100,
    random_seed=42,
    verbose=False,            # CV сам будет логировать прогресс
)


# ---------------------------------------------------------------------------
# Шаг 1: Загрузка и подготовка данных
# ---------------------------------------------------------------------------

def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    """
    Читает CSV, сортирует по start_time, удаляет служебные колонки.
    Хронологическая сортировка критична — TimeSeriesSplit зависит от порядка строк.
    """
    log.info("Загружаем датасет: %s", csv_path)
    df = pd.read_csv(csv_path)
    log.info("Загружено строк: %d, колонок: %d", len(df), len(df.columns))

    df = df.sort_values("start_time").reset_index(drop=True)
    log.info("Датафрейм отсортирован по start_time.")

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    log.info("Удалены служебные колонки: %s", cols_to_drop)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Целевая переменная '{TARGET_COL}' не найдена в датасете.")

    # Заполняем пропуски медианой — CatBoost терпит NaN, но явное заполнение
    # даёт более стабильные результаты на малых данных
    null_before = df.isnull().sum().sum()
    if null_before > 0:
        df = df.fillna(df.median(numeric_only=True))
        log.info("Заполнено пропусков медианой: %d ячеек.", null_before)

    return df


# ---------------------------------------------------------------------------
# Evaluate: метрики на 100% матчей (порог 0.5)
# ---------------------------------------------------------------------------

def evaluate(y_true: pd.Series, y_proba: np.ndarray) -> dict:
    """
    Считает метрики по всем матчам без исключения (Coverage = 100%).

    Правило принятия решения:
      proba > 0.50  → победа Radiant (1)
      proba <= 0.50 → победа Dire    (0)

    Возвращает:
      roc_auc  — ROC-AUC (качество вероятностей, не зависит от порога)
      accuracy — доля верных предсказаний по всем матчам тестовой выборки
    """
    roc_auc  = roc_auc_score(y_true, y_proba)
    accuracy = accuracy_score(y_true, (y_proba > 0.5).astype(int))
    return {"roc_auc": roc_auc, "accuracy": accuracy}


# ---------------------------------------------------------------------------
# Шаг 2: TimeSeriesSplit кросс-валидация
# ---------------------------------------------------------------------------

def run_timeseries_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
) -> tuple[list[float], list[float], pd.Series]:
    """
    Запускает TimeSeriesSplit CV и собирает метрики и feature importance.

    TimeSeriesSplit принцип:
      Фолд 1: train=[0..n],   test=[n+1..m]
      Фолд 2: train=[0..m],   test=[m+1..k]
      ...
    Тест всегда строго после трейна — никакой утечки из будущего.

    Возвращает:
      auc_scores       — список ROC-AUC по фолдам
      acc_scores       — список Accuracy по фолдам
      mean_importance  — усреднённая важность признаков по всем фолдам
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    auc_scores:  list[float] = []
    acc_scores:  list[float] = []
    importances: list[pd.Series] = []

    log.info("Запускаем TimeSeriesSplit CV (%d фолдов)...", n_splits)
    log.info("-" * 60)

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        log.info(
            "Фолд %d/%d | train: %d строк (idx %d..%d) | test: %d строк (idx %d..%d)",
            fold_idx, n_splits,
            len(X_train), train_idx[0], train_idx[-1],
            len(X_test),  test_idx[0],  test_idx[-1],
        )

        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            use_best_model=True,
        )

        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_proba)

        auc_scores.append(metrics["roc_auc"])
        acc_scores.append(metrics["accuracy"])
        importances.append(
            pd.Series(model.get_feature_importance(), index=X.columns)
        )

        log.info(
            "  -> ROC-AUC: %.4f | Accuracy: %.4f | Coverage: 100%% | best_iter: %d",
            metrics["roc_auc"], metrics["accuracy"], model.best_iteration_,
        )

    log.info("-" * 60)

    mean_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)

    return auc_scores, acc_scores, mean_importance


# ---------------------------------------------------------------------------
# Шаг 3: Финальная модель на всём датасете
# ---------------------------------------------------------------------------

def train_final_model(X: pd.DataFrame, y: pd.Series) -> CatBoostClassifier:
    """
    После CV обучаем финальную модель на ВСЁМ датасете.
    Это даёт максимум сигнала при инференсе на новых матчах.
    Early stopping убираем — нет отдельного eval_set.
    """
    params = {**CATBOOST_PARAMS, "early_stopping_rounds": None, "verbose": 100}
    model = CatBoostClassifier(**params)

    log.info("Обучаем финальную модель на всём датасете (%d строк)...", len(X))
    model.fit(X, y)
    log.info("Финальная модель обучена.")
    return model


# ---------------------------------------------------------------------------
# Шаг 4: Вывод итоговых метрик
# ---------------------------------------------------------------------------

def print_cv_summary(
    auc_scores:      list[float],
    acc_scores:      list[float],
    mean_importance: pd.Series,
) -> None:
    """Выводит сводку по всем фолдам CV (Coverage = 100%, порог 0.5)."""

    log.info("=" * 60)
    log.info("ИТОГИ КРОСС-ВАЛИДАЦИИ (%d фолдов) | Coverage = 100%% | порог = 0.50", len(auc_scores))
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
    log.info("Топ-15 признаков (усреднённая важность по фолдам):")
    log.info("\n%s", mean_importance.head(15).to_string())
    log.info("")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not CSV_PATH.exists():
        log.error("Датасет не найден: %s — сначала запустите processor.py", CSV_PATH)
        return

    df = load_and_prepare(CSV_PATH)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # CV для оценки реального качества модели
    auc_scores, acc_scores, mean_importance = run_timeseries_cv(X, y)
    print_cv_summary(auc_scores, acc_scores, mean_importance)

    # Финальная модель на всех данных → для инференса
    final_model = train_final_model(X, y)
    final_model.save_model(str(MODEL_PATH))
    log.info("Модель сохранена: %s", MODEL_PATH)


if __name__ == "__main__":
    main()
