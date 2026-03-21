# -*- coding: utf-8 -*-
"""
poc_synthetic_feature.py — Proof-of-Concept: synthetic team history feature
============================================================================
Задача 3: Проверяем, насколько CatBoost отреагирует на историческую фичу
команды, ПРЕЖДЕ чем парсить реальные данные.

Метод:
  1. Берём duration_min как "идеальную" историческую фичу команды
  2. Добавляем гауссовый шум так, чтобы Pearson r с таргетом ≈ 0.15–0.20
  3. Обучаем две модели (с фичей и без) на одинаковом пайплайне
  4. Сравниваем RMSE на TimeSeriesCV

Интерпретация:
  Δ RMSE > 0.3 мин → гипотеза жизнеспособна, стоит строить реальные фичи
  Δ RMSE < 0.1 мин → сигнал слишком слаб при текущем объёме данных (441 матч)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CSV_PATH   = Path(__file__).parent.parent / "dota_ml_duration.csv"
TARGET_COL = "duration_min"
DROP_COLS  = ["match_id", "start_time", "patch_version", "duration_min", "radiant_win"]
N_SPLITS   = 4
TARGET_R   = 0.175   # целевая корреляция синтетической фичи с таргетом

CATBOOST_PARAMS = dict(
    iterations=1000,
    learning_rate=0.01,
    depth=3,
    l2_leaf_reg=15,
    loss_function="RMSE",
    eval_metric="RMSE",
    early_stopping_rounds=100,
    random_seed=42,
    verbose=False,
)

# ---------------------------------------------------------------------------
# 1. Генерация синтетической фичи с заданной корреляцией
# ---------------------------------------------------------------------------

def make_synthetic_feature(y: pd.Series, target_r: float, seed: int = 42) -> pd.Series:
    """
    Генерирует фичу f = alpha * y_normalized + beta * noise
    так что Pearson(f, y) ≈ target_r.

    Формула: если сигнал s = y_norm, шум n ~ N(0,1):
      f = s + k * n  →  r = 1 / sqrt(1 + k²)
      k = sqrt((1 - r²) / r²)
    """
    rng = np.random.default_rng(seed)
    y_norm = (y - y.mean()) / y.std()

    k = np.sqrt((1 - target_r**2) / (target_r**2))
    noise = rng.standard_normal(len(y))
    f_raw = y_norm + k * noise

    # Нормируем обратно к scale duration_min для читаемости
    f_scaled = f_raw * y.std() + y.mean()
    return pd.Series(f_scaled, index=y.index, name="simulated_avg_past_duration")


def verify_correlation(f: pd.Series, y: pd.Series) -> float:
    r, p = pearsonr(f, y)
    print(f"  Synthetic feature: Pearson r = {r:.3f}  (p={p:.4f})")
    print(f"  Target was r ≈ {TARGET_R:.3f}  — {'OK' if abs(r - TARGET_R) < 0.03 else 'check seed'}")
    return r


# ---------------------------------------------------------------------------
# 2. Загрузка и подготовка данных
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(CSV_PATH).sort_values("start_time").reset_index(drop=True)
    y  = df[TARGET_COL].copy()
    X  = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X  = X.fillna(X.median(numeric_only=True))

    # Добавляем combined total/diff фичи (как в train_duration.py)
    new_cols = {}
    for col in list(X.columns):
        if col.startswith("radiant_") and col.endswith("_count"):
            base = col[len("radiant_"):-len("_count")]
            dc   = f"dire_{base}_count"
            if dc in X.columns:
                new_cols[f"total_{base}"] = X[col] + X[dc]
                new_cols[f"diff_{base}"]  = X[col] - X[dc]
    X = pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1)
    return X, y


# ---------------------------------------------------------------------------
# 3. TimeSeriesCV на заданном наборе фич
# ---------------------------------------------------------------------------

def run_cv(X: pd.DataFrame, y: pd.Series, label: str) -> tuple[float, float]:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rmse_list, mae_list = [], []

    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        model = CatBoostRegressor(**CATBOOST_PARAMS)
        model.fit(X.iloc[tr], y.iloc[tr],
                  eval_set=(X.iloc[te], y.iloc[te]),
                  use_best_model=True)
        pred = model.predict(X.iloc[te])
        rmse_list.append(np.sqrt(mean_squared_error(y.iloc[te], pred)))
        mae_list.append(mean_absolute_error(y.iloc[te], pred))

    mean_rmse = float(np.mean(rmse_list))
    mean_mae  = float(np.mean(mae_list))
    print(f"  [{label}]  folds={rmse_list}  MEAN RMSE={mean_rmse:.3f}  MAE={mean_mae:.3f}")
    return mean_rmse, mean_mae


# ---------------------------------------------------------------------------
# 4. Выбор топ-N фич (простой Spearman-ранк, без BH)
#    Используем тот же TOP_N что и в train_duration.py
# ---------------------------------------------------------------------------

def pick_top_features(X: pd.DataFrame, y: pd.Series, top_n: int = 15) -> list[str]:
    from scipy.stats import spearmanr
    rows = []
    for col in X.columns:
        r, _ = spearmanr(X[col], y)
        rows.append((col, abs(r)))
    ranked = sorted(rows, key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_n]]


# ---------------------------------------------------------------------------
# 5. Main PoC
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("PoC: Does CatBoost benefit from team history feature?")
    print("=" * 65)

    X, y = load_data()
    print(f"\nDataset: {len(X)} rows, {X.shape[1]} features after combined")

    # --- Baseline: топ-15 без исторической фичи ---
    print("\n--- STEP 1: Baseline (no history feature) ---")
    top_feats_base = pick_top_features(X, y, top_n=15)
    X_base = X[top_feats_base]
    rmse_base, mae_base = run_cv(X_base, y, label="BASELINE")

    # --- Синтетическая фича ---
    print("\n--- STEP 2: Generating synthetic team history feature ---")
    synth = make_synthetic_feature(y, TARGET_R)
    real_r = verify_correlation(synth, y)

    # Добавляем фичу в датасет и пересчитываем топ
    X_with = X.copy()
    X_with["simulated_avg_past_duration"] = synth.values
    top_feats_with = pick_top_features(X_with, y, top_n=15)

    is_synth_selected = "simulated_avg_past_duration" in top_feats_with
    print(f"  Synthetic feature selected in top-15: {is_synth_selected}")
    if not is_synth_selected:
        # Принудительно включаем — тестируем именно её влияние
        top_feats_with = ["simulated_avg_past_duration"] + top_feats_base[:14]
        print("  (Forced into feature set, replacing 15th feature)")

    X_with_top = X_with[top_feats_with]
    print("\n--- STEP 3: CV with synthetic history feature ---")
    rmse_with, mae_with = run_cv(X_with_top, y, label="+SYNTH")

    # --- Результат ---
    delta_rmse = rmse_base - rmse_with
    delta_mae  = mae_base  - mae_with

    print("\n" + "=" * 65)
    print("RESULT SUMMARY")
    print("=" * 65)
    print(f"  BASELINE   RMSE = {rmse_base:.3f} min | MAE = {mae_base:.3f} min")
    print(f"  +SYNTH     RMSE = {rmse_with:.3f} min | MAE = {mae_with:.3f} min")
    print(f"  Δ RMSE = {delta_rmse:+.3f} min  ({'IMPROVED' if delta_rmse > 0 else 'DEGRADED'})")
    print(f"  Δ MAE  = {delta_mae:+.3f} min")
    print()
    print("  Synthetic feature real Pearson r = {:.3f}".format(real_r))
    print()

    # Интерпретация
    if delta_rmse > 0.5:
        verdict = "STRONG SIGNAL. Real historical features will likely reduce RMSE by 1-2+ min."
    elif delta_rmse > 0.15:
        verdict = "MODERATE SIGNAL. Worth engineering real features — expect 0.5-1 min gain."
    elif delta_rmse > 0.0:
        verdict = "WEAK SIGNAL. Feature helps slightly. Real data may do better (less noise)."
    else:
        verdict = "NO SIGNAL at this dataset size. More data needed before history features pay off."

    print(f"  VERDICT: {verdict}")
    print()

    # Сколько нужно данных для стабильного сигнала?
    # При r=0.175 и alpha=0.05: n = (z_alpha/2 + z_beta)^2 / r^2 * (1 + r^2/2) ≈ (1.96+0.84)^2 / 0.175^2
    z = 1.96 + 0.842   # alpha=0.05, power=0.80
    n_needed = int(np.ceil((z / real_r)**2 * (1 + real_r**2 / 2)))
    print(f"  Statistical power note:")
    print(f"    For r={real_r:.3f} to be detectable at 80% power: n ≈ {n_needed} matches")
    print(f"    Current dataset: {len(X)} matches")
    if len(X) >= n_needed:
        print(f"    -> Sufficient. If real r ≈ synthetic r, feature should be stable.")
    else:
        print(f"    -> Insufficient by {n_needed - len(X)} matches. Collect more data first.")
    print("=" * 65)


if __name__ == "__main__":
    main()
