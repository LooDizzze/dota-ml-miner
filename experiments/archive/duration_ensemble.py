"""
duration_ensemble.py — Ансамбль OLS + Cox PH + Gaussian Process

Три лучших подхода по отдельности дали RMSE ~10.5 (baseline = 10.57).
Комбинируем их OOF-предсказания:
  - простое среднее
  - взвешенное среднее (веса = 1/RMSE каждого фолда)
  - мета-Ridge (стекинг: OOF как фичи → Ridge → финал)
"""

import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

BASE = Path(__file__).parent.parent
CSV  = BASE / "dota_ml_duration.csv"
N_SPLITS = 4


def load():
    df = pd.read_csv(CSV).sort_values("start_time").reset_index(drop=True)
    y  = df["duration_min"].astype(float)
    DROP = ["match_id", "start_time", "patch_version", "radiant_win", "duration_min"]
    X  = df.drop(columns=[c for c in DROP if c in df.columns]).fillna(df.median(numeric_only=True))
    return df, X, y


def top_features(X, y, n=8):
    from scipy.stats import spearmanr
    corrs = [(c, abs(spearmanr(X[c], y)[0])) for c in X.columns]
    corrs.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in corrs[:n]]


def oof_ols(X, y, feats, tscv):
    import statsmodels.api as sm
    oof = np.full(len(y), np.nan)
    for train_i, test_i in tscv.split(X):
        m = sm.OLS(y.iloc[train_i], sm.add_constant(X[feats].iloc[train_i])).fit()
        oof[test_i] = m.predict(sm.add_constant(X[feats].iloc[test_i]))
    return oof


def oof_cox(X, y, feats, tscv):
    from lifelines import CoxPHFitter
    oof = np.full(len(y), np.nan)
    for train_i, test_i in tscv.split(X):
        df_tr = X[feats].iloc[train_i].copy()
        df_te = X[feats].iloc[test_i].copy()
        df_tr["duration"] = y.iloc[train_i].values
        df_tr["event"]    = 1
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df_tr, duration_col="duration", event_col="event", show_progress=False)
        sf  = cph.predict_survival_function(df_te)
        med = []
        for col in sf.columns:
            below = sf[col][sf[col] < 0.5].index
            med.append(float(below[0]) if len(below) else float(sf[col].index[-1]))
        oof[test_i] = med
    return oof


def oof_gp(X, y, feats, tscv):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    oof   = np.full(len(y), np.nan)
    sigma = np.full(len(y), np.nan)
    scaler = StandardScaler()
    kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1.0)
    for train_i, test_i in tscv.split(X):
        Xtr = scaler.fit_transform(X[feats].iloc[train_i])
        Xte = scaler.transform(X[feats].iloc[test_i])
        gp  = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                       normalize_y=True, random_state=42)
        gp.fit(Xtr, y.iloc[train_i].values)
        mu, std = gp.predict(Xte, return_std=True)
        oof[test_i]   = mu
        sigma[test_i] = std
    return oof, sigma


def score(y_true, y_pred, label):
    valid = ~np.isnan(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true[valid], y_pred[valid]))
    mae  = mean_absolute_error(y_true[valid], y_pred[valid])
    print(f"  {label:40s}  RMSE={rmse:.3f}  MAE={mae:.3f}")
    return rmse, mae


def main():
    print("=" * 60)
    print("АНСАМБЛЬ: OLS + Cox PH + GP")
    print("=" * 60)

    df, X, y = load()
    tscv  = TimeSeriesSplit(N_SPLITS)
    feats = top_features(X, y, n=8)

    print(f"\nФичи: {feats}\n")

    print("Считаем OOF для каждой модели...")
    oof_o       = oof_ols(X, y, feats, tscv)
    print("  OLS — готово")
    oof_c       = oof_cox(X, y, feats, tscv)
    print("  Cox PH — готово")
    oof_g, sig  = oof_gp(X, y, feats, tscv)
    print("  GP — готово\n")

    print("=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    baseline = np.sqrt(mean_squared_error(y, np.full(len(y), y.mean())))
    print(f"  {'Baseline (всегда среднее)':40s}  RMSE={baseline:.3f}")
    print()

    # Отдельные модели
    score(y.values, oof_o, "OLS")
    score(y.values, oof_c, "Cox PH")
    score(y.values, oof_g, "GP")
    print()

    # Комбинации
    valid = ~(np.isnan(oof_o) | np.isnan(oof_c) | np.isnan(oof_g))

    # Простое среднее
    avg = (oof_o + oof_c + oof_g) / 3
    score(y.values, avg, "Среднее (OLS + Cox + GP)")

    # Взвешенное по обратному RMSE
    r_o = np.sqrt(mean_squared_error(y.values[valid], oof_o[valid]))
    r_c = np.sqrt(mean_squared_error(y.values[valid], oof_c[valid]))
    r_g = np.sqrt(mean_squared_error(y.values[valid], oof_g[valid]))
    w   = np.array([1/r_o, 1/r_c, 1/r_g])
    w  /= w.sum()
    wavg = w[0]*oof_o + w[1]*oof_c + w[2]*oof_g
    score(y.values, wavg, f"Взвеш. среднее (w={w.round(3)})")

    # OLS + GP (без Cox — хуже обоих)
    avg2 = (oof_o + oof_g) / 2
    score(y.values, avg2, "Среднее (OLS + GP)")

    # GP взвешенный по своей уверенности: чем меньше σ — тем больше доверие GP
    gp_weight = 1 / (sig + 1e-6)
    gp_weight = gp_weight / (gp_weight + 1)          # нормируем в [0,1]
    blend_gp  = gp_weight * oof_g + (1 - gp_weight) * oof_o
    score(y.values, blend_gp, "GP*σ-weight + OLS*(1-σ-weight)")

    # Мета-Ridge (стекинг)
    print()
    meta_X = pd.DataFrame({"ols": oof_o, "cox": oof_c, "gp": oof_g}).fillna(
        pd.DataFrame({"ols": oof_o, "cox": oof_c, "gp": oof_g}).median()
    )
    tscv2 = TimeSeriesSplit(N_SPLITS)
    meta_oof = np.full(len(y), np.nan)
    for train_i, test_i in tscv2.split(meta_X):
        r = Ridge(alpha=1.0)
        r.fit(meta_X.iloc[train_i], y.iloc[train_i])
        meta_oof[test_i] = r.predict(meta_X.iloc[test_i])
    score(y.values, meta_oof, "Мета-Ridge (стекинг)")

    # Финальные веса мета-Ridge
    r_final = Ridge(alpha=1.0).fit(meta_X, y)
    print(f"\n  Веса мета-Ridge: OLS={r_final.coef_[0]:.3f}  "
          f"Cox={r_final.coef_[1]:.3f}  GP={r_final.coef_[2]:.3f}  "
          f"intercept={r_final.intercept_:.2f}")

    # GP с доверительным интервалом — показываем пример
    print()
    print("=" * 60)
    print("GP ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ (финальная модель, пример)")
    print("=" * 60)
    # Финальная GP на всех данных
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    scaler_f = StandardScaler()
    Xf = scaler_f.fit_transform(X[feats])
    gp_f = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0)*RBF(1.0)+WhiteKernel(1.0),
        n_restarts_optimizer=3, normalize_y=True, random_state=42
    )
    gp_f.fit(Xf, y.values)
    mu_f, std_f = gp_f.predict(Xf, return_std=True)

    # Лучшее взвешенное (OLS + GP)
    best_oof = avg2
    best_valid = ~np.isnan(best_oof)
    best_rmse = np.sqrt(mean_squared_error(y.values[best_valid], best_oof[best_valid]))
    best_mae  = mean_absolute_error(y.values[best_valid], best_oof[best_valid])

    print(f"\n  σ (неопределённость GP): mean={std_f.mean():.2f} мин  "
          f"min={std_f.min():.2f}  max={std_f.max():.2f}")
    print(f"\n  Примеры предсказаний (последние 8 матчей):")
    print(f"  {'#':4s}  {'Реальная':10s}  {'OLS':8s}  {'GP':8s}  "
          f"{'GP 80%CI':16s}  {'Среднее':8s}")
    print("  " + "-"*60)
    for i in range(-8, 0):
        real  = y.values[i]
        p_ols = oof_o[i] if not np.isnan(oof_o[i]) else np.nan
        p_gp  = mu_f[i]
        s_gp  = std_f[i]
        p_avg = (p_ols + p_gp) / 2 if not np.isnan(p_ols) else p_gp
        lo    = p_gp - 1.28 * s_gp
        hi    = p_gp + 1.28 * s_gp
        hit   = "✓" if lo <= real <= hi else "✗"
        print(f"  {i:4d}  {real:8.1f}    {p_ols if not np.isnan(p_ols) else '—':6.1f}    "
              f"{p_gp:6.1f}    [{lo:5.1f}—{hi:5.1f}] {hit}   {p_avg if not np.isnan(p_ols) else p_gp:6.1f}")

    print(f"\n  ЛУЧШИЙ АНСАМБЛЬ: OLS + GP среднее  "
          f"RMSE={best_rmse:.3f}  MAE={best_mae:.3f}  (baseline={baseline:.3f})")


if __name__ == "__main__":
    main()
