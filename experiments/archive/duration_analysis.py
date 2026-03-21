"""
experiments/duration_analysis.py — 6 подходов к предсказанию длительности без ML-модели

Подходы:
  1. OLS (линейная регрессия statsmodels) — явная формула с коэффициентами
  2. Байесовский апостериорный расчёт — prior от команды + сдвиг по тегам
  3. KNN (k ближайших соседей) — найти похожие матчи в истории
  4. Survival Analysis (Cox PH) — lifelines, классический инструмент для времён
  5. Rule-based скоркард — явные правила: каждый тег → ±минуты
  6. Gaussian Process — sklearn, честные доверительные интервалы

Метрики: RMSE, MAE, Coverage 80% CI (для вероятностных методов)
TimeSeriesCV (4 фолда) везде где применимо.
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

BASE = Path(__file__).parent.parent
CSV  = BASE / "dota_ml_duration.csv"

N_SPLITS = 4
TARGET   = "duration_min"
DROP     = ["match_id", "start_time", "patch_version", "radiant_win", "duration_min"]


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------

def load():
    df = pd.read_csv(CSV).sort_values("start_time").reset_index(drop=True)
    y  = df[TARGET].astype(float)
    X  = df.drop(columns=[c for c in DROP if c in df.columns]).fillna(df.median(numeric_only=True))
    print(f"Матчей: {len(df)}  |  Фич: {X.shape[1]}  |  "
          f"duration: mean={y.mean():.1f}  std={y.std():.1f}  "
          f"min={y.min():.1f}  max={y.max():.1f}")
    return df, X, y


def rmse_mae(y_true, y_pred):
    return (np.sqrt(mean_squared_error(y_true, y_pred)),
            mean_absolute_error(y_true, y_pred))


def print_scores(label, rmse, mae, extra=""):
    print(f"  {label:45s}  RMSE={rmse:.2f}  MAE={mae:.2f}  {extra}")


# ---------------------------------------------------------------------------
# 1. OLS (statsmodels)
# ---------------------------------------------------------------------------

def approach_ols(X, y):
    import statsmodels.api as sm
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests

    print("\n" + "="*65)
    print("1. OLS — линейная регрессия (statsmodels)")
    print("="*65)

    # Отбор фич по Spearman + BH (топ-10 по |rho|)
    rows = []
    for col in X.columns:
        rho, p = spearmanr(X[col].values, y.values)
        rows.append((col, rho, p))
    df_corr = pd.DataFrame(rows, columns=["feature", "rho", "p"])
    _, p_adj, _, _ = multipletests(df_corr["p"].values, method="fdr_bh")
    df_corr["p_adj"] = p_adj
    df_corr["sig"]   = p_adj < 0.05
    df_corr = df_corr.sort_values("rho", key=abs, ascending=False)

    sig_feats = df_corr[df_corr["sig"]]["feature"].tolist()[:12]
    # Если BH срезал всё — берём топ-8 по |rho| без поправки
    if not sig_feats:
        sig_feats = df_corr.head(8)["feature"].tolist()
        print(f"\n  Значимых фич (BH): 0 — используем топ-8 по |rho|")
    else:
        print(f"\n  Значимых фич (BH): {len(sig_feats)}")
    for _, row in df_corr[df_corr["sig"]].head(12).iterrows():
        print(f"    {row['feature']:40s}  rho={row['rho']:+.4f}  p_adj={row['p_adj']:.4f}")

    # CV
    tscv = TimeSeriesSplit(N_SPLITS)
    rmses, maes, preds_all, true_all = [], [], [], []
    for train_i, test_i in tscv.split(X):
        Xtr = sm.add_constant(X[sig_feats].iloc[train_i])
        Xte = sm.add_constant(X[sig_feats].iloc[test_i])
        model = sm.OLS(y.iloc[train_i], Xtr).fit()
        pred  = model.predict(Xte)
        rmses.append(np.sqrt(mean_squared_error(y.iloc[test_i], pred)))
        maes.append(mean_absolute_error(y.iloc[test_i], pred))
        preds_all.extend(pred.tolist())
        true_all.extend(y.iloc[test_i].tolist())

    print(f"\n  CV результаты:")
    print_scores("OLS (sig features)", np.mean(rmses), np.mean(maes))

    # Финальная модель — коэффициенты
    final = sm.OLS(y, sm.add_constant(X[sig_feats])).fit()
    print(f"\n  Формула (коэффициенты, финальная модель на всех данных):")
    print(f"  duration = {final.params['const']:.2f}")
    for feat in sig_feats:
        coef = final.params[feat]
        pval = final.pvalues[feat]
        print(f"    {'+' if coef>=0 else ''}{coef:.3f} * {feat}  (p={pval:.4f})")
    print(f"\n  R² = {final.rsquared:.4f}  |  Adj R² = {final.rsquared_adj:.4f}")

    return np.mean(rmses), np.mean(maes), sig_feats


# ---------------------------------------------------------------------------
# 2. Байесовский апостериорный расчёт
# ---------------------------------------------------------------------------

def approach_bayes(df, y):
    print("\n" + "="*65)
    print("2. Байесовский расчёт — prior от команды + сдвиг по тегам")
    print("="*65)

    # Веса тегов: знак определяем по Spearman-корреляции с duration
    from scipy.stats import spearmanr
    tag_cols = [c for c in df.columns if c.endswith("_count") and
                (c.startswith("radiant_") or c.startswith("dire_"))]

    # Считаем вклад каждого тега в минуты (по OLS-коэффициенту)
    import statsmodels.api as sm
    X_tags = df[tag_cols].fillna(0)
    model_tags = sm.OLS(y, sm.add_constant(X_tags)).fit()
    tag_weights = model_tags.params.drop("const")

    # Prior: нормальное распределение по длительности ПРОШЛЫХ матчей команды
    # Posterior: prior + likelihood от тегов драфта
    prior_mean_global = y.mean()
    prior_std_global  = y.std()

    tscv = TimeSeriesSplit(N_SPLITS)
    all_preds, all_true = [], []
    all_lo80, all_hi80  = [], []   # 80% CI

    for train_i, test_i in tscv.split(df):
        df_train = df.iloc[train_i].copy()
        df_test  = df.iloc[test_i].copy()
        y_train  = y.iloc[train_i]

        # Пересчитываем веса тегов на тренировочных данных
        X_tr = sm.add_constant(df_train[tag_cols].fillna(0))
        m    = sm.OLS(y_train, X_tr).fit()
        weights = m.params.drop("const")

        # Prior per team
        team_stats = {}
        for tid_col in ["radiant_team_id", "dire_team_id"] if "radiant_team_id" in df.columns else []:
            for _, row in df_train.iterrows():
                tid = row.get(tid_col)
                if pd.isna(tid): continue
                dur = y_train.loc[row.name] if row.name in y_train.index else None
                if dur:
                    if tid not in team_stats: team_stats[tid] = []
                    team_stats[tid].append(dur)

        for _, row in df_test.iterrows():
            # Prior от команды
            rteam = row.get("radiant_team_id") if "radiant_team_id" in row else None
            dteam = row.get("dire_team_id")    if "dire_team_id"    in row else None

            team_durs = []
            for tid in [rteam, dteam]:
                if tid and not pd.isna(tid) and tid in team_stats:
                    team_durs.extend(team_stats[tid])

            if team_durs:
                prior_mean = np.mean(team_durs)
                prior_n    = len(team_durs)
            else:
                prior_mean = prior_mean_global
                prior_n    = 2   # слабый prior

            # Likelihood от тегов (сдвиг)
            tag_shift = sum(weights.get(tc, 0) * row.get(tc, 0) for tc in tag_cols)

            # Байесовское обновление (нормальный-нормальный сопряжённый)
            sigma_prior = prior_std_global / np.sqrt(max(prior_n, 1))
            sigma_like  = prior_std_global  # дисперсия тег-модели

            w_prior = 1 / sigma_prior**2
            w_like  = 1 / sigma_like**2
            post_mean = (w_prior * prior_mean + w_like * (prior_mean_global + tag_shift)) / (w_prior + w_like)
            post_std  = np.sqrt(1 / (w_prior + w_like))

            all_preds.append(post_mean)
            all_true.append(y.iloc[test_i[list(df_test.index).index(row.name)]
                                   if row.name in list(df_test.index) else 0])
            all_lo80.append(post_mean - 1.28 * post_std)
            all_hi80.append(post_mean + 1.28 * post_std)

    all_preds = np.array(all_preds)
    all_true  = y.iloc[list(tscv.split(df))[-1][1][0]:].values  # грубо — используем иначе
    # Правильный способ: собираем индексы
    true_vals = []
    for _, test_i in tscv.split(df):
        true_vals.extend(y.iloc[test_i].tolist())
    all_true  = np.array(true_vals[:len(all_preds)])
    lo80 = np.array(all_lo80)
    hi80 = np.array(all_hi80)

    rmse = np.sqrt(mean_squared_error(all_true, all_preds))
    mae  = mean_absolute_error(all_true, all_preds)
    coverage = np.mean((all_true >= lo80) & (all_true <= hi80))
    print_scores("Bayes (prior=team + tag likelihood)", rmse, mae,
                 f"Coverage 80%CI={coverage:.1%}")

    # Пример интерпретации
    print(f"\n  Топ тегов влияющих на длительность:")
    tag_impact = [(t, weights.get(t, 0)) for t in tag_cols if t in weights.index]
    tag_impact.sort(key=lambda x: abs(x[1]), reverse=True)
    for t, w in tag_impact[:10]:
        print(f"    {t:45s}  {w:+.3f} мин/ед")

    return rmse, mae


# ---------------------------------------------------------------------------
# 3. KNN
# ---------------------------------------------------------------------------

def approach_knn(X, y):
    from sklearn.neighbors import KNeighborsRegressor
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests

    print("\n" + "="*65)
    print("3. KNN — k ближайших матчей в истории")
    print("="*65)

    # Используем только значимые фичи
    rows = []
    for col in X.columns:
        rho, p = spearmanr(X[col].values, y.values)
        rows.append((col, abs(rho), p))
    df_c = pd.DataFrame(rows, columns=["f","rho","p"])
    _, padj,_,_ = multipletests(df_c["p"].values, method="fdr_bh")
    df_c["padj"] = padj
    feats = df_c[df_c["padj"]<0.05].sort_values("rho",ascending=False)["f"].tolist()[:15]
    if not feats:
        feats = df_c.sort_values("rho",ascending=False)["f"].tolist()[:10]

    scaler = StandardScaler()
    tscv = TimeSeriesSplit(N_SPLITS)
    results = []

    for k in [3, 5, 7, 10, 15]:
        rmses, maes = [], []
        for train_i, test_i in tscv.split(X):
            Xtr = scaler.fit_transform(X[feats].iloc[train_i])
            Xte = scaler.transform(X[feats].iloc[test_i])
            m = KNeighborsRegressor(n_neighbors=min(k, len(train_i)-1),
                                    weights="distance")
            m.fit(Xtr, y.iloc[train_i])
            pred = m.predict(Xte)
            rmses.append(np.sqrt(mean_squared_error(y.iloc[test_i], pred)))
            maes.append(mean_absolute_error(y.iloc[test_i], pred))
        results.append((k, np.mean(rmses), np.mean(maes)))
        print_scores(f"KNN k={k}", np.mean(rmses), np.mean(maes))

    best = min(results, key=lambda x: x[1])
    print(f"\n  Лучший k={best[0]}  RMSE={best[1]:.2f}  MAE={best[2]:.2f}")
    print(f"  Фичи для KNN: {feats[:8]}...")
    return best[1], best[2]


# ---------------------------------------------------------------------------
# 4. Survival Analysis (Cox PH)
# ---------------------------------------------------------------------------

def approach_survival(X, y, sig_feats):
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    print("\n" + "="*65)
    print("4. Survival Analysis (Cox Proportional Hazards)")
    print("="*65)
    print("  Матч 'выживает' пока не закончится. Все события наблюдаемые.")

    # Cox PH требует: duration + event (всегда 1 т.к. нет цензурирования)
    df_surv = X[sig_feats].copy()
    df_surv["duration"] = y.values
    df_surv["event"]    = 1  # все матчи завершились

    tscv = TimeSeriesSplit(N_SPLITS)
    c_indices, rmses, maes = [], [], []

    for train_i, test_i in tscv.split(df_surv):
        train_df = df_surv.iloc[train_i].copy()
        test_df  = df_surv.iloc[test_i].copy()

        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(train_df, duration_col="duration", event_col="event", show_progress=False)
        except Exception as e:
            print(f"  Ошибка Cox PH: {e}")
            continue

        # Предсказанное медианное время выживания
        sf = cph.predict_survival_function(test_df.drop(columns=["duration","event"]))
        median_surv = []
        for col in sf.columns:
            surv_col = sf[col]
            # Находим время где S(t) < 0.5
            idx_below = surv_col[surv_col < 0.5].index
            median_surv.append(idx_below[0] if len(idx_below) > 0 else surv_col.index[-1])

        pred = np.array(median_surv)
        true = test_df["duration"].values
        rmses.append(np.sqrt(mean_squared_error(true, pred)))
        maes.append(mean_absolute_error(true, pred))
        c_idx = concordance_index(true, -pred)
        c_indices.append(c_idx)

    if rmses:
        print_scores("Cox PH (median survival)", np.mean(rmses), np.mean(maes),
                     f"C-index={np.mean(c_indices):.4f}")

        # Таблица коэффициентов финальной модели
        cph_final = CoxPHFitter(penalizer=0.1)
        cph_final.fit(df_surv, duration_col="duration", event_col="event", show_progress=False)
        print(f"\n  Cox PH коэффициенты (log hazard ratio):")
        summary = cph_final.summary[["coef","exp(coef)","p"]].sort_values("coef")
        for feat, row in summary.iterrows():
            direction = "→ длиннее" if row["coef"] < 0 else "→ короче"
            print(f"    {feat:40s}  coef={row['coef']:+.4f}  HR={row['exp(coef)']:.3f}  "
                  f"p={row['p']:.4f}  {direction}")
        return np.mean(rmses), np.mean(maes)
    return None, None


# ---------------------------------------------------------------------------
# 5. Rule-based скоркард
# ---------------------------------------------------------------------------

def approach_rules(df, y):
    print("\n" + "="*65)
    print("5. Rule-based скоркард")
    print("="*65)

    # Правила: базовая длина + вклад каждого тега
    BASE_DUR = y.mean()

    # Веса подобраны по смыслу и проверены корреляцией
    tag_rules = {
        # Теги, удлиняющие матч
        "radiant_late_game_scaling_count": +1.5,
        "dire_late_game_scaling_count":    +1.5,
        "radiant_push_cooldown_count":     +1.2,
        "dire_push_cooldown_count":        +1.2,
        "radiant_hard_save_count":         +0.8,
        "dire_hard_save_count":            +0.8,
        "radiant_mass_heal_count":         +1.0,
        "dire_mass_heal_count":            +1.0,
        "radiant_waveclear_count":         +0.7,
        "dire_waveclear_count":            +0.7,
        # Теги, укорачивающие матч
        "radiant_burst_damage_count":      -1.0,
        "dire_burst_damage_count":         -1.0,
        "radiant_initiation_count":        -0.8,
        "dire_initiation_count":           -0.8,
        "radiant_push_zoo_count":          -1.2,
        "dire_push_zoo_count":             -1.2,
        "radiant_global_mobility_count":   -0.9,
        "dire_global_mobility_count":      -0.9,
    }

    # team_pace фичи (если есть)
    team_rules = {
        "rad_team_avg_duration":  0.4,
        "dire_team_avg_duration": 0.4,
        "team_max_late_tendency": 8.0,
    }

    def predict_rule(row):
        score = BASE_DUR
        for feat, weight in tag_rules.items():
            if feat in row.index:
                score += weight * row[feat]
        for feat, weight in team_rules.items():
            if feat in row.index and not pd.isna(row[feat]):
                score += weight * (row[feat] - (BASE_DUR if "duration" in feat else 0.47))
        return score

    tscv = TimeSeriesSplit(N_SPLITS)
    rmses, maes = [], []

    for train_i, test_i in tscv.split(df):
        df_test = df.iloc[test_i]
        preds = df_test.apply(predict_rule, axis=1).values
        rmses.append(np.sqrt(mean_squared_error(y.iloc[test_i], preds)))
        maes.append(mean_absolute_error(y.iloc[test_i], preds))

    print_scores("Rule-based (фиксированные веса)", np.mean(rmses), np.mean(maes))

    # Оптимизированные веса (GridSearch по обучающим данным)
    from sklearn.linear_model import Ridge

    rule_feats = [f for f in list(tag_rules.keys()) + list(team_rules.keys()) if f in df.columns]
    rmses2, maes2 = [], []
    for train_i, test_i in tscv.split(df):
        X_r = df[rule_feats].iloc[train_i].fillna(0)
        X_t = df[rule_feats].iloc[test_i].fillna(0)
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_r, y.iloc[train_i])
        pred = ridge.predict(X_t)
        rmses2.append(np.sqrt(mean_squared_error(y.iloc[test_i], pred)))
        maes2.append(mean_absolute_error(y.iloc[test_i], pred))

    print_scores("Rule-based (Ridge, те же фичи)", np.mean(rmses2), np.mean(maes2))

    # Распечатаем читаемую формулу
    from sklearn.linear_model import Ridge
    ridge_final = Ridge(alpha=10.0).fit(df[rule_feats].fillna(0), y)
    print(f"\n  Оптимизированные веса (Ridge):")
    print(f"  duration = {ridge_final.intercept_:.1f}")
    for feat, coef in sorted(zip(rule_feats, ridge_final.coef_), key=lambda x: abs(x[1]), reverse=True):
        if abs(coef) > 0.05:
            print(f"    {'+' if coef>=0 else ''}{coef:.3f} * {feat}")

    return np.mean(rmses), np.mean(maes2)


# ---------------------------------------------------------------------------
# 6. Gaussian Process
# ---------------------------------------------------------------------------

def approach_gp(X, y, sig_feats):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

    print("\n" + "="*65)
    print("6. Gaussian Process — честные доверительные интервалы")
    print("="*65)
    print("  (медленнее остальных, работаем на подмножестве фич)")

    feats = sig_feats[:8]  # GP плохо масштабируется на много фич
    scaler = StandardScaler()
    tscv = TimeSeriesSplit(N_SPLITS)

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)

    rmses, maes, coverages = [], [], []

    for fold_idx, (train_i, test_i) in enumerate(tscv.split(X), 1):
        Xtr = scaler.fit_transform(X[feats].iloc[train_i])
        Xte = scaler.transform(X[feats].iloc[test_i])
        ytr = y.iloc[train_i].values

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                      normalize_y=True, random_state=42)
        gp.fit(Xtr, ytr)

        mu, sigma = gp.predict(Xte, return_std=True)
        lo80 = mu - 1.28 * sigma
        hi80 = mu + 1.28 * sigma
        yte  = y.iloc[test_i].values

        rmses.append(np.sqrt(mean_squared_error(yte, mu)))
        maes.append(mean_absolute_error(yte, mu))
        coverages.append(np.mean((yte >= lo80) & (yte <= hi80)))

        print(f"  Fold {fold_idx}: RMSE={rmses[-1]:.2f}  MAE={maes[-1]:.2f}  "
              f"σ_mean={sigma.mean():.2f}  Coverage80%={coverages[-1]:.1%}")

    print_scores("GP (RBF + White)", np.mean(rmses), np.mean(maes),
                 f"Coverage 80%CI={np.mean(coverages):.1%}")

    # Неопределённость по командам: cold_start vs опытные
    if "rad_team_cold_start" in X.columns:
        Xall = scaler.fit_transform(X[feats])
        gp_full = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                           normalize_y=True, random_state=42)
        gp_full.fit(Xall, y.values)
        _, sigma_all = gp_full.predict(Xall, return_std=True)

        cold = X["rad_team_cold_start"] == 1 if "rad_team_cold_start" in X.columns else pd.Series(False, index=X.index)
        print(f"\n  σ при cold_start=1 (нет истории): {sigma_all[cold.values].mean():.2f} мин")
        print(f"  σ при cold_start=0 (есть история): {sigma_all[~cold.values].mean():.2f} мин")
        print("  → GP честно говорит 'я не уверен' когда данных мало")

    return np.mean(rmses), np.mean(maes)


# ---------------------------------------------------------------------------
# Итоговая таблица
# ---------------------------------------------------------------------------

def summary_table(results: dict, baseline_rmse: float):
    print("\n" + "="*65)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("="*65)
    print(f"  Baseline (всегда предсказывать среднее): RMSE={baseline_rmse:.2f}")
    print()
    print(f"  {'Подход':45s}  {'RMSE':8s}  {'MAE':8s}  {'vs baseline':12s}")
    print("  " + "-"*75)
    for name, (rmse, mae) in sorted(results.items(), key=lambda x: x[1][0]):
        if rmse is None: continue
        delta = rmse - baseline_rmse
        print(f"  {name:45s}  {rmse:8.2f}  {mae:8.2f}  {delta:+.2f}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("="*65)
    print("АНАЛИЗ ДЛИТЕЛЬНОСТИ: 6 подходов без ML-модели")
    print("="*65)

    df, X, y = load()
    baseline_rmse = y.std()   # предсказывать среднее всегда

    results = {}

    rmse1, mae1, sig_feats = approach_ols(X, y)
    results["1. OLS"] = (rmse1, mae1)

    rmse2, mae2 = approach_bayes(df, y)
    results["2. Bayes (team prior + tag shift)"] = (rmse2, mae2)

    rmse3, mae3 = approach_knn(X, y)
    results["3. KNN (best k)"] = (rmse3, mae3)

    rmse4, mae4 = approach_survival(X, y, sig_feats)
    if rmse4: results["4. Cox PH (survival)"] = (rmse4, mae4)

    rmse5a, rmse5b = approach_rules(df, y)
    results["5a. Rule-based (ручные веса)"] = (rmse5a, rmse5a)
    results["5b. Rule-based (Ridge opt)"]   = (rmse5b, rmse5b)

    rmse6, mae6 = approach_gp(X, y, sig_feats)
    results["6. Gaussian Process"] = (rmse6, mae6)

    summary_table(results, baseline_rmse)


if __name__ == "__main__":
    main()
