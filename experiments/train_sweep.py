"""
train_sweep.py — Feature sweep: Wald-ranked top-N vs ALL features

Approach:
  1. Rank all features by Wald W statistic (univariate logistic regression)
     W = (beta / SE(beta))^2 — best statistically-grounded univariate ranking
     confirmed by our 4-test analysis (MW + Chi2 + LRT + Wald)
  2. Train CatBoost with TimeSeriesSplit(4) on subsets:
     TOP-10, TOP-15, TOP-20, TOP-32, ALL
  3. Compare ROC-AUC and Accuracy across all configurations
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2 as chi2dist
import statsmodels.api as sm
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
CSV_PATH   = Path(__file__).parent / "dota_ml_features_final.csv"
TARGET_COL = "radiant_win"
DROP_COLS  = ["match_id", "start_time", "patch_version"]

N_SPLITS = 4
SWEEP_SIZES = [10, 15, 20, 32, "ALL"]

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


def load():
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("start_time").reset_index(drop=True)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.fillna(df.median(numeric_only=True))
    return df


def wald_ranking(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Rank features by Wald W statistic from univariate logistic regression."""
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
            beta, se, W, p, OR = 0, 1, 0, 1.0, 1.0
        rows.append(dict(feature=col, beta=beta, SE=se, W=W, p_wald=p, OR=OR))

    ranking = pd.DataFrame(rows).sort_values("W", ascending=False).reset_index(drop=True)
    ranking["rank"] = ranking.index + 1
    return ranking


def run_cv(X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    """TimeSeriesSplit(4) CV, returns mean AUC/Acc + per-fold lists."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    auc_list, acc_list, iters = [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

        y_proba = model.predict_proba(X_test)[:, 1]
        auc_list.append(roc_auc_score(y_test, y_proba))
        acc_list.append(accuracy_score(y_test, (y_proba > 0.5).astype(int)))
        iters.append(model.best_iteration_)

    return dict(
        label=label,
        n_features=X.shape[1],
        auc_mean=np.mean(auc_list),
        auc_std=np.std(auc_list),
        acc_mean=np.mean(acc_list),
        acc_std=np.std(acc_list),
        auc_folds=auc_list,
        acc_folds=acc_list,
        best_iters=iters,
    )


def main():
    df = load()
    y  = df[TARGET_COL]
    X  = df.drop(columns=[TARGET_COL])

    print(f"Dataset: {len(df)} matches, {X.shape[1]} features")
    print()

    # ── Step 1: Wald ranking ─────────────────────────────────────────────────
    print("Computing Wald ranking (univariate logistic regression)...")
    ranking = wald_ranking(X, y)

    print()
    print("=" * 72)
    print("WALD RANKING — Top-32 features by W statistic")
    print("W = (beta/SE)^2 ~ chi2(1) | OR = exp(beta) for 1-sigma change in X")
    print("=" * 72)
    print(f"  {'#':>3}  {'Feature':<42}  {'W':>7}  {'p_wald':>8}  {'OR':>6}")
    print("  " + "-" * 65)
    for _, row in ranking.head(32).iterrows():
        star = " *" if row["p_wald"] < 0.05 else "  "
        print(f"  {int(row['rank']):>3}. {row['feature']:<42}  {row['W']:>7.3f}  "
              f"{row['p_wald']:>8.4f}  {row['OR']:>6.3f}{star}")
    print("  (* p_wald < 0.05 uncorrected)")
    print()

    # ── Step 2: Build feature subsets ────────────────────────────────────────
    all_ranked = ranking["feature"].tolist()
    subsets = {}
    for size in SWEEP_SIZES:
        if size == "ALL":
            subsets["ALL"] = X.columns.tolist()
        else:
            subsets[f"TOP-{size}"] = all_ranked[:size]

    # ── Step 3: CV sweep ─────────────────────────────────────────────────────
    print("=" * 72)
    print("TRAINING SWEEP — CatBoost TimeSeriesSplit(4)")
    print("=" * 72)

    results = []
    for label, feats in subsets.items():
        print(f"\n--- {label} ({len(feats)} features) ---")
        r = run_cv(X[feats], y, label)
        results.append(r)
        for i, (auc, acc, best) in enumerate(zip(r["auc_folds"], r["acc_folds"], r["best_iters"]), 1):
            print(f"  Fold {i}: AUC={auc:.4f}  Acc={acc:.4f}  best_iter={best}")
        print(f"  MEAN:   AUC={r['auc_mean']:.4f} ± {r['auc_std']:.4f}  "
              f"Acc={r['acc_mean']:.4f} ± {r['acc_std']:.4f}")

    # ── Step 4: Summary table ────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    print(f"  {'Config':<10}  {'N':>5}  {'AUC mean':>10}  {'AUC std':>8}  "
          f"{'Acc mean':>10}  {'Acc std':>8}  {'Folds AUC'}")
    print("  " + "-" * 85)

    best_auc = max(r["auc_mean"] for r in results)
    best_acc = max(r["acc_mean"] for r in results)

    for r in results:
        mark_auc = " <-- best AUC" if abs(r["auc_mean"] - best_auc) < 1e-6 else ""
        mark_acc = " <-- best Acc" if abs(r["acc_mean"] - best_acc) < 1e-6 else ""
        folds_str = "  ".join(f"{a:.4f}" for a in r["auc_folds"])
        print(f"  {r['label']:<10}  {r['n_features']:>5}  {r['auc_mean']:>10.4f}  "
              f"{r['auc_std']:>8.4f}  {r['acc_mean']:>10.4f}  {r['acc_std']:>8.4f}  "
              f"[{folds_str}]{mark_auc}{mark_acc}")

    print()

    # ── Step 5: Per-fold breakdown ────────────────────────────────────────────
    print("=" * 72)
    print("PER-FOLD BREAKDOWN")
    print("=" * 72)
    configs = [r["label"] for r in results]
    print(f"  {'Fold':<6}", end="")
    for r in results:
        print(f"  {r['label']:<10}", end="")
    print()
    print("  " + "-" * (6 + len(results) * 12))

    for fold_i in range(N_SPLITS):
        print(f"  {fold_i+1:<6}", end="")
        for r in results:
            print(f"  {r['auc_folds'][fold_i]:.4f}    ", end="")
        print()
    print(f"  {'MEAN':<6}", end="")
    for r in results:
        print(f"  {r['auc_mean']:.4f}    ", end="")
    print()

    # ── Step 6: Feature breakdown for each TOP-N ─────────────────────────────
    print()
    print("=" * 72)
    print("FEATURE SETS — what each config adds over the previous")
    print("=" * 72)
    prev_set = set()
    for size in SWEEP_SIZES:
        label = "ALL" if size == "ALL" else f"TOP-{size}"
        curr_set = set(subsets[label])
        added = [f for f in subsets[label] if f not in prev_set]
        if added:
            print(f"\n  {label} adds {len(added)} features:")
            for f in added:
                row = ranking[ranking["feature"] == f]
                if not row.empty:
                    w = row["W"].values[0]
                    p = row["p_wald"].values[0]
                    OR = row["OR"].values[0]
                    rank = int(row["rank"].values[0])
                    print(f"    #{rank:>2}  {f:<42}  W={w:.3f}  p={p:.4f}  OR={OR:.3f}")
                else:
                    print(f"    {f}")
        prev_set = curr_set


if __name__ == "__main__":
    main()
