"""
stat_analysis.py — Statistical analysis of Dota ML features

Four complementary tests:
  1. Mann-Whitney U test (non-parametric, each feature vs radiant_win)
     Effect size: rank-biserial r = 1 - 2U/(n0*n1)
  2. Chi-squared test (Bernoulli model: tag count > 0 → binary)
     Odds ratio shows how much more likely tag appears in winning team
  3. Bernoulli Likelihood Ratio Test
     Model X ~ Bernoulli(p_win) and X ~ Bernoulli(p_loss) separately
     D = 2*(log L(p_win, p_loss) - log L(p_pool)) ~ chi2(1)
  4. Wald test (univariate logistic regression per feature)
     Fit logit(radiant_win) ~ beta_0 + beta_1 * X_j
     W = (beta_1 / SE(beta_1))^2 ~ chi2(1) under H0: beta_1 = 0
     Also reports OR = exp(beta_1) with 95% CI

All p-values corrected via Benjamini-Hochberg (FDR < 5%).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, chi2_contingency, chi2 as chi2dist
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

# ---------------------------------------------------------------------------
CSV_PATH = Path(__file__).parent / "dota_ml_features_final.csv"
TARGET = "radiant_win"
DROP_COLS = ["match_id", "start_time", "patch_version"]
# ---------------------------------------------------------------------------

def load():
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("start_time").reset_index(drop=True)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.fillna(df.median(numeric_only=True))
    return df


def part1_mannwhitney(X, y):
    """Mann-Whitney U + rank-biserial r + BH correction."""
    group0 = X[y == 0]
    group1 = X[y == 1]
    n0, n1 = len(group0), len(group1)

    rows = []
    for col in X.columns:
        u, p = mannwhitneyu(group1[col], group0[col], alternative="two-sided")
        r = 1 - 2 * u / (n1 * n0)
        rows.append(dict(feature=col, U=u, p_raw=p, r=r,
                         mean_win=group1[col].mean(),
                         mean_loss=group0[col].mean()))

    res = pd.DataFrame(rows).sort_values("p_raw")
    _, p_adj, _, _ = multipletests(res["p_raw"].values, method="fdr_bh")
    res["p_adj"] = p_adj
    res["sig"]   = p_adj < 0.05
    return res


def part2_chi2_bernoulli(X, y):
    """Chi-squared test on binarized tag counts."""
    tag_cols = [c for c in X.columns if c.endswith("_count")]
    rows = []
    for col in tag_cols:
        binary = (X[col] > 0).astype(int)
        ct = pd.crosstab(binary, y)
        if ct.shape == (2, 2):
            chi2_val, p, _, _ = chi2_contingency(ct, correction=True)
            # P(has_tag | radiant_win=1) vs P(has_tag | radiant_win=0)
            p_tag_win  = ct.loc[1, 1] / (ct[1].sum()) if 1 in ct.index else 0
            p_tag_loss = ct.loc[1, 0] / (ct[0].sum()) if 1 in ct.index else 0
            # Odds ratio
            a = ct.loc[1, 1]; b = ct.loc[1, 0]
            c = ct.loc[0, 1]; d = ct.loc[0, 0]
            or_val = (a * d) / (b * c) if (b * c) > 0 else float("nan")
        else:
            chi2_val, p, or_val = 0.0, 1.0, 1.0
            p_tag_win = p_tag_loss = 0.0
        rows.append(dict(feature=col, chi2=chi2_val, p_raw=p,
                         OR=or_val, p_tag_win=p_tag_win, p_tag_loss=p_tag_loss))

    res = pd.DataFrame(rows).sort_values("p_raw")
    _, p_adj, _, _ = multipletests(res["p_raw"].values, method="fdr_bh")
    res["p_adj"] = p_adj
    res["sig"]   = p_adj < 0.05
    return res


def part3_lrt_bernoulli(X, y):
    """Bernoulli Likelihood Ratio Test per tag feature."""
    tag_cols = [c for c in X.columns if c.endswith("_count")]
    rows = []
    for col in tag_cols:
        x = (X[col] > 0).astype(float).values
        x_win  = x[y == 1]; x_loss = x[y == 0]
        nw, nl = len(x_win), len(x_loss)
        kw, kl = x_win.sum(), x_loss.sum()
        p_w = kw / nw if nw > 0 else 0.5
        p_l = kl / nl if nl > 0 else 0.5
        p_pool = (kw + kl) / (nw + nl)

        eps = 1e-10
        ll_sep = (kw  * np.log(p_w    + eps) + (nw - kw) * np.log(1 - p_w    + eps) +
                  kl  * np.log(p_l    + eps) + (nl - kl) * np.log(1 - p_l    + eps))
        ll_pool= ((kw + kl) * np.log(p_pool + eps) +
                  (nw + nl - kw - kl) * np.log(1 - p_pool + eps))
        D = 2 * (ll_sep - ll_pool)
        p_lrt = chi2dist.sf(D, df=1)
        rows.append(dict(feature=col, p_win=p_w, p_loss=p_l,
                         p_pool=p_pool, D=D, p_lrt=p_lrt))

    res = pd.DataFrame(rows).sort_values("p_lrt")
    _, p_adj, _, _ = multipletests(res["p_lrt"].values, method="fdr_bh")
    res["p_adj"] = p_adj
    res["sig"]   = p_adj < 0.05
    return res


def part4_wald(X, y):
    """
    Univariate Wald test via logistic regression.

    For each feature X_j, fit:
        logit(P(radiant_win=1)) = beta_0 + beta_1 * X_j

    Wald statistic: W = (beta_1 / SE(beta_1))^2  ~  chi2(1)
    Odds ratio: OR = exp(beta_1)
    95% CI for OR: exp(beta_1 +/- 1.96 * SE)

    Advantage over Chi2/LRT: works on continuous features too,
    gives direction and magnitude (OR) in a single model.
    """
    rows = []
    for col in X.columns:
        x_scaled = (X[col] - X[col].mean()) / (X[col].std() + 1e-10)
        Xc = sm.add_constant(x_scaled)
        try:
            res = sm.Logit(y, Xc).fit(disp=False, maxiter=200)
            beta   = res.params[col]
            se     = res.bse[col]
            W      = (beta / se) ** 2
            p_wald = chi2dist.sf(W, df=1)
            OR     = np.exp(beta)
            ci_lo  = np.exp(beta - 1.96 * se)
            ci_hi  = np.exp(beta + 1.96 * se)
        except Exception:
            beta, se, W, p_wald, OR, ci_lo, ci_hi = 0, 1, 0, 1.0, 1.0, 1.0, 1.0
        rows.append(dict(feature=col, beta=beta, SE=se, W=W,
                         p_raw=p_wald, OR=OR, CI_lo=ci_lo, CI_hi=ci_hi))

    res_df = pd.DataFrame(rows).sort_values("p_raw")
    _, p_adj, _, _ = multipletests(res_df["p_raw"].values, method="fdr_bh")
    res_df["p_adj"] = p_adj
    res_df["sig"]   = p_adj < 0.05
    return res_df


def main():
    df  = load()
    y   = df[TARGET]
    X   = df.drop(columns=[TARGET])
    n0, n1 = (y == 0).sum(), (y == 1).sum()

    print(f"Dataset: {len(df)} matches | Radiant wins: {n1} ({n1/len(df):.1%}) | Dire wins: {n0} ({n0/len(df):.1%})")
    print()

    # ── Part 1 ───────────────────────────────────────────────────────────────
    mw = part1_mannwhitney(X, y)

    print("=" * 85)
    print("PART 1: Mann-Whitney U test")
    print("H0: identical distributions in win/loss groups | Effect size r = rank-biserial")
    print("|r| > 0.1 = small | > 0.3 = medium | > 0.5 = large")
    print("=" * 85)
    print(f"Significant after BH: {mw['sig'].sum()} / {len(mw)}")
    print()

    sig_mw = mw[mw["sig"]]
    if not sig_mw.empty:
        hdr = f"  {'Feature':<42} {'r':>7}  {'p_raw':>8}  {'p_adj':>8}  {'mean_win':>9}  {'mean_loss':>9}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for _, row in sig_mw.iterrows():
            direction = "+Radiant" if row["r"] > 0 else "+Dire   "
            print(f"  {row['feature']:<42} {row['r']:>+7.4f}  {row['p_raw']:>8.4f}"
                  f"  {row['p_adj']:>8.4f}  {row['mean_win']:>9.4f}  {row['mean_loss']:>9.4f}  {direction}")

    print()
    print("Top-15 by |r|:")
    top15 = mw.reindex(mw["r"].abs().sort_values(ascending=False).index).head(15)
    print(f"  {'Feature':<42} {'r':>7}  {'p_raw':>8}  {'p_adj':>8}  sig")
    print("  " + "-" * 72)
    for _, row in top15.iterrows():
        mark = "YES" if row["sig"] else "   "
        print(f"  {row['feature']:<42} {row['r']:>+7.4f}  {row['p_raw']:>8.4f}  {row['p_adj']:>8.4f}  {mark}")

    # ── Part 2 ───────────────────────────────────────────────────────────────
    chi = part2_chi2_bernoulli(X, y)

    print()
    print("=" * 85)
    print("PART 2: Chi-squared test (Bernoulli model: has_tag = count > 0)")
    print("H0: P(has_tag | win) == P(has_tag | loss)")
    print("OR > 1: tag is associated with Radiant win | OR < 1: associated with Dire win")
    print("=" * 85)
    print(f"Significant after BH: {chi['sig'].sum()} / {len(chi)}")
    print()
    hdr2 = f"  {'Feature':<42} {'chi2':>7}  {'p_raw':>8}  {'p_adj':>8}  {'OR':>6}  {'P(tag|win)':>10}  {'P(tag|loss)':>11}  sig"
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for _, row in chi.iterrows():
        mark = "YES" if row["sig"] else "   "
        print(f"  {row['feature']:<42} {row['chi2']:>7.3f}  {row['p_raw']:>8.4f}"
              f"  {row['p_adj']:>8.4f}  {row['OR']:>6.3f}  {row['p_tag_win']:>10.3f}"
              f"  {row['p_tag_loss']:>11.3f}  {mark}")

    # ── Part 3 ───────────────────────────────────────────────────────────────
    lrt = part3_lrt_bernoulli(X, y)

    print()
    print("=" * 85)
    print("PART 3: Bernoulli Likelihood Ratio Test")
    print("X ~ Bernoulli(p_win) vs Bernoulli(p_loss) | D ~ chi2(1) under H0")
    print("=" * 85)
    print(f"Significant after BH: {lrt['sig'].sum()} / {len(lrt)}")
    print()
    hdr3 = f"  {'Feature':<42} {'p(tag|win)':>10}  {'p(tag|loss)':>11}  {'D':>7}  {'p_lrt':>8}  {'p_adj':>8}  sig"
    print(hdr3)
    print("  " + "-" * (len(hdr3) - 2))
    for _, row in lrt.iterrows():
        mark = "YES" if row["sig"] else "   "
        print(f"  {row['feature']:<42} {row['p_win']:>10.3f}  {row['p_loss']:>11.3f}"
              f"  {row['D']:>7.3f}  {row['p_lrt']:>8.4f}  {row['p_adj']:>8.4f}  {mark}")

    # ── Part 4: Wald test ────────────────────────────────────────────────────
    wald = part4_wald(X, y)

    print()
    print("=" * 85)
    print("PART 4: Wald test (univariate logistic regression per feature, standardized)")
    print("logit(P(win)) = b0 + b1*X_j  |  W = (b1/SE)^2 ~ chi2(1)  |  OR = exp(b1)")
    print("Feature standardized before fitting: OR reflects 1-sigma change in X_j")
    print("=" * 85)
    print(f"Significant after BH: {wald['sig'].sum()} / {len(wald)}")
    print()
    hdr4 = (f"  {'Feature':<42} {'beta':>7}  {'SE':>6}  {'W':>7}  "
            f"{'p_raw':>8}  {'p_adj':>8}  {'OR':>6}  {'95% CI':>16}  sig")
    print(hdr4)
    print("  " + "-" * (len(hdr4) - 2))
    for _, row in wald.iterrows():
        mark = "YES" if row["sig"] else "   "
        ci_str = f"[{row['CI_lo']:.3f}, {row['CI_hi']:.3f}]"
        print(f"  {row['feature']:<42} {row['beta']:>+7.4f}  {row['SE']:>6.4f}  {row['W']:>7.3f}  "
              f"{row['p_raw']:>8.4f}  {row['p_adj']:>8.4f}  {row['OR']:>6.3f}  {ci_str:>16}  {mark}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 85)
    print("SUMMARY: Features significant in >= 2 out of 4 tests (MW + Chi2 + LRT + Wald)")
    print("=" * 85)
    sig_mw_set   = set(mw[mw["sig"]]["feature"])
    sig_chi_set  = set(chi[chi["sig"]]["feature"])
    sig_lrt_set  = set(lrt[lrt["sig"]]["feature"])
    sig_wald_set = set(wald[wald["sig"]]["feature"])

    found_any = False
    for col in X.columns:
        in_mw   = col in sig_mw_set
        in_chi  = col in sig_chi_set
        in_lrt  = col in sig_lrt_set
        in_wald = col in sig_wald_set
        count = sum([in_mw, in_chi, in_lrt, in_wald])
        if count >= 2:
            found_any = True
            tests = (("MW "   if in_mw   else "   ") +
                     ("Chi2 " if in_chi  else "     ") +
                     ("LRT "  if in_lrt  else "    ") +
                     ("Wald"  if in_wald else "    "))
            r_val = mw[mw["feature"] == col]["r"].values
            r_str = f"{r_val[0]:>+7.4f}" if len(r_val) else "   N/A "
            w_row = wald[wald["feature"] == col]
            or_str = f"OR={w_row['OR'].values[0]:.3f}" if len(w_row) else ""
            print(f"  {col:<42}  [{tests}]  r={r_str}  {or_str}")

    if not found_any:
        print("  No features survive in >= 2 tests after BH correction.")

    print()
    print("Top-10 by Wald W statistic (uncorrected), for reference:")
    print(f"  {'Feature':<42} {'W':>7}  {'p_raw':>8}  {'OR':>6}  {'95% CI':>16}  sig")
    print("  " + "-" * 75)
    for _, row in wald.head(10).iterrows():
        mark = "YES" if row["sig"] else "   "
        ci_str = f"[{row['CI_lo']:.3f}, {row['CI_hi']:.3f}]"
        print(f"  {row['feature']:<42} {row['W']:>7.3f}  {row['p_raw']:>8.4f}  "
              f"{row['OR']:>6.3f}  {ci_str:>16}  {mark}")


if __name__ == "__main__":
    main()
