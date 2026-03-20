"""
bayes_analysis.py — Bayesian analysis of Dota ML features

Three parts:

  PART 1: Beta-Binomial conjugate model for tag features
    Each hero tag ~ Bernoulli(theta)
    Prior: theta ~ Beta(1, 1)  — uniform, we know nothing
    Posterior: Beta(1 + k, 1 + n - k)  — closed form, no MCMC needed
    Compute: P(theta_win > theta_loss | data) via Monte Carlo
             Bayes Factor BF_10 (H1: different thetas vs H0: shared theta)
             95% HDI for the difference (theta_win - theta_loss)

  PART 2: Bayesian logistic regression (Laplace approximation)
    Posterior ~ N(beta_MLE, H^-1)  where H = observed Fisher information
    Report: posterior mean, 95% credible interval, P(beta > 0 | data)

  PART 3: Posterior predictive — expected win probability per match
    For the top 3 confirmed features, visualize prior -> likelihood -> posterior
    and compute E[radiant_win | feature_values] with uncertainty

All computation uses scipy + numpy only (no PyMC).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import betaln
from scipy.stats import beta as beta_dist, norm
import statsmodels.api as sm
from scipy.stats import chi2 as chi2dist

# ---------------------------------------------------------------------------
CSV_PATH = Path(__file__).parent / "dota_ml_features_final.csv"
TARGET   = "radiant_win"
DROP_COLS = ["match_id", "start_time", "patch_version"]
RNG = np.random.default_rng(42)
N_SAMPLES = 100_000   # Monte Carlo samples for posterior
# ---------------------------------------------------------------------------


def load():
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("start_time").reset_index(drop=True)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.fillna(df.median(numeric_only=True))
    return df


# ---------------------------------------------------------------------------
# PART 1: Beta-Binomial
# ---------------------------------------------------------------------------

def log_beta_binomial_marginal(k: int, n: int, a: float, b: float) -> float:
    """log P(k successes from n | Beta(a,b) prior) = log BetaBinomial(k,n,a,b)"""
    from scipy.special import gammaln
    log_binom = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return float(log_binom + betaln(a + k, b + n - k) - betaln(a, b))


def bayes_factor_10(k_win, n_win, k_loss, n_loss, a=1.0, b=1.0) -> float:
    """
    BF_10: H1 (independent thetas) vs H0 (shared theta)

    log BF_10 = log P(data|H1) - log P(data|H0)

    H0: theta ~ Beta(a,b), shared for both groups
        log P(data|H0) = log BetaBin(k_win+k_loss, n_win+n_loss, a, b)

    H1: theta_win ~ Beta(a,b), theta_loss ~ Beta(a,b) independently
        log P(data|H1) = log BetaBin(k_win, n_win, a, b)
                       + log BetaBin(k_loss, n_loss, a, b)
    """
    log_h1 = (log_beta_binomial_marginal(k_win,  n_win,  a, b) +
              log_beta_binomial_marginal(k_loss, n_loss, a, b))
    log_h0 = log_beta_binomial_marginal(k_win + k_loss, n_win + n_loss, a, b)
    return float(np.exp(log_h1 - log_h0))


def beta_posterior_stats(k: int, n: int, a=1.0, b=1.0) -> dict:
    """Posterior Beta(a+k, b+n-k): mean, mode, 95% HDI"""
    a_post = a + k
    b_post = b + n - k
    mean = a_post / (a_post + b_post)
    mode = (a_post - 1) / (a_post + b_post - 2) if (a_post > 1 and b_post > 1) else mean
    lo, hi = beta_dist.ppf([0.025, 0.975], a_post, b_post)
    return dict(a=a_post, b=b_post, mean=mean, mode=mode, hdi_lo=lo, hdi_hi=hi)


def part1_beta_binomial(X, y):
    tag_cols = [c for c in X.columns if c.endswith("_count")]

    rows = []
    for col in tag_cols:
        binary  = (X[col] > 0).astype(int)
        x_win   = binary[y == 1]
        x_loss  = binary[y == 0]
        k_win,  n_win  = int(x_win.sum()),  len(x_win)
        k_loss, n_loss = int(x_loss.sum()), len(x_loss)

        post_win  = beta_posterior_stats(k_win,  n_win)
        post_loss = beta_posterior_stats(k_loss, n_loss)

        # P(theta_win > theta_loss | data) via Monte Carlo
        samples_win  = RNG.beta(post_win["a"],  post_win["b"],  N_SAMPLES)
        samples_loss = RNG.beta(post_loss["a"], post_loss["b"], N_SAMPLES)
        p_win_gt_loss = (samples_win > samples_loss).mean()

        # E[theta_win - theta_loss | data] with 95% HDI of difference
        diff = samples_win - samples_loss
        diff_mean = diff.mean()
        diff_hdi  = np.percentile(diff, [2.5, 97.5])

        # Bayes Factor
        bf = bayes_factor_10(k_win, n_win, k_loss, n_loss)

        rows.append(dict(
            feature=col,
            k_win=k_win, n_win=n_win, theta_win=post_win["mean"],
            k_loss=k_loss, n_loss=n_loss, theta_loss=post_loss["mean"],
            p_win_gt_loss=p_win_gt_loss,
            diff_mean=diff_mean,
            diff_hdi_lo=diff_hdi[0], diff_hdi_hi=diff_hdi[1],
            BF10=bf,
        ))

    res = pd.DataFrame(rows)
    res["log_BF"] = np.log10(res["BF10"].clip(1e-10))
    return res.sort_values("p_win_gt_loss", ascending=False)


BF_LABELS = [
    (100,  "Extreme evidence for H1"),
    (30,   "Very strong for H1"),
    (10,   "Strong for H1"),
    (3,    "Moderate for H1"),
    (1,    "Anecdotal for H1"),
    (1/3,  "Anecdotal for H0"),
    (1/10, "Moderate for H0"),
    (0,    "Strong for H0"),
]

def bf_label(bf: float) -> str:
    for threshold, label in BF_LABELS:
        if bf >= threshold:
            return label
    return "Strong for H0"


# ---------------------------------------------------------------------------
# PART 2: Bayesian logistic regression (Laplace approximation)
# ---------------------------------------------------------------------------

def part2_bayesian_logreg(X, y):
    """
    Laplace approximation to Bayesian logistic regression.

    Posterior: p(beta | data) ~= N(beta_MAP, Sigma)
    where beta_MAP = MLE (for flat prior) and Sigma = -H^-1 (inverse Hessian)

    statsmodels Logit gives us MLE and SE = sqrt(diag(Sigma)).
    So credible interval = beta ± 1.96 * SE  (same as confidence interval
    under flat prior — Laplace approximation).

    Extra: P(beta > 0 | data) = Phi(beta / SE)
    """
    rows = []
    for col in X.columns:
        x_sc = (X[col] - X[col].mean()) / (X[col].std() + 1e-10)
        Xc = sm.add_constant(x_sc)
        try:
            res = sm.Logit(y, Xc).fit(disp=False, maxiter=200)
            beta_map = res.params[col]
            se       = res.bse[col]
            ci_lo, ci_hi = res.conf_int().loc[col]
            # P(beta > 0) = P(Z > -beta/se) where Z ~ N(0,1) for flat prior
            p_pos = float(norm.cdf(beta_map / se))
            # Posterior samples
            beta_samples = RNG.normal(beta_map, se, 10_000)
            hdi = np.percentile(beta_samples, [2.5, 97.5])
        except Exception:
            beta_map, se, ci_lo, ci_hi = 0, 1, -2, 2
            p_pos = 0.5
            hdi = [-2, 2]

        rows.append(dict(
            feature=col,
            beta_map=beta_map, SE=se,
            ci_lo=ci_lo, ci_hi=ci_hi,
            p_positive=p_pos,
            OR=np.exp(beta_map),
            OR_lo=np.exp(hdi[0]), OR_hi=np.exp(hdi[1]),
        ))

    res = pd.DataFrame(rows)
    # Sort by "certainty about sign": max(P(>0), P(<0))
    res["sign_certainty"] = res["p_positive"].apply(lambda p: max(p, 1 - p))
    return res.sort_values("sign_certainty", ascending=False)


# ---------------------------------------------------------------------------
# PART 3: Posterior predictive for top features
# ---------------------------------------------------------------------------

def part3_posterior_predictive(X, y, top_features: list[str]):
    """
    For each confirmed feature, show:
      - Prior: Beta(1,1)
      - Posterior for wins group and losses group
      - Probability that a randomly chosen win-match has higher feature value
        than a randomly chosen loss-match
    And compute a simple Bayesian ensemble win probability for a hypothetical
    new match given its feature values.
    """
    print("=" * 85)
    print("PART 3: Posterior Predictive Analysis")
    print("For each top feature: prior -> posterior (wins) vs posterior (losses)")
    print("=" * 85)

    for feat in top_features:
        x = X[feat]
        x_win  = x[y == 1]
        x_loss = x[y == 0]

        # Check if it's a binary-ish feature
        is_tag = feat.endswith("_count")

        print(f"\n  Feature: {feat}")
        print(f"  Type: {'binary tag (Bernoulli)' if is_tag else 'continuous'}")

        if is_tag:
            k_w, n_w = int((x_win > 0).sum()), len(x_win)
            k_l, n_l = int((x_loss > 0).sum()), len(x_loss)
            pw = beta_posterior_stats(k_w, n_w)
            pl = beta_posterior_stats(k_l, n_l)

            print(f"  Prior:          Beta(1, 1)  — uniform on [0,1]")
            print(f"  Posterior (win):   Beta({pw['a']:.0f}, {pw['b']:.0f})"
                  f"  mean={pw['mean']:.3f}  95%HDI=[{pw['hdi_lo']:.3f}, {pw['hdi_hi']:.3f}]")
            print(f"  Posterior (loss):  Beta({pl['a']:.0f}, {pl['b']:.0f})"
                  f"  mean={pl['mean']:.3f}  95%HDI=[{pl['hdi_lo']:.3f}, {pl['hdi_hi']:.3f}]")

            s_w = RNG.beta(pw["a"], pw["b"], N_SAMPLES)
            s_l = RNG.beta(pl["a"], pl["b"], N_SAMPLES)
            diff = s_w - s_l
            print(f"  P(theta_win > theta_loss | data) = {(s_w > s_l).mean():.4f}")
            print(f"  E[theta_win - theta_loss | data] = {diff.mean():+.4f}"
                  f"  95%HDI=[{np.percentile(diff,2.5):+.4f}, {np.percentile(diff,97.5):+.4f}]")

            # Posterior predictive: if new team HAS this tag, what's P(win)?
            # P(win | has_tag) = k_w / (k_w + k_l) adjusted by group sizes
            p_win_given_tag    = k_w / n_w / (k_w / n_w + k_l / n_l)
            p_win_given_notag  = (n_w - k_w) / n_w / ((n_w-k_w)/n_w + (n_l-k_l)/n_l)
            print(f"  Posterior predictive:")
            print(f"    P(radiant_win | HAS tag) ~= {p_win_given_tag:.3f}")
            print(f"    P(radiant_win | NO  tag) ~= {p_win_given_notag:.3f}")

        else:
            # Normal approximation for continuous features
            mu_w, std_w = x_win.mean(), x_win.std()
            mu_l, std_l = x_loss.mean(), x_loss.std()
            n_w, n_l = len(x_win), len(x_loss)

            # Posterior mean with flat prior (= sample mean) and posterior SE
            se_w = std_w / np.sqrt(n_w)
            se_l = std_l / np.sqrt(n_l)

            s_w = RNG.normal(mu_w, se_w, N_SAMPLES)
            s_l = RNG.normal(mu_l, se_l, N_SAMPLES)
            diff = s_w - s_l

            print(f"  Prior:          flat (improper uniform)")
            print(f"  Posterior (win):   N({mu_w:.4f}, {se_w:.4f})"
                  f"  95%CI=[{mu_w-1.96*se_w:.4f}, {mu_w+1.96*se_w:.4f}]")
            print(f"  Posterior (loss):  N({mu_l:.4f}, {se_l:.4f})"
                  f"  95%CI=[{mu_l-1.96*se_l:.4f}, {mu_l+1.96*se_l:.4f}]")
            print(f"  P(mu_win > mu_loss | data) = {(s_w > s_l).mean():.4f}")
            print(f"  E[mu_win - mu_loss | data] = {diff.mean():+.4f}"
                  f"  95%HDI=[{np.percentile(diff,2.5):+.4f}, {np.percentile(diff,97.5):+.4f}]")


# ---------------------------------------------------------------------------
# PART 4: Bayesian ensemble win probability for a hypothetical match
# ---------------------------------------------------------------------------

def part4_bayesian_ensemble(X, y, features: list[str]):
    """
    Given the posteriors from Beta-Binomial / logistic regression,
    estimate P(radiant_win) for a 'typical' match and a 'strong Radiant' match.
    Uses posterior samples and logistic regression coefficients.
    """
    print()
    print("=" * 85)
    print("PART 4: Bayesian Ensemble Win Probability")
    print("Fit full Bayesian logistic model on top features, get posterior")
    print("predictive P(radiant_win) with uncertainty for different scenarios")
    print("=" * 85)

    X_sel = X[features].copy()
    # Standardize
    means = X_sel.mean()
    stds  = X_sel.std().replace(0, 1)
    X_sc  = (X_sel - means) / stds

    Xc = sm.add_constant(X_sc)
    try:
        fit = sm.Logit(y, Xc).fit(disp=False, maxiter=500)
    except Exception as e:
        print(f"  Logit fit failed: {e}")
        return

    beta_map = fit.params.values
    cov      = fit.cov_params().values   # posterior covariance (Laplace)

    # Sample from posterior
    beta_samples = RNG.multivariate_normal(beta_map, cov, size=10_000)

    def predict_proba(x_new_raw: dict) -> np.ndarray:
        """x_new_raw: feature_name -> raw value. Returns array of P(win) samples."""
        x_new = np.array([(x_new_raw.get(f, means[f]) - means[f]) / stds[f]
                          for f in features])
        x_new_c = np.concatenate([[1.0], x_new])  # add intercept
        logits = beta_samples @ x_new_c
        return 1 / (1 + np.exp(-logits))

    scenarios = {
        "Average match (all features at mean)": {f: float(means[f]) for f in features},
        "Strong Radiant (winrate_adv +2 sigma)": {
            **{f: float(means[f]) for f in features},
            **{f: float(means[f] + 2 * stds[f]) for f in features
               if "radiant" in f and "winrate" in f},
        },
        "Strong Dire (winrate_adv -2 sigma)": {
            **{f: float(means[f]) for f in features},
            **{f: float(means[f] - 2 * stds[f]) for f in features
               if "radiant" in f and "winrate" in f},
        },
    }

    print()
    for name, scenario in scenarios.items():
        probs = predict_proba(scenario)
        print(f"  Scenario: {name}")
        print(f"    P(radiant_win) = {probs.mean():.4f}"
              f"  95%CrI=[{np.percentile(probs,2.5):.4f}, {np.percentile(probs,97.5):.4f}]")
        print(f"    P(radiant_win > 0.5 | posterior) = {(probs > 0.5).mean():.4f}")
        print()

    # Posterior coefficient summary
    print("  Posterior coefficient summary (standardized features):")
    print(f"  {'Feature':<42}  {'beta_mean':>10}  {'95% CrI':>22}  {'P(beta>0)':>10}  direction")
    print("  " + "-" * 95)
    for i, feat in enumerate(["const"] + features):
        b_samples = beta_samples[:, i]
        b_mean = b_samples.mean()
        b_lo, b_hi = np.percentile(b_samples, [2.5, 97.5])
        p_pos = (b_samples > 0).mean()
        direction = "+Radiant" if p_pos > 0.5 else "+Dire   "
        certainty = max(p_pos, 1 - p_pos)
        if feat != "const":
            print(f"  {feat:<42}  {b_mean:>+10.4f}  [{b_lo:>+8.4f}, {b_hi:>+8.4f}]  "
                  f"{p_pos:>10.4f}  {direction}  ({certainty:.1%} certain)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    df = load()
    y  = df[TARGET]
    X  = df.drop(columns=[TARGET])
    n0, n1 = (y == 0).sum(), (y == 1).sum()

    print(f"Dataset: {len(df)} matches | Radiant wins: {n1} ({n1/len(df):.1%})"
          f" | Dire wins: {n0} ({n0/len(df):.1%})")
    print()

    # ── PART 1 ───────────────────────────────────────────────────────────────
    chi = part1_beta_binomial(X, y)

    print("=" * 85)
    print("PART 1: Beta-Binomial Bayesian Analysis for tag features")
    print("Prior: Beta(1,1) — uniform | Posterior updated from data")
    print("BF interpretation: >10=Strong, >30=Very Strong, >100=Extreme")
    print("=" * 85)
    print()
    hdr = (f"  {'Feature':<42}  {'theta_win':>9}  {'theta_loss':>10}  "
           f"{'P(win>loss)':>11}  {'E[diff]':>8}  {'BF10':>8}  Evidence")
    print(hdr)
    print("  " + "-" * (len(hdr)))
    for _, row in chi.iterrows():
        mark = " <<" if row["p_win_gt_loss"] > 0.95 or row["p_win_gt_loss"] < 0.05 else ""
        print(f"  {row['feature']:<42}  {row['theta_win']:>9.3f}  {row['theta_loss']:>10.3f}  "
              f"{row['p_win_gt_loss']:>11.4f}  {row['diff_mean']:>+8.4f}  "
              f"{row['BF10']:>8.3f}  {bf_label(row['BF10'])}{mark}")

    print()
    print("Top features by BF10 (strongest Bayesian evidence H1 vs H0):")
    top_bf = chi.sort_values("BF10", ascending=False).head(10)
    print(f"  {'Feature':<42}  {'BF10':>8}  {'log10(BF)':>10}  {'P(win>loss)':>11}")
    print("  " + "-" * 80)
    for _, row in top_bf.iterrows():
        print(f"  {row['feature']:<42}  {row['BF10']:>8.3f}  {row['log_BF']:>+10.3f}  "
              f"{row['p_win_gt_loss']:>11.4f}")

    # ── PART 2 ───────────────────────────────────────────────────────────────
    logreg = part2_bayesian_logreg(X, y)

    print()
    print("=" * 85)
    print("PART 2: Bayesian Logistic Regression (Laplace approximation)")
    print("Posterior p(beta|data) ~ N(beta_MLE, H^-1) — flat prior")
    print("P(beta>0|data): probability that feature positively affects Radiant win")
    print("=" * 85)
    print()
    print("Top-20 by sign certainty:")
    print(f"  {'Feature':<42}  {'beta':>7}  {'OR':>6}  {'95% CrI (OR)':>18}  {'P(b>0)':>7}  direction")
    print("  " + "-" * 90)
    for _, row in logreg.head(20).iterrows():
        direction = "+Radiant" if row["p_positive"] > 0.5 else "+Dire   "
        certainty = max(row["p_positive"], 1 - row["p_positive"])
        ci_str = f"[{row['OR_lo']:.3f}, {row['OR_hi']:.3f}]"
        print(f"  {row['feature']:<42}  {row['beta_map']:>+7.4f}  {row['OR']:>6.3f}  "
              f"{ci_str:>18}  {row['p_positive']:>7.4f}  {direction}  ({certainty:.1%})")

    # ── PART 3 ───────────────────────────────────────────────────────────────
    top_confirmed = ["radiant_winrate_advantage", "radiant_basic_dispel_count",
                     "radiant_recent_winrate"]
    part3_posterior_predictive(X, y, top_confirmed)

    # ── PART 4 ───────────────────────────────────────────────────────────────
    top15_wald = logreg.head(15)["feature"].tolist()
    part4_bayesian_ensemble(X, y, top15_wald)


if __name__ == "__main__":
    main()
