"""Statistics for an online controlled experiment: power, SRM, z-test, Welch, bootstrap, CUPED, Holm, ROI."""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TABLES = ["users", "assignments", "exposures", "bill_payments", "guardrails"]


# ------------------------------------------------------------------ data
def user_metrics(raw: Path = ROOT / "data" / "raw") -> pd.DataFrame:
    con = duckdb.connect()
    for t in TABLES:
        con.sql(f"CREATE VIEW {t} AS SELECT * FROM read_csv('{(raw / f'{t}.csv').as_posix()}', header = true)")
    return con.sql((ROOT / "sql" / "user_metrics.sql").read_text()).df()


# ------------------------------------------------------------------ design
def mde_for_power(p_base: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable absolute lift for a two-proportion test (normal approximation)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(2 * p_base * (1 - p_base) / n_per_arm))


def sample_size_per_arm(p_base: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    p2 = p_base + mde_abs
    pbar = (p_base + p2) / 2
    return int(np.ceil(z ** 2 * 2 * pbar * (1 - pbar) / mde_abs ** 2))


# ------------------------------------------------------------------ validity
def srm_test(counts: dict[str, int], expected_split: dict[str, float] | None = None) -> dict:
    """Chi-square sample-ratio-mismatch test. p < 0.001 is the usual alarm threshold."""
    names = list(counts)
    obs = np.array([counts[k] for k in names], dtype=float)
    exp_split = np.array([(expected_split or {k: 1 / len(names) for k in names})[k] for k in names])
    chi2, p = stats.chisquare(obs, obs.sum() * exp_split)
    return {"counts": counts, "observed_split": {k: round(float(v), 4) for k, v in zip(names, obs / obs.sum())},
            "chi2": round(float(chi2), 2), "p_value": float(p), "srm_detected": bool(p < 0.001)}


def balance_table(df: pd.DataFrame, covariates: list[str], arm: str = "variant") -> pd.DataFrame:
    """Standardised mean differences of pre-period covariates; |SMD| < 0.1 means balanced."""
    t, c = df[df[arm] == "treatment"], df[df[arm] == "control"]
    rows = []
    for col in covariates:
        x = df[[col]].astype(float) if pd.api.types.is_numeric_dtype(df[col]) \
            else pd.get_dummies(df[col], prefix=col, dtype=float)
        for k in x.columns:
            mt, mc = x.loc[t.index, k].mean(), x.loc[c.index, k].mean()
            sd = np.sqrt((x.loc[t.index, k].var() + x.loc[c.index, k].var()) / 2)
            rows.append({"covariate": k, "treatment": mt, "control": mc, "smd": (mt - mc) / sd if sd > 0 else 0.0})
    out = pd.DataFrame(rows).round(4)
    out["balanced"] = out.smd.abs() < 0.1
    return out


# ------------------------------------------------------------------ effects
def proportion_test(x_t: int, n_t: int, x_c: int, n_c: int, alpha: float = 0.05) -> dict:
    p_t, p_c = x_t / n_t, x_c / n_c
    p_pool = (x_t + x_c) / (n_t + n_c)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    z = (p_t - p_c) / se_pool
    se = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    zc = stats.norm.ppf(1 - alpha / 2)
    diff = p_t - p_c
    out = {"control": p_c, "treatment": p_t, "abs_lift": diff, "rel_lift": diff / p_c,
           "ci_low": diff - zc * se, "ci_high": diff + zc * se, "z": z, "p_value": 2 * stats.norm.sf(abs(z))}
    return {k: float(v) for k, v in out.items()}


def welch_test(t: np.ndarray, c: np.ndarray, alpha: float = 0.05) -> dict:
    diff = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    res = stats.ttest_ind(t, c, equal_var=False)
    zc = stats.norm.ppf(1 - alpha / 2)
    out = {"control": c.mean(), "treatment": t.mean(), "abs_lift": diff, "rel_lift": diff / c.mean(),
           "ci_low": diff - zc * se, "ci_high": diff + zc * se, "se": se, "p_value": res.pvalue}
    return {k: float(v) for k, v in out.items()}


def bootstrap_diff_ci(t: np.ndarray, c: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for a difference in means (robust to heavy-tailed GMV)."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(t, len(t)).mean() - rng.choice(c, len(c)).mean()
    return tuple(np.quantile(diffs, [alpha / 2, 1 - alpha / 2]))


def cuped(df: pd.DataFrame, metric: str, covariate: str, arm: str = "variant") -> tuple[pd.Series, float]:
    """CUPED: y_adj = y - theta * (x - mean(x)), theta = cov(y, x) / var(x) estimated on pooled data.
    Returns the adjusted metric and the variance reduction achieved."""
    y, x = df[metric].astype(float), df[covariate].astype(float)
    theta = np.cov(y, x)[0, 1] / x.var()
    adj = y - theta * (x - x.mean())
    return adj, float(1 - adj.var() / y.var())


def holm(pvals: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Holm-Bonferroni step-down: returns adjusted p-values."""
    order = pvals.sort_values()
    m = len(order)
    adj = np.maximum.accumulate([(m - i) * p for i, p in enumerate(order.to_numpy())])
    return pd.Series(np.minimum(adj, 1), index=order.index).reindex(pvals.index)


def segment_effects(df: pd.DataFrame, by: str, metric: str = "converted") -> pd.DataFrame:
    rows = []
    for g, d in df.groupby(by):
        t, c = d[d.variant == "treatment"][metric], d[d.variant == "control"][metric]
        r = proportion_test(int(t.sum()), len(t), int(c.sum()), len(c))
        keep = ("control", "treatment", "abs_lift", "rel_lift", "ci_low", "ci_high", "p_value")
        rows.append({"segment": g, "users": len(d), **{k: r[k] for k in keep}})
    out = pd.DataFrame(rows).set_index("segment")
    return out


def daily_effect(df: pd.DataFrame, start: pd.Timestamp, days: int = 28) -> pd.DataFrame:
    """Cumulative conversion by day of experiment for each arm, and the running lift (novelty check)."""
    rows = []
    for d in range(1, days + 1):
        cutoff = start + pd.Timedelta(days=d)
        conv = df.assign(c=(df.first_bill_ts.notna() & (df.first_bill_ts < cutoff)).astype(int)).groupby("variant").c.mean()
        rows.append({"day": d, "control": conv["control"], "treatment": conv["treatment"]})
    out = pd.DataFrame(rows)
    out["abs_lift"] = out.treatment - out.control
    out["incremental_per_day"] = out.abs_lift.diff().fillna(out.abs_lift)
    return out


# ------------------------------------------------------------------ economics
def roi(df: pd.DataFrame, value_per_retained_payer: float = 180.0, revenue_per_bill: float = 4.0) -> dict:
    """Unit economics of the cashback, per 1,000 users exposed to the offer.

    * cost: Rs 25 cashback paid to EVERY first-time payer in treatment (including those who would have paid anyway)
    * benefit: biller commission on incremental bill payments + value of incremental users who keep paying bills
      after the cashback ends (value_per_retained_payer = 12-month commission estimate, an assumption)
    """
    t, c = df[df.variant == "treatment"], df[df.variant == "control"]
    per_k = 1000
    cost = t.cashback_paid.mean() * per_k
    incr_bills = (t.bill_payments.mean() - c.bill_payments.mean()) * per_k
    incr_retained = (t.paid_bill_post_period.mean() - c.paid_bill_post_period.mean()) * per_k
    benefit = incr_bills * revenue_per_bill + incr_retained * value_per_retained_payer
    payers_t = t.converted.mean() * per_k
    incr_payers = (t.converted.mean() - c.converted.mean()) * per_k
    out = {"cashback_cost_per_1k": cost, "payers_per_1k_treatment": payers_t, "incremental_payers_per_1k": incr_payers,
           "subsidy_leakage_pct": 100 * (1 - incr_payers / payers_t),
           "cost_per_incremental_payer": cost / incr_payers if incr_payers > 0 else np.inf,
           "incremental_retained_payers_per_1k": incr_retained,
           "benefit_per_1k": benefit, "net_per_1k": benefit - cost, "roi": (benefit - cost) / cost}
    return {k: round(float(v), 2) for k, v in out.items()}
