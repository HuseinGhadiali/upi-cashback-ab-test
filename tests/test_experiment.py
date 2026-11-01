import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import experiment as X  # noqa: E402
from generate_data import generate  # noqa: E402


def test_srm_detects_imbalance():
    assert not X.srm_test({"t": 50_000, "c": 50_100})["srm_detected"]
    assert X.srm_test({"t": 49_000, "c": 51_000})["srm_detected"]


def test_proportion_test_matches_statsmodels_style_math():
    r = X.proportion_test(600, 10_000, 500, 10_000)
    assert r["abs_lift"] == pytest.approx(0.01)
    assert r["ci_low"] < 0.01 < r["ci_high"]
    assert 0 < r["p_value"] < 0.01


def test_power_and_sample_size_are_consistent():
    n = X.sample_size_per_arm(0.05, 0.005)
    mde = X.mde_for_power(0.05, n)
    assert mde == pytest.approx(0.005, rel=0.05)


def test_holm_adjustment():
    p = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03})
    adj = X.holm(p)
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert (adj >= p).all()


def test_cuped_reduces_variance_and_keeps_effect():
    rng = np.random.default_rng(0)
    x = rng.normal(10, 3, 20_000)
    arm = np.where(rng.random(20_000) < 0.5, "treatment", "control")
    y = 2 * x + rng.normal(0, 1, 20_000) + np.where(arm == "treatment", 0.5, 0)
    df = pd.DataFrame({"y": y, "x": x, "variant": arm})
    adj, vr = X.cuped(df, "y", "x")
    assert vr > 0.9
    eff = adj[df.variant == "treatment"].mean() - adj[df.variant == "control"].mean()
    assert eff == pytest.approx(0.5, abs=0.05)


def test_bootstrap_ci_covers_difference():
    rng = np.random.default_rng(1)
    t, c = rng.exponential(110, 5000), rng.exponential(100, 5000)
    lo, hi = X.bootstrap_diff_ci(t, c, n_boot=500)
    assert lo < t.mean() - c.mean() < hi


def test_generated_experiment_end_to_end(tmp_path):
    for k, v in generate(seed=3, n_users=30_000).items():
        v.to_csv(tmp_path / f"{k}.csv", index=False)
    df = X.user_metrics(tmp_path)
    assert len(df) == 30_000 and df.user_id.is_unique
    assert (df.loc[df.variant == "control", "cashback_paid"] == 0).all()        # control never gets cashback
    assert (df.cashback_paid <= 25).all()                                        # only the first bill
    assert X.balance_table(df, ["city_tier", "pre_upi_txn_28d"]).balanced.all()
    r = X.roi(df)
    assert 0 <= r["subsidy_leakage_pct"] <= 100
