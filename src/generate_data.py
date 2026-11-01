"""Synthetic experiment logs for a UPI payments app: "Rs 25 cashback on your first bill payment".

Fully synthetic. 200K eligible users (active on UPI, never paid a bill in-app) randomised 50/50 for 28 days
(3-Nov-2025 to 30-Nov-2025), followed by a 28-day post-period without cashback.

Built-in truths the analysis has to recover:
* overall lift in bill-payment conversion, much larger in Tier-2/3 cities and for newer users, ~0 for Metro iOS users
* a logging bug: iOS app v5.2 failed to record treatment exposures on 7-8 Nov -> sample-ratio mismatch in raw exposure logs
* a novelty effect: the lift is biggest in week 1 and decays
* many treatment payers would have paid anyway (subsidy leakage), and cashback-induced payers retain worse afterwards
* pre-period UPI activity predicts bill GMV (so CUPED reduces variance)

Usage: python src/generate_data.py --out data/raw --seed 41
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXP_START, EXP_DAYS, POST_DAYS = pd.Timestamp("2025-11-03"), 28, 28
BILLERS = {"Mobile Postpaid": (0.30, 550), "Electricity": (0.34, 1600), "Broadband": (0.14, 850),
           "DTH": (0.12, 380), "Piped Gas": (0.06, 700), "Water": (0.04, 450)}
CASHBACK = 25


def generate(seed: int = 41, n_users: int = 200_000) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    tier = rng.choice(["Metro", "Tier 1", "Tier 2", "Tier 3"], n_users, p=[0.34, 0.26, 0.24, 0.16])
    platform = np.where(rng.random(n_users) < np.select([tier == "Metro"], [0.24], 0.08), "iOS", "Android")
    tenure_days = rng.gamma(1.4, 260, n_users).astype(int) + 7
    age_band = rng.choice(["18-24", "25-34", "35-44", "45+"], n_users, p=[0.28, 0.38, 0.22, 0.12])
    activity = rng.lognormal(0, 0.8, n_users)                                   # latent engagement
    pre_txn = rng.poisson(6 * activity)
    pre_gmv = np.round(pre_txn * rng.lognormal(6.0, 0.6, n_users), -1)
    variant = np.where(rng.random(n_users) < 0.5, "treatment", "control")
    users = pd.DataFrame({
        "user_id": [f"U{i:07d}" for i in range(n_users)], "city_tier": tier, "platform": platform,
        "tenure_days": tenure_days, "age_band": age_band, "pre_upi_txn_28d": pre_txn, "pre_upi_gmv_28d": pre_gmv,
    })
    assign = pd.DataFrame({"user_id": users.user_id, "variant": variant,
                           "assigned_ts": EXP_START + pd.to_timedelta(rng.integers(0, 3 * 86400, n_users), unit="s")})

    # ---------------- exposures (first time the bill-pay screen was seen) ----------------
    seen = rng.random(n_users) < 0.86
    expo_day = np.minimum(rng.exponential(5, n_users).astype(int), EXP_DAYS - 1)
    expo_ts = EXP_START + pd.to_timedelta(expo_day * 86400 + rng.integers(8 * 3600, 23 * 3600, n_users), unit="s")
    app_version = np.where(platform == "iOS", rng.choice(["5.1", "5.2"], n_users, p=[0.4, 0.6]),
                           rng.choice(["5.1.3", "5.2.0"], n_users, p=[0.5, 0.5]))
    bug = (platform == "iOS") & (app_version == "5.2") & (variant == "treatment") \
        & expo_ts.normalize().isin(pd.date_range("2025-11-06", "2025-11-10"))
    logged = seen & ~bug
    exposures = pd.DataFrame({"user_id": users.user_id[logged], "variant": variant[logged],
                              "exposure_ts": expo_ts[logged], "platform": platform[logged], "app_version": app_version[logged]})

    # ---------------- outcome: did the user pay a bill during the experiment? ----------------
    base = 0.035 + 0.035 * np.tanh(activity / 2) + np.select([tier == "Metro", tier == "Tier 1"], [0.012, 0.006], 0.0)
    lift = np.select([tier == "Metro", tier == "Tier 1", tier == "Tier 2"], [0.008, 0.017, 0.030], 0.034)
    lift = lift * np.where(tenure_days < 180, 1.5, 0.85) * np.where(platform == "iOS", 0.35, 1.0)
    treat = (variant == "treatment") & seen
    p_organic = base * np.where(seen, 1.0, 0.6)
    organic = rng.random(n_users) < p_organic
    induced = treat & ~organic & (rng.random(n_users) < lift / (1 - p_organic))
    payer = organic | induced
    # timing: induced payers react early (novelty), organic payers spread evenly
    day_org = rng.integers(0, EXP_DAYS, n_users)
    day_ind = np.minimum(expo_day + rng.exponential(3.5, n_users).astype(int), EXP_DAYS - 1)
    pay_day = np.where(induced, day_ind, day_org)
    n_bills = 1 + rng.poisson(0.25 + 0.15 * np.tanh(activity), n_users)

    rows = []
    biller_names = list(BILLERS)
    bp = np.array([v[0] for v in BILLERS.values()])
    for idx in np.flatnonzero(payer):
        for k in range(n_bills[idx]):
            d = min(pay_day[idx] + k * rng.integers(3, 12), EXP_DAYS - 1)
            b = rng.choice(len(biller_names), p=bp)
            amt = round(BILLERS[biller_names[b]][1] * rng.lognormal(0, 0.55) * (0.7 + 0.3 * min(activity[idx], 3)), -1)
            rows.append((users.user_id.iat[idx], EXP_START + pd.Timedelta(days=int(d), seconds=int(rng.integers(7 * 3600, 23 * 3600))),
                         biller_names[b], amt, "experiment"))
    # ---------------- post-period: does the bill-pay habit survive without cashback? ----------------
    keep_p = np.where(induced, 0.38, 0.66)
    stays = payer & (rng.random(n_users) < keep_p)
    for idx in np.flatnonzero(stays):
        d = EXP_DAYS + int(rng.integers(0, POST_DAYS))
        b = rng.choice(len(biller_names), p=bp)
        amt = round(BILLERS[biller_names[b]][1] * rng.lognormal(0, 0.55), -1)
        rows.append((users.user_id.iat[idx], EXP_START + pd.Timedelta(days=int(d), seconds=int(rng.integers(7 * 3600, 23 * 3600))),
                     biller_names[b], amt, "post"))
    bills = pd.DataFrame(rows, columns=["user_id", "payment_ts", "biller_category", "amount", "period"])
    bills = bills.sort_values("payment_ts").reset_index(drop=True)
    bills.insert(0, "payment_id", [f"BP{i:08d}" for i in range(len(bills))])
    # cashback: first successful bill payment in the experiment window, treatment only
    bills = bills.merge(assign[["user_id", "variant"]], on="user_id")
    first = bills[bills.period == "experiment"].groupby("user_id").payment_ts.idxmin()
    bills["cashback"] = 0
    is_first_treat = bills.index.isin(first.values) & (bills.variant == "treatment")
    bills.loc[is_first_treat, "cashback"] = CASHBACK
    # refunds / disputes: a small share, slightly higher for cashback hunters (guardrail)
    bills["status"] = np.where(rng.random(len(bills)) < np.where(bills.cashback > 0, 0.016, 0.012), "Refunded", "Success")
    bills = bills.drop(columns="variant")

    # ---------------- guardrail events ----------------
    uninstall = rng.random(n_users) < 0.021
    tickets = rng.poisson(np.where(variant == "treatment", 0.030, 0.027))
    p2p_after = rng.poisson(6.2 * activity)                     # P2P transfers during the test (no true effect)
    guard = pd.DataFrame({"user_id": users.user_id, "uninstalled_during_test": uninstall.astype(int),
                          "support_tickets": tickets, "p2p_txn_during_test": p2p_after})
    return {"users": users, "assignments": assign, "exposures": exposures, "bill_payments": bills, "guardrails": guard}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--seed", type=int, default=41)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, v in generate(a.seed).items():
        v.to_csv(out / f"{k}.csv", index=False)
        print(f"{k:<14}{len(v):>9,} rows")


if __name__ == "__main__":
    main()
