"""Charts for the experiment readout."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from experiment import ROOT
from viz_style import MUTED, SERIES, TEXT, TEXT_2, apply_style, finish

IMG = ROOT / "images"
apply_style()


def srm_chart(e: pd.DataFrame, path=IMG / "srm_by_slice.png"):
    g = e.groupby(["platform", "app_version"]).variant.apply(lambda s: (s == "treatment").mean()).reset_index()
    g["label"] = g.platform + " " + g.app_version
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = [SERIES[1] if v < 0.45 else MUTED for v in g.variant]
    ax.barh(g.label, g.variant, color=colors, height=0.55)
    ax.axvline(0.5, color=TEXT_2, lw=1)
    for y, v in enumerate(g.variant):
        ax.text(v + 0.005, y, f"{v:.1%}", va="center", fontsize=9, color=TEXT)
    ax.set_xlim(0.3, 0.58)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Treatment share of logged exposures (expected 50%)")
    ax.grid(axis="y", visible=False)
    finish(fig, ax, "SRM traced to one slice: iOS 5.2 lost treatment exposure logs",
           "Exposure logs by platform and app version", path)


def segment_forest(seg: pd.DataFrame, path=IMG / "segment_lift.png"):
    seg = seg.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(seg))
    sig = seg.p_holm < 0.05
    ax.hlines(y, seg.ci_low, seg.ci_high, color=np.where(sig, SERIES[0], MUTED), lw=2.2)
    ax.scatter(seg.abs_lift, y, s=60, color=np.where(sig, SERIES[0], MUTED), zorder=3)
    ax.axvline(0, color=TEXT_2, lw=1)
    for yi, (lift, rel, p) in enumerate(zip(seg.abs_lift, seg.rel_lift, seg.p_holm)):
        ax.text(seg.ci_high.iloc[yi] + 0.001, yi, f"+{lift * 100:.1f} pp ({rel:+.0%})" + ("" if p < 0.05 else "  n.s."),
                va="center", fontsize=8.5, color=TEXT)
    ax.set_yticks(y, seg.index)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_xlim(min(-0.004, seg.ci_low.min() - 0.002), seg.ci_high.max() + 0.014)
    ax.set_xlabel("Absolute lift in bill-payment conversion (95% CI); grey = not significant after Holm correction")
    finish(fig, ax, "The cashback works in Tier-2/3 and for new users, not in Metro",
           "Treatment effect on conversion by segment", path)


def novelty_chart(daily: pd.DataFrame, path=IMG / "novelty.png"):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.bar(daily.day, daily.incremental_per_day, color=SERIES[0], width=0.7)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.0005))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    ax.set_xlabel("Day of experiment")
    ax.set_ylabel("Incremental conversions that day\n(share of users)")
    wk1 = daily.incremental_per_day.iloc[:7].sum() / daily.abs_lift.iloc[-1]
    finish(fig, ax, f"{wk1:.0%} of the total lift arrived in week 1, then it faded",
           "Daily increase in the treatment–control conversion gap (novelty check)", path)


def roi_chart(scen: pd.DataFrame, path=IMG / "rollout_roi.png"):
    scen = scen.sort_values("net_per_1k")
    fig, ax = plt.subplots(figsize=(8.5, 4))
    colors = [SERIES[0] if v > 0 else SERIES[1] for v in scen.net_per_1k]
    ax.barh(scen.index, scen.net_per_1k, color=colors, height=0.55)
    ax.axvline(0, color=TEXT_2, lw=1)
    for y, (v, r) in enumerate(zip(scen.net_per_1k, scen.roi)):
        ax.text(v + (25 if v >= 0 else -25), y, f"Rs {v:+,.0f}  (ROI {r:+.0%})", va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=TEXT)
    ax.set_xlim(scen.net_per_1k.min() * 1.45, scen.net_per_1k.max() * 2.4)
    ax.set_xlabel("Net value per 1,000 users offered the cashback (Rs)")
    ax.grid(axis="y", visible=False)
    finish(fig, ax, "Ship it to Tier-2/3 only: a full rollout loses money",
           "Cashback cost vs value of incremental (and retained) bill payers, by rollout scenario", path)


def cuped_chart(rows: pd.DataFrame, path=IMG / "cuped.png"):
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    y = np.arange(len(rows))
    ax.barh(y + 0.18, rows.ci_width_raw, height=0.34, color=MUTED, label="Raw")
    ax.barh(y - 0.18, rows.ci_width_cuped, height=0.34, color=SERIES[0], label="CUPED-adjusted")
    for yi, vr in enumerate(rows.variance_reduction):
        ax.text(rows.ci_width_raw.iloc[yi] * 1.02, yi + 0.18, f"variance −{vr:.0%}", va="center", fontsize=9, color=TEXT)
    ax.set_yticks(y, rows.index)
    ax.set_xlabel("Width of the 95% confidence interval, relative to the raw interval")
    ax.set_xlim(0, 1.35)
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    finish(fig, ax, "CUPED shrinks intervals where pre-period data predicts the metric",
           "Pre-period UPI activity as covariate", path)
