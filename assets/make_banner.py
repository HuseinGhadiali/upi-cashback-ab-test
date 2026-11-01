"""Render assets/banner.png: bootstrap distributions of conversion for control vs treatment (python assets/make_banner.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from banner import glow_line, make_banner  # noqa: E402
from experiment import user_metrics  # noqa: E402

df = user_metrics()
rng = np.random.default_rng(0)
dists = {}
for arm in ("control", "treatment"):
    y = df.loc[df.variant == arm, "converted"].to_numpy()
    dists[arm] = np.array([rng.choice(y, len(y)).mean() for _ in range(600)])


def art(ax, acc):
    lo, hi = min(d.min() for d in dists.values()) - 0.002, max(d.max() for d in dists.values()) + 0.002
    xs = np.linspace(lo, hi, 400)
    for i, (arm, d) in enumerate(dists.items()):
        bw = d.std() * 0.6
        dens = np.exp(-((xs[:, None] - d[None, :]) ** 2) / (2 * bw ** 2)).sum(1)
        dens = dens / dens.max()
        color = acc[0] if i == 0 else acc[3]
        ax.fill_between(xs, 0, dens, color=color, alpha=0.18, lw=0)
        glow_line(ax, xs, dens, color, lw=2.2)
        ax.text(d.mean(), 1.08, f"{arm}\n{d.mean():.2%}", color=color, ha="center", fontsize=11, fontweight="bold")
    # tiny 'user' dots underneath: each dot a bootstrap replicate
    for i, d in enumerate(dists.values()):
        ax.scatter(d, -0.12 - 0.1 * i + rng.normal(0, 0.02, len(d)), s=4, color=acc[0] if i == 0 else acc[3], alpha=0.5, lw=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.4, 1.35)
    ax.text(lo, -0.38, "bootstrap distributions of 28-day bill-payment conversion, 100K users per arm",
            color="white", alpha=0.55, fontsize=9)


make_banner(Path(__file__).parent / "banner.png", "UPI Cashback A/B Test",
            "SRM · CUPED · segments · novelty · ROI decision",
            ["Experimentation", "SciPy stats", "DuckDB SQL", "Product analytics"], art, theme="aurora",
            emoji_free_kicker="Fintech · 200K-user experiment")
