import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Experiment Readout: ₹25 Cashback on the First Bill Payment (UPI app)\n"
       "**Hypothesis:** a ₹25 cashback on the first in-app bill payment converts UPI-only users into bill payers, and the habit sticks.\n\n"
       "**Design:** 200,000 eligible users (active on UPI, never paid a bill in-app), user-level 50/50 randomisation, "
       "28 days (3–30 Nov 2025) plus a 28-day post-period with no cashback.\n\n"
       "| Metric type | Metric |\n|---|---|\n| Primary | Bill-payment conversion within 28 days |\n"
       "| Secondary | Bill payments per user, bill GMV per user, still paying bills in the post-period (habit) |\n"
       "| Guardrails | Uninstalls, support tickets, refunds, P2P transfers (cannibalisation) |\n\n"
       "> Synthetic data: `src/generate_data.py` (seed 41). Metrics built in SQL: `sql/user_metrics.sql`."),
    code("import sys, warnings\nsys.path.insert(0, '../src')\nwarnings.filterwarnings('ignore')\nimport pandas as pd\n"
         "from IPython.display import Image, display\nfrom experiment import *\nimport charts\n"
         "df = user_metrics()\nprint(df.shape)\ndf.head(3)"),
    md("## 0. Pre-registration: power\nThe historical 28-day conversion is about 5.5%. With 100K users per arm the test detects:"),
    code("print(f\"MDE at 80% power, alpha 0.05: {mde_for_power(0.055, 100_000):.2%} absolute\")\n"
         "print(f\"Users per arm needed for a 0.5 pp MDE: {sample_size_per_arm(0.055, 0.005):,}\")"),
    md("## 1. Validity checks\n### 1a. Sample-ratio mismatch (SRM)"),
    code("print('Assignment table:', srm_test(df.variant.value_counts().to_dict()))\n"
         "e = df[df.exposure_logged == 1]\nprint('Exposure logs:   ', srm_test(e.variant.value_counts().to_dict()))"),
    md("Assignment is balanced, but the **exposure logs fail SRM**. Slicing by platform and app version localises the problem:"),
    code("charts.srm_chart(e)\ndisplay(e.groupby(['platform', 'app_version', 'variant']).size().unstack())\n"
         "display(Image('../images/srm_by_slice.png'))"),
    code("clean = e[~((e.platform == 'iOS') & (e.app_version == '5.2'))]\nsrm_test(clean.variant.value_counts().to_dict())"),
    md("**Diagnosis:** iOS v5.2 dropped treatment exposure events on 6–10 Nov (a client logging bug, not a randomisation bug). "
       "Excluding that slice restores the 50/50 ratio.\n\n**Decision:** the **primary analysis is intention-to-treat on the assignment table**, "
       "which does not depend on exposure logging. Exposure-based (triggered) numbers are reported only with iOS 5.2 excluded from both arms."),
    md("### 1b. Covariate balance (pre-period)"),
    code("bal = balance_table(df, ['city_tier', 'platform', 'age_band', 'tenure_group', 'pre_upi_txn_28d', 'pre_upi_gmv_28d'])\n"
         "print('All |SMD| < 0.1:', bal.balanced.all())\nbal"),
    md("## 2. Primary metric: conversion"),
    code("t, c = df[df.variant == 'treatment'], df[df.variant == 'control']\n"
         "primary = proportion_test(int(t.converted.sum()), len(t), int(c.converted.sum()), len(c))\npd.Series(primary).round(5)"),
    md("## 3. Secondary metrics\nBill GMV is heavy-tailed and mostly zero, so a bootstrap CI is reported next to Welch's t-test."),
    code("rows = {}\nfor m in ['bill_payments', 'bill_gmv', 'paid_bill_post_period']:\n"
         "    r = welch_test(t[m].to_numpy(float), c[m].to_numpy(float))\n"
         "    if m == 'bill_gmv':\n        r['boot_ci_low'], r['boot_ci_high'] = bootstrap_diff_ci(t[m].to_numpy(float), c[m].to_numpy(float))\n"
         "    rows[m] = r\npd.DataFrame(rows).T.round(4)"),
    md("## 4. CUPED variance reduction\nAdjust each metric with its pre-period counterpart. CUPED helps a lot when the covariate predicts the metric "
       "(P2P transfers), and barely helps for a sparse first-time-behaviour metric (conversion), which is itself a useful finding."),
    code("res = {}\nfor metric, cov in [('p2p_txn_during_test', 'pre_upi_txn_28d'), ('bill_gmv', 'pre_upi_gmv_28d'), ('converted', 'pre_upi_txn_28d')]:\n"
         "    raw = welch_test(t[metric].to_numpy(float), c[metric].to_numpy(float))\n"
         "    adj, vr = cuped(df, metric, cov)\n"
         "    a = welch_test(adj[t.index].to_numpy(), adj[c.index].to_numpy())\n"
         "    res[metric] = {'variance_reduction': vr, 'lift_raw': raw['abs_lift'], 'p_raw': raw['p_value'], 'lift_cuped': a['abs_lift'],\n"
         "                   'p_cuped': a['p_value'], 'ci_width_raw': 1.0, 'ci_width_cuped': a['se'] / raw['se']}\n"
         "cup = pd.DataFrame(res).T\ncharts.cuped_chart(cup)\ndisplay(cup.round(4))\ndisplay(Image('../images/cuped.png'))"),
    md("## 5. Who does it work for? Segments with Holm correction\nMultiple segments mean multiple tests, so p-values are Holm-adjusted across every segment shown."),
    code("seg = pd.concat([segment_effects(df, 'city_tier'), segment_effects(df, 'tenure_group'), segment_effects(df, 'platform')])\n"
         "seg['p_holm'] = holm(seg.p_value)\ncharts.segment_forest(seg)\ndisplay(seg.round(4))\ndisplay(Image('../images/segment_lift.png'))"),
    md("## 6. Novelty: is the effect front-loaded?"),
    code("daily = daily_effect(df, pd.Timestamp('2025-11-03'))\ncharts.novelty_chart(daily)\ndisplay(Image('../images/novelty.png'))"),
    md("## 7. Guardrails"),
    code("g = {m: welch_test(t[m].to_numpy(float), c[m].to_numpy(float)) for m in ['uninstalled_during_test', 'support_tickets', 'refunds', 'p2p_txn_during_test']}\n"
         "pd.DataFrame(g).T[['control', 'treatment', 'rel_lift', 'p_value']].round(4)"),
    md("## 8. Does it pay? ROI by rollout scenario\nCost = ₹25 to *every* first-time treatment payer, including those who would have paid anyway (subsidy leakage). "
       "Benefit = biller commission on incremental bills (₹4 each, an assumption) plus the 12-month value of incremental users still paying bills after the "
       "cashback ends (₹180 each, an assumption)."),
    code("scen = {'All users': df, 'Metro': df[df.city_tier == 'Metro'], 'Tier 1': df[df.city_tier == 'Tier 1'],\n"
         "        'Tier 2': df[df.city_tier == 'Tier 2'], 'Tier 3': df[df.city_tier == 'Tier 3'],\n"
         "        'Tier 2 + 3 (recommended)': df[df.city_tier.isin(['Tier 2', 'Tier 3'])]}\n"
         "roi_tbl = pd.DataFrame({k: roi(v) for k, v in scen.items()}).T\ncharts.roi_chart(roi_tbl)\nroi_tbl.to_csv('../reports/rollout_roi.csv')\n"
         "display(roi_tbl)\ndisplay(Image('../images/rollout_roi.png'))"),
    code("# sensitivity: how valuable must a retained bill payer be for a FULL rollout to break even?\n"
         "pd.Series({v: roi(df, value_per_retained_payer=v)['roi'] for v in (180, 240, 300, 360)}, name='roi_all_users')"),
    md("## Decision\n**Do not ship to everyone. Ship to Tier-2/3 users**, and re-test Metro with a smaller incentive. "
       "Fix iOS 5.2 exposure logging before the next test, and watch the support-ticket guardrail. See the README for the summary."),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "experiment_readout.ipynb")
