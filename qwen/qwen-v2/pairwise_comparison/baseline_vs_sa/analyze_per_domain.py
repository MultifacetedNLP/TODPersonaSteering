"""
Per-domain comparison: Baseline vs SA (SA) — Qwen2.5-7B-Instruct

Combines:
  - Standard evaluation metrics (from evaluation/baseline & evaluation/sa)
  - Pairwise LLM-judge results (from pairwise_comparison/baseline_vs_sa)

Domains: Movies_1_Movies_3, Restaurants_2
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT   = Path(__file__).resolve().parents[4]   # Activation-Steered-Personas
QV2       = PROJECT / "qwen" / "qwen-v2"
EVAL_BASE = QV2 / "evaluation" / "baseline" / "baseline"
EVAL_SA   = QV2 / "evaluation" / "sa"   / "sa"
PAIR_DIR  = QV2 / "pairwise_comparison" / "baseline_vs_sa"
OUT       = PAIR_DIR / "analysis_output_baseline_vs_sa"
OUT.mkdir(exist_ok=True)

DOMAINS     = ["Movies_1_Movies_3", "Restaurants_2"]
DOM_LABELS  = {"Movies_1_Movies_3": "Movies", "Restaurants_2": "Restaurants"}
METRICS_STD = ["method_accuracy", "full_api_accuracy", "successful_dialogs_rate",
               "dialog_completion_rate", "inform_accuracy", "bleu"]
METRIC_NICE = {
    "method_accuracy":        "Method Accuracy",
    "full_api_accuracy":      "Full API Accuracy",
    "successful_dialogs_rate":"Dialog Success Rate",
    "dialog_completion_rate": "Dialog Completion Rate",
    "inform_accuracy":        "Inform Accuracy",
    "bleu":                   "BLEU",
}
PAIR_METRICS = ["usersat", "constraint", "truthfulness"]
PALETTE      = {"baseline": "#4C72B0", "sa": "#DD8452", "tie": "#8c8c8c"}

# ── load standard eval ─────────────────────────────────────────────────────────
std = {}
for domain in DOMAINS:
    base_f = EVAL_BASE / domain / "metrics.json"
    sa_f   = EVAL_SA   / domain / "metrics.json"
    with open(base_f) as f: base_d = json.load(f)["summary"]
    with open(sa_f)   as f: sa_d   = json.load(f)["summary"]
    std[domain] = {"baseline": base_d, "sa": sa_d}

# ── load pairwise ──────────────────────────────────────────────────────────────
pair_summaries = {}
pair_dialogs   = {}
for m in PAIR_METRICS:
    with open(PAIR_DIR / m / "summary.json") as f:
        pair_summaries[m] = json.load(f)
    rows = []
    for fpath in (PAIR_DIR / m).rglob("dialog_*.json"):
        with open(fpath) as f: rows.append(json.load(f))
    pair_dialogs[m] = rows

# ── helpers ────────────────────────────────────────────────────────────────────
def pct(x): return f"{x*100:.1f}%"
def pp(a, b): return (b - a) * 100

def pair_stats(metric, domain=None):
    """Return (win_a%, win_b%, tie%, n) for pairwise metric, optionally filtered by domain."""
    s = pair_summaries[metric]
    if domain:
        s = s.get("per_domain", {}).get(domain, {})
    return (s.get("win_rate_a", 0)*100,
            s.get("win_rate_b", 0)*100,
            s.get("tie_rate",   0)*100,
            s.get("n_pairs",    0))

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("BASELINE vs SA — PER-DOMAIN COMPARISON  (Qwen2.5-7B-Instruct)")
print("=" * 70)

# ── 1. Standard eval table ────────────────────────────────────────────────────
print("\n── 1. STANDARD EVALUATION METRICS ──────────────────────────────────────")
print(f"\n{'Metric':<26} {'Movies BASE':>12} {'Movies SA':>10} {'Δ':>7}"
      f" │ {'Rest BASE':>10} {'Rest SA':>9} {'Δ':>7}")
print("-" * 80)
for m in METRICS_STD:
    mv_b = std["Movies_1_Movies_3"]["baseline"][m]
    mv_p = std["Movies_1_Movies_3"]["sa"][m]
    rs_b = std["Restaurants_2"]["baseline"][m]
    rs_p = std["Restaurants_2"]["sa"][m]
    d_mv = pp(mv_b, mv_p)
    d_rs = pp(rs_b, rs_p)
    flag_mv = "▲" if d_mv > 0 else ("▼" if d_mv < 0 else " ")
    flag_rs = "▲" if d_rs > 0 else ("▼" if d_rs < 0 else " ")
    print(f"{METRIC_NICE[m]:<26} {pct(mv_b):>12} {pct(mv_p):>10} "
          f"{flag_mv}{d_mv:>+5.1f}pp"
          f" │ {pct(rs_b):>10} {pct(rs_p):>9} {flag_rs}{d_rs:>+5.1f}pp")

# ── 2. Pairwise table ─────────────────────────────────────────────────────────
print("\n── 2. PAIRWISE JUDGE RESULTS ────────────────────────────────────────────")
print(f"\n{'Metric':<14} │ {'Movies':^38} │ {'Restaurants':^38}")
print(f"{'':14} │ {'BASE win%':>10} {'SA win%':>10} {'tie%':>8} {'Δ':>8} │"
      f" {'BASE win%':>10} {'SA win%':>10} {'tie%':>8} {'Δ':>8}")
print("-" * 96)
for m in PAIR_METRICS:
    mv = pair_stats(m, "Movies_1_Movies_3")
    rs = pair_stats(m, "Restaurants_2")
    d_mv = mv[1] - mv[0]
    d_rs = rs[1] - rs[0]
    print(f"{m:<14} │ {mv[0]:>10.1f}% {mv[1]:>10.1f}% {mv[2]:>8.1f}% {d_mv:>+7.1f}pp │"
          f" {rs[0]:>10.1f}% {rs[1]:>10.1f}% {rs[2]:>8.1f}% {d_rs:>+7.1f}pp")

# ── 3. Per-domain narrative ───────────────────────────────────────────────────
print("\n── 3. DOMAIN NARRATIVE ──────────────────────────────────────────────────")
for domain in DOMAINS:
    d_label = DOM_LABELS[domain]
    b  = std[domain]["baseline"]
    sa = std[domain]["sa"]
    print(f"\n  [{d_label}]")
    for m in METRICS_STD:
        delta = pp(b[m], sa[m])
        arrow = "▲" if delta > 0.05 else ("▼" if delta < -0.05 else "→")
        print(f"    {arrow} {METRIC_NICE[m]:<28} "
              f"BASE={pct(b[m])}  SA={pct(sa[m])}  Δ={delta:>+5.1f}pp")
    print(f"    ── Pairwise ──")
    for m in PAIR_METRICS:
        wa, wb, wt, n = pair_stats(m, domain)
        delta = wb - wa
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "→")
        print(f"    {arrow} {m:<28} BASE={wa:.1f}%  SA={wb:.1f}%  tie={wt:.1f}%  Δ={delta:>+5.1f}pp  n={n}")

# ── 4. Metric-level delta heatmap data ───────────────────────────────────────
print("\n── 4. DELTA SUMMARY TABLE (SA − Baseline, all metrics) ─────────────────")
print(f"\n{'Metric':<30} {'Movies Δ':>12} {'Restaurants Δ':>15} {'Domain gap':>12}")
print("-" * 70)
all_deltas = {}
for m in METRICS_STD:
    d_mv = pp(std["Movies_1_Movies_3"]["baseline"][m],
              std["Movies_1_Movies_3"]["sa"][m])
    d_rs = pp(std["Restaurants_2"]["baseline"][m],
              std["Restaurants_2"]["sa"][m])
    all_deltas[METRIC_NICE[m]] = (d_mv, d_rs)
    print(f"{METRIC_NICE[m]:<30} {d_mv:>+11.2f}pp {d_rs:>+14.2f}pp {d_rs-d_mv:>+11.2f}pp")
for m in PAIR_METRICS:
    mv_a, mv_b, *_ = pair_stats(m, "Movies_1_Movies_3")
    rs_a, rs_b, *_ = pair_stats(m, "Restaurants_2")
    d_mv = mv_b - mv_a
    d_rs = rs_b - rs_a
    all_deltas[m] = (d_mv, d_rs)
    print(f"{m:<30} {d_mv:>+11.2f}pp {d_rs:>+14.2f}pp {d_rs-d_mv:>+11.2f}pp")

# ══════════════════════════════════════════════════════════════════════════════
print("\n── PLOTS ────────────────────────────────────────────────────────────────")
sns.set_theme(style="whitegrid", font_scale=1.0)

# ── Plot 1. Side-by-side: standard metrics per domain ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
for ax, domain in zip(axes, DOMAINS):
    d_label = DOM_LABELS[domain]
    b_vals  = [std[domain]["baseline"][m] * 100 for m in METRICS_STD]
    sa_vals = [std[domain]["sa"][m]       * 100 for m in METRICS_STD]
    x   = np.arange(len(METRICS_STD))
    w   = 0.35
    bars_b  = ax.bar(x - w/2, b_vals,  w, label="baseline", color=PALETTE["baseline"], edgecolor="white")
    bars_sa = ax.bar(x + w/2, sa_vals, w, label="sa",       color=PALETTE["sa"],       edgecolor="white")
    for bar, v in list(zip(bars_b, b_vals)) + list(zip(bars_sa, sa_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_NICE[m].replace(" ", "\n") for m in METRICS_STD], fontsize=7)
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"{d_label} — Standard Metrics")
    ax.set_ylim(0, 80)
    ax.legend(fontsize=8)
fig.suptitle("Baseline vs SA — Standard Evaluation by Domain", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "domain_01_standard_metrics.png", dpi=150)
plt.close(fig)
print("  Saved: domain_01_standard_metrics.png")

# ── Plot 2. Delta bars (SA − Baseline) for all metrics, both domains ──────────
metric_names = list(all_deltas.keys())
mv_deltas    = [all_deltas[n][0] for n in metric_names]
rs_deltas    = [all_deltas[n][1] for n in metric_names]
x2 = np.arange(len(metric_names))
w2 = 0.35
fig, ax = plt.subplots(figsize=(13, 5))
bars_mv = ax.bar(x2 - w2/2, mv_deltas, w2, label="Movies",
                 color="#2196F3", edgecolor="white", alpha=0.85)
bars_rs = ax.bar(x2 + w2/2, rs_deltas, w2, label="Restaurants",
                 color="#FF9800", edgecolor="white", alpha=0.85)
for bar, v in list(zip(bars_mv, mv_deltas)) + list(zip(bars_rs, rs_deltas)):
    yoff = 0.15 if v >= 0 else -0.6
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + yoff,
            f"{v:+.1f}", ha="center", va="bottom", fontsize=7)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x2)
ax.set_xticklabels(metric_names, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("SA − Baseline (pp)")
ax.set_title("SA Advantage per Metric × Domain  (positive = SA wins)")
ax.legend()
# shade pairwise vs standard
ax.axvspan(len(METRICS_STD) - 0.5, len(metric_names) - 0.5,
           alpha=0.07, color="purple", label="pairwise metrics")
ax.text(len(METRICS_STD) + 0.5*(len(PAIR_METRICS)-1), ax.get_ylim()[1]*0.92,
        "◀ pairwise (LLM judge)", ha="center", fontsize=8, color="purple")
fig.tight_layout()
fig.savefig(OUT / "domain_02_delta_all_metrics.png", dpi=150)
plt.close(fig)
print("  Saved: domain_02_delta_all_metrics.png")

# ── Plot 3. Pairwise winner stacked bars — Movies vs Restaurants ──────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, m in zip(axes, PAIR_METRICS):
    cats = ["Movies", "Restaurants", "Overall"]
    rows = pair_dialogs[m]
    a_pct, b_pct, tie_pct = [], [], []
    for cat in cats:
        if cat == "Overall":
            sub = rows
        else:
            sub = [r for r in rows if (("Movie" in r.get("domain",""))
                   if cat == "Movies" else ("Restaurant" in r.get("domain","")))]
        n = len(sub)
        a_pct.append(   sum(1 for r in sub if (r.get("winner") or "").lower() == "a") / n * 100 if n else 0)
        b_pct.append(   sum(1 for r in sub if (r.get("winner") or "").lower() == "b") / n * 100 if n else 0)
        tie_pct.append( sum(1 for r in sub if (r.get("winner") or "").lower() == "tie") / n * 100 if n else 0)
    x3 = np.arange(3)
    bottom_b   = np.array(a_pct)
    bottom_tie = bottom_b + np.array(b_pct)
    b1 = ax.bar(x3, a_pct,   0.5, label="baseline", color=PALETTE["baseline"], edgecolor="white")
    b2 = ax.bar(x3, b_pct,   0.5, bottom=bottom_b,   label="sa",       color=PALETTE["sa"],       edgecolor="white")
    b3 = ax.bar(x3, tie_pct, 0.5, bottom=bottom_tie, label="tie",      color=PALETTE["tie"],      edgecolor="white")
    # label each segment if > 3%
    for xi, (av, bv, tv) in enumerate(zip(a_pct, b_pct, tie_pct)):
        if av > 3:   ax.text(xi, av/2,            f"{av:.0f}%",  ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        if bv > 3:   ax.text(xi, av + bv/2,       f"{bv:.0f}%",  ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        if tv > 5:   ax.text(xi, av + bv + tv/2,  f"{tv:.0f}%",  ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(x3); ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylim(0, 100); ax.set_ylabel("% of dialogs")
    ax.set_title(m)
    ax.legend(fontsize=7, loc="upper right")
fig.suptitle("Pairwise Winner Distribution — Baseline vs SA, by Domain", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "domain_03_pairwise_winners.png", dpi=150)
plt.close(fig)
print("  Saved: domain_03_pairwise_winners.png")

# ── Plot 4. Radar / spider chart — Movies vs Restaurants ──────────────────────
# Normalise each metric to [0,1] range across the 4 values (base_mv, sa_mv, base_rs, sa_rs)
radar_metrics = METRICS_STD + PAIR_METRICS
labels = [METRIC_NICE.get(m, m) for m in radar_metrics]
N = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), subplot_kw=dict(polar=True))
for ax, domain in zip(axes, DOMAINS):
    d_label = DOM_LABELS[domain]
    b_vals_raw  = [std[domain]["baseline"][m] for m in METRICS_STD]
    sa_vals_raw = [std[domain]["sa"][m]       for m in METRICS_STD]
    # add pairwise win_rate_b as proxy for SA quality
    wa, wb, wt, _ = zip(*[pair_stats(m, domain) for m in PAIR_METRICS])
    b_vals_raw  += [w/100 for w in wa]
    sa_vals_raw += [w/100 for w in wb]

    # close the polygon
    b_vals  = b_vals_raw  + b_vals_raw[:1]
    sa_vals = sa_vals_raw + sa_vals_raw[:1]

    ax.plot(angles, b_vals,  "o-", color=PALETTE["baseline"], linewidth=1.5, label="baseline")
    ax.fill(angles, b_vals,  alpha=0.15, color=PALETTE["baseline"])
    ax.plot(angles, sa_vals, "s-", color=PALETTE["sa"],       linewidth=1.5, label="sa")
    ax.fill(angles, sa_vals, alpha=0.15, color=PALETTE["sa"])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=7)
    ax.set_title(d_label, size=11, pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
fig.suptitle("Baseline vs SA — Radar Profile by Domain", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "domain_04_radar.png", dpi=150)
plt.close(fig)
print("  Saved: domain_04_radar.png")

# ── Plot 5. Heatmap: SA − Baseline delta, metric × domain ────────────────────
import pandas as pd
delta_data = {}
for m in METRICS_STD:
    delta_data[METRIC_NICE[m]] = {
        "Movies":      pp(std["Movies_1_Movies_3"]["baseline"][m], std["Movies_1_Movies_3"]["sa"][m]),
        "Restaurants": pp(std["Restaurants_2"]["baseline"][m],     std["Restaurants_2"]["sa"][m]),
    }
for m in PAIR_METRICS:
    mv_a, mv_b, *_ = pair_stats(m, "Movies_1_Movies_3")
    rs_a, rs_b, *_ = pair_stats(m, "Restaurants_2")
    delta_data[m] = {"Movies": mv_b - mv_a, "Restaurants": rs_b - rs_a}

df_delta = pd.DataFrame(delta_data).T
fig, ax = plt.subplots(figsize=(6, 7))
sns.heatmap(df_delta, annot=True, fmt=".1f", center=0,
            cmap="RdYlGn", linewidths=0.5, ax=ax,
            cbar_kws={"label": "SA − Baseline (pp)"})
ax.set_title("SA Advantage (pp) per Metric × Domain\n(Qwen — Baseline vs SA)")
ax.set_xlabel("Domain")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(OUT / "domain_05_heatmap.png", dpi=150)
plt.close(fig)
print("  Saved: domain_05_heatmap.png")

# ── final summary ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)

for domain in DOMAINS:
    d_label = DOM_LABELS[domain]
    b = std[domain]["baseline"]
    p = std[domain]["sa"]
    gains   = {m: pp(b[m], p[m]) for m in METRICS_STD}
    winners = [m for m, d in gains.items() if d > 0.05]
    losers  = [m for m, d in gains.items() if d < -0.05]
    print(f"\n  {d_label}:")
    print(f"    SA improves: {', '.join(METRIC_NICE[m] for m in winners) or 'none'}")
    print(f"    SA hurts:    {', '.join(METRIC_NICE[m] for m in losers)  or 'none'}")
    wa, wb, wt, n = pair_stats("usersat", domain)
    print(f"    Pairwise usersat: SA +{wb-wa:.1f}pp  (BASE={wa:.1f}% SA={wb:.1f}% tie={wt:.1f}%)")
    wa, wb, wt, n = pair_stats("truthfulness", domain)
    print(f"    Pairwise truth:   SA +{wb-wa:.1f}pp  (BASE={wa:.1f}% SA={wb:.1f}% tie={wt:.1f}%)")
    wa, wb, wt, n = pair_stats("constraint", domain)
    print(f"    Pairwise constr:  SA {wb-wa:+.1f}pp  (BASE={wa:.1f}% SA={wb:.1f}% tie={wt:.1f}%)")

print(f"\n  All plots saved to: {OUT}/\n")
