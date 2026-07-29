"""
Analysis of Baseline vs SA (SA) pairwise comparison results.

Conditions:
  A = baseline (no personality, no user steering)
  B = sa      (SA only, no user steering)

Metrics: usersat, constraint, truthfulness
Coverage: 50 dialog pairs (25 Movies, 25 Restaurants) — no trait/scalar variation.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent
METRICS = ["usersat", "constraint", "truthfulness"]
OUT     = BASE / "analysis_output_baseline_vs_sa"
OUT.mkdir(exist_ok=True)

COND_A  = "baseline"
COND_B  = "sa"
PALETTE = {COND_A: "#4C72B0", COND_B: "#DD8452", "tie": "#8c8c8c"}

# ── load ───────────────────────────────────────────────────────────────────────
summaries = {}
dialogs   = {}   # metric -> list of per-dialog dicts

for m in METRICS:
    with open(BASE / m / "summary.json") as f:
        summaries[m] = json.load(f)
    rows = []
    for fpath in (BASE / m).rglob("dialog_*.json"):
        with open(fpath) as f:
            d = json.load(f)
        d["metric"] = m
        rows.append(d)
    dialogs[m] = rows

print(f"Loaded: { {m: len(dialogs[m]) for m in METRICS} }")

def fmt(x):
    return f"{x*100:.1f}%"

def adv(s):
    return (s["win_rate_b"] - s["win_rate_a"]) * 100

# ── 1. Overall summary ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("1. OVERALL WIN RATES BY METRIC")
print("="*65)
print(f"{'Metric':<14} {'A(baseline)':>12} {'B(sa)':>10} {'tie':>8} {'n':>6} {'B−A Δ':>8} {'Winner'}")
print("-"*65)
for m in METRICS:
    s = summaries[m]
    winner = COND_B if s["win_rate_b"] > s["win_rate_a"] else (COND_A if s["win_rate_a"] > s["win_rate_b"] else "tie")
    delta  = adv(s)
    print(f"{m:<14} {fmt(s['win_rate_a']):>12} {fmt(s['win_rate_b']):>10} "
          f"{fmt(s['tie_rate']):>8} {s['n_pairs']:>6} {delta:>+7.1f}pp  {winner}")

# ── 2. Per-domain ──────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("2. PER-DOMAIN BREAKDOWN")
print("="*65)
for m in METRICS:
    print(f"\n  [{m}]")
    for domain, s in sorted(summaries[m].get("per_domain", {}).items()):
        d_short = "Movies" if "Movie" in domain else "Restaurants"
        delta   = adv(s)
        print(f"    {d_short:<14} A={fmt(s['win_rate_a'])} ({s['wins_a']})  "
              f"B={fmt(s['win_rate_b'])} ({s['wins_b']})  "
              f"tie={fmt(s['tie_rate'])} ({s['ties']})  "
              f"n={s['n_pairs']}  Δ={delta:+.1f}pp")

# ── 3. Winner distribution from dialog files ───────────────────────────────────
print("\n" + "="*65)
print("3. WINNER DISTRIBUTION FROM INDIVIDUAL FILES")
print("="*65)
for m in METRICS:
    winners = [d["winner"].lower() if d.get("winner") else "none" for d in dialogs[m]]
    c = Counter(winners)
    total = sum(c.values())
    print(f"\n  [{m}]  total={total}")
    for label in ["a", "b", "tie", "none"]:
        n = c.get(label, 0)
        bar = "█" * int(n / total * 40)
        print(f"    {label:<6} {n:3d}  ({n/total*100:5.1f}%)  {bar}")

# ── 4. Reason analysis ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("4. JUDGE REASONS FOR NON-TIE VERDICTS")
print("="*65)
for m in METRICS:
    non_ties = [d for d in dialogs[m] if (d.get("winner") or "").lower() not in ("tie", "none", "")]
    print(f"\n  [{m}]  ({len(non_ties)} non-tie verdicts)")
    for d in non_ties:
        w = (d["winner"] or "").upper()
        dom = "Movies" if "Movie" in d.get("domain","") else "Restaurants"
        print(f"    Winner={w} ({dom})  dialog={d['dialog_id']}")
        print(f"      Reason: {d.get('reason','')[:120]}")

# ── 5. Constraint slot counts ──────────────────────────────────────────────────
print("\n" + "="*65)
print("5. CONSTRAINT SLOT COUNTS (count_a vs count_b)")
print("="*65)
slot_rows = [d for d in dialogs["constraint"] if d.get("count_a") is not None]
print(f"  Dialogs with slot counts: {len(slot_rows)}")
if slot_rows:
    diffs = [d["count_b"] - d["count_a"] for d in slot_rows]
    sa_better   = sum(1 for x in diffs if x > 0)
    base_better = sum(1 for x in diffs if x < 0)
    equal       = sum(1 for x in diffs if x == 0)
    print(f"  SA fills more slots:      {sa_better} dialogs")
    print(f"  Baseline fills more slots: {base_better} dialogs")
    print(f"  Equal slot counts:         {equal} dialogs")
    avg_a = np.mean([d["count_a"] for d in slot_rows])
    avg_b = np.mean([d["count_b"] for d in slot_rows])
    print(f"  Avg slots filled — baseline: {avg_a:.2f}  sa: {avg_b:.2f}  Δ={avg_b-avg_a:+.2f}")

    # by domain
    for dom in ["Movies_1_Movies_3", "Restaurants_2"]:
        sub = [d for d in slot_rows if d["domain"] == dom]
        if sub:
            a_avg = np.mean([d["count_a"] for d in sub])
            b_avg = np.mean([d["count_b"] for d in sub])
            d_short = "Movies" if "Movie" in dom else "Restaurants"
            print(f"  {d_short}: baseline avg={a_avg:.2f}  sa avg={b_avg:.2f}  Δ={b_avg-a_avg:+.2f}  n={len(sub)}")

# ── 6. Context: Baseline vs SA vs US vs US+SA ──────────────────────────────────
# Load US vs US+SA summaries for comparison
us_us_sa_dir = BASE.parent   # pairwise_comparison/
print("\n" + "="*65)
print("6. CONTEXT: BASELINE vs SA  vs  US vs US+SA  (B−A advantage)")
print("="*65)
print(f"  {'Metric':<14} {'Baseline→SA Δ':>16} {'US→US+SA Δ':>14}")
print("  " + "-"*44)
for m in METRICS:
    bpf_adv = adv(summaries[m])
    us_sa_path = us_us_sa_dir / m / "summary.json"
    if us_sa_path.exists():
        with open(us_sa_path) as f:
            us_sa_s = json.load(f)
        us_sa_adv = (us_sa_s["win_rate_b"] - us_sa_s["win_rate_a"]) * 100
    else:
        us_sa_adv = float("nan")
    print(f"  {m:<14} {bpf_adv:>+14.1f}pp  {us_sa_adv:>+12.1f}pp")

# ── 7. PLOTS ───────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("7. GENERATING PLOTS")
print("="*65)

import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.0)

# ── 7a. Overall grouped bar chart ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
x  = np.arange(len(METRICS))
w  = 0.25
for i, (key, label, color) in enumerate([
    ("win_rate_a", COND_A,  PALETTE[COND_A]),
    ("win_rate_b", COND_B,  PALETTE[COND_B]),
    ("tie_rate",   "tie",   PALETTE["tie"]),
]):
    vals = [summaries[m][key] * 100 for m in METRICS]
    bars = ax.bar(x + (i-1)*w, vals, w, label=label, color=color, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(METRICS)
ax.set_ylabel("Rate (%)"); ax.set_ylim(0, 105)
ax.set_title("Overall Win / Tie Rates — Baseline (A) vs SA (B)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "01_overall_win_rates.png", dpi=150)
plt.close(fig)
print("  Saved: 01_overall_win_rates.png")

# ── 7b. Per-domain grouped bars ────────────────────────────────────────────────
domains = ["Movies_1_Movies_3", "Restaurants_2"]
dom_labels = ["Movies", "Restaurants"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, m in zip(axes, METRICS):
    x2 = np.arange(len(domains))
    pd_data = summaries[m].get("per_domain", {})
    a_vals = [pd_data.get(d, {}).get("win_rate_a", 0)*100 for d in domains]
    b_vals = [pd_data.get(d, {}).get("win_rate_b", 0)*100 for d in domains]
    ax.bar(x2 - 0.2, a_vals, 0.35, label=COND_A, color=PALETTE[COND_A], edgecolor="white")
    ax.bar(x2 + 0.2, b_vals, 0.35, label=COND_B, color=PALETTE[COND_B], edgecolor="white")
    for xi, (av, bv) in enumerate(zip(a_vals, b_vals)):
        ax.text(xi - 0.2, av + 0.3, f"{av:.0f}%", ha="center", fontsize=8)
        ax.text(xi + 0.2, bv + 0.3, f"{bv:.0f}%", ha="center", fontsize=8)
    ax.set_title(m); ax.set_xticks(x2); ax.set_xticklabels(dom_labels)
    ax.set_ylabel("Win Rate (%)"); ax.set_ylim(0, 30)
    ax.legend(fontsize=7)
fig.suptitle("Win Rates by Domain — Baseline vs SA", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "02_domain_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: 02_domain_comparison.png")

# ── 7c. Stacked winner bars per metric × domain ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, m in zip(axes, METRICS):
    rows = dialogs[m]
    cats = ["Movies", "Restaurants", "Overall"]
    a_pct, b_pct, tie_pct = [], [], []
    for cat in cats:
        if cat == "Overall":
            sub = rows
        elif cat == "Movies":
            sub = [r for r in rows if "Movie" in r.get("domain","")]
        else:
            sub = [r for r in rows if "Restaurant" in r.get("domain","")]
        n = len(sub)
        if n == 0:
            a_pct.append(0); b_pct.append(0); tie_pct.append(0)
            continue
        a_pct.append(sum(1 for r in sub if (r.get("winner") or "").lower() == "a") / n * 100)
        b_pct.append(sum(1 for r in sub if (r.get("winner") or "").lower() == "b") / n * 100)
        tie_pct.append(sum(1 for r in sub if (r.get("winner") or "").lower() == "tie") / n * 100)
    x3 = np.arange(len(cats))
    ax.bar(x3, a_pct,   0.5, label=COND_A,  color=PALETTE[COND_A], edgecolor="white")
    ax.bar(x3, b_pct,   0.5, bottom=a_pct,  label=COND_B,  color=PALETTE[COND_B], edgecolor="white")
    ax.bar(x3, tie_pct, 0.5,
           bottom=[a+b for a,b in zip(a_pct, b_pct)],
           label="tie", color=PALETTE["tie"], edgecolor="white")
    ax.set_xticks(x3); ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylim(0, 100); ax.set_ylabel("% of dialogs")
    ax.set_title(m); ax.legend(fontsize=7)
fig.suptitle("Winner Distribution — Baseline vs SA", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "03_winner_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: 03_winner_distribution.png")

# ── 7d. B−A advantage comparison: Baseline→SA vs US→US+SA ─────────────────────
bpf_advs  = [adv(summaries[m]) for m in METRICS]
us_sa_advs = []
for m in METRICS:
    p = us_us_sa_dir / m / "summary.json"
    if p.exists():
        with open(p) as f:
            s = json.load(f)
        us_sa_advs.append((s["win_rate_b"] - s["win_rate_a"]) * 100)
    else:
        us_sa_advs.append(0)

fig, ax = plt.subplots(figsize=(9, 4))
x4 = np.arange(len(METRICS))
w2 = 0.3
bars_bpf  = ax.bar(x4 - w2/2, bpf_advs,  w2, label="Baseline→SA",  color="#2ca02c", edgecolor="white")
bars_us_sa = ax.bar(x4 + w2/2, us_sa_advs, w2, label="US→US+SA",      color="#9467bd", edgecolor="white")
for bar, v in list(zip(bars_bpf, bpf_advs)) + list(zip(bars_us_sa, us_sa_advs)):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.1 if v >= 0 else -0.8),
            f"{v:+.1f}pp", ha="center", va="bottom", fontsize=8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x4); ax.set_xticklabels(METRICS)
ax.set_ylabel("SA advantage (pp)  [positive = SA/US+SA wins]")
ax.set_title("Effect of SA: Baseline→SA vs US→US+SA")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "04_sa_effect_comparison.png", dpi=150)
plt.close(fig)
print("  Saved: 04_sa_effect_comparison.png")

# ── 7e. Constraint slot count scatter ─────────────────────────────────────────
slot_rows = [d for d in dialogs["constraint"] if d.get("count_a") is not None]
if slot_rows:
    ca = [d["count_a"] for d in slot_rows]
    cb = [d["count_b"] for d in slot_rows]
    colors_sc = [PALETTE[COND_B] if b > a else (PALETTE[COND_A] if a > b else PALETTE["tie"])
                 for a, b in zip(ca, cb)]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(ca, cb, c=colors_sc, s=60, alpha=0.8, edgecolors="white", zorder=3)
    lim = max(max(ca), max(cb)) + 0.5
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="tie line")
    ax.set_xlabel("Slots filled — baseline"); ax.set_ylabel("Slots filled — sa")
    ax.set_title("Constraint: Slot Counts per Dialog\n(above line = SA fills more)")
    patches = [mpatches.Patch(color=PALETTE[COND_B], label="SA wins"),
               mpatches.Patch(color=PALETTE[COND_A], label="Baseline wins"),
               mpatches.Patch(color=PALETTE["tie"],  label="Tie")]
    ax.legend(handles=patches, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "05_constraint_slots.png", dpi=150)
    plt.close(fig)
    print("  Saved: 05_constraint_slots.png")

# ── summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY")
print("="*65)

for m in METRICS:
    s = summaries[m]
    delta = adv(s)
    winner = COND_B if delta > 0 else (COND_A if delta < 0 else "exact tie")
    print(f"\n  [{m}]")
    print(f"    Overall: {winner} leads by {abs(delta):.1f}pp  "
          f"(A={fmt(s['win_rate_a'])}, B={fmt(s['win_rate_b'])}, tie={fmt(s['tie_rate'])})")
    for domain, ds in sorted(s.get("per_domain", {}).items()):
        d_short = "Movies" if "Movie" in domain else "Restaurants"
        print(f"    {d_short}: A={fmt(ds['win_rate_a'])} B={fmt(ds['win_rate_b'])} "
              f"Δ={adv(ds):+.1f}pp")

print(f"\n  All plots saved to: {OUT}/\n")
