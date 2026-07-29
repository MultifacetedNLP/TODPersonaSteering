"""
Pairwise comparison analysis for Activation-Steered-Personas qwen-v2 results.

Compares condition_a (user-steer) vs condition_b (user-steer+sa) across:
  - Metrics:  constraint, truthfulness, usersat
  - Traits:   calm, careless, compassionate, consistent, dependable,
              inventive, nervous, outgoing, self-interested, solitary
  - Scalars:  -1, 0, 1, 2
  - Domains:  Movies_1_Movies_3, Restaurants_2
"""

import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
METRICS = ["constraint", "truthfulness", "usersat"]
OUTPUT_DIR = BASE / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

COND_A_LABEL = "user-steer"
COND_B_LABEL = "user-steer+sa"

PALETTE = {"user-steer": "#4C72B0", "user-steer+sa": "#DD8452", "tie": "#8c8c8c"}

# ── helpers ────────────────────────────────────────────────────────────────────

def load_summary(metric: str) -> dict:
    path = BASE / metric / "summary.json"
    with open(path) as f:
        return json.load(f)


def load_all_dialogs(metric: str) -> pd.DataFrame:
    """Walk every per-run dialog JSON and return a flat DataFrame."""
    rows = []
    metric_dir = BASE / metric
    for fpath in metric_dir.rglob("*.json"):
        if fpath.name == "summary.json":
            continue
        with open(fpath) as f:
            d = json.load(f)
        d["metric"] = metric
        rows.append(d)
    return pd.DataFrame(rows)


def advantage(row, col_a="win_rate_a", col_b="win_rate_b"):
    """Return B − A win-rate advantage (positive → B wins more)."""
    return row[col_b] - row[col_a]


def winner_label(row):
    if row["win_rate_b"] > row["win_rate_a"]:
        return COND_B_LABEL
    elif row["win_rate_a"] > row["win_rate_b"]:
        return COND_A_LABEL
    return "tie"


def fmt_pct(x):
    return f"{x*100:.1f}%"


# ── load data ──────────────────────────────────────────────────────────────────

summaries = {m: load_summary(m) for m in METRICS}

print("Loading individual dialog files …")
dfs = {m: load_all_dialogs(m) for m in METRICS}
all_df = pd.concat(dfs.values(), ignore_index=True)
all_df["winner"] = all_df["winner"].str.lower()   # normalise A/B/tie → a/b/tie
print(f"  Total dialog comparisons loaded: {len(all_df):,}\n")

# ── 1. Overall summary table ───────────────────────────────────────────────────

print("=" * 70)
print("1. OVERALL WIN RATES BY METRIC")
print("=" * 70)

overall_rows = []
for m, s in summaries.items():
    overall_rows.append({
        "metric": m,
        "cond_a (user-steer) win%": fmt_pct(s["win_rate_a"]),
        "cond_b (user-steer+sa) win%": fmt_pct(s["win_rate_b"]),
        "tie%": fmt_pct(s["tie_rate"]),
        "n_pairs": s["n_pairs"],
        "B−A advantage": fmt_pct(s["win_rate_b"] - s["win_rate_a"]),
        "winner": winner_label(s),
    })

overall_df = pd.DataFrame(overall_rows).set_index("metric")
print(overall_df.to_string())
print()

# ── 2. Per-trait analysis ──────────────────────────────────────────────────────

print("=" * 70)
print("2. PER-TRAIT WIN RATES (across all metrics)")
print("=" * 70)

trait_rows = []
for m, s in summaries.items():
    for trait, stats in s.get("per_trait", {}).items():
        trait_rows.append({
            "metric": m,
            "trait": trait,
            "win_rate_a": stats["win_rate_a"],
            "win_rate_b": stats["win_rate_b"],
            "tie_rate": stats["tie_rate"],
            "n_pairs": stats["n_pairs"],
            "B_advantage": stats["win_rate_b"] - stats["win_rate_a"],
        })

trait_df = pd.DataFrame(trait_rows)

pivot_trait = trait_df.pivot(index="trait", columns="metric", values="B_advantage")
print("\nB−A advantage (positive → user-steer+sa wins more):")
print(pivot_trait.map(lambda x: f"{x*100:+.1f}%").to_string())
print()

# Per-metric trait ranking
for m in METRICS:
    sub = trait_df[trait_df["metric"] == m].sort_values("B_advantage", ascending=False)
    print(f"  [{m}] traits ranked by B−A advantage:")
    for _, r in sub.iterrows():
        bar = "█" * int(abs(r["B_advantage"]) * 200)
        sign = "+" if r["B_advantage"] >= 0 else ""
        print(f"    {r['trait']:18s}  A={fmt_pct(r['win_rate_a'])}  "
              f"B={fmt_pct(r['win_rate_b'])}  "
              f"tie={fmt_pct(r['tie_rate'])}  "
              f"Δ={sign}{fmt_pct(r['B_advantage'])}  {bar}")
    print()

# ── 3. Per-domain analysis ─────────────────────────────────────────────────────

print("=" * 70)
print("3. PER-DOMAIN WIN RATES")
print("=" * 70)

domain_rows = []
for m, s in summaries.items():
    for domain, stats in s.get("per_domain", {}).items():
        domain_rows.append({
            "metric": m,
            "domain": domain,
            "win_rate_a": stats["win_rate_a"],
            "win_rate_b": stats["win_rate_b"],
            "tie_rate": stats["tie_rate"],
            "n_pairs": stats["n_pairs"],
            "B_advantage": stats["win_rate_b"] - stats["win_rate_a"],
        })

domain_df = pd.DataFrame(domain_rows)
pivot_domain = domain_df.pivot(index="domain", columns="metric", values="B_advantage")
print("\nB−A advantage by domain × metric:")
print(pivot_domain.map(lambda x: f"{x*100:+.1f}%").to_string())
print()

for m in METRICS:
    sub = domain_df[domain_df["metric"] == m]
    print(f"  [{m}]")
    for _, r in sub.iterrows():
        delta = f"{r['B_advantage']*100:+.1f}%"
        print(f"    {r['domain']:30s}  A={fmt_pct(r['win_rate_a'])}  "
              f"B={fmt_pct(r['win_rate_b'])}  Δ={delta:>7s}  n={r['n_pairs']}")
    print()

# ── 4. Per-scalar analysis ─────────────────────────────────────────────────────

print("=" * 70)
print("4. PER-SCALAR (persona strength) WIN RATES")
print("=" * 70)
print("  scalar: -1=strongly negative, 0=neutral, 1=moderate, 2=strongly positive")
print()

scalar_rows = []
for m, s in summaries.items():
    for scalar, stats in s.get("per_scalar", {}).items():
        scalar_rows.append({
            "metric": m,
            "scalar": int(scalar),
            "win_rate_a": stats["win_rate_a"],
            "win_rate_b": stats["win_rate_b"],
            "tie_rate": stats["tie_rate"],
            "n_pairs": stats["n_pairs"],
            "B_advantage": stats["win_rate_b"] - stats["win_rate_a"],
        })

scalar_df = pd.DataFrame(scalar_rows).sort_values(["metric", "scalar"])
pivot_scalar = scalar_df.pivot(index="scalar", columns="metric", values="B_advantage")
print("\nB−A advantage by scalar × metric:")
print(pivot_scalar.map(lambda x: f"{x*100:+.1f}%").to_string())
print()

for m in METRICS:
    sub = scalar_df[scalar_df["metric"] == m].sort_values("scalar")
    print(f"  [{m}]")
    for _, r in sub.iterrows():
        print(f"    scalar={int(r['scalar']):+d}  "
              f"A={fmt_pct(r['win_rate_a'])}  "
              f"B={fmt_pct(r['win_rate_b'])}  "
              f"tie={fmt_pct(r['tie_rate'])}  "
              f"Δ={r['B_advantage']*100:>+7.1f}%  n={r['n_pairs']}")
    print()

# ── 5. Per-run (trait × scalar) heatmap data ──────────────────────────────────

print("=" * 70)
print("5. PER-RUN (trait × scalar) B−A ADVANTAGE")
print("=" * 70)

for m, s in summaries.items():
    print(f"\n  [{m}]")
    run_data = []
    for run_label, stats in s.get("per_run", {}).items():
        parts = run_label.rsplit("__s", 1)
        trait = parts[0]
        scalar = int(parts[1]) if len(parts) > 1 else None
        run_data.append({
            "trait": trait,
            "scalar": scalar,
            "B_advantage": stats["win_rate_b"] - stats["win_rate_a"],
            "win_rate_a": stats["win_rate_a"],
            "win_rate_b": stats["win_rate_b"],
        })
    run_df = pd.DataFrame(run_data)
    pivot = run_df.pivot(index="trait", columns="scalar", values="B_advantage")
    pivot.columns = [f"s={c}" for c in pivot.columns]
    print(pivot.map(lambda x: f"{x*100:+.1f}%").to_string())

# ── 6. Individual dialog-level analysis from flat files ───────────────────────

print("\n" + "=" * 70)
print("6. DIALOG-LEVEL WINNER DISTRIBUTION (from individual files)")
print("=" * 70)

winner_counts = (
    all_df.groupby(["metric", "winner"])
    .size()
    .reset_index(name="count")
)
winner_pct = winner_counts.copy()
totals = winner_counts.groupby("metric")["count"].transform("sum")
winner_pct["pct"] = winner_counts["count"] / totals * 100

print()
print(winner_pct.pivot(index="metric", columns="winner", values="pct")
      .map(lambda x: f"{x:.1f}%").to_string())

# trait × winner breakdown per metric
print("\n  Winner distribution by trait:")
for m in METRICS:
    sub = all_df[all_df["metric"] == m]
    tw = (sub.groupby(["trait", "winner"])
            .size()
            .unstack(fill_value=0))
    tw_pct = tw.div(tw.sum(axis=1), axis=0) * 100
    print(f"\n  [{m}]")
    print(tw_pct.map(lambda x: f"{x:.1f}%").to_string())

# scalar × winner breakdown per metric
print("\n  Winner distribution by scalar:")
for m in METRICS:
    sub = all_df[all_df["metric"] == m]
    # scalar stored as string in files
    sub = sub.copy()
    sub["scalar"] = pd.to_numeric(sub["scalar"], errors="coerce")
    sw = (sub.groupby(["scalar", "winner"])
            .size()
            .unstack(fill_value=0)
            .sort_index())
    sw_pct = sw.div(sw.sum(axis=1), axis=0) * 100
    print(f"\n  [{m}]")
    print(sw_pct.map(lambda x: f"{x:.1f}%").to_string())

# domain × winner breakdown per metric
print("\n  Winner distribution by domain:")
for m in METRICS:
    sub = all_df[all_df["metric"] == m]
    dw = (sub.groupby(["domain", "winner"])
            .size()
            .unstack(fill_value=0))
    dw_pct = dw.div(dw.sum(axis=1), axis=0) * 100
    print(f"\n  [{m}]")
    print(dw_pct.map(lambda x: f"{x:.1f}%").to_string())

# ── 7. VISUALIZATIONS ─────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("7. GENERATING PLOTS …")
print("=" * 70)

sns.set_theme(style="whitegrid", font_scale=1.0)

# ── 7a. Overall grouped bar chart ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(METRICS))
w = 0.25
for i, (col, label, color) in enumerate([
    ("win_rate_a", COND_A_LABEL, PALETTE[COND_A_LABEL]),
    ("win_rate_b", COND_B_LABEL, PALETTE[COND_B_LABEL]),
    ("tie_rate",   "tie",        PALETTE["tie"]),
]):
    vals = [summaries[m][col] * 100 for m in METRICS]
    ax.bar(x + (i - 1) * w, vals, w, label=label, color=color, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(METRICS)
ax.set_ylabel("Rate (%)")
ax.set_title("Overall Win / Tie Rates by Metric")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_overall_win_rates.png", dpi=150)
plt.close(fig)
print("  Saved: 01_overall_win_rates.png")

# ── 7b. Trait × metric heatmap (B−A advantage) ────────────────────────────────
pivot_heat = trait_df.pivot(index="trait", columns="metric", values="B_advantage") * 100
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(pivot_heat, annot=True, fmt=".1f", center=0,
            cmap="RdYlGn", linewidths=0.5, ax=ax,
            cbar_kws={"label": "B−A advantage (pp)"})
ax.set_title("B−A Win-Rate Advantage (pp)\nby Trait × Metric")
ax.set_xlabel("Metric")
ax.set_ylabel("Trait")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_trait_metric_heatmap.png", dpi=150)
plt.close(fig)
print("  Saved: 02_trait_metric_heatmap.png")

# ── 7c. Scalar trend lines ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
for ax, m in zip(axes, METRICS):
    sub = scalar_df[scalar_df["metric"] == m].sort_values("scalar")
    ax.plot(sub["scalar"], sub["win_rate_a"] * 100, "o-",
            color=PALETTE[COND_A_LABEL], label=COND_A_LABEL)
    ax.plot(sub["scalar"], sub["win_rate_b"] * 100, "s-",
            color=PALETTE[COND_B_LABEL], label=COND_B_LABEL)
    ax.set_title(m)
    ax.set_xlabel("Scalar (persona strength)")
    ax.set_ylabel("Win Rate (%)")
    ax.set_xticks([-1, 0, 1, 2])
    ax.legend(fontsize=7)
fig.suptitle("Win Rates vs Persona Scalar Strength", y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_scalar_trend.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: 03_scalar_trend.png")

# ── 7d. Domain comparison grouped bars ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
domains = sorted(domain_df["domain"].unique())
x = np.arange(len(domains))
w = 0.3
for ax, m in zip(axes, METRICS):
    sub = domain_df[domain_df["metric"] == m].set_index("domain").reindex(domains)
    ax.bar(x - w / 2, sub["win_rate_a"] * 100, w,
           color=PALETTE[COND_A_LABEL], label=COND_A_LABEL, edgecolor="white")
    ax.bar(x + w / 2, sub["win_rate_b"] * 100, w,
           color=PALETTE[COND_B_LABEL], label=COND_B_LABEL, edgecolor="white")
    ax.set_title(m)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in domains], fontsize=8)
    ax.set_ylabel("Win Rate (%)")
    ax.legend(fontsize=7)
fig.suptitle("Win Rates by Domain × Metric", y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_domain_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: 04_domain_comparison.png")

# ── 7e. Per-run heatmap for each metric ────────────────────────────────────────
for m, s in summaries.items():
    run_data = []
    for run_label, stats in s.get("per_run", {}).items():
        parts = run_label.rsplit("__s", 1)
        trait = parts[0]
        scalar = int(parts[1]) if len(parts) > 1 else None
        run_data.append({
            "trait": trait,
            "scalar": scalar,
            "B_advantage": (stats["win_rate_b"] - stats["win_rate_a"]) * 100,
        })
    rdf = pd.DataFrame(run_data)
    pivot = rdf.pivot(index="trait", columns="scalar", values="B_advantage")
    pivot.columns.name = "scalar"
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", center=0,
                cmap="RdYlGn", linewidths=0.5, ax=ax,
                cbar_kws={"label": "B−A advantage (pp)"})
    ax.set_title(f"[{m}] B−A Advantage per Run (trait × scalar)")
    fig.tight_layout()
    fname = f"05_per_run_heatmap_{m}.png"
    fig.savefig(OUTPUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

# ── 7f. Stacked bar: winner distribution by trait (dialog level) ──────────────
for m in METRICS:
    sub = all_df[all_df["metric"] == m].copy()
    tw = (sub.groupby(["trait", "winner"])
            .size()
            .unstack(fill_value=0))
    # ensure all winner columns present
    for col in ["a", "b", "tie"]:
        if col not in tw.columns:
            tw[col] = 0
    tw_pct = tw.div(tw.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(len(tw_pct))
    colors = [PALETTE[COND_A_LABEL], PALETTE[COND_B_LABEL], PALETTE["tie"]]
    labels_map = {"a": COND_A_LABEL, "b": COND_B_LABEL, "tie": "tie"}
    for col, color in zip(["a", "b", "tie"], colors):
        if col in tw_pct.columns:
            vals = tw_pct[col].values
            ax.bar(tw_pct.index, vals, bottom=bottom,
                   label=labels_map[col], color=color, edgecolor="white")
            bottom += vals
    ax.set_ylabel("% of dialogs")
    ax.set_title(f"[{m}] Winner Distribution by Trait")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 100)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fname = f"06_stacked_trait_{m}.png"
    fig.savefig(OUTPUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

# ── 7g. Box + labeled dots: B_advantage per metric, annotated by trait ────────
trait_long = trait_df.copy()
metric_order = METRICS
metric_positions = {m: i for i, m in enumerate(metric_order)}

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=trait_long, x="metric", y="B_advantage", order=metric_order,
            hue="metric", palette="Set2", legend=False, ax=ax,
            width=0.5, fliersize=0)

# Use a fixed x-offset per trait so labels don't stack
np.random.seed(42)
traits_sorted = sorted(trait_long["trait"].unique())
n_traits = len(traits_sorted)
offsets = np.linspace(-0.28, 0.28, n_traits)
trait_offset = {t: offsets[i] for i, t in enumerate(traits_sorted)}

for _, row in trait_long.iterrows():
    m = row["metric"]
    xpos = metric_positions[m] + trait_offset[row["trait"]]
    ypos = row["B_advantage"]
    ax.scatter(xpos, ypos, color="black", s=30, zorder=5, alpha=0.8)
    ax.annotate(
        row["trait"],
        xy=(xpos, ypos),
        xytext=(4, 2),
        textcoords="offset points",
        fontsize=7,
        color="#333333",
        va="center",
    )

ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set_xticks(range(len(metric_order)))
ax.set_xticklabels(metric_order)
ax.set_xlabel("metric")
ax.set_ylabel("B−A advantage")
ax.set_title("B−A Advantage per Trait × Metric\n(user-steer+sa minus user-steer win rate)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "07_trait_advantage_box.png", dpi=150)
plt.close(fig)
print("  Saved: 07_trait_advantage_box.png")

# ── summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for m, s in summaries.items():
    adv = s["win_rate_b"] - s["win_rate_a"]
    winner = COND_B_LABEL if adv > 0 else (COND_A_LABEL if adv < 0 else "tie")
    print(f"\n  [{m}]  overall: {winner} leads by {abs(adv)*100:.2f}pp  "
          f"(A={fmt_pct(s['win_rate_a'])}, B={fmt_pct(s['win_rate_b'])}, "
          f"tie={fmt_pct(s['tie_rate'])})")

    # best trait for B
    pt = s.get("per_trait", {})
    best_trait = max(pt, key=lambda t: pt[t]["win_rate_b"] - pt[t]["win_rate_a"])
    worst_trait = min(pt, key=lambda t: pt[t]["win_rate_b"] - pt[t]["win_rate_a"])
    print(f"    Trait where B gains most:  {best_trait}  "
          f"(Δ={fmt_pct(pt[best_trait]['win_rate_b'] - pt[best_trait]['win_rate_a'])})")
    print(f"    Trait where B gains least: {worst_trait}  "
          f"(Δ={fmt_pct(pt[worst_trait]['win_rate_b'] - pt[worst_trait]['win_rate_a'])})")

    # best scalar
    ps = s.get("per_scalar", {})
    best_scalar = max(ps, key=lambda sc: ps[sc]["win_rate_b"] - ps[sc]["win_rate_a"])
    print(f"    Scalar where B gains most: {best_scalar}  "
          f"(Δ={fmt_pct(ps[best_scalar]['win_rate_b'] - ps[best_scalar]['win_rate_a'])})")

print(f"\nAll plots saved to: {OUTPUT_DIR}/\n")
