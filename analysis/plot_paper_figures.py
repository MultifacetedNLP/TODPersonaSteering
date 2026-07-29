"""
Reproduce Figures 1-3 from "Activation-Steered Personas in Task-Oriented LLM
Agent Simulations" from the checked-in evaluation/pairwise-comparison JSON and
markdown data. No re-running of the LLM pipeline is required.

Usage:
    python analysis/plot_paper_figures.py --model qwen --out paper_figures/qwen
    python analysis/plot_paper_figures.py --model llama --out paper_figures/llama

Figure 1 (target-trait presence rate by scalar) needs the blind trait-recognition
judge markdown table and is only available for the model it was run on
(<model>/analysis/trait_expression_us.md or equivalent) -- skipped if absent.
Figures 2 and 3 are built directly from <model>/pairwise_comparison/*/summary.json
and work for any model with pairwise-judge data.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIT_ORDER = ["outgoing", "compassionate", "dependable", "nervous", "inventive",
               "solitary", "self-interested", "careless", "calm", "consistent"]
METRICS = ["constraint", "truthfulness", "usersat"]
METRIC_LABEL = {"constraint": "Constraint Satisfaction", "truthfulness": "Truthfulness",
                "usersat": "User Satisfaction"}
SCALARS = ["-1", "0", "1", "2"]


def model_root(model: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / model if model != "qwen" else repo_root / "qwen" / "qwen-v2"


# --------------------------------------------------------------------------
# Figure 2: US+SA minus US pairwise win-rate advantage by trait
# --------------------------------------------------------------------------
def trait_deltas_all_scalars(root: Path, metric: str) -> dict:
    """Qwen convention (Figure 2 / Table 3): averaged over all 4 scalars, s=0 included."""
    summary = json.loads((root / "pairwise_comparison" / metric / "summary.json").read_text())
    per_trait = summary["per_trait"]
    return {t: 100 * (per_trait[t]["win_rate_b"] - per_trait[t]["win_rate_a"])
            for t in TRAIT_ORDER if t in per_trait}


def trait_deltas_exclude_s0(root: Path, metric: str) -> dict:
    """Llama convention (Table 7): directional scalars only, s=0 excluded.
    Recomputed from the raw per-dialogue pairwise-judge files because
    summary.json's per_trait breakdown does not separate out s=0."""
    metric_dir = root / "pairwise_comparison" / metric
    counts = {t: {"a": 0, "b": 0, "n": 0} for t in TRAIT_ORDER}
    for run_dir in metric_dir.iterdir():
        if not run_dir.is_dir() or "__s" not in run_dir.name:
            continue
        trait, scalar = run_dir.name.rsplit("__s", 1)
        if scalar == "0" or trait not in counts:
            continue
        for domain_dir in run_dir.iterdir():
            if not domain_dir.is_dir():
                continue
            for f in domain_dir.glob("*.json"):
                winner = json.loads(f.read_text()).get("winner")
                if winner is None:
                    continue
                winner = str(winner).strip().lower()
                counts[trait]["n"] += 1
                if winner == "a":
                    counts[trait]["a"] += 1
                elif winner == "b":
                    counts[trait]["b"] += 1
    return {t: 100 * (c["b"] - c["a"]) / c["n"] for t, c in counts.items() if c["n"]}


def plot_figure2(root: Path, out_path: Path, exclude_s0: bool):
    data = {}
    fn = trait_deltas_exclude_s0 if exclude_s0 else trait_deltas_all_scalars
    for metric in METRICS:
        deltas = fn(root, metric)
        for trait, delta_pp in deltas.items():
            data.setdefault(trait, {})[metric] = delta_pp

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    for i, trait in enumerate(TRAIT_ORDER):
        ax = axes[i // 5, i % 5]
        vals = [data[trait][m] for m in METRICS]
        labels = ["C", "T", "U"]
        bars = ax.bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.5 if v >= 0 else -1.5),
                    f"{v:+.1f}", ha="center", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(trait, fontsize=13, fontweight="bold")
        if i % 5 == 0:
            ax.set_ylabel("SA - US (pp)")
    fig.text(0.5, 0.02, "C = Constraint Satisfaction,  T = Truthfulness,  U = User Satisfaction",
              ha="center", fontsize=11)
    scalar_note = "s ∈ {-1,1,2}, s=0 excluded" if exclude_s0 else "all scalars incl. s=0"
    fig.suptitle(f"Figure 2: US+SA minus US pairwise win-rate advantage by trait ({scalar_note})",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------
# Figure 3: pairwise win rates across scalar strengths
# --------------------------------------------------------------------------
def plot_figure3(root: Path, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, metric in zip(axes, METRICS):
        summary = json.loads((root / "pairwise_comparison" / metric / "summary.json").read_text())
        per_scalar = summary["per_scalar"]
        us = [100 * per_scalar[s]["win_rate_a"] for s in SCALARS]
        us_sa = [100 * per_scalar[s]["win_rate_b"] for s in SCALARS]
        ax.plot(SCALARS, us, "o-", label="US")
        ax.plot(SCALARS, us_sa, "s-", label="US+SA")
        ax.set_title(METRIC_LABEL[metric])
        ax.set_xlabel("Scalar (α)")
        ax.set_ylabel("Win Rate (%)")
        ax.legend()
    fig.suptitle("Figure 3: Pairwise win rates across persona scalar strengths", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------
# Figure 1: target-trait presence rate by scalar (blind trait-recognition judge)
# --------------------------------------------------------------------------
ROW_RE = re.compile(
    r"^\|\s*(?:\*\*(?P<trait>[\w-]+)\*\*)?\s*\|\s*(?P<scalar>-?\d+)\s*\|.*?\|"
    r"\s*\**(?P<present>[\d.]+)%\**\s*\|"
)


def parse_trait_scalar_table(md_path: Path):
    """Parses the '## 3. Steering Accuracy by Trait x Scalar' table."""
    lines = md_path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("## 3."))
    data = {}
    current_trait = None
    for line in lines[start:]:
        if line.startswith("---") and current_trait is not None:
            break
        m = ROW_RE.match(line)
        if not m:
            continue
        if m.group("trait"):
            current_trait = m.group("trait")
        if current_trait is None:
            continue
        data.setdefault(current_trait, {})[m.group("scalar")] = float(m.group("present"))
    return data


def plot_figure1(md_path: Path, out_path: Path):
    if not md_path.is_file():
        print(f"skip figure1: {md_path} not found")
        return
    data = parse_trait_scalar_table(md_path)
    fig, ax = plt.subplots(figsize=(10, 7))
    for trait, by_scalar in data.items():
        xs = SCALARS
        ys = [by_scalar.get(s, 0.0) for s in xs]
        ax.plot(xs, ys, marker="o", label=trait)
    ax.set_xlabel("Scalar strength (s)")
    ax.set_ylabel("Target-trait presence rate (%)")
    ax.set_title("Figure 1: Target-trait presence rate by steering scalar strength")
    ax.legend(title="Trait", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen", "llama"], required=True)
    ap.add_argument("--out", default=None, help="output directory (default: paper_figures/<model>)")
    args = ap.parse_args()

    root = model_root(args.model)
    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "paper_figures" / args.model

    # Matches each model's reported convention in the paper: Qwen's Figure 2 /
    # Table 3 average over all 4 scalars; Llama's Table 7 excludes s=0.
    exclude_s0 = (args.model == "llama")
    plot_figure2(root, out_dir / "figure2_trait_advantage.png", exclude_s0=exclude_s0)
    plot_figure3(root, out_dir / "figure3_scalar_winrates.png")
    plot_figure1(root / "analysis" / "trait_expression_us.md", out_dir / "figure1_trait_observability.png")


if __name__ == "__main__":
    main()
