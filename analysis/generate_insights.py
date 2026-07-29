"""
Generate sa_insights.md for any Activation-Steered-Personas experiment set.

Usage:
  python analysis/generate_insights.py \\
    --exp-root /work/hdd/beto/balvesrodrigues/experiments-persona/llama-v2 \\
    --baseline     llama-baseline-2026-05-27-07-42 \\
    --sa llama-sa-2026-05-27-07-52 \\
    --user-steer   llama-user-steer-2026-05-27-07-43 \\
    --user-steer-sa llama-user-steer-sa-both-2026-05-27-07-43 \\
    --output /work/hdd/beto/balvesrodrigues/experiments-persona/llama-v2/analysis/sa_insights.md \\
    --model "LLaMA-3.1-8B-Instruct" \\
    --version "llama-v2"
"""

import argparse
import json
from pathlib import Path

TRAITS  = ["calm", "careless", "compassionate", "consistent", "dependable",
           "inventive", "nervous", "outgoing", "self-interested", "solitary"]
SCALARS = ["s-1", "s0", "s1", "s2"]


def load_summary(path: Path) -> dict:
    with open(path / "metrics.json") as f:
        return json.load(f)["summary"]


def pct(v, decimals=1):
    if v is None:
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def pct100(v, decimals=1):
    """For values already stored as 0-100."""
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}%"


def fmt(v, decimals=3):
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def signed(v, mode="pct", decimals=2):
    if v is None:
        return "N/A"
    sign = "+" if v >= 0 else ""
    if mode == "pct":
        return f"{sign}{v * 100:.{decimals}f} pp"
    elif mode == "pct100":
        return f"{sign}{v:.{decimals}f} pp"
    else:
        return f"{sign}{v:.{decimals}f}"


def load_all(experiments):
    data = {}

    data["baseline"]     = load_summary(experiments["baseline"])
    data["sa"] = load_summary(experiments["sa"])

    data["user-steer"]    = {}
    data["user-steer-sa"] = {}
    for trait in TRAITS:
        for scalar in SCALARS:
            run = f"{trait}__{scalar}"
            p = experiments["user-steer"] / run
            if (p / "metrics.json").exists():
                data["user-steer"][run] = load_summary(p)
            p = experiments["user-steer-sa"] / run
            if (p / "metrics.json").exists():
                data["user-steer-sa"][run] = load_summary(p)

    return data


def avg(vals):
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None


def generate(data, model_name, version, analysis_date):
    b  = data["baseline"]
    sa = data["sa"]
    us = data["user-steer"]
    us_sa = data["user-steer-sa"]

    lines = []
    A = lines.append

    # ── header ────────────────────────────────────────────────────────────────
    A(f"# Persona-Flow as a System Intervention — Research Insights")
    A(f"")
    A(f"**Experiment:** {version} · **Model:** {model_name} · **Analysis date:** {analysis_date}")
    A(f"")
    A(f"---")
    A(f"")

    # ── framing ───────────────────────────────────────────────────────────────
    A(f"## Framing")
    A(f"")
    A(f"**Persona-flow (SA)** is the experimental intervention applied to the *system model* — the service agent. "
      f"It is a prompt-based conditioning strategy that makes the system more adaptive, proactive, and personality-aware.")
    A(f"")
    A(f"**User-steer** is the experimental variable used to test robustness: steering vectors are injected into the "
      f"*user model* to simulate users with different Big-5-derived personality traits "
      f"(10 traits × 4 scalars: s-1, s0, s1, s2). This creates 40 distinct user behavior profiles.")
    A(f"")
    A(f"The central research question is: **Does applying sa to the system model improve "
      f"task-oriented dialog outcomes, and does the answer depend on the user's personality?**")
    A(f"")
    A(f"---")
    A(f"")

    # ── section 1: SA standalone ──────────────────────────────────────────────
    A(f"## 1. SA standalone: effect against a standard user")
    A(f"")
    A(f"Against a standard (non-steered) user, sa produces the following changes:")
    A(f"")

    def delta_row(label, b_val, sa_val, mode="pct", decimals=1):
        d = (sa_val - b_val) if (b_val is not None and sa_val is not None) else None
        if mode == "pct":
            b_s  = pct(b_val, decimals)
            sa_s = pct(sa_val, decimals)
            d_s  = signed(d, "pct", decimals)
        elif mode == "pct100":
            b_s  = pct100(b_val, decimals)
            sa_s = pct100(sa_val, decimals)
            d_s  = signed(d, "pct100", decimals)
        else:
            b_s  = fmt(b_val, 3)
            sa_s = fmt(sa_val, 3)
            d_s  = signed(d, "raw", 3)
        return f"| {label:<30} | {b_s:>12} | {sa_s:>14} | {d_s:>10} |"

    A("| Metric                         |    Baseline  |  Persona-Flow  |      Δ     |")
    A("| ------------------------------ | ------------ | -------------- | ---------- |")
    A(delta_row("Dialog success rate",          b.get("successful_dialogs_rate"), sa.get("successful_dialogs_rate")))
    A(delta_row("Method accuracy",              b.get("method_accuracy"),         sa.get("method_accuracy")))
    A(delta_row("Full API accuracy",            b.get("full_api_accuracy"),       sa.get("full_api_accuracy")))
    A(delta_row("BLEU",                         b.get("bleu"),                    sa.get("bleu")))
    A(delta_row("BERT F1",                      b.get("bert_score", {}).get("avg_f1"), sa.get("bert_score", {}).get("avg_f1")))
    A(delta_row("Dialog completion",            b.get("dialog_completion_rate"),  sa.get("dialog_completion_rate")))
    A(delta_row("Constraint satisfaction",      b.get("constraint_satisfaction_pct"), sa.get("constraint_satisfaction_pct"), "pct100"))
    A(delta_row("Inform score (LLM)",           b.get("inform_score"),            sa.get("inform_score"), "raw"))
    A(delta_row("Truthfulness (LLM)",           b.get("truthfulness_score"),      sa.get("truthfulness_score"), "raw"))
    A(delta_row("User satisfaction (LLM)",      b.get("user_satisfaction_score"), sa.get("user_satisfaction_score"), "raw"))
    ppl_b  = b.get("perplexity", {}).get("perplexity")  if isinstance(b.get("perplexity"), dict) else b.get("perplexity")
    ppl_sa = sa.get("perplexity", {}).get("perplexity") if isinstance(sa.get("perplexity"), dict) else sa.get("perplexity")
    A(delta_row("Perplexity",                   ppl_b,                            ppl_sa, "raw"))
    A(f"")

    # Narrative
    b_succ = b.get("successful_dialogs_rate", 0) or 0
    sa_succ = sa.get("successful_dialogs_rate", 0) or 0
    d_succ = sa_succ - b_succ
    rel = (d_succ / b_succ * 100) if b_succ else 0
    A(f"The headline result is the dialog success rate change "
      f"({'improvement' if d_succ >= 0 else 'drop'} of {abs(d_succ)*100:.1f} pp, "
      f"{abs(rel):.0f}% {'relative gain' if d_succ >= 0 else 'relative loss'}).")
    A(f"")
    A(f"---")
    A(f"")

    # ── section 2: SA across user personalities ────────────────────────────────
    A(f"## 2. SA effect across user personalities — 40-condition analysis")
    A(f"")
    A(f"Comparing user-steer alone (US) vs user-steer + sa (US+SA) across all 40 conditions:")
    A(f"")

    csat_deltas = []
    succ_deltas = []
    truth_deltas = []
    usat_deltas = []
    for run, s in us.items():
        s2 = us_sa.get(run)
        if s2 is None:
            continue
        c1 = s.get("constraint_satisfaction_pct")
        c2 = s2.get("constraint_satisfaction_pct")
        if c1 is not None and c2 is not None:
            csat_deltas.append(c2 - c1)
        d1 = s.get("successful_dialogs_rate")
        d2 = s2.get("successful_dialogs_rate")
        if d1 is not None and d2 is not None:
            succ_deltas.append(d2 - d1)
        t1 = s.get("truthfulness_score")
        t2 = s2.get("truthfulness_score")
        if t1 is not None and t2 is not None:
            truth_deltas.append(t2 - t1)
        u1 = s.get("user_satisfaction_score")
        u2 = s2.get("user_satisfaction_score")
        if u1 is not None and u2 is not None:
            usat_deltas.append(u2 - u1)

    threshold = 1.0
    improves = sum(1 for d in csat_deltas if d > threshold)
    neutral  = sum(1 for d in csat_deltas if abs(d) <= threshold)
    worsens  = sum(1 for d in csat_deltas if d < -threshold)
    total_c  = len(csat_deltas)

    A(f"| Outcome                                           | Count               |")
    A(f"| ------------------------------------------------- | ------------------- |")
    A(f"| SA improves constraint satisfaction (>+{threshold:.0f} pp)  | **{improves}** / {total_c} conditions |")
    A(f"| SA is neutral (±{threshold:.0f} pp)                        | **{neutral}** / {total_c} conditions |")
    A(f"| SA worsens constraint satisfaction (>−{threshold:.0f} pp)  | **{worsens}** / {total_c} conditions |")
    A(f"")

    us_csat_mean   = avg([s.get("constraint_satisfaction_pct") for s in us.values()])
    us_sa_csat_mean = avg([s.get("constraint_satisfaction_pct") for s in us_sa.values()])
    us_succ_mean   = avg([s.get("successful_dialogs_rate") for s in us.values()])
    us_sa_succ_mean = avg([s.get("successful_dialogs_rate") for s in us_sa.values()])
    us_truth_mean  = avg([s.get("truthfulness_score") for s in us.values()])
    us_sa_truth_mean= avg([s.get("truthfulness_score") for s in us_sa.values()])
    us_usat_mean   = avg([s.get("user_satisfaction_score") for s in us.values()])
    us_sa_usat_mean = avg([s.get("user_satisfaction_score") for s in us_sa.values()])

    A(f"The aggregate mean over all 40 conditions:")
    A(f"")
    A(f"|                         | US mean       | US+SA mean    | Δ         |")
    A(f"| ----------------------- | ------------- | ------------- | --------- |")
    A(f"| Constraint satisfaction  | {pct100(us_csat_mean):>13} | {pct100(us_sa_csat_mean):>13} | {signed(us_sa_csat_mean - us_csat_mean if us_csat_mean and us_sa_csat_mean else None, 'pct100'):>9} |")
    A(f"| Dialog success rate      | {pct(us_succ_mean):>13} | {pct(us_sa_succ_mean):>13} | {signed(us_sa_succ_mean - us_succ_mean if us_succ_mean and us_sa_succ_mean else None, 'pct'):>9} |")
    A(f"| Truthfulness             | {fmt(us_truth_mean):>13} | {fmt(us_sa_truth_mean):>13} | {signed(us_sa_truth_mean - us_truth_mean if us_truth_mean and us_sa_truth_mean else None, 'raw'):>9} |")
    A(f"| User satisfaction        | {fmt(us_usat_mean):>13} | {fmt(us_sa_usat_mean):>13} | {signed(us_sa_usat_mean - us_usat_mean if us_usat_mean and us_sa_usat_mean else None, 'raw'):>9} |")
    A(f"")
    A(f"---")
    A(f"")

    # ── section 3: SA by user performance tier ─────────────────────────────────
    A(f"## 3. SA effect by user performance tier")
    A(f"")
    A(f"Segmenting by baseline constraint satisfaction (without SA):")
    A(f"")

    tiers = {"low": [], "mid": [], "high": []}
    for run, s in us.items():
        s2 = us_sa.get(run)
        if s2 is None:
            continue
        c1 = s.get("constraint_satisfaction_pct")
        c2 = s2.get("constraint_satisfaction_pct")
        if c1 is None or c2 is None:
            continue
        if c1 < 70:
            tiers["low"].append((c1, c2))
        elif c1 <= 75:
            tiers["mid"].append((c1, c2))
        else:
            tiers["high"].append((c1, c2))

    A(f"| User type                           | US csat (mean) | US+SA csat (mean) | SA Δ       |")
    A(f"| ----------------------------------- | -------------- | ----------------- | ---------- |")
    for tier_name, label in [("low", "Low-performing (csat < 70%)"),
                              ("mid", "Mid-performing (csat 70–75%)"),
                              ("high","High-performing (csat > 75%)")]:
        pairs = tiers[tier_name]
        if not pairs:
            A(f"| {label:<35} | {'N/A':>14} | {'N/A':>17} | {'N/A':>10} |")
            continue
        us_m   = avg([p[0] for p in pairs])
        us_sa_m = avg([p[1] for p in pairs])
        d      = (us_sa_m - us_m) if us_m is not None and us_sa_m is not None else None
        A(f"| {label:<35} | {pct100(us_m):>14} | {pct100(us_sa_m):>17} | {signed(d, 'pct100'):>10} |")
    A(f"")

    # ── section 4: per-trait analysis ─────────────────────────────────────────
    A(f"## 4. Per-trait SA effect on constraint satisfaction")
    A(f"")
    A(f"Mean constraint satisfaction across all scalars for each trait:")
    A(f"")
    A(f"| Trait            | US csat | US+SA csat | Δ          |")
    A(f"| ---------------- | ------- | ---------- | ---------- |")

    trait_results = []
    for trait in TRAITS:
        us_vals   = [us.get(f"{trait}__{s}", {}).get("constraint_satisfaction_pct")
                     for s in SCALARS]
        us_sa_vals = [us_sa.get(f"{trait}__{s}", {}).get("constraint_satisfaction_pct")
                     for s in SCALARS]
        us_m   = avg([v for v in us_vals   if v is not None])
        us_sa_m = avg([v for v in us_sa_vals if v is not None])
        d      = (us_sa_m - us_m) if us_m is not None and us_sa_m is not None else None
        trait_results.append((trait, us_m, us_sa_m, d))

    # sort by delta descending
    trait_results.sort(key=lambda x: (x[3] if x[3] is not None else 0), reverse=True)
    for trait, us_m, us_sa_m, d in trait_results:
        marker = "**" if d is not None and abs(d) >= 2.0 else ""
        A(f"| {marker}{trait:<16}{marker} | {pct100(us_m):>7} | {pct100(us_sa_m):>10} | {signed(d, 'pct100'):>10} |")
    A(f"")

    # ── section 5: scalar analysis ─────────────────────────────────────────────
    A(f"## 5. Effect of steering intensity (scalar analysis)")
    A(f"")
    A(f"Mean constraint satisfaction by scalar across all traits:")
    A(f"")
    A(f"| Scalar | US csat | US+SA csat | SA Δ       |")
    A(f"| ------ | ------- | ---------- | ---------- |")
    for scalar in SCALARS:
        us_vals   = [us.get(f"{t}__{scalar}", {}).get("constraint_satisfaction_pct")
                     for t in TRAITS]
        us_sa_vals = [us_sa.get(f"{t}__{scalar}", {}).get("constraint_satisfaction_pct")
                     for t in TRAITS]
        us_m   = avg([v for v in us_vals   if v is not None])
        us_sa_m = avg([v for v in us_sa_vals if v is not None])
        d      = (us_sa_m - us_m) if us_m is not None and us_sa_m is not None else None
        A(f"| {scalar:<6} | {pct100(us_m):>7} | {pct100(us_sa_m):>10} | {signed(d, 'pct100'):>10} |")
    A(f"")

    # ── section 6: degenerate conditions ──────────────────────────────────────
    A(f"## 6. Degenerate / collapse conditions")
    A(f"")
    A(f"Conditions where user-steer produces very low constraint satisfaction (< 20%):")
    A(f"")
    A(f"| Condition      | US csat | US+SA csat | US ppl | US+SA ppl |")
    A(f"| -------------- | ------- | ---------- | ------ | --------- |")
    for run, s in sorted(us.items()):
        c = s.get("constraint_satisfaction_pct", 100)
        if c is not None and c < 20:
            s2 = us_sa.get(run, {})
            ppl1 = s.get("perplexity", {})
            ppl2 = s2.get("perplexity", {})
            if isinstance(ppl1, dict):
                ppl1 = ppl1.get("perplexity")
            if isinstance(ppl2, dict):
                ppl2 = ppl2.get("perplexity")
            A(f"| {run:<14} | {pct100(c):>7} | {pct100(s2.get('constraint_satisfaction_pct')):>10} | {fmt(ppl1, 2):>6} | {fmt(ppl2, 2):>9} |")
    A(f"")

    # ── section 7: top conditions ──────────────────────────────────────────────
    A(f"## 7. Best and worst conditions")
    A(f"")
    all_us = [(run, s.get("constraint_satisfaction_pct", 0) or 0) for run, s in us.items()]
    all_us.sort(key=lambda x: x[1], reverse=True)
    A(f"**Top 5 user-steer conditions by constraint satisfaction:**")
    A(f"")
    A(f"| Condition      | US csat | US+SA csat | SA Δ       |")
    A(f"| -------------- | ------- | ---------- | ---------- |")
    for run, c in all_us[:5]:
        s2 = us_sa.get(run, {})
        c2 = s2.get("constraint_satisfaction_pct")
        d  = (c2 - c) if c2 is not None else None
        A(f"| {run:<14} | {pct100(c):>7} | {pct100(c2):>10} | {signed(d, 'pct100'):>10} |")
    A(f"")
    A(f"**Bottom 5 user-steer conditions by constraint satisfaction:**")
    A(f"")
    A(f"| Condition      | US csat | US+SA csat | SA Δ       |")
    A(f"| -------------- | ------- | ---------- | ---------- |")
    for run, c in all_us[-5:]:
        s2 = us_sa.get(run, {})
        c2 = s2.get("constraint_satisfaction_pct")
        d  = (c2 - c) if c2 is not None else None
        A(f"| {run:<14} | {pct100(c):>7} | {pct100(c2):>10} | {signed(d, 'pct100'):>10} |")
    A(f"")

    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-root",      required=True)
    p.add_argument("--baseline",      required=True)
    p.add_argument("--sa",  required=True)
    p.add_argument("--user-steer",    required=True)
    p.add_argument("--user-steer-sa", required=True)
    p.add_argument("--output",        required=True)
    p.add_argument("--model",         default="Unknown")
    p.add_argument("--version",       default="experiment")
    p.add_argument("--date",          default=None)
    return p.parse_args()


def main():
    import datetime
    args   = parse_args()
    root   = Path(args.exp_root)
    date   = args.date or datetime.date.today().isoformat()

    experiments = {
        "baseline":      root / getattr(args, "baseline")     / "runs" / "baseline",
        "sa":  root / getattr(args, "sa") / "runs" / "sa",
        "user-steer":    root / getattr(args, "user_steer")   / "runs",
        "user-steer-sa": root / getattr(args, "user_steer_sa")/ "runs",
    }

    print("Loading metrics …")
    data = load_all(experiments)

    print("Generating insights …")
    md = generate(data, model_name=args.model, version=args.version, analysis_date=date)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
