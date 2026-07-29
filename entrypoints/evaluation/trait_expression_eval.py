"""Trait expression evaluation using Trait_Expression_Prompt.

For each dialog, runs GPT-4o-mini to identify the top-3 most evident
USER personality traits and their expression scores.

Two modes:
  Flat   (baseline/sa):  --dialogs-dir {root}/{domain}/dialog_*.json
  Nested (US/US+SA):      --dialogs-dir {runs_dir}/{trait}__s{scalar}/dialogs/{domain}/
                         --nested

For US/US+SA the script additionally computes steering accuracy:
whether the intended (target) trait was among the predicted top traits.

Saves:
    {output_dir}/{domain}/{id}.json       -- per-dialog judge result + accuracy
    {output_dir}/summary.json             -- domain analysis + steering accuracy (US only)

Usage (baseline):
    python entrypoints/evaluation/trait_expression_eval.py \
        --dialogs-dir qwen/qwen-v2/qwen-baseline-2026-05-20-09-19/runs/baseline/dialogs \
        --output-dir  qwen/qwen-v2/evaluation/trait_expression/baseline

Usage (US):
    python entrypoints/evaluation/trait_expression_eval.py \
        --dialogs-dir qwen/qwen-v2/qwen-user-steer-2026-05-25-15-49/runs \
        --output-dir  qwen/qwen-v2/evaluation/trait_expression/us \
        --nested
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if not sys.path or sys.path[0] != _PROJECT_ROOT:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.big5.judge_json import OpenAiJsonJudge
from entrypoints.evaluation.quality_metrics import (
    Trait_Expression_Prompt,
    TRAIT_DESCRIPTIONS_STR,
    _render_dialog_text,
)

DOMAINS = ["Movies_1_Movies_3", "Restaurants_2"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_run_label(run_label: str) -> Tuple[str, str]:
    """'calm__s-1' -> ('calm', '-1')"""
    if "__s" in run_label:
        parts = run_label.split("__s", 1)
        return parts[0], parts[1]
    return run_label, "0"


def _canonical(trait: Optional[str]) -> Optional[str]:
    if not trait:
        return None
    return trait.strip().lower()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dialogs_flat(dialogs_dir: str, domains: List[str] = None) -> List[Dict[str, Any]]:
    """Load from {dialogs_dir}/{domain}/dialog_*.json  (baseline/sa)."""
    if domains is None:
        domains = DOMAINS
    base = Path(dialogs_dir)
    records: List[Dict[str, Any]] = []
    for domain in domains:
        domain_dir = base / domain
        if not domain_dir.exists():
            print(f"[trait_eval] Domain dir not found: {domain_dir}", file=sys.stderr)
            continue
        for f in sorted(domain_dir.glob("dialog_*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                records.append({
                    "dialog_id":        f.stem,
                    "domain":           domain,
                    "run_label":        None,
                    "target_trait":     None,
                    "scalar":           None,
                    "filepath":         str(f),
                    "generated_dialog": data.get("generated_dialog", []),
                })
            except (json.JSONDecodeError, IOError) as e:
                print(f"[trait_eval] Error loading {f}: {e}", file=sys.stderr)
    return records


def load_dialogs_nested(runs_dir: str, domains: List[str] = None) -> List[Dict[str, Any]]:
    """Load from {runs_dir}/{run_label}/dialogs/{domain}/  (US/US+SA)."""
    if domains is None:
        domains = DOMAINS
    base = Path(runs_dir)
    records: List[Dict[str, Any]] = []
    for run_dir in sorted(d for d in base.iterdir() if d.is_dir()):
        run_label = run_dir.name
        trait, scalar = _parse_run_label(run_label)
        for domain in domains:
            domain_dir = run_dir / "dialogs" / domain
            if not domain_dir.exists():
                continue
            for f in sorted(domain_dir.glob("dialog_*.json")):
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    records.append({
                        "dialog_id":        f.stem,
                        "domain":           domain,
                        "run_label":        run_label,
                        "target_trait":     trait,
                        "scalar":           scalar,
                        "filepath":         str(f),
                        "generated_dialog": data.get("generated_dialog", []),
                    })
                except (json.JSONDecodeError, IOError) as e:
                    print(f"[trait_eval] Error loading {f}: {e}", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

async def _eval_one(
    *,
    sem: asyncio.Semaphore,
    record: Dict[str, Any],
    judge: OpenAiJsonJudge,
) -> Dict[str, Any]:
    dialog_text = _render_dialog_text(record["generated_dialog"])
    async with sem:
        try:
            result = await judge(
                trait_descriptions=TRAIT_DESCRIPTIONS_STR,
                dialogue=dialog_text,
            )
            if "error" in result:
                raise RuntimeError(result["error"])
            return {
                "dialog_id":    record["dialog_id"],
                "domain":       record["domain"],
                "run_label":    record["run_label"],
                "target_trait": record["target_trait"],
                "scalar":       record["scalar"],
                "filepath":     record["filepath"],
                **result,
            }
        except Exception as exc:
            print(
                f"[trait_eval] Failed {record['dialog_id']} "
                f"({record.get('run_label', '')} / {record['domain']}): {exc}",
                file=sys.stderr,
            )
            return {
                "dialog_id":    record["dialog_id"],
                "domain":       record["domain"],
                "run_label":    record["run_label"],
                "target_trait": record["target_trait"],
                "scalar":       record["scalar"],
                "filepath":     record["filepath"],
                "error":        str(exc),
            }


def run_eval(
    records: List[Dict[str, Any]],
    judge_model: str = "openai/gpt-4o-mini",
    max_concurrent: int = 30,
) -> List[Dict[str, Any]]:
    judge = OpenAiJsonJudge(
        judge_model, Trait_Expression_Prompt, max_new_tokens=512, temperature=0.0
    )
    print("judge")
    async def _run():
        sem = asyncio.Semaphore(max_concurrent)
        tasks = [
            asyncio.create_task(_eval_one(sem=sem, record=r, judge=judge))
            for r in records
        ]
        return await asyncio.gather(*tasks)

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Domain-level trait frequency analysis (baseline + US)
# ---------------------------------------------------------------------------

def _freq_stats(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Trait frequency and avg score across a set of dialog results."""
    top1_counts: Counter = Counter()
    top2_counts: Counter = Counter()
    top3_counts: Counter = Counter()
    any_counts:  Counter = Counter()
    scores_by_trait: Dict[str, List[float]] = defaultdict(list)

    for r in subset:
        if "error" in r:
            continue
        for rank, key in enumerate(["top1", "top2", "top3"], start=1):
            trait = _canonical(r.get(f"{key}_evident_trait"))
            expr  = r.get(f"{key}_evident_trait_expression") or {}
            score = expr.get("score")
            if not trait:
                continue
            if rank == 1:
                top1_counts[trait] += 1
            elif rank == 2:
                top2_counts[trait] += 1
            else:
                top3_counts[trait] += 1
            any_counts[trait] += 1
            if isinstance(score, (int, float)):
                scores_by_trait[trait].append(float(score))

    n = len([r for r in subset if "error" not in r])
    return {
        "n_dialogs":             n,
        "top1_trait_counts":     dict(top1_counts.most_common()),
        "top2_trait_counts":     dict(top2_counts.most_common()),
        "top3_trait_counts":     dict(top3_counts.most_common()),
        "any_top3_trait_counts": dict(any_counts.most_common()),
        "avg_score_by_trait": {
            t: round(sum(v) / len(v), 3)
            for t, v in sorted(scores_by_trait.items())
        },
    }


def analyze_frequency(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_domain: Dict[str, list] = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r)
    return {
        "overall":   _freq_stats(results),
        "by_domain": {d: _freq_stats(v) for d, v in sorted(by_domain.items())},
    }


# ---------------------------------------------------------------------------
# Steering accuracy analysis (US/US+SA only)
# ---------------------------------------------------------------------------

def _add_accuracy(result: Dict[str, Any]) -> Dict[str, Any]:
    """Embed per-dialog accuracy fields (intended rank/score) into the result."""
    if "error" in result or not result.get("target_trait"):
        return result

    target = _canonical(result["target_trait"])

    def _score(key: str) -> Optional[float]:
        expr = result.get(f"{key}_evident_trait_expression") or {}
        s = expr.get("score")
        return float(s) if isinstance(s, (int, float)) else None

    top_traits = [
        (1, _canonical(result.get("top1_evident_trait")), _score("top1")),
        (2, _canonical(result.get("top2_evident_trait")), _score("top2")),
        (3, _canonical(result.get("top3_evident_trait")), _score("top3")),
    ]
    top_traits = [(r, t, s) for r, t, s in top_traits if t is not None]

    intended_rank  = 4
    intended_score = 0.0
    for rank, pred_trait, score in top_traits:
        if pred_trait == target:
            intended_rank  = rank
            intended_score = score or 0.0
            break

    top_names = [t for _, t, _ in top_traits]
    result["accuracy"] = {
        "target_trait":     target,
        "top1_match":       top_traits[0][1] == target if top_traits else False,
        "top2_match":       target in top_names[:2],
        "intended_rank":    intended_rank,
        "intended_score":   intended_score,
        "intended_present": intended_rank <= 3,
    }
    return result


def _steering_stats(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(subset)
    if total == 0:
        return {}

    top1_match_count = 0
    top2_match_count = 0
    intended_score_sum_all     = 0.0
    intended_score_sum_present = 0.0
    intended_present_count     = 0
    intended_rank_sum          = 0.0
    top1_score_sum             = 0.0
    predicted_trait_counts: Dict[str, int]   = defaultdict(int)
    confusion_counts:       Dict[str, int]   = defaultdict(int)
    confusion_score_sums:   Dict[str, float] = defaultdict(float)

    for r in subset:
        acc        = r.get("accuracy", {})
        top1_trait = _canonical(r.get("top1_evident_trait"))
        top1_expr  = r.get("top1_evident_trait_expression") or {}
        top1_score = float(top1_expr.get("score") or 0)

        top1_score_sum += top1_score
        if top1_trait:
            predicted_trait_counts[top1_trait] += 1

        if acc.get("top1_match"):
            top1_match_count += 1
        elif top1_trait:
            confusion_counts[top1_trait]      += 1
            confusion_score_sums[top1_trait]  += top1_score

        if acc.get("top2_match"):
            top2_match_count += 1

        intended_rank  = acc.get("intended_rank", 4)
        intended_score = acc.get("intended_score", 0.0)
        intended_rank_sum      += intended_rank
        intended_score_sum_all += intended_score

        if acc.get("intended_present"):
            intended_present_count     += 1
            intended_score_sum_present += intended_score

    avg_score_present = (
        intended_score_sum_present / intended_present_count
        if intended_present_count > 0 else 0.0
    )

    return {
        "total_dialogs":                   total,
        "top1_match_rate":                 round(top1_match_count / total, 4),
        "top2_match_rate":                 round(top2_match_count / total, 4),
        "intended_present_rate":           round(intended_present_count / total, 4),
        "avg_intended_rank":               round(intended_rank_sum / total, 3),
        "avg_intended_score_all":          round(intended_score_sum_all / total, 3),
        "avg_intended_score_when_present": round(avg_score_present, 3),
        "avg_top1_predicted_score":        round(top1_score_sum / total, 3),
        "predicted_trait_distribution": {
            k: round(v / total, 4)
            for k, v in sorted(predicted_trait_counts.items(), key=lambda x: -x[1])
        },
        "confusion_distribution": {
            k: round(v / total, 4)
            for k, v in sorted(confusion_counts.items(), key=lambda x: -x[1])
        },
        "confusion_avg_score": {
            k: round(confusion_score_sums[k] / confusion_counts[k], 3)
            for k in sorted(confusion_counts, key=lambda x: -confusion_counts[x])
            if confusion_counts[k] > 0
        },
    }


def analyze_steering(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in results if "error" not in r and r.get("target_trait")]

    by_trait: Dict[str, list] = defaultdict(list)
    for r in valid:
        by_trait[r["target_trait"]].append(r)

    per_trait: Dict[str, Any] = {}
    for trait in sorted(by_trait):
        subset = by_trait[trait]
        per_trait[trait] = _steering_stats(subset)

        by_domain: Dict[str, list] = defaultdict(list)
        for r in subset:
            by_domain[r["domain"]].append(r)
        per_trait[trait]["by_domain"] = {
            d: _steering_stats(v) for d, v in sorted(by_domain.items())
        }

    return {
        "overall":   _steering_stats(valid),
        "per_trait": per_trait,
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    out_root: Path,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    for r in results:
        domain_dir = out_root / r["domain"]
        domain_dir.mkdir(parents=True, exist_ok=True)
        fname = r["dialog_id"]
        if r.get("run_label"):
            fname = f"{r['run_label']}__{r['dialog_id']}"
        with open(domain_dir / f"{fname}.json", "w") as f:
            json.dump(r, f, indent=2)

    summary_path = out_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[trait_eval] Summary:    {summary_path}")
    print(f"[trait_eval] Per-dialog: {out_root}/{{domain}}/{{id}}.json")


# ---------------------------------------------------------------------------
# CLI printing
# ---------------------------------------------------------------------------

def _print_freq(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("TRAIT FREQUENCY ANALYSIS")
    print("=" * 60)
    ov = summary["overall"]
    print(f"Overall (n={ov['n_dialogs']}) — top-1 most frequent:")
    for t, c in list(ov["top1_trait_counts"].items())[:5]:
        print(f"  {t}: {c}")
    for domain, st in summary["by_domain"].items():
        print(f"\n  {domain} (n={st['n_dialogs']}) — top-1:")
        for t, c in list(st["top1_trait_counts"].items())[:5]:
            print(f"    {t}: {c}")


def _print_steering(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 65)
    print("STEERING ACCURACY — US (target trait vs predicted top traits)")
    print("=" * 65)
    ov = summary["overall"]
    print(
        f"Overall ({ov['total_dialogs']} dialogs): "
        f"top1={ov['top1_match_rate']:.1%}  "
        f"top2={ov['top2_match_rate']:.1%}  "
        f"present={ov['intended_present_rate']:.1%}  "
        f"avg_rank={ov['avg_intended_rank']:.2f}"
    )
    print()
    print(f"{'Trait':<18} {'top1%':>6} {'top2%':>6} {'rank':>5} {'score_all':>10} {'present%':>9}")
    print("-" * 65)
    for trait, st in summary["per_trait"].items():
        print(
            f"{trait:<18} "
            f"{st['top1_match_rate']:>6.1%} "
            f"{st['top2_match_rate']:>6.1%} "
            f"{st['avg_intended_rank']:>5.2f} "
            f"{st['avg_intended_score_all']:>10.3f} "
            f"{st['intended_present_rate']:>9.1%}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Trait expression eval (+ US steering accuracy) on dialog files."
    )
    parser.add_argument("--dialogs-dir", required=True,
                        help="Dialogs root (flat) or runs root (use --nested for US/US+SA)")
    parser.add_argument("--output-dir", required=True,
                        help="Root output directory for results and summary")
    parser.add_argument("--nested", action="store_true",
                        help="Nested loader for US/US+SA runs/ structure")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini")
    parser.add_argument("--max-concurrent", type=int, default=30)
    parser.add_argument("--domain", choices=DOMAINS,
                        help="Restrict to one domain (default: both)")
    args = parser.parse_args()

    domains = [args.domain] if args.domain else DOMAINS

    print(f"[trait_eval] Dialogs dir:  {args.dialogs_dir}")
    print(f"[trait_eval] Output dir:   {args.output_dir}")
    print(f"[trait_eval] Mode:         {'nested (US/US+SA)' if args.nested else 'flat (baseline/sa)'}")
    print(f"[trait_eval] Judge model:  {args.judge_model}")
    print(f"[trait_eval] Domains:      {domains}")

    records = (
        load_dialogs_nested(args.dialogs_dir, domains=domains)
        if args.nested
        else load_dialogs_flat(args.dialogs_dir, domains=domains)
    )
    print(f"[trait_eval] Loaded {len(records)} dialogs")
    if not records:
        print("[trait_eval] No dialogs found. Exiting.")
        return

    print("[trait_eval] Running judge …")
    results = run_eval(records, judge_model=args.judge_model, max_concurrent=args.max_concurrent)

    errors = sum(1 for r in results if "error" in r)
    print(f"[trait_eval] Done: {len(results) - errors} succeeded, {errors} errors.")

    # Domain/frequency analysis (both modes)
    freq_summary = analyze_frequency(results)
    _print_freq(freq_summary)

    combined_summary: Dict[str, Any] = {"frequency_analysis": freq_summary}

    # Steering accuracy (US/US+SA only)
    if args.nested:
        results = [_add_accuracy(r) for r in results]
        steering_summary = analyze_steering(results)
        _print_steering(steering_summary)
        combined_summary["steering_accuracy"] = steering_summary

    save_results(results, combined_summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
