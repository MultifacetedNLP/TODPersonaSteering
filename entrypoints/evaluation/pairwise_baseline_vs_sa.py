"""Pairwise LLM judge for Baseline vs SA (SA) dialogs.

Reads paired dialog files from:
    paired_dialogues/baseline_vs_sa/{domain}/dialog_{id}_baseline.json
    paired_dialogues/baseline_vs_sa/{domain}/dialog_{id}_sa.json

Runs the same three metrics as the US vs US+SA pairwise evaluation:
    - truthfulness
    - constraint
    - usersat  (adapted: no trait/scalar, general satisfaction)

Saves results to:
    pairwise_comparison/baseline_vs_sa/{metric}/summary.json
    pairwise_comparison/baseline_vs_sa/{metric}/{domain}/dialog_{id}.json
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if not sys.path or sys.path[0] != _PROJECT_ROOT:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.big5.judge_json import OpenAiJsonJudge

# ── prompts ────────────────────────────────────────────────────────────────────

CONSTRAINT_PROMPT = """You are comparing two task-oriented dialogues produced with the same user request.

Evaluate only constraint satisfaction.

Required slots:
{slot_list}

Total slots: {total_slots}

Dialogue A:
[DIALOGUE A START]
{dialogue_a}
[DIALOGUE A END]

Dialogue B:
[DIALOGUE B START]
{dialogue_b}
[DIALOGUE B END]

For each dialogue, count how many required slots the SYSTEM provides correctly.

Count a slot only if:
* The system gives the value clearly
* The value matches the search results
* Repeated values are counted once

Choose the dialogue with the higher count. Choose "tie" if the counts are equal.

Return only valid JSON:
{{"winner": "A" | "B" | "tie", "count_a": 0, "count_b": 0, "reason": "Brief reason."}}"""

TRUTHFULNESS_PROMPT = """You are comparing two task-oriented dialogues produced with the same user request.

Evaluate only truthfulness.

Shared API calls:
[API CALLS START]
{api_calls}
[API CALLS END]

Shared search results:
[SEARCH RESULTS START]
{search_results}
[SEARCH RESULTS END]

Dialogue A:
[DIALOGUE A START]
{dialogue_a}
[DIALOGUE A END]

Dialogue B:
[DIALOGUE B START]
{dialogue_b}
[DIALOGUE B END]

Compare the SYSTEM's factual claims with the API calls and search results.

Check whether the system:
* Reports facts correctly
* Avoids unsupported or contradictory claims
* Does not falsely claim that an API action succeeded or failed

Choose the more truthful dialogue. Choose "tie" if both are equally truthful.

Return only valid JSON:
{{"winner": "A" | "B" | "tie", "reason": "Brief reason."}}"""

USERSAT_PROMPT = """You are comparing two task-oriented dialogues. Both dialogues use a neutral user (no specific personality trait applied).

Evaluate overall user satisfaction with the SYSTEM's responses.

Task domain: {domain}

Dialogue A:
[DIALOGUE A START]
{dialogue_a}
[DIALOGUE A END]

Dialogue B:
[DIALOGUE B START]
{dialogue_b}
[DIALOGUE B END]

Choose which SYSTEM better satisfies the user.

Check whether:
* The tone and wording are helpful and natural
* The system responds well to the user's needs
* The user is likely to feel understood and supported
* The response is clear and appropriately informative

Choose "tie" if neither dialogue has a meaningful advantage.

Return only valid JSON:
{{"winner": "A" | "B" | "tie", "reason": "Brief reason."}}"""

_METRIC_PROMPTS = {
    "usersat":      USERSAT_PROMPT,
    "constraint":   CONSTRAINT_PROMPT,
    "truthfulness": TRUTHFULNESS_PROMPT,
}

DOMAINS = ["Movies_1_Movies_3", "Restaurants_2"]


# ── helpers ────────────────────────────────────────────────────────────────────

def _render_dialog(generated_dialog: List[str], max_chars: int = 12000) -> str:
    parts: List[str] = []
    for turn in generated_dialog or []:
        if not isinstance(turn, str):
            continue
        t = turn.strip()
        if not t:
            continue
        if t.startswith("Search Results"):
            lines = t.splitlines()
            head = lines[0] if lines else ""
            body = t[len(head):].strip()
            if len(body) > 800:
                body = body[:800] + "\n...[search results truncated]"
            parts.append(f"{head}\n{body}".strip())
        else:
            parts.append(t)
    rendered = "\n".join(parts)
    if len(rendered) > max_chars:
        half = max_chars // 2
        rendered = f"{rendered[:half]}\n\n[...dialog truncated...]\n\n{rendered[-half:]}"
    return rendered


def _flatten_req_slots(user_req_slots) -> List[str]:
    flat: List[str] = []
    for group in (user_req_slots or []):
        if isinstance(group, (list, tuple)):
            flat.extend(str(s) for s in group)
        else:
            flat.append(str(group))
    return flat


def _domain_display(domain: str) -> str:
    if "Restaurant" in domain:
        return "Restaurant booking"
    if "Movie" in domain:
        return "Movie finding"
    return domain


def _extract_id(filename: str) -> Optional[str]:
    """'dialog_7_00080_baseline.json' -> '7_00080'"""
    m = re.search(r'dialog_(\d+_\d+)_(?:baseline|sa)\.json', filename)
    return m.group(1) if m else None


# ── pair loading ───────────────────────────────────────────────────────────────

def load_pairs(paired_dir: str, domains: List[str] = None) -> List[Dict[str, Any]]:
    """Read paired baseline/sa dialog files and return a flat list of pair dicts."""
    if domains is None:
        domains = DOMAINS

    base = Path(paired_dir)
    pairs: List[Dict[str, Any]] = []

    for domain in domains:
        domain_dir = base / domain
        if not domain_dir.exists():
            print(f"[judge] Domain dir not found: {domain_dir}", file=sys.stderr)
            continue

        # Group files by dialog_id
        by_id: Dict[str, Dict[str, Path]] = defaultdict(dict)
        for f in sorted(domain_dir.glob("dialog_*.json")):
            did = _extract_id(f.name)
            if not did:
                continue
            if "_baseline.json" in f.name:
                by_id[did]["baseline"] = f
            elif "_sa.json" in f.name:
                by_id[did]["sa"] = f

        n_matched = 0
        for did, files in sorted(by_id.items()):
            if "baseline" not in files or "sa" not in files:
                continue
            try:
                with open(files["baseline"]) as fh:
                    da = json.load(fh)
                with open(files["sa"]) as fh:
                    db = json.load(fh)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[judge] Error loading {did}: {e}", file=sys.stderr)
                continue

            slot_list = _flatten_req_slots(da.get("user_req_slots", []))
            api_calls_str = "\n".join(str(c) for c in da.get("api_calls", []))
            search_results_str = "\n".join(str(r) for r in da.get("search_results", []))

            pairs.append({
                "dialog_id":          did,
                "domain":             domain,
                "domain_display":     _domain_display(domain),
                "condition_a":        "baseline",
                "condition_b":        "sa",
                "dialogue_a":         _render_dialog(da.get("generated_dialog", [])),
                "dialogue_b":         _render_dialog(db.get("generated_dialog", [])),
                "filepath_a":         str(files["baseline"]),
                "filepath_b":         str(files["sa"]),
                "slot_list":          slot_list,
                "total_slots":        len(slot_list),
                "api_calls_str":      api_calls_str,
                "search_results_str": search_results_str,
            })
            n_matched += 1

        print(f"[judge] {domain}: {n_matched} matched pairs")

    return pairs


# ── judge ──────────────────────────────────────────────────────────────────────

def _build_kwargs(pair: Dict[str, Any], metric: str) -> Optional[Dict[str, Any]]:
    if metric == "usersat":
        return {
            "dialogue_a": pair["dialogue_a"],
            "dialogue_b": pair["dialogue_b"],
            "domain":     pair["domain_display"],
        }
    if metric == "constraint":
        if pair["total_slots"] == 0:
            return None
        return {
            "slot_list":   ", ".join(pair["slot_list"]) if pair["slot_list"] else "(none)",
            "total_slots": pair["total_slots"],
            "dialogue_a":  pair["dialogue_a"],
            "dialogue_b":  pair["dialogue_b"],
        }
    if metric == "truthfulness":
        if not pair["api_calls_str"].strip():
            return None
        return {
            "api_calls":      pair["api_calls_str"],
            "search_results": pair["search_results_str"],
            "dialogue_a":     pair["dialogue_a"],
            "dialogue_b":     pair["dialogue_b"],
        }
    raise ValueError(f"Unknown metric: {metric!r}")


async def _judge_one(
    *,
    sem: asyncio.Semaphore,
    pair: Dict[str, Any],
    judge: OpenAiJsonJudge,
    metric: str,
) -> Dict[str, Any]:
    base = {
        "dialog_id":   pair["dialog_id"],
        "domain":      pair["domain"],
        "condition_a": pair["condition_a"],
        "condition_b": pair["condition_b"],
        "filepath_a":  pair["filepath_a"],
        "filepath_b":  pair["filepath_b"],
        "metric":      metric,
    }
    kwargs = _build_kwargs(pair, metric)
    if kwargs is None:
        return {**base, "winner": None, "reason": "skipped (no slots / no api calls)"}

    async with sem:
        try:
            result = await judge(**kwargs)
            if "error" in result:
                raise RuntimeError(result["error"])
            return {
                **base,
                "winner":  result.get("winner"),
                "reason":  result.get("reason", ""),
                "count_a": result.get("count_a"),
                "count_b": result.get("count_b"),
            }
        except Exception as exc:
            print(f"[judge] failed {pair['dialog_id']} ({pair['domain']}): {exc}", file=sys.stderr)
            return {**base, "winner": None, "reason": f"Error: {exc}"}


def _win_rates(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"win_rate_a": None, "win_rate_b": None, "tie_rate": None, "n_pairs": 0}
    win_a = sum(1 for r in results if (r["winner"] or "").upper() == "A")
    win_b = sum(1 for r in results if (r["winner"] or "").upper() == "B")
    ties  = sum(1 for r in results if (r["winner"] or "").lower() == "tie")
    return {
        "win_rate_a": round(win_a / n, 4),
        "win_rate_b": round(win_b / n, 4),
        "tie_rate":   round(ties  / n, 4),
        "wins_a": win_a,
        "wins_b": win_b,
        "ties":   ties,
        "n_pairs": n,
    }


def run_metric(
    pairs: List[Dict[str, Any]],
    metric: str,
    judge_model: str = "openai/gpt-4o-mini",
    max_concurrent: int = 30,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    judge = OpenAiJsonJudge(
        judge_model,
        _METRIC_PROMPTS[metric],
        max_new_tokens=256,
        temperature=0.0,
    )

    async def _run():
        sem = asyncio.Semaphore(max_concurrent)
        tasks = [
            asyncio.create_task(_judge_one(sem=sem, pair=p, judge=judge, metric=metric))
            for p in pairs
        ]
        return await asyncio.gather(*tasks)

    per_pair = asyncio.run(_run())
    valid = [r for r in per_pair if r["winner"] is not None]

    summary: Dict[str, Any] = {
        "condition_a": "baseline",
        "condition_b": "sa",
        **_win_rates(valid),
    }

    by_domain: Dict[str, list] = defaultdict(list)
    for r in valid:
        by_domain[r["domain"]].append(r)
    summary["per_domain"] = {d: _win_rates(v) for d, v in sorted(by_domain.items())}

    return summary, per_pair


# ── save ───────────────────────────────────────────────────────────────────────

def save_results(
    summary: Dict[str, Any],
    per_pair: List[Dict[str, Any]],
    out_root: Path,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    summary_path = out_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[judge] Summary saved: {summary_path}")

    for r in per_pair:
        if r["winner"] is None:
            continue
        domain_dir = out_root / r["domain"]
        domain_dir.mkdir(parents=True, exist_ok=True)
        with open(domain_dir / f"dialog_{r['dialog_id']}.json", "w") as f:
            json.dump(r, f, indent=2)

    print(f"[judge] Per-dialogue results in: {out_root}/{{domain}}/dialog_{{id}}.json")


def _print_summary(metric: str, summary: Dict[str, Any]) -> None:
    def _fmt(stats):
        if stats["n_pairs"] == 0:
            return "no data"
        return (
            f"A(baseline)={stats['win_rate_a']:.1%} ({stats['wins_a']})  "
            f"B(sa)={stats['win_rate_b']:.1%} ({stats['wins_b']})  "
            f"tie={stats['tie_rate']:.1%} ({stats['ties']})  "
            f"n={stats['n_pairs']}"
        )
    print(f"\n{'='*65}")
    print(f"PAIRWISE {metric.upper()} — Baseline (A) vs SA (B)")
    print(f"{'='*65}")
    print(f"Overall: {_fmt(summary)}")
    for d, s in sorted(summary.get("per_domain", {}).items()):
        print(f"  {d}: {_fmt(s)}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pairwise LLM judge: Baseline vs SA."
    )
    parser.add_argument(
        "--paired-dir", required=True,
        help="Path to paired_dialogues/baseline_vs_sa/ directory",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Root output directory (e.g. qwen/qwen-v2/pairwise_comparison/baseline_vs_sa)",
    )
    parser.add_argument(
        "--metric", choices=["usersat", "constraint", "truthfulness", "all"],
        default="all",
        help="Metric(s) to evaluate (default: all three)",
    )
    parser.add_argument(
        "--judge-model", default="openai/gpt-4o-mini",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=30,
    )
    parser.add_argument(
        "--domain", choices=DOMAINS,
        help="Restrict to one domain (default: both)",
    )
    args = parser.parse_args()

    domains = [args.domain] if args.domain else DOMAINS
    metrics = ["usersat", "constraint", "truthfulness"] if args.metric == "all" else [args.metric]
    out_root = Path(args.output_dir)

    print(f"[judge] Paired dir:   {args.paired_dir}")
    print(f"[judge] Output dir:   {out_root}")
    print(f"[judge] Metrics:      {metrics}")
    print(f"[judge] Judge model:  {args.judge_model}")
    print(f"[judge] Domains:      {domains}")

    pairs = load_pairs(args.paired_dir, domains=domains)
    print(f"[judge] Total pairs:  {len(pairs)}")
    if not pairs:
        print("[judge] No pairs found. Exiting.")
        return

    for metric in metrics:
        print(f"\n[judge] Running '{metric}' …")
        summary, per_pair = run_metric(
            pairs,
            metric=metric,
            judge_model=args.judge_model,
            max_concurrent=args.max_concurrent,
        )
        _print_summary(metric, summary)
        save_results(summary, per_pair, out_root / metric)

    print("\n[judge] Done.")


if __name__ == "__main__":
    main()
