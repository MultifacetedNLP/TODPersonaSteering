"""Pairwise dialog comparison metrics.

Compares matched pairs of dialogs (by dialog ID) across two experimental conditions:
  User-Steering (US) vs User-Steering+SA (US+SA)

For each (run_label, domain) pair the trait and scalar are extracted from the run
directory name (format: {trait}__s{scalar}) and forwarded to the judge prompt so
the model knows who the user is before evaluating.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path as _Path

# Ensure project root is on sys.path so `evaluation.big5` resolves correctly.
_PROJECT_ROOT = str(_Path(__file__).resolve().parents[2])
if not sys.path or sys.path[0] != _PROJECT_ROOT:
    sys.path.insert(0, _PROJECT_ROOT)
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_trait_descriptions() -> Dict[str, str]:
    """Extract plain-text trait definitions from evaluation/big5/trait_prompts.json."""
    path = Path(__file__).resolve().parents[2] / "evaluation" / "big5" / "trait_prompts.json"
    with open(path) as f:
        raw = json.load(f)
    descriptions: Dict[str, str] = {}
    for trait, prompt in raw.items():
        # Each value has the form: "...displays the trait: **name**. <description>\n\nPrompt:..."
        m = re.search(r'\*\*[^*]+\*\*\.\s*(.*?)\n\nPrompt:', prompt, re.DOTALL)
        descriptions[trait] = m.group(1).strip() if m else ""
    return descriptions

TRAIT_DESCRIPTIONS: Dict[str, str] = _load_trait_descriptions()

PAIRWISE_CONSTRAINT_PROMPT = """You are comparing two task-oriented dialogues produced with the same user setup.

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

PAIRWISE_TRUTHFULNESS_PROMPT = """You are comparing two task-oriented dialogues produced with the same user setup.

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

PAIRWISE_USER_SAT_PROMPT = """You are comparing two task-oriented dialogues produced with the same user setup.

Evaluate only personality-conditioned user satisfaction.

Personality trait: {trait}
Steering coefficient: {scalar}
Trait definition: {trait_description}
Task domain: {domain}

Dialogue A:
[DIALOGUE A START]
{dialogue_a}
[DIALOGUE A END]

Dialogue B:
[DIALOGUE B START]
{dialogue_b}
[DIALOGUE B END]

Choose which SYSTEM better satisfies this user.

Check whether:

* The tone and wording fit the user's personality
* The system responds well to the user's preferences and reactions
* The user is likely to feel understood and supported
* The adaptation is natural, not exaggerated or stereotypical

Do not judge task completion unless it clearly harms the interaction.
Choose "tie" if neither dialogue has a meaningful advantage.

Return only valid JSON:
{{"winner": "A" | "B" | "tie", "reason": "Brief reason."}}"""



# ---------------------------------------------------------------------------
# Dialog text rendering
# ---------------------------------------------------------------------------

def _render_dialog_for_comparison(generated_dialog: List[str], max_chars: int = 12000) -> str:
    """Render dialog turns for pairwise comparison, truncating long search blocks."""
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


def _extract_dialog_id(filename: str) -> Optional[str]:
    """Extract dialog ID from filename like 'dialog_0_00001.json' -> '0_00001'."""
    match = re.search(r'dialog_(\d+_\d+)\.json', filename)
    return match.group(1) if match else None


def _parse_run_label(run_label: str) -> Tuple[str, str]:
    """Extract (trait, scalar) from run label like 'calm__s-1' -> ('calm', '-1')."""
    if "__s" in run_label:
        parts = run_label.split("__s")
        return parts[0], parts[-1]
    return run_label, "0"


def _domain_display(domain: str) -> str:
    if "Restaurant" in domain:
        return "Restaurant booking"
    if "Movie" in domain:
        return "Movie finding"
    return domain


# ---------------------------------------------------------------------------
# Dialog loading
# ---------------------------------------------------------------------------

def _flatten_req_slots(user_req_slots) -> List[str]:
    flat: List[str] = []
    if not user_req_slots:
        return flat
    for group in user_req_slots:
        if isinstance(group, (list, tuple)):
            flat.extend(str(s) for s in group)
        else:
            flat.append(str(group))
    return flat


def _load_dialogs_from_directory(directory: str) -> Dict[str, Dict[str, Any]]:
    """Load all dialog_*.json files from a single flat directory.

    Returns:
        Dict mapping dialog_id -> {"filename", "filepath", "generated_dialog",
                                    "user_req_slots", "api_calls", "search_results"}
    """
    dialogs: Dict[str, Dict[str, Any]] = {}
    d = Path(directory)
    if not d.exists():
        return dialogs

    for f in d.glob("dialog_*.json"):
        dialog_id = _extract_dialog_id(f.name)
        if not dialog_id:
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            dialogs[dialog_id] = {
                "filename":        f.name,
                "filepath":        str(f),
                "generated_dialog": data.get("generated_dialog", []),
                "user_req_slots":  data.get("user_req_slots", []),
                "api_calls":       data.get("api_calls", []),
                "search_results":  data.get("search_results", []),
            }
        except (json.JSONDecodeError, IOError) as e:
            print(f"[pairwise] Error loading {f}: {e}", file=sys.stderr)

    return dialogs


# ---------------------------------------------------------------------------
# Pair matching
# ---------------------------------------------------------------------------

def create_matched_pairs(
    us_base: str,
    us_sa_base: str,
    domains: List[str] = None,
    run_label_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Walk all runs/domains under us_base and us_sa_base and build matched pairs.

    Pairs are matched by (run_label, domain, dialog_id).  Trait and scalar are
    extracted from the run_label directory name.

    Args:
        us_base:          Root of the User-Steering experiment
                          (e.g. "qwen/qwen-v2/qwen-user-steer-2026-05-25-15-49")
        us_sa_base:        Root of the User-Steering+SA experiment
        domains:          Domains to process (default: both Restaurants_2 and Movies)
        run_label_filter: If given, only process this one run label.

    Returns:
        List of pair dicts, each containing:
            dialog_id, run_label, trait, scalar, domain,
            condition_a, condition_b,
            dialogue_a, dialogue_b,
            filepath_a, filepath_b
    """
    if domains is None:
        domains = ["Restaurants_2", "Movies_1_Movies_3"]

    us_runs_dir   = Path(us_base)   / "runs"
    us_sa_runs_dir = Path(us_sa_base) / "runs"

    if not us_runs_dir.exists():
        raise FileNotFoundError(f"US runs directory not found: {us_runs_dir}")
    if not us_sa_runs_dir.exists():
        raise FileNotFoundError(f"US+SA runs directory not found: {us_sa_runs_dir}")

    run_labels = sorted(d.name for d in us_runs_dir.iterdir() if d.is_dir())
    if run_label_filter:
        run_labels = [r for r in run_labels if r == run_label_filter]

    pairs: List[Dict[str, Any]] = []

    for run_label in run_labels:
        trait, scalar = _parse_run_label(run_label)

        for domain in domains:
            us_dir   = us_runs_dir   / run_label / "dialogs" / domain
            us_sa_dir = us_sa_runs_dir / run_label / "dialogs" / domain

            if not us_dir.exists() or not us_sa_dir.exists():
                continue

            dialogs_us   = _load_dialogs_from_directory(str(us_dir))
            dialogs_us_sa = _load_dialogs_from_directory(str(us_sa_dir))

            common_ids = set(dialogs_us) & set(dialogs_us_sa)
            if not common_ids:
                print(
                    f"[pairwise] No common IDs for {run_label}/{domain} "
                    f"(US={len(dialogs_us)}, US+SA={len(dialogs_us_sa)})",
                    file=sys.stderr,
                )
                continue

            for dialog_id in sorted(common_ids):
                da = dialogs_us[dialog_id]
                db = dialogs_us_sa[dialog_id]

                # user_req_slots, api_calls, search_results are SGD ground truth —
                # identical for both conditions (same dialogue_id), so we use A's.
                slot_list = _flatten_req_slots(da["user_req_slots"])
                api_calls_str = "\n".join(str(c) for c in da["api_calls"])
                search_results_str = "\n".join(str(r) for r in da["search_results"])

                pairs.append({
                    "dialog_id":          dialog_id,
                    "run_label":          run_label,
                    "trait":              trait,
                    "scalar":             scalar,
                    "domain":             domain,
                    "domain_display":     _domain_display(domain),
                    "condition_a":        "user-steer",
                    "condition_b":        "user-steer+sa",
                    "dialogue_a":         _render_dialog_for_comparison(da["generated_dialog"]),
                    "dialogue_b":         _render_dialog_for_comparison(db["generated_dialog"]),
                    "filepath_a":         da["filepath"],
                    "filepath_b":         db["filepath"],
                    "slot_list":          slot_list,
                    "total_slots":        len(slot_list),
                    "api_calls_str":      api_calls_str,
                    "search_results_str": search_results_str,
                })

    return pairs


# ---------------------------------------------------------------------------
# Pairwise evaluation
# ---------------------------------------------------------------------------

def _build_judge_kwargs(pair: Dict[str, Any], metric: str) -> Optional[Dict[str, Any]]:
    """Return the kwargs dict to pass to the judge for the given metric.

    Returns None when the pair should be skipped (e.g. no slots for constraint).
    """
    if metric == "usersat":
        return {
            "dialogue_a":        pair["dialogue_a"],
            "dialogue_b":        pair["dialogue_b"],
            "trait":             pair["trait"],
            "scalar":            pair["scalar"],
            "domain":            pair["domain_display"],
            "trait_description": TRAIT_DESCRIPTIONS.get(pair["trait"], ""),
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
    raise ValueError(f"Unknown metric: {metric!r}. Choose usersat, constraint, or truthfulness.")


async def _judge_pair(
    *,
    sem: asyncio.Semaphore,
    pair: Dict[str, Any],
    judge,
    metric: str,
) -> Dict[str, Any]:
    """Run one pairwise judge on a single pair."""
    base = {
        "dialog_id":   pair["dialog_id"],
        "run_label":   pair["run_label"],
        "trait":       pair["trait"],
        "scalar":      pair["scalar"],
        "domain":      pair["domain"],
        "condition_a": pair["condition_a"],
        "condition_b": pair["condition_b"],
        "filepath_a":  pair["filepath_a"],
        "filepath_b":  pair["filepath_b"],
        "metric":      metric,
    }

    kwargs = _build_judge_kwargs(pair, metric)
    if kwargs is None:
        return {**base, "winner": None, "reason": "skipped (no slots)"}

    async with sem:
        try:
            result = await judge(**kwargs)
            if "error" in result:
                raise RuntimeError(result["error"])
            return {
                **base,
                "winner":      result.get("winner"),
                "reason":      result.get("reason", ""),
                "count_a":     result.get("count_a"),
                "count_b":     result.get("count_b"),
            }
        except Exception as exc:
            print(
                f"[pairwise] judge failed for {pair['dialog_id']} "
                f"({pair['run_label']}/{pair['domain']}): {exc}",
                file=sys.stderr,
            )
            return {**base, "winner": None, "reason": f"Error: {exc}"}


def _win_rates(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute win/tie rates from a list of result dicts."""
    n = len(results)
    if n == 0:
        return {"win_rate_a": None, "win_rate_b": None, "tie_rate": None, "n_pairs": 0}
    win_a = sum(1 for r in results if r["winner"] == "A")
    win_b = sum(1 for r in results if r["winner"] == "B")
    ties  = sum(1 for r in results if r["winner"] == "tie")
    return {
        "win_rate_a": win_a / n,
        "win_rate_b": win_b / n,
        "tie_rate":   ties  / n,
        "wins_a": win_a,
        "wins_b": win_b,
        "ties":   ties,
        "n_pairs": n,
    }


_METRIC_PROMPTS = {
    "usersat":      PAIRWISE_USER_SAT_PROMPT,
    "constraint":   PAIRWISE_CONSTRAINT_PROMPT,
    "truthfulness": PAIRWISE_TRUTHFULNESS_PROMPT,
}


def evaluate_pairwise_metrics(
    pairs: List[Dict[str, Any]],
    *,
    metric: str = "usersat",
    judge_model: str = "openai/gpt-4o-mini",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    max_concurrent_requests: int = 30,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run a single pairwise metric across matched dialog pairs.

    Args:
        pairs:                   Output of ``create_matched_pairs()``.
        metric:                  One of ``"usersat"``, ``"constraint"``, ``"truthfulness"``.
        judge_model:             OpenAI / OpenRouter model identifier.
        max_new_tokens:          Max tokens in judge response (default 256).
        temperature:             Judge sampling temperature (default 0.0).
        max_concurrent_requests: Semaphore limit for async API calls.

    Returns:
        (summary, per_pair_results)
        summary contains overall win rates and breakdowns by domain, trait, scalar.
    """
    if metric not in _METRIC_PROMPTS:
        raise ValueError(f"Unknown metric {metric!r}. Choose: {list(_METRIC_PROMPTS)}")

    from evaluation.big5.judge_json import OpenAiJsonJudge

    judge = OpenAiJsonJudge(
        judge_model,
        _METRIC_PROMPTS[metric],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    async def _run() -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(max_concurrent_requests)
        tasks = [
            asyncio.create_task(_judge_pair(sem=sem, pair=p, judge=judge, metric=metric))
            for p in pairs
        ]
        return await asyncio.gather(*tasks)

    per_pair = asyncio.run(_run())

    valid = [r for r in per_pair if r["winner"] is not None]

    # Overall
    summary: Dict[str, Any] = {
        "condition_a": pairs[0]["condition_a"] if pairs else "user-steer",
        "condition_b": pairs[0]["condition_b"] if pairs else "user-steer+sa",
        **_win_rates(valid),
    }

    # Per domain
    by_domain: Dict[str, List] = defaultdict(list)
    for r in valid:
        by_domain[r["domain"]].append(r)
    summary["per_domain"] = {d: _win_rates(v) for d, v in sorted(by_domain.items())}

    # Per trait
    by_trait: Dict[str, List] = defaultdict(list)
    for r in valid:
        by_trait[r["trait"]].append(r)
    summary["per_trait"] = {t: _win_rates(v) for t, v in sorted(by_trait.items())}

    # Per scalar
    by_scalar: Dict[str, List] = defaultdict(list)
    for r in valid:
        by_scalar[r["scalar"]].append(r)
    summary["per_scalar"] = {
        s: _win_rates(v)
        for s, v in sorted(by_scalar.items(), key=lambda x: int(x[0]))
    }

    # Per (trait, scalar) — finest granularity
    by_trait_scalar: Dict[str, List] = defaultdict(list)
    for r in valid:
        by_trait_scalar[f"{r['trait']}__s{r['scalar']}"].append(r)
    summary["per_run"] = {k: _win_rates(v) for k, v in sorted(by_trait_scalar.items())}

    return summary, per_pair


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_win_rates(label: str, stats: Dict[str, Any], indent: int = 2) -> None:
    pad = " " * indent
    if stats["n_pairs"] == 0:
        print(f"{pad}{label}: no data")
        return
    print(
        f"{pad}{label}: "
        f"A={stats['win_rate_a']:.1%} ({stats.get('wins_a',0)})  "
        f"B={stats['win_rate_b']:.1%} ({stats.get('wins_b',0)})  "
        f"tie={stats['tie_rate']:.1%} ({stats.get('ties',0)})  "
        f"n={stats['n_pairs']}"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pairwise user satisfaction: User-Steering vs User-Steering+SA."
    )
    parser.add_argument(
        "--us-dir", required=True,
        help="Root of the User-Steering experiment "
             "(e.g. qwen/qwen-v2/qwen-user-steer-2026-05-25-15-49)",
    )
    parser.add_argument(
        "--us_sa-dir", required=True,
        help="Root of the User-Steering+SA experiment",
    )
    parser.add_argument(
        "--domain", choices=["Restaurants_2", "Movies_1_Movies_3"],
        help="Restrict to a single domain (default: both)",
    )
    parser.add_argument(
        "--run-label",
        help="Restrict to a single run label (e.g. calm__s-1)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file (default: pairwise_results.json next to --us-dir)",
    )
    parser.add_argument(
        "--metric",
        choices=["usersat", "constraint", "truthfulness"],
        default="usersat",
        help="Which pairwise metric to evaluate (default: usersat)",
    )
    parser.add_argument(
        "--judge-model", default="openai/gpt-4o-mini",
        help="OpenAI / OpenRouter model to use as judge",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Max tokens in judge response (default 256)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Judge sampling temperature (default 0.0)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=30,
        help="Max concurrent API requests",
    )

    args = parser.parse_args()

    domains = [args.domain] if args.domain else ["Restaurants_2", "Movies_1_Movies_3"]

    print(f"[pairwise] User-Steering (A):    {args.us_dir}")
    print(f"[pairwise] User-Steering+SA (B): {args.us_sa_dir}")
    print(f"[pairwise] Metric:  {args.metric}")
    print(f"[pairwise] Domains: {domains}")

    pairs = create_matched_pairs(
        args.us_dir,
        args.us_sa_dir,
        domains=domains,
        run_label_filter=args.run_label,
    )

    print(f"[pairwise] Matched pairs: {len(pairs)}")
    if not pairs:
        print("[pairwise] No pairs found. Exiting.")
        return

    # Show pair breakdown
    domain_counts: Dict[str, int] = defaultdict(int)
    trait_counts:  Dict[str, int] = defaultdict(int)
    for p in pairs:
        domain_counts[p["domain"]] += 1
        trait_counts[p["trait"]]   += 1
    for d, n in sorted(domain_counts.items()):
        print(f"  {d}: {n} pairs")

    print(f"\n[pairwise] Running {args.metric!r} evaluation with {args.judge_model}...")

    summary, per_pair = evaluate_pairwise_metrics(
        pairs,
        metric=args.metric,
        judge_model=args.judge_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_concurrent_requests=args.max_concurrent,
    )

    # Print results
    print("\n" + "=" * 65)
    print(f"PAIRWISE {args.metric.upper()} — User-Steer (A) vs User-Steer+SA (B)")
    print("=" * 65)
    _print_win_rates("Overall", summary, indent=0)

    print("\nBy domain:")
    for d, stats in sorted(summary.get("per_domain", {}).items()):
        _print_win_rates(d, stats)

    print("\nBy scalar:")
    for s, stats in sorted(summary.get("per_scalar", {}).items(), key=lambda x: int(x[0])):
        _print_win_rates(f"scalar={s}", stats)

    print("\nBy trait:")
    for t, stats in sorted(summary.get("per_trait", {}).items()):
        _print_win_rates(t, stats)

    print("\nBy run (trait × scalar):")
    for run, stats in sorted(summary.get("per_run", {}).items()):
        _print_win_rates(run, stats)

    # Save
    # Root output dir: pairwise_comparison/{metric}/
    out_root = Path(
        args.output
        or str(Path(__file__).resolve().parents[2] / "qwen" / "qwen-v2" / "pairwise_comparison")
    ) / args.metric
    out_root.mkdir(parents=True, exist_ok=True)

    # Summary
    summary_path = out_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Per-dialogue files: {run_label}/{domain}/dialog_{id}.json
    for r in per_pair:
        pair_dir = out_root / r["run_label"] / r["domain"]
        pair_dir.mkdir(parents=True, exist_ok=True)
        pair_path = pair_dir / f"dialog_{r['dialog_id']}.json"
        with open(pair_path, "w") as f:
            json.dump(r, f, indent=2)

    print(f"\n[pairwise] Summary saved to:      {summary_path}")
    print(f"[pairwise] Per-dialogue results in: {out_root}/<run_label>/<domain>/dialog_<id>.json")


if __name__ == "__main__":
    main()
