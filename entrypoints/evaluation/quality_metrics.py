"""LLM-judged dialog quality metrics.

Adds four metrics on top of trait scoring:

* ``inform``       per system text turn (1-5 with SKIP).
* ``truthfulness`` per APICall-with-results (1-5 with SKIP).
* ``constraint``   one judge call per dialog returning a count of fulfilled
                   user-required slots; divided programmatically by total.
* ``user_sat``     one judge call per dialog scoring overall user satisfaction
                   from the user's perspective (1-5).

All averages are computed per-dialog first, then averaged across dialogs so that
long dialogs do not dominate corpus-level numbers.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
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

TRAIT_DESCRIPTIONS_STR: str = "\n".join(
    f"- {trait}: {desc}" for trait, desc in TRAIT_DESCRIPTIONS.items()
)


Trait_Expression_Prompt = (
    "You are an expert evaluator of personality expression in task-oriented dialogue.\n"
    "Your task is to identify the top three most evident personality traits expressed by the USER "
    "across the full dialogue and rate how strongly each trait is expressed, in ranked order.\n\n"
    "Possible User traits and their personality definitions:\n"
    "{trait_descriptions}\n\n"
    "Rate the response on a scale of 1-5 based on how much the trait is evident:\n"
    "1 = Very low level of this trait (strongly does not demonstrate this characteristic)\n"
    "2 = Low level of this trait (mostly does not demonstrate this characteristic)\n"
    "3 = Moderate/neutral level (mixed or unclear evidence)\n"
    "4 = High level of this trait (mostly demonstrates this characteristic)\n"
    "5 = Very high level of this trait (strongly demonstrates this characteristic)\n\n"
    "Instructions:\n"
    "1. Identify the top three most evident USER traits across the dialogue.\n"
    "2. Rank them from most evident to third most evident.\n"
    "3. Score each selected trait on a scale from 1 to 5.\n"
    "4. Use the full dialogue, not just one utterance.\n"
    "5. Base all judgments only on observable USER behavior, not on SYSTEM behavior.\n"
    "6. Do not infer broad personality characteristics beyond what is clearly supported by the USER’s utterances.\n"
    "7. Be strict, evidence-based, and concise.\n"
    "8. The three returned traits must be different from each other.\n"
    "9. Return only valid JSON.\n"
    "10. In the reason fields, do not use double quotation marks.\n"
    "11. If referring to dialogue text, use only single quotation marks ‘ ‘.\n"
    "12. Do not include any extra commentary, explanation, or markdown.\n\n"
    "Here is the current dialogue between a USER and a SYSTEM:\n"
    "{dialogue}\n\n"
    "After reading the entire dialogue, PLEASE THINK STEP BY STEP and respond ONLY in the "
    "following JSON format. Do not add any extra keys or text.\n\n"
    "{{\n"
    "  \"top1_evident_trait\": \"\",\n"
    "  \"top1_evident_trait_expression\": {{\n"
    "    \"score\": 1,\n"
    "    \"reason\": \"\"\n"
    "  }},\n"
    "  \"top2_evident_trait\": \"\",\n"
    "  \"top2_evident_trait_expression\": {{\n"
    "    \"score\": 1,\n"
    "    \"reason\": \"\"\n"
    "  }},\n"
    "  \"top3_evident_trait\": \"\",\n"
    "  \"top3_evident_trait_expression\": {{\n"
    "    \"score\": 1,\n"
    "    \"reason\": \"\"\n"
    "  }}\n"
    "}}"
)
                    
                    
                    
                   
INFORM_PROMPT = (
    "You are evaluating whether a system response answers the user's most recent "
    "message in a task-oriented dialog.\n\n"
    "User message:\n[USER START]\n{question}\n[USER END]\n\n"
    "System response:\n[SYSTEM START]\n{answer}\n[SYSTEM END]\n\n"
    "Rate how well the system response provides the information or service the user asked for.\n\n"
    "Respond with SKIP if any of the following apply:\n"
    "- the user did not ask for information or service (e.g., a confirmation, greeting, goodbye);\n"
    "- the system response is itself a clarifying question rather than an answer;\n"
    "- the system response is an API call placeholder rather than a user-facing reply.\n\n"
    "Otherwise respond with a single digit:\n"
    "1 = does not address the request\n"
    "2 = barely addresses it / mostly off-topic\n"
    "3 = partial answer, missing key info\n"
    "4 = mostly answers, minor gaps\n"
    "5 = fully and accurately addresses the request\n\n"
    "Respond with one of: SKIP, 1, 2, 3, 4, 5. Nothing else."
)


TRUTHFULNESS_PROMPT = (
    "You are evaluating whether a system response faithfully uses information from API "
    "search results in a task-oriented dialog.\n\n"
    "API call:\n[CALL START]\n{api_call}\n[CALL END]\n\n"
    "Search results returned:\n[RESULTS START]\n{search_results}\n[RESULTS END]\n\n"
    "System response to the user:\n[SYSTEM START]\n{system_response}\n[SYSTEM END]\n\n"
    "Rate how truthfully the system response uses the search results -- i.e., whether facts "
    "it states (names, numbers, attributes) appear in the results, with no fabrication or "
    "contradiction.\n\n"
    "1 = mostly fabricated or contradicts the results\n"
    "2 = significant hallucinations or wrong facts\n"
    "3 = partly accurate, partly wrong\n"
    "4 = mostly accurate, minor slips\n"
    "5 = fully grounded in the results\n\n"
    "Respond with SKIP if the system response asserts no checkable facts (e.g., it only "
    "asks a follow-up question).\n\n"
    "Respond with one of: SKIP, 1, 2, 3, 4, 5. Nothing else."
)


CONSTRAINT_PROMPT = (
    "You are evaluating a task-oriented dialog. The user requested specific information "
    "slots from the system across one or more API calls.\n\n"
    "Required slots (in order across API calls):\n{slot_list}\n\n"
    "Total required slots: {total_slots}\n\n"
    "Full dialog:\n[DIALOG START]\n{dialog_text}\n[DIALOG END]\n\n"
    "Count how many of the {total_slots} required slots the system successfully provided "
    "to the user, correctly and unambiguously (a slot counts only if the system's stated "
    "value is consistent with the search results shown in the dialog). Do not over-count: "
    "if a slot is restated multiple times, count it once.\n\n"
    "Respond with a single integer from 0 to {total_slots}. Nothing else."
)


USER_SAT_PROMPT = (
    "You are role-playing as the user in the task-oriented dialog below. The user's goal "
    "was to obtain specific information from the system.\n\n"
    "User's required slots (in order across API calls):\n{slot_list}\n\n"
    "Full dialog:\n[DIALOG START]\n{dialog_text}\n[DIALOG END]\n\n"
    "From the user's perspective, rate your overall satisfaction with how the system "
    "handled the conversation. Consider whether you got the information you asked for, "
    "whether the system was efficient, accurate, and easy to interact with.\n\n"
    "1 = very unsatisfied\n"
    "2 = unsatisfied\n"
    "3 = neutral\n"
    "4 = satisfied\n"
    "5 = very satisfied\n\n"
    "Respond with a single digit 1-5. Nothing else."
)


# ---------------------------------------------------------------------------
# Per-dialog item builders
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int = 6000) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars // 2]}\n\n[...truncated...]\n\n{text[-max_chars // 2 :]}"


def _render_dialog_text(generated_dialog: List[str], max_chars: int = 12000) -> str:
    """Compact dialog rendering for whole-dialog judge prompts."""
    parts: List[str] = []
    for turn in generated_dialog or []:
        if not isinstance(turn, str):
            continue
        t = turn.strip()
        if not t:
            continue
        # Trim huge search-result blocks so the dialog stays readable.
        if t.startswith("Search Results"):
            head = t.splitlines()[0]
            body = t[len(head):].strip()
            if len(body) > 800:
                body = body[:800] + " ...[truncated]"
            parts.append(f"{head}\n{body}".strip())
        else:
            parts.append(t)
    return _truncate("\n".join(parts), max_chars=max_chars)


def _build_inform_items(generated_dialog: List[str], dialog_file: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    last_user = ""
    for turn_idx, turn in enumerate(generated_dialog or []):
        if not isinstance(turn, str):
            continue
        if turn.startswith("User:"):
            last_user = turn[len("User:"):].strip()
            continue
        if not turn.startswith("System:"):
            continue
        sys_text = turn[len("System:"):].strip()
        if not sys_text or sys_text.startswith("APICall(") or sys_text.startswith("ApiCall("):
            continue
        if not last_user:
            continue
        items.append({
            "dialog_file": dialog_file,
            "turn_index": str(turn_idx),
            "question": _truncate(last_user),
            "answer": _truncate(sys_text),
        })
    return items


def _build_truthfulness_items(generated_dialog: List[str], dialog_file: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    turns = generated_dialog or []
    for turn_idx, turn in enumerate(turns):
        if not isinstance(turn, str) or not turn.startswith("System:"):
            continue
        body = turn[len("System:"):].strip()
        if not (body.startswith("APICall(") or body.startswith("ApiCall(")):
            continue
        # Find the next Search Results block and the next System: text turn.
        search_block = ""
        next_system_text = ""
        for j in range(turn_idx + 1, len(turns)):
            nt = turns[j]
            if not isinstance(nt, str):
                continue
            if nt.startswith("Search Results"):
                search_block = nt.strip()
                continue
            if nt.startswith("System:"):
                sub = nt[len("System:"):].strip()
                if sub.startswith("APICall(") or sub.startswith("ApiCall("):
                    break  # another api call before any text response
                next_system_text = sub
                break
            # User turn before a system text reply -> no truthfulness pairing
            if nt.startswith("User:"):
                break
        if not search_block or not next_system_text:
            continue
        items.append({
            "dialog_file": dialog_file,
            "turn_index": str(turn_idx),
            "api_call": _truncate(body, max_chars=2000),
            "search_results": _truncate(search_block, max_chars=6000),
            "system_response": _truncate(next_system_text, max_chars=3000),
        })
    return items


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


# ---------------------------------------------------------------------------
# Per-dialog orchestration
# ---------------------------------------------------------------------------

async def _score_dialog(
    *,
    sem: asyncio.Semaphore,
    dialog_file: str,
    generated_dialog: List[str],
    user_req_slots,
    inform_judge,
    truth_judge,
    constraint_judge,
    usersat_judge,
) -> Dict[str, Any]:
    async def call(judge, **kw):
        async with sem:
            try:
                return await judge(**kw)
            except Exception as exc:  # noqa: BLE001
                print(f"[quality] judge failed for {dialog_file}: {type(exc).__name__}: {exc}", file=sys.stderr)
                return None

    inform_items = _build_inform_items(generated_dialog, dialog_file)
    truth_items = _build_truthfulness_items(generated_dialog, dialog_file)

    inform_tasks = [
        asyncio.create_task(call(inform_judge, question=it["question"], answer=it["answer"]))
        for it in inform_items
    ]
    truth_tasks = [
        asyncio.create_task(
            call(
                truth_judge,
                api_call=it["api_call"],
                search_results=it["search_results"],
                system_response=it["system_response"],
            )
        )
        for it in truth_items
    ]

    slot_list = _flatten_req_slots(user_req_slots)
    total_slots = len(slot_list)
    dialog_text = _render_dialog_text(generated_dialog)
    slot_list_str = ", ".join(slot_list) if slot_list else "(none)"

    constraint_task: Optional[asyncio.Task] = None
    if total_slots > 0:
        constraint_task = asyncio.create_task(
            call(
                constraint_judge,
                slot_list=slot_list_str,
                total_slots=total_slots,
                dialog_text=dialog_text,
            )
        )

    usersat_task = asyncio.create_task(
        call(
            usersat_judge,
            slot_list=slot_list_str,
            dialog_text=dialog_text,
        )
    )

    inform_scores = [s for s in await asyncio.gather(*inform_tasks) if s is not None]
    truth_scores = [s for s in await asyncio.gather(*truth_tasks) if s is not None]
    usersat_score = await usersat_task
    constraint_raw = await constraint_task if constraint_task is not None else None

    inform_avg = float(sum(inform_scores) / len(inform_scores)) if inform_scores else None
    truth_avg = float(sum(truth_scores) / len(truth_scores)) if truth_scores else None

    if constraint_raw is None or total_slots == 0:
        constraint_count = None
        constraint_pct = None
    else:
        # Clamp the judge's count into [0, total_slots] before turning into a %.
        constraint_count = max(0.0, min(float(constraint_raw), float(total_slots)))
        constraint_pct = 100.0 * constraint_count / total_slots

    return {
        "dialog_file": dialog_file,
        "inform_score": inform_avg,
        "inform_n_items": len(inform_scores),
        "truthfulness_score": truth_avg,
        "truthfulness_n_items": len(truth_scores),
        "constraint_count_llm": constraint_count,
        "constraint_total_slots": total_slots,
        "constraint_satisfaction_pct": constraint_pct,
        "user_satisfaction_score": float(usersat_score) if usersat_score is not None else None,
    }


def evaluate_quality_metrics(
    dialogs: List[Dict[str, Any]],
    *,
    judge_model: str = "openai/gpt-4o-mini",
    max_concurrent_requests: int = 30,
) -> Tuple[Dict[str, Optional[float]], List[Dict[str, Any]]]:
    """Run the 4 quality judges across ``dialogs``.

    ``dialogs`` is a list of dicts with keys ``dialog_file``, ``generated_dialog``,
    and ``user_req_slots``. Returns a ``(summary, per_dialog)`` tuple.
    """
    from evaluation.big5.judge import OpenAiJudge  # noqa: E402

    inform_judge = OpenAiJudge(judge_model, INFORM_PROMPT, eval_type="1_5_skip")
    truth_judge = OpenAiJudge(judge_model, TRUTHFULNESS_PROMPT, eval_type="1_5_skip")
    constraint_judge = OpenAiJudge(judge_model, CONSTRAINT_PROMPT, eval_type="0_100")
    usersat_judge = OpenAiJudge(judge_model, USER_SAT_PROMPT, eval_type="1_5_skip")

    async def _run() -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(max_concurrent_requests)
        tasks = [
            asyncio.create_task(
                _score_dialog(
                    sem=sem,
                    dialog_file=d["dialog_file"],
                    generated_dialog=d.get("generated_dialog") or [],
                    user_req_slots=d.get("user_req_slots"),
                    inform_judge=inform_judge,
                    truth_judge=truth_judge,
                    constraint_judge=constraint_judge,
                    usersat_judge=usersat_judge,
                )
            )
            for d in dialogs
        ]
        return await asyncio.gather(*tasks)

    per_dialog = asyncio.run(_run())

    def _mean(key: str) -> Optional[float]:
        vals = [r[key] for r in per_dialog if r.get(key) is not None]
        return float(sum(vals) / len(vals)) if vals else None

    summary = {
        "inform_score": _mean("inform_score"),
        "truthfulness_score": _mean("truthfulness_score"),
        "constraint_satisfaction_pct": _mean("constraint_satisfaction_pct"),
        "user_satisfaction_score": _mean("user_satisfaction_score"),
        "n_dialogs_scored": len(per_dialog),
    }
    return summary, per_dialog


# ---------------------------------------------------------------------------
# Trait expression evaluation
# ---------------------------------------------------------------------------

def evaluate_trait_expression(
    dialogs: List[Dict[str, Any]],
    *,
    judge_model: str = "openai/gpt-4o-mini",
    max_concurrent_requests: int = 30,
) -> List[Dict[str, Any]]:
    """Run Trait_Expression_Prompt on each dialog.

    ``dialogs`` is a list of dicts with keys ``dialog_file`` and
    ``generated_dialog``.  Returns a list of per-dialog result dicts.
    """
    from evaluation.big5.judge_json import OpenAiJsonJudge

    judge = OpenAiJsonJudge(judge_model, Trait_Expression_Prompt, max_new_tokens=256, temperature=0.0)

    async def _run_one(sem: asyncio.Semaphore, d: Dict[str, Any]) -> Dict[str, Any]:
        dialog_text = _render_dialog_text(d.get("generated_dialog") or [])
        async with sem:
            try:
                result = await judge(
                    trait_descriptions=TRAIT_DESCRIPTIONS_STR,
                    dialogue=dialog_text,
                )
                if "error" in result:
                    raise RuntimeError(result["error"])
                return {"dialog_file": d["dialog_file"], **result}
            except Exception as exc:
                print(f"[trait_expr] judge failed for {d['dialog_file']}: {exc}", file=sys.stderr)
                return {"dialog_file": d["dialog_file"], "error": str(exc)}

    async def _run():
        sem = asyncio.Semaphore(max_concurrent_requests)
        tasks = [asyncio.create_task(_run_one(sem, d)) for d in dialogs]
        return await asyncio.gather(*tasks)

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Trait_Expression_Prompt on one or more dialog JSON files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dialog-file", help="Path to a single dialog_*.json file")
    group.add_argument("--dialogs-dir", help="Directory containing dialog_*.json files")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini")
    parser.add_argument("--max-concurrent", type=int, default=30)
    args = parser.parse_args()

    if args.dialog_file:
        files = [Path(args.dialog_file)]
    else:
        files = sorted(Path(args.dialogs_dir).glob("dialog_*.json"))

    dialogs = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            dialogs.append({"dialog_file": str(f), "generated_dialog": data.get("generated_dialog", [])})
        except Exception as e:
            print(f"[trait_expr] Error loading {f}: {e}", file=sys.stderr)

    print(f"[trait_expr] Loaded {len(dialogs)} dialog(s). Running judge ({args.judge_model}) ...")
    results = evaluate_trait_expression(dialogs, judge_model=args.judge_model, max_concurrent_requests=args.max_concurrent)

    for r in results:
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
