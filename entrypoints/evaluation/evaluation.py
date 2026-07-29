"""Evaluation entrypoint for SGD inference runs.

Reads the per-dialog JSON files produced by inference runs and emits
task-success metrics (API accuracy, dialog completion, requested-slot
answering, NLG fluency) and, when enabled, Big-Five trait scores from an
LLM judge.

Run as a module:

    python -m entrypoints.evaluation.evaluation --json-dir storage/<run_dir>

Use ``--evaluate-traits`` / ``--only-traits`` for trait scoring. See
``docs/ARCHITECTURE.md`` for where this fits in the broader pipeline.
"""

import json
import os
import re
import argparse
import logging
import utils
from tqdm import tqdm
from utilities import text_utilities
import ast
from fuzzywuzzy import fuzz
import evaluate
import numpy as np
from collections import deque
import sys
import asyncio
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def _flatten_metric_rows(prefix: str, value):
    if isinstance(value, dict):
        rows = []
        for key, nested in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_metric_rows(nested_prefix, nested))
        return rows
    return [{"metric": prefix, "value": value}]


def _write_metrics_xlsx(json_directory: str, metrics: dict, trait_payload: Optional[dict] = None) -> None:
    """Write a compact Excel companion for quick inspection outside Python."""
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas is not installed; skipping metrics.xlsx export")
        return

    output_path = Path(json_directory) / "metrics.xlsx"
    summary = metrics.get("summary", {})
    summary_rows = []
    for metric_name, metric_value in summary.items():
        summary_rows.extend(_flatten_metric_rows(metric_name, metric_value))

    try:
        with pd.ExcelWriter(output_path) as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
            if trait_payload:
                trait_summary = trait_payload.get("summary", {})
                trait_rows = []
                for metric_name, metric_value in trait_summary.items():
                    trait_rows.extend(_flatten_metric_rows(metric_name, metric_value))
                pd.DataFrame(trait_rows).to_excel(writer, sheet_name="trait_summary", index=False)
                items = trait_payload.get("items") or []
                if items:
                    pd.DataFrame(items).to_excel(writer, sheet_name="trait_items", index=False)
    except Exception as exc:
        logger.warning("failed to write metrics.xlsx: %s", exc)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_search_results_block(results: str):
    """Parse a 'Search Results' payload that may be a Python repr of list[dict].

    Generated outputs sometimes contain numpy scalar reprs like 'np.int64(30)',
    which breaks ast.literal_eval. This helper sanitizes common cases.
    """
    if results is None:
        return []
    if not isinstance(results, str):
        if isinstance(results, dict):
            return [results]
        if isinstance(results, list):
            return results
        return []

    s = results.strip()
    if s == "":
        return []

    s = re.sub(r"\b(?:np|numpy)\.nan\b", "None", s)
    s = re.sub(r"\bmath\.nan\b", "None", s)
    s = re.sub(r"float\(\s*['\"]nan['\"]\s*\)", "None", s)
    s = re.sub(r"\bNaN\b", "None", s)
    s = re.sub(r"\bnan\b", "None", s)
    np_scalar_re = re.compile(
        r"\b(?:np|numpy)\.(?:int64|int32|int16|int8|uint64|uint32|uint16|uint8|float64|float32|float16|bool_)\(([^()]+)\)"
    )
    while True:
        s2, n = np_scalar_re.subn(r"\1", s)
        s = s2
        if n == 0:
            break

    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

    if parsed is None:
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


def adjust_list_length(my_list, m):
    n = len(my_list)
    if m > n:
        return my_list + [""] * (m - n)
    elif m < n:
        return my_list[:m]
    return my_list


def process_dialogs(generated_dialog, reference_dialog):
    pattern = r"Search Results\n(.*?)\nEnd Search Results"
    last_system_utterance_pattern = r"Last System Utterance:.*$"
    cleaned_generated_dialog = [t for t in generated_dialog if not t.startswith("Search Results\n")]
    generated_dialog_text = "\n".join(cleaned_generated_dialog).lower().replace("|endofdialog|>", "")
    reference_dialog_text = re.sub(pattern, "", reference_dialog)
    reference_dialog_text = re.sub(
        last_system_utterance_pattern, "", reference_dialog_text, flags=re.MULTILINE
    ).replace("Last User Utterance:", "User: ").strip().lower()
    return generated_dialog_text, reference_dialog_text


def remove_user_turns(generated_dialog, reference_dialog):
    reference_dialog_processed = (
        reference_dialog.replace("Last User Utterance:", "User: ").replace("Last System Utterance:", "System:")
    )
    reference_dialog_processed = re.sub(r"User:.*\n", "", reference_dialog_processed)
    generated_dialog_processed = [t for t in generated_dialog if not t.startswith("User:")]
    return generated_dialog_processed, reference_dialog_processed


def remove_system_turns(generated_dialog, reference_dialog):
    reference_dialog_processed = (
        reference_dialog.replace("Last User Utterance:", "User: ").replace("Last System Utterance:", "System:")
    )
    reference_dialog_processed = re.sub(r"System:.*\n", "", reference_dialog_processed + "\n")
    generated_dialog_processed = [t for t in generated_dialog if not t.startswith("System:")]
    return generated_dialog_processed, reference_dialog_processed


def calculate_num_answered_slots(generated_dialog, user_req_slots, search_results, api_call_marker="APICall("):
    """Count how many requested slots appear in the system response after each API call."""
    api_id = 0
    num_answered_slots = 0
    queue = deque(maxlen=2)
    for turn in generated_dialog:
        if not turn.startswith("System:"):
            continue
        if api_call_marker in turn:
            api_id += 1
        if api_call_marker not in turn and api_id >= 1:
            queue.append(turn)
            current_api_id = api_id - 1
            if current_api_id >= len(user_req_slots) or current_api_id >= len(search_results):
                continue
            requested_slots = user_req_slots[current_api_id]
            raw_service_results = search_results[current_api_id]
            if isinstance(raw_service_results, str):
                service_results_payload = (
                    raw_service_results.replace("Search Results\n", "").replace("\nEnd Search Results", "").strip()
                )
            else:
                service_results_payload = raw_service_results
            if not service_results_payload:
                continue
            service_results = _parse_search_results_block(service_results_payload)
            if not service_results:
                continue
            answered_slots_all_result = []
            for result in service_results:
                answered_slots_per_result = []
                for slot in requested_slots:
                    if slot in result and result[slot] is not None:
                        for q_turn in list(queue):
                            t = q_turn.replace("System: ", "")
                            slot_value = result[slot]
                            match_ratio = fuzz.partial_ratio(str(slot_value), t)
                            if (
                                match_ratio >= 80
                                or str(slot_value).lower() in t.lower()
                                or (slot_value in (True, False) and fuzz.partial_ratio(slot, t) > 50)
                            ):
                                answered_slots_per_result.append(slot)
                                break
                answered_slots_all_result.append(answered_slots_per_result)
            if not answered_slots_all_result:
                continue
            num_answered_slots += max(len(sl) for sl in answered_slots_all_result)
            longest_answered = max(answered_slots_all_result, key=len)
            user_req_slots[current_api_id] = [s for s in requested_slots if s not in longest_answered]
    return num_answered_slots


def calculate_slots_with_search_results(user_req_slots, search_results):
    total_slots_with_search_results = 0
    num_slots = len(user_req_slots)
    if len(search_results) > num_slots:
        search_results = search_results[:num_slots]
    for i, turn in enumerate(search_results):
        results = turn.replace("Search Results\n", "").replace("\nEnd Search Results", "")
        if results != "":
            parsed = _parse_search_results_block(results)
            if not parsed:
                continue
            keys: set = set()
            for r in parsed:
                if isinstance(r, dict):
                    keys.update(r.keys())
            if keys and any(slot in keys for slot in user_req_slots[i]):
                total_slots_with_search_results += len(user_req_slots[i])
    return total_slots_with_search_results


# ---------------------------------------------------------------------------
# Metric calculators
# ---------------------------------------------------------------------------

class SGDMetricCalculator:
    """Metric calculator for SGD-style (flat string value) API calls."""

    def get_method_metric(self, pred_method, ref_method):
        return 1 if pred_method == ref_method else 0

    def get_key_value_metric(self, pred_params, ref_params):
        if ref_params == {}:
            if pred_params == {}:
                return 1.0, 1.0, 1.0, 1.0
            return 0.0, 0.0, 0.0, 0.0

        ref_date_keys = [k for k in ref_params if "date" in k.lower()]
        param_accs, value_accs = [], []
        pred_keys = set(pred_params)
        ref_keys = set(ref_params)
        intersection_keys = pred_keys & ref_keys
        union_keys = pred_keys | ref_keys

        for k, v in ref_params.items():
            if k in pred_params:
                param_accs.append(1)
                if k in ref_date_keys:
                    value_accs.append(1)
                else:
                    value_accs.append(utils.fuzzy_string_match(pred_params[k], v))
            else:
                param_accs.append(0)
                value_accs.append(0)

        param_acc = np.mean(param_accs)
        value_acc = np.mean(value_accs)
        param_iou = len(intersection_keys) / len(union_keys)
        value_iou = sum(value_accs) / len(union_keys)
        return param_acc, value_acc, param_iou, value_iou

    def calculate(self, APIcall, action):
        pred_method = text_utilities.get_apicall_method_from_text(action)
        ref_method = text_utilities.get_apicall_method_from_text(APIcall)
        method_accuracy = self.get_method_metric(pred_method, ref_method)
        pred_params = text_utilities.get_parameters_from_gpt_generated_text_gpt(action)
        ref_params = text_utilities.get_parameters_from_text(APIcall)
        key_accuracy, value_accuracy, key_iou, value_iou = self.get_key_value_metric(pred_params, ref_params)
        full_api_accuracy = 1 if method_accuracy == 1 and key_accuracy == 1 and value_accuracy == 1 else 0
        full_api_accuracy_iou = 1 if method_accuracy == 1 and key_iou == 1 and value_iou == 1 else 0
        return method_accuracy, key_accuracy, value_accuracy, full_api_accuracy, key_iou, value_iou, full_api_accuracy_iou



# ---------------------------------------------------------------------------
# Per-dialog metric aggregation
# ---------------------------------------------------------------------------

def _calculate_metrics_sgd(api_calls, predicted_api_calls):
    """Return 7 per-call metric lists for SGD."""
    mc = SGDMetricCalculator()
    method_acc, key_acc, val_acc, full_api_acc, key_iou, val_iou, full_iou = [], [], [], [], [], [], []
    for ref_call, pred_call in zip(api_calls, adjust_list_length(predicted_api_calls, len(api_calls))):
        m, ka, va, fa, ki, vi, fi = mc.calculate(ref_call, pred_call)
        method_acc.append(m)
        key_acc.append(ka)
        val_acc.append(va)
        full_api_acc.append(fa)
        key_iou.append(ki)
        val_iou.append(vi)
        full_iou.append(fi)
    return method_acc, key_acc, val_acc, full_api_acc, key_iou, val_iou, full_iou


calculate_metrics = _calculate_metrics_sgd


# ---------------------------------------------------------------------------
# Trait evaluation (SGD only)
# ---------------------------------------------------------------------------

def _extract_role_text(
    turns: List[str],
    role_prefix: str,
    *,
    exclude_substrings: Optional[List[str]] = None,
) -> str:
    exclude_substrings = exclude_substrings or []
    out: List[str] = []
    for t in turns:
        if not isinstance(t, str) or not t.startswith(role_prefix):
            continue
        if any(s in t for s in exclude_substrings):
            continue
        content = t[len(role_prefix):].strip()
        if content:
            out.append(content)
    return "\n".join(out).strip()


def _truncate_for_judge(text: str, max_chars: int = 6000) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n\n[...truncated...]\n\n{tail}"


def evaluate_trait_metrics(
    items: List[Dict[str, str]],
    *,
    judge_model: str = "openai/gpt-4o-mini",
    max_concurrent_requests: int = 30,
) -> Tuple[Dict[str, Optional[float]], List[Dict[str, Optional[float]]]]:
    """Evaluate trait-pole scores for individual user messages."""
    persona_tod_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, persona_tod_dir)

    from evaluation.big5.judge import OpenAiJudge  # noqa: E402
    from evaluation import big5 as _big5_pkg  # noqa: E402

    prompts_path = os.path.join(os.path.dirname(_big5_pkg.__file__), "trait_prompts.json")
    with open(prompts_path, "r") as f:
        trait_prompts = json.load(f)

    traits = [
        "calm", "careless", "compassionate", "consistent", "dependable",
        "inventive", "nervous", "outgoing", "self-interested", "solitary",
    ]
    missing = [t for t in traits if t not in trait_prompts]
    if missing:
        raise ValueError(f"Missing trait prompts in {prompts_path}: {missing}")

    trait_judges = {t: OpenAiJudge(judge_model, trait_prompts[t], eval_type="0_100") for t in traits}
    coherence_judges = {
        t: OpenAiJudge(
            judge_model,
            (
                "I am evaluating a different language model. I want to assess the coherence of its answer.\n"
                f"The model is attempting to display the personality trait: **{t}**.\n"
                "Please rate only coherence (clarity, internal consistency, and fit to the question), not whether the content is correct.\n\n"
                "Evaluation question:\n[QUESTION START]\n{question}\n[QUESTION END]\n\n"
                "Model response:\n[ANSWER START]\n{answer}\n[ANSWER END]\n\n"
                "Respond with a number between 0 and 100. Don't say anything else, just the number."
            ),
            eval_type="0_100",
        )
        for t in traits
    }

    async def _run() -> List[Dict[str, Optional[float]]]:
        sem = asyncio.Semaphore(max_concurrent_requests)
        per_item_results: List[Dict[str, Optional[float]]] = []

        async def bounded_call(judge, *, question: str, answer: str):
            async with sem:
                return await judge(question=question, answer=answer)

        tasks: List[asyncio.Task] = []
        item_refs: List[Dict[str, str]] = []
        for item in items:
            question = _truncate_for_judge(item.get("question", ""))
            answer = _truncate_for_judge(item.get("answer", ""))
            if not answer.strip():
                continue
            item_refs.append(item)

            async def eval_one(q=question, a=answer):
                trait_vals: Dict[str, Optional[float]] = {}
                trait_cohs: Dict[str, Optional[float]] = {}
                for t in traits:
                    trait_vals[t] = await bounded_call(trait_judges[t], question=q, answer=a)
                    trait_cohs[t] = await bounded_call(coherence_judges[t], question=q, answer=a)
                return trait_vals, trait_cohs

            tasks.append(asyncio.create_task(eval_one()))

        for item, task in zip(item_refs, tasks):
            trait_vals, trait_cohs = await task
            out_item: Dict[str, Optional[float]] = dict(item)
            for t in traits:
                v = trait_vals.get(t)
                out_item[f"{t}_score"] = float(v) if v is not None else None
                c = trait_cohs.get(t)
                out_item[f"{t}_coherence"] = float(c) if c is not None else None
            per_item_results.append(out_item)

        return per_item_results

    per_item_results = asyncio.run(_run())
    summary: Dict[str, Optional[float]] = {}
    for t in traits:
        vals = [float(r[f"{t}_score"]) for r in per_item_results if r.get(f"{t}_score") is not None]
        summary[f"{t}_score"] = float(np.mean(vals)) if vals else None
        coh_vals = [float(r[f"{t}_coherence"]) for r in per_item_results if r.get(f"{t}_coherence") is not None]
        summary[f"{t}_coherence"] = float(np.mean(coh_vals)) if coh_vals else None
    return summary, per_item_results


def _build_trait_eval_items_from_dialog(
    generated_dialog: List[str],
    *,
    dialog_file: str,
) -> List[Dict[str, str]]:
    """Collect user turns for trait scoring; preceding System turn is used as the
    judge {question}. APICall turns are skipped when picking the preceding system
    turn so that traits are evaluated against an actual system utterance."""
    items: List[Dict[str, str]] = []
    last_system = ""
    user_msg_idx = 0
    for turn_idx, turn in enumerate(generated_dialog or []):
        if not isinstance(turn, str):
            continue
        if turn.startswith("System:") and "APICall(" not in turn:
            last_system = turn[len("System:"):].strip()
            continue
        if not turn.startswith("User:"):
            continue
        user_text = turn[len("User:"):].strip()
        if not user_text:
            continue
        items.append({
            "dialog_file": dialog_file,
            "user_message_index": str(user_msg_idx),
            "turn_index": str(turn_idx),
            "question": last_system or "Start of a task-oriented dialog.",
            "answer": user_text,
        })
        user_msg_idx += 1
    return items


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_dialogs(
    directory: str,
    bert_model: str = "microsoft/mpnet-base",
    evaluate_traits: bool = False,
    only_traits: bool = False,
    evaluate_quality: bool = False,
    only_quality: bool = False,
):
    """Process JSON files in *directory* and calculate task-success metrics.

    Parameters
    ----------
    directory:
        Path to a directory containing per-dialog JSON files.
    bert_model:
        HuggingFace model id for BERTScore.
    evaluate_traits:
        If True, evaluate trait-pole scores for user messages.
    only_traits:
        If True, only compute trait metrics.
    evaluate_quality:
        If True, evaluate LLM-judged dialog quality metrics (inform,
        truthfulness, constraint satisfaction, user satisfaction).
    only_quality:
        If True, only compute quality metrics.
    """
    api_call_marker = "APICall("
    only_llm = only_traits or only_quality

    all_method_accuracy: list = []
    all_key_accuracy: list = []
    all_value_accuracy: list = []
    all_full_api_accuracy: list = []
    all_key_iou: list = []
    all_value_iou: list = []
    all_full_api_accuracy_iou: list = []

    total_slots = 0
    total_slots_with_search_results = 0
    num_answered_slots = 0
    all_generated_dialogs: list = []
    all_reference_dialogs: list = []
    all_user_generated_dialogs: list = []
    all_user_reference_dialogs: list = []
    all_system_generated_dialogs: list = []
    all_system_reference_dialogs: list = []
    trait_eval_items: List[Dict[str, str]] = []
    per_message_trait_results: List[Dict[str, Optional[float]]] = []
    quality_dialogs: List[Dict[str, object]] = []
    per_dialog_quality_results: List[Dict[str, object]] = []
    num_finished_dialogs = 0
    total_dialogs = 0
    bleu_metric = None
    bert_score_metric = None
    if not only_llm:
        bleu_metric = evaluate.load("bleu")
        bert_score_metric = evaluate.load("bertscore")
    successful_dialogs = 0
    successful_dialogs_iou = 0
    successful_dialogs_per_api: dict = {}
    total_dialogs_per_api: dict = {}

    dialog_files = [
        (root, fname)
        for root, _, files in os.walk(directory)
        for fname in files
        if fname.endswith(".json") and fname.startswith("dialog_")
    ]

    for filepath_dir, filename in tqdm(dialog_files):
        if "config" in filename or "metrics" in filename:
            continue
        filepath = os.path.join(filepath_dir, filename)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            if "predicted_api_calls" not in data:
                continue
            if data["predicted_api_calls"] is None:
                data["predicted_api_calls"] = []
            if "predicted_search_results" not in data:
                data["predicted_search_results"] = []
            if data["predicted_search_results"] is None:
                data["predicted_search_results"] = []

            if not only_llm:
                total_slots += sum(len(slots) for slots in data["user_req_slots"])
                total_slots_with_search_results += calculate_slots_with_search_results(
                    data["user_req_slots"], data["predicted_search_results"]
                )
                num_answered_slots += calculate_num_answered_slots(
                    data["generated_dialog"],
                    data["user_req_slots"],
                    data["predicted_search_results"],
                    api_call_marker=api_call_marker,
                )

            dialog_full_api_accuracy: list = []
            dialog_full_api_accuracy_iou: list = []

            if not only_llm and "api_calls" in data and "predicted_api_calls" in data:
                data["predicted_api_calls"] = [
                    c for c in data["predicted_api_calls"] if c is not None
                ]
                (dm, dk, dv, dfa, dki, dvi, dfi) = _calculate_metrics_sgd(
                    data["api_calls"], data["predicted_api_calls"]
                )
                all_method_accuracy.extend(dm)
                all_key_accuracy.extend(dk)
                all_value_accuracy.extend(dv)
                all_full_api_accuracy.extend(dfa)
                all_key_iou.extend(dki)
                all_value_iou.extend(dvi)
                all_full_api_accuracy_iou.extend(dfi)
                dialog_full_api_accuracy = dfa
                dialog_full_api_accuracy_iou = dfi

            if not only_llm and dialog_full_api_accuracy:
                if all(dialog_full_api_accuracy):
                    successful_dialogs += 1
                if all(dialog_full_api_accuracy_iou):
                    successful_dialogs_iou += 1
                num_apis = len(dialog_full_api_accuracy)
                if all(dialog_full_api_accuracy):
                    successful_dialogs_per_api[num_apis] = successful_dialogs_per_api.get(num_apis, 0) + 1
                total_dialogs_per_api[num_apis] = total_dialogs_per_api.get(num_apis, 0) + 1

            if not only_llm:
                gen_text, ref_text = process_dialogs(data["generated_dialog"], data["dialog"])
                all_generated_dialogs.append(gen_text)
                all_reference_dialogs.append([ref_text])
                gen_sys, ref_sys = remove_system_turns(data["generated_dialog"], data["dialog"])
                gen_sys, ref_sys = process_dialogs(gen_sys, ref_sys)
                all_user_generated_dialogs.append(gen_sys)
                all_user_reference_dialogs.append(ref_sys)
                gen_usr, ref_usr = remove_user_turns(data["generated_dialog"], data["dialog"])
                gen_usr, ref_usr = process_dialogs(gen_usr, ref_usr)
                all_system_generated_dialogs.append(gen_usr)
                all_system_reference_dialogs.append(ref_usr)

            if evaluate_traits:
                trait_eval_items.extend(
                    _build_trait_eval_items_from_dialog(
                        data.get("generated_dialog", []), dialog_file=filename
                    )
                )

            if evaluate_quality:
                quality_dialogs.append({
                    "dialog_file": filename,
                    "generated_dialog": data.get("generated_dialog", []),
                    "user_req_slots": data.get("user_req_slots"),
                })

            if not only_llm:
                total_dialogs += 1
                if data.get("dialog_completed", False):
                    num_finished_dialogs += 1

        except (json.JSONDecodeError, IOError) as e:
            logger.error("Error processing %s: %s", filename, e)

    successful_dialogs_per_api_rate = {}
    if not only_llm:
        successful_dialogs_per_api_rate = dict(sorted({
            k: v / total_dialogs_per_api[k]
            for k, v in successful_dialogs_per_api.items()
        }.items()))

    results = {"bleu": None}
    bert_score = {"precision": [], "recall": [], "f1": []}
    bert_score_user = {"precision": [], "recall": [], "f1": []}
    bert_score_system = {"precision": [], "recall": [], "f1": []}
    if not only_llm and bleu_metric is not None:
        results = bleu_metric.compute(predictions=all_generated_dialogs, references=all_reference_dialogs)
        logger.info("Computing BERTScore...")
        bert_score = bert_score_metric.compute(
            predictions=all_generated_dialogs, references=all_reference_dialogs, model_type=bert_model
        )
        bert_score_user = bert_score_metric.compute(
            predictions=all_user_generated_dialogs, references=all_user_reference_dialogs, model_type=bert_model
        )
        bert_score_system = bert_score_metric.compute(
            predictions=all_system_generated_dialogs, references=all_system_reference_dialogs, model_type=bert_model
        )

    inform_accuracy = num_answered_slots / total_slots if total_slots else 0.0
    inform_accuracy_with_search_results = (
        num_answered_slots / total_slots_with_search_results if total_slots_with_search_results else 0.0
    )
    dialog_completion_rate = num_finished_dialogs / total_dialogs if total_dialogs else 0.0
    successful_dialogs_rate = successful_dialogs / total_dialogs if total_dialogs else 0.0
    successful_dialogs_iou_rate = successful_dialogs_iou / total_dialogs if total_dialogs else 0.0

    trait_metrics = None
    if evaluate_traits:
        trait_metrics, per_message_trait_results = evaluate_trait_metrics(trait_eval_items)

    quality_metrics = None
    if evaluate_quality and quality_dialogs:
        from entrypoints.evaluation.quality_metrics import evaluate_quality_metrics  # noqa: E402
        quality_metrics, per_dialog_quality_results = evaluate_quality_metrics(quality_dialogs)

    return (
        all_method_accuracy, all_key_accuracy, all_value_accuracy, all_full_api_accuracy,
        all_key_iou, all_value_iou, all_full_api_accuracy_iou,
        inform_accuracy, inform_accuracy_with_search_results, results, dialog_completion_rate,
        successful_dialogs_rate, successful_dialogs_iou_rate, bert_score, bert_score_user,
        bert_score_system, successful_dialogs_per_api_rate, trait_metrics, per_message_trait_results,
        quality_metrics, per_dialog_quality_results,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated dialogs from a directory of JSON files.")
    parser.add_argument(
        "--json-dir", "--json_directory", dest="json_directory",
        default="storage/interactive_generated_both_deepseek-deepseek-chat",
        help="Path to directory containing JSON files",
    )
    parser.add_argument(
        "--bertscore-model", dest="bertscore_model", default="microsoft/mpnet-base",
        help="HuggingFace model id to use for BERTScore",
    )
    parser.add_argument(
        "--evaluate-traits", dest="evaluate_traits", action="store_true",
        help="Evaluate trait-pole scores for user messages",
    )
    parser.add_argument(
        "--only-traits", dest="only_traits", action="store_true",
        help="Only compute trait-pole metrics and append to metrics.json",
    )
    parser.add_argument(
        "--evaluate-quality", dest="evaluate_quality", action="store_true",
        help="Evaluate LLM-judged dialog quality metrics: inform, truthfulness, "
             "constraint satisfaction, user satisfaction",
    )
    parser.add_argument(
        "--only-quality", dest="only_quality", action="store_true",
        help="Only compute LLM-judged quality metrics and append to metrics.json",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="Directory to write metrics.json (and related files) to. "
             "Defaults to --json-dir when not set.",
    )
    parser.add_argument(
        "--dataset-type", dest="dataset_type", default="sgd",
        help="Dataset type (accepted for compatibility; currently unused).",
    )
    args = parser.parse_args()

    only_llm = args.only_traits or args.only_quality

    (
        all_method_accuracy, all_key_accuracy, all_value_accuracy, all_full_api_accuracy,
        all_key_iou, all_value_iou, all_full_api_accuracy_iou,
        inform_accuracy, inform_accuracy_with_search_results, results, dialog_completion_rate,
        successful_dialogs_rate, successful_dialogs_iou_rate, bert_score, bert_score_user,
        bert_score_system, successful_dialogs_per_api_rate, trait_metrics, per_message_trait_results,
        quality_metrics, per_dialog_quality_results,
    ) = evaluate_dialogs(
        args.json_directory,
        args.bertscore_model,
        evaluate_traits=args.evaluate_traits or args.only_traits,
        only_traits=args.only_traits,
        evaluate_quality=args.evaluate_quality or args.only_quality,
        only_quality=args.only_quality,
    )

    json_directory = args.json_directory
    output_directory = args.output_dir if args.output_dir else args.json_directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if not only_llm:
        print(f"Method Accuracy: {sum(all_method_accuracy) / len(all_method_accuracy)}")
        print(f"Key Accuracy: {sum(all_key_accuracy) / len(all_key_accuracy)}")
        print(f"Key IoU: {sum(all_key_iou) / len(all_key_iou)}")
        print(f"Value Accuracy: {sum(all_value_accuracy) / len(all_value_accuracy)}")
        print(f"Value IoU: {sum(all_value_iou) / len(all_value_iou)}")
        print(f"Full API Accuracy: {sum(all_full_api_accuracy) / len(all_full_api_accuracy)}")
        print(f"Full API Accuracy IoU: {sum(all_full_api_accuracy_iou) / len(all_full_api_accuracy_iou)}")
        print(f"Successful dialogs rate: {successful_dialogs_rate}")
        print(f"Successful Dialogs per API Rate: {successful_dialogs_per_api_rate}")
        print(f"Successful dialogs rate IoU: {successful_dialogs_iou_rate}")
        print(f"Inform Accuracy: {inform_accuracy}")
        print(f"Inform Accuracy with Search Results: {inform_accuracy_with_search_results}")
        print(f"BLEU Score: {results['bleu']}")
        print(f"Dialog Completion Rate: {dialog_completion_rate}")
        print(f"Bert Score (avg_precision): {np.mean(bert_score['precision'])}")
        print(f"Bert Score (avg_recall): {np.mean(bert_score['recall'])}")
        print(f"Bert Score (avg_f1): {np.mean(bert_score['f1'])}")
        print(f"Bert Score User (avg_precision): {np.mean(bert_score_user['precision'])}")
        print(f"Bert Score User (avg_recall): {np.mean(bert_score_user['recall'])}")
        print(f"Bert Score User (avg_f1): {np.mean(bert_score_user['f1'])}")
        print(f"Bert Score System (avg_precision): {np.mean(bert_score_system['precision'])}")
        print(f"Bert Score System (avg_recall): {np.mean(bert_score_system['recall'])}")
        print(f"Bert Score System (avg_f1): {np.mean(bert_score_system['f1'])}")

    metrics_path = os.path.join(output_directory, "metrics.json")
    trait_metrics_path = os.path.join(output_directory, "trait_metrics.json")
    quality_metrics_path = os.path.join(output_directory, "quality_metrics.json")

    def _write_quality_files(quality_summary, per_dialog_results):
        if not quality_summary:
            return
        payload = {
            "summary": quality_summary,
            "num_dialogs": len(per_dialog_results) if per_dialog_results else 0,
            "items": per_dialog_results or [],
        }
        with open(quality_metrics_path, "w") as f:
            json.dump(payload, f, indent=2)

    if only_llm:
        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        if "summary" not in metrics:
            metrics["summary"] = {}
        if trait_metrics:
            metrics["summary"].update(trait_metrics)
        if quality_metrics:
            metrics["summary"].update(quality_metrics)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        trait_payload = None
        if trait_metrics:
            trait_payload = {
                "summary": trait_metrics,
                "num_items": len(per_message_trait_results) if per_message_trait_results else 0,
                "items": per_message_trait_results or [],
            }
            with open(trait_metrics_path, "w") as f:
                json.dump(trait_payload, f, indent=2)
        _write_quality_files(quality_metrics, per_dialog_quality_results)
        _write_metrics_xlsx(json_directory, metrics, trait_payload)
    else:
        summary: dict = {
            "method_accuracy": sum(all_method_accuracy) / len(all_method_accuracy) if all_method_accuracy else None,
            "key_accuracy": sum(all_key_accuracy) / len(all_key_accuracy) if all_key_accuracy else None,
            "key_iou": sum(all_key_iou) / len(all_key_iou) if all_key_iou else None,
            "value_accuracy": sum(all_value_accuracy) / len(all_value_accuracy) if all_value_accuracy else None,
            "value_iou": sum(all_value_iou) / len(all_value_iou) if all_value_iou else None,
            "full_api_accuracy": sum(all_full_api_accuracy) / len(all_full_api_accuracy) if all_full_api_accuracy else None,
            "full_api_accuracy_iou": sum(all_full_api_accuracy_iou) / len(all_full_api_accuracy_iou) if all_full_api_accuracy_iou else None,
            "successful_dialogs_rate": successful_dialogs_rate,
            "successful_dialogs_per_api_rate": successful_dialogs_per_api_rate,
            "successful_dialogs_iou_rate": successful_dialogs_iou_rate,
            "inform_accuracy": inform_accuracy,
            "inform_accuracy_with_search_results": inform_accuracy_with_search_results,
            "bleu": results["bleu"],
            "dialog_completion_rate": dialog_completion_rate,
            "bert_score": {
                "avg_precision": float(np.mean(bert_score["precision"])),
                "avg_recall": float(np.mean(bert_score["recall"])),
                "avg_f1": float(np.mean(bert_score["f1"])),
            },
            "bert_score_user": {
                "avg_precision": float(np.mean(bert_score_user["precision"])),
                "avg_recall": float(np.mean(bert_score_user["recall"])),
                "avg_f1": float(np.mean(bert_score_user["f1"])),
            },
            "bert_score_system": {
                "avg_precision": float(np.mean(bert_score_system["precision"])),
                "avg_recall": float(np.mean(bert_score_system["recall"])),
                "avg_f1": float(np.mean(bert_score_system["f1"])),
            },
        }
        if trait_metrics:
            summary.update(trait_metrics)
        if quality_metrics:
            summary.update(quality_metrics)
        metrics = {"summary": summary}
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        trait_payload = None
        if trait_metrics:
            trait_payload = {
                "summary": trait_metrics,
                "num_items": len(per_message_trait_results) if per_message_trait_results else 0,
                "items": per_message_trait_results or [],
            }
            with open(trait_metrics_path, "w") as f:
                json.dump(trait_payload, f, indent=2)
        _write_quality_files(quality_metrics, per_dialog_quality_results)
        _write_metrics_xlsx(json_directory, metrics, trait_payload)


if __name__ == "__main__":
    main()
