"""
Per-domain metric comparison across the 4 experimental conditions.

Metrics reported per domain:
  - Dialog success rate   (all predicted API calls correct)
  - Truthfulness          (LLM judge)
  - Inform score          (LLM judge)
  - Constraint satisfaction %
  - User satisfaction     (LLM judge)

Usage:
    python analysis/domain_comparison.py
"""

import json
import os
import glob
import re
from collections import defaultdict
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOMAINS = ["Restaurants_2", "Movies_1_Movies_3"]

EXPERIMENTS = {
    "llama": {
        "baseline":     "llama/llama-baseline-2026-05-27-07-42",
        "sa": "llama/llama-sa-2026-05-27-07-52",
        "user_steer":   "llama/llama-user-steer-2026-05-27-07-43",
        "user_steer_sa":"llama/llama-user-steer-sa-both-2026-05-27-07-43",
    },
    "qwen": {
        "baseline":     "qwen/qwen-v2/qwen-baseline-2026-05-20-09-19",
        "sa": "qwen/qwen-v2/qwen-sa-2026-05-20-15-47",
        "user_steer":   "qwen/qwen-v2/qwen-user-steer-2026-05-25-15-49",
        "user_steer_sa":"qwen/qwen-v2/qwen-user-steer-sa-both-2026-05-20-15-47",
    },
}

DISPLAY_METRICS = [
    ("dialog_success_rate",        "Dialog success rate  (%)"),
    ("truthfulness_score",         "Truthfulness         (0-5)"),
    ("inform_score",               "Inform score         (0-5)"),
    ("constraint_satisfaction_pct","Constraint satisf.   (%)"),
    ("user_satisfaction_score",    "User satisfaction    (0-5)"),
]


# --------------------------------------------------------------------------
# Success rate: recompute per dialog from stored api_calls / predicted_api_calls
# --------------------------------------------------------------------------
_METHOD_RE = re.compile(r"method='([^']+)'")
_PARAM_RE  = re.compile(r"parameters=(\{[^}]*\})")


def _parse_api_call(call_str: str) -> tuple[str, dict]:
    """Very light parser for 'ApiCall(method=..., parameters={...})' strings."""
    m = _METHOD_RE.search(str(call_str))
    method = m.group(1) if m else ""
    p = _PARAM_RE.search(str(call_str))
    try:
        params = eval(p.group(1)) if p else {}  # noqa: S307 – controlled internal data
    except Exception:
        params = {}
    return method, params


def _full_api_match(gt_str: str, pred_str: str) -> bool:
    """True when method and ALL parameters match exactly (case-insensitive values)."""
    gm, gp = _parse_api_call(gt_str)
    pm, pp = _parse_api_call(pred_str)
    if gm.lower() != pm.lower():
        return False
    for k, v in gp.items():
        if str(pp.get(k, "")).lower() != str(v).lower():
            return False
    return True


def _dialog_success(dialog_data: dict) -> bool:
    """True when every ground-truth API call has a matching predicted call."""
    gt_calls   = [c for c in (dialog_data.get("api_calls") or []) if c]
    pred_calls = [c for c in (dialog_data.get("predicted_api_calls") or []) if c]
    if not gt_calls:
        return False
    for gt in gt_calls:
        if not any(_full_api_match(gt, p) for p in pred_calls):
            return False
    return True


# --------------------------------------------------------------------------
# Load per-dialog metrics for one run directory
# --------------------------------------------------------------------------
def load_run(run_dir: str) -> tuple[dict[str, str], dict[str, dict]]:
    """
    Returns:
      domain_map   : {filename -> domain}
      per_dialog   : {filename -> {metric: value}}
    """
    domain_map: dict[str, str] = {}
    per_dialog: dict[str, dict] = defaultdict(dict)

    # Build domain map and compute success rate from dialog JSONs
    dialogs_dir = os.path.join(run_dir, "dialogs")
    if os.path.isdir(dialogs_dir):
        for domain in os.listdir(dialogs_dir):
            domain_dir = os.path.join(dialogs_dir, domain)
            if not os.path.isdir(domain_dir):
                continue
            for fpath in glob.glob(os.path.join(domain_dir, "dialog_*.json")):
                fname = os.path.basename(fpath)
                domain_map[fname] = domain
                try:
                    d = json.load(open(fpath))
                    per_dialog[fname]["dialog_success_rate"] = float(_dialog_success(d))
                except Exception:
                    pass

    # LLM quality metrics (inform, truthfulness, constraint, satisfaction)
    qm_path = os.path.join(run_dir, "quality_metrics.json")
    if os.path.isfile(qm_path):
        qm = json.load(open(qm_path))
        for item in qm.get("items", []):
            fname = item["dialog_file"]
            per_dialog[fname]["inform_score"]              = item.get("inform_score")
            per_dialog[fname]["truthfulness_score"]        = item.get("truthfulness_score")
            per_dialog[fname]["constraint_satisfaction_pct"] = item.get("constraint_satisfaction_pct")
            per_dialog[fname]["user_satisfaction_score"]   = item.get("user_satisfaction_score")

    return domain_map, per_dialog


# --------------------------------------------------------------------------
# Aggregate across all runs in an experiment condition
# --------------------------------------------------------------------------
def aggregate_condition(exp_dir: str) -> dict[str, dict[str, float | None]]:
    """Returns {domain -> {metric -> mean}} averaged over all runs."""
    runs_dir = os.path.join(exp_dir, "runs")
    if not os.path.isdir(runs_dir):
        return {}

    domain_accum: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for run_name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_dir):
            continue
        domain_map, per_dialog = load_run(run_dir)
        for fname, metrics in per_dialog.items():
            domain = domain_map.get(fname)
            if domain is None:
                continue
            for metric, value in metrics.items():
                if value is not None:
                    domain_accum[domain][metric].append(value)

    def _aggregate(metric: str, vals: list) -> float | None:
        if not vals:
            return None
        m = mean(vals)
        # dialog_success_rate is stored as 0/1 → convert to percentage
        if metric == "dialog_success_rate":
            return round(m * 100, 1)
        # constraint_satisfaction_pct is already 0-100
        if metric == "constraint_satisfaction_pct":
            return round(m, 1)
        return round(m, 3)

    return {
        domain: {
            metric: _aggregate(metric, vals)
            for metric, vals in metrics.items()
        }
        for domain, metrics in domain_accum.items()
    }


# --------------------------------------------------------------------------
# Pretty-print
# --------------------------------------------------------------------------
def print_table(model_name: str, conditions: dict[str, str]):
    print(f"\n{'='*88}")
    print(f"  MODEL: {model_name.upper()}")
    print(f"{'='*88}")

    cond_names = list(conditions.keys())
    results: dict[str, dict[str, dict]] = {}

    for cond, rel_path in conditions.items():
        exp_dir = os.path.join(ROOT, rel_path)
        if not os.path.isdir(exp_dir):
            print(f"  [SKIP] {cond}: {exp_dir} not found")
            results[cond] = {}
        else:
            results[cond] = aggregate_condition(exp_dir)

    col_w = 18
    label_w = 32

    for domain in DOMAINS:
        print(f"\n  Domain: {domain}")
        print(f"  {'-'*84}")
        header = f"  {'Metric':<{label_w}}" + "".join(f"{c:<{col_w}}" for c in cond_names)
        print(header)
        print(f"  {'-'*84}")

        for metric_key, metric_label in DISPLAY_METRICS:
            row = f"  {metric_label:<{label_w}}"
            for cond in cond_names:
                val = results.get(cond, {}).get(domain, {}).get(metric_key)
                if val is None:
                    cell = "N/A"
                elif metric_key == "dialog_success_rate":
                    cell = f"{val:.1f}%"
                elif metric_key == "constraint_satisfaction_pct":
                    cell = f"{val:.1f}%"
                else:
                    cell = f"{val:.3f}"
                row += f"{cell:<{col_w}}"
            print(row)
        print()


def main():
    for model_name, conditions in EXPERIMENTS.items():
        print_table(model_name, conditions)


if __name__ == "__main__":
    main()
