"""Aggregate per-run-label/domain metrics.json files from two conditions and
print a comparison table (US vs US+SA) broken down by trait, scalar, domain.

Usage:
    python entrypoints/evaluation/aggregate_std_metrics.py \
        --eval-dir qwen/qwen-v2/evaluation \
        --condition-a us \
        --condition-b us_sa \
        [--output qwen/qwen-v2/evaluation/comparison.json]
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict


DISPLAY_METRICS = [
    ("method_accuracy",         "Method Acc"),
    ("full_api_accuracy",       "Full API Acc"),
    ("successful_dialogs_rate", "Dialog Succ"),
    ("dialog_completion_rate",  "Dialog Comp"),
    ("inform_accuracy",         "Inform Acc"),
    ("bleu",                    "BLEU"),
]


def _load_metrics(eval_dir: Path, condition: str):
    """Return dict keyed by (run_label, domain) → metrics summary dict."""
    cond_dir = eval_dir / condition
    results = {}
    if not cond_dir.exists():
        return results
    for run_label_dir in sorted(cond_dir.iterdir()):
        if not run_label_dir.is_dir():
            continue
        run_label = run_label_dir.name
        for domain_dir in sorted(run_label_dir.iterdir()):
            if not domain_dir.is_dir():
                continue
            domain = domain_dir.name
            mfile = domain_dir / "metrics.json"
            if not mfile.exists():
                continue
            with open(mfile) as f:
                data = json.load(f)
            summary = data.get("summary", data)
            results[(run_label, domain)] = summary
    return results


def _parse_run_label(run_label: str):
    """Return (trait, scalar_str) from e.g. 'calm__s-1'."""
    if "__s" in run_label:
        trait, s = run_label.rsplit("__s", 1)
        return trait, s
    return run_label, "?"


def _avg(values):
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def _fmt(v):
    if v is None:
        return "  N/A  "
    return f"{v * 100:6.2f}%"


def _print_section(title, rows_a, rows_b, keys_a, keys_b, metric_keys):
    print(f"\nBy {title}:")
    header_parts = [f"{'':25}"]
    for _, label in metric_keys:
        header_parts.append(f"{'A:'+label:>14} {'B:'+label:>14}")
    print("  " + " | ".join(header_parts))
    for key in sorted(set(list(rows_a.keys()) + list(rows_b.keys()))):
        row = f"  {str(key):25}"
        for mkey, _ in metric_keys:
            va = rows_a.get(key, {}).get(mkey)
            vb = rows_b.get(key, {}).get(mkey)
            row += f" {_fmt(va):>14} {_fmt(vb):>14}"
        print(row)


def compare(eval_dir: Path, cond_a: str, cond_b: str, output_path=None):
    data_a = _load_metrics(eval_dir, cond_a)
    data_b = _load_metrics(eval_dir, cond_b)

    all_keys = sorted(set(list(data_a.keys()) + list(data_b.keys())))
    if not all_keys:
        print("ERROR: no metrics.json files found. Run submit_std_eval.sh first.")
        return

    print(f"\n{'=' * 80}")
    print(f"STANDARD METRICS — {cond_a.upper()} (A)  vs  {cond_b.upper()} (B)")
    print(f"{'=' * 80}")

    # --- Overall ---
    print("\nOverall:")
    header = f"  {'metric':30} {'A':>14} {'B':>14}"
    print(header)
    for mkey, label in DISPLAY_METRICS:
        va = _avg([d.get(mkey) for d in data_a.values()])
        vb = _avg([d.get(mkey) for d in data_b.values()])
        print(f"  {label:30} {_fmt(va):>14} {_fmt(vb):>14}")

    # --- By domain ---
    domains = sorted({k[1] for k in all_keys})
    by_domain_a = defaultdict(dict)
    by_domain_b = defaultdict(dict)
    for mkey, _ in DISPLAY_METRICS:
        domain_vals_a = defaultdict(list)
        domain_vals_b = defaultdict(list)
        for (rl, dom) in all_keys:
            if (rl, dom) in data_a:
                domain_vals_a[dom].append(data_a[(rl, dom)].get(mkey))
            if (rl, dom) in data_b:
                domain_vals_b[dom].append(data_b[(rl, dom)].get(mkey))
        for dom in domains:
            by_domain_a[dom][mkey] = _avg(domain_vals_a[dom])
            by_domain_b[dom][mkey] = _avg(domain_vals_b[dom])

    _print_section("domain", by_domain_a, by_domain_b, domains, domains, DISPLAY_METRICS)

    # --- By trait ---
    traits = sorted({_parse_run_label(k[0])[0] for k in all_keys})
    by_trait_a = defaultdict(dict)
    by_trait_b = defaultdict(dict)
    for mkey, _ in DISPLAY_METRICS:
        trait_vals_a = defaultdict(list)
        trait_vals_b = defaultdict(list)
        for (rl, dom) in all_keys:
            trait, _ = _parse_run_label(rl)
            if (rl, dom) in data_a:
                trait_vals_a[trait].append(data_a[(rl, dom)].get(mkey))
            if (rl, dom) in data_b:
                trait_vals_b[trait].append(data_b[(rl, dom)].get(mkey))
        for t in traits:
            by_trait_a[t][mkey] = _avg(trait_vals_a[t])
            by_trait_b[t][mkey] = _avg(trait_vals_b[t])

    _print_section("trait", by_trait_a, by_trait_b, traits, traits, DISPLAY_METRICS)

    # --- By scalar ---
    scalars = sorted({_parse_run_label(k[0])[1] for k in all_keys},
                     key=lambda s: int(s) if s.lstrip("-").isdigit() else 0)
    by_scalar_a = defaultdict(dict)
    by_scalar_b = defaultdict(dict)
    for mkey, _ in DISPLAY_METRICS:
        scalar_vals_a = defaultdict(list)
        scalar_vals_b = defaultdict(list)
        for (rl, dom) in all_keys:
            _, sc = _parse_run_label(rl)
            if (rl, dom) in data_a:
                scalar_vals_a[sc].append(data_a[(rl, dom)].get(mkey))
            if (rl, dom) in data_b:
                scalar_vals_b[sc].append(data_b[(rl, dom)].get(mkey))
        for sc in scalars:
            by_scalar_a[sc][mkey] = _avg(scalar_vals_a[sc])
            by_scalar_b[sc][mkey] = _avg(scalar_vals_b[sc])

    _print_section("scalar", by_scalar_a, by_scalar_b, scalars, scalars, DISPLAY_METRICS)

    # --- By run_label (trait × scalar) ---
    run_labels = sorted({k[0] for k in all_keys})
    by_run_a = defaultdict(dict)
    by_run_b = defaultdict(dict)
    for mkey, _ in DISPLAY_METRICS:
        run_vals_a = defaultdict(list)
        run_vals_b = defaultdict(list)
        for (rl, dom) in all_keys:
            if (rl, dom) in data_a:
                run_vals_a[rl].append(data_a[(rl, dom)].get(mkey))
            if (rl, dom) in data_b:
                run_vals_b[rl].append(data_b[(rl, dom)].get(mkey))
        for rl in run_labels:
            by_run_a[rl][mkey] = _avg(run_vals_a[rl])
            by_run_b[rl][mkey] = _avg(run_vals_b[rl])

    _print_section("run (trait × scalar)", by_run_a, by_run_b,
                   run_labels, run_labels, DISPLAY_METRICS)

    # --- Save JSON ---
    if output_path:
        payload = {
            "conditions": {cond_a: str(eval_dir / cond_a), cond_b: str(eval_dir / cond_b)},
            "n_entries": {cond_a: len(data_a), cond_b: len(data_b)},
            "overall": {
                cond_a: {mkey: _avg([d.get(mkey) for d in data_a.values()]) for mkey, _ in DISPLAY_METRICS},
                cond_b: {mkey: _avg([d.get(mkey) for d in data_b.values()]) for mkey, _ in DISPLAY_METRICS},
            },
            "by_domain": {
                cond_a: dict(by_domain_a),
                cond_b: dict(by_domain_b),
            },
            "by_trait": {
                cond_a: dict(by_trait_a),
                cond_b: dict(by_trait_b),
            },
            "by_scalar": {
                cond_a: dict(by_scalar_a),
                cond_b: dict(by_scalar_b),
            },
            "by_run": {
                cond_a: dict(by_run_a),
                cond_b: dict(by_run_b),
            },
        }
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nComparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="qwen/qwen-v2/evaluation",
                        help="Base directory containing per-condition subdirs")
    parser.add_argument("--condition-a", default="us", help="Label for condition A")
    parser.add_argument("--condition-b", default="us_sa", help="Label for condition B")
    parser.add_argument("--output", default=None,
                        help="Path to write comparison JSON (default: <eval-dir>/comparison.json)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output = args.output or str(eval_dir / "comparison.json")
    compare(eval_dir, args.condition_a, args.condition_b, output)


if __name__ == "__main__":
    main()
