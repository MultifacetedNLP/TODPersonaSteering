"""Perplexity evaluation entrypoint.

Computes per-token cross-entropy perplexity on the natural-language
utterances of all dialog JSONs under a run directory, then amends the
run's metrics.json with a "perplexity" key.

Usage:
    python entrypoints/evaluation/perplexity_eval.py \\
        --run-dir <path/to/run_dir> \\
        [--model-name Qwen/Qwen2.5-7B-Instruct] \\
        [--batch-size 8] \\
        [--max-length 256] \\
        [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics.perplexity import DialogPerplexityEvaluator


def _load_or_init_metrics(metrics_path: Path) -> dict:
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text())
        except Exception:
            pass
    return {}


def _save_metrics(metrics_path: Path, metrics: dict) -> None:
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[ppl] Metrics written to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Amend run metrics.json with perplexity score")
    parser.add_argument("--run-dir", required=True, help="Path to the run directory (contains dialog_*.json files)")
    parser.add_argument("--model-name", default=os.environ.get("PERSONA_TOD_SYSTEM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("PPL_BATCH_SIZE", "8")))
    parser.add_argument("--max-length", type=int, default=int(os.environ.get("PPL_MAX_LENGTH", "256")))
    parser.add_argument("--device", default=None, help="Force device (cuda/cpu). Auto-detected if omitted.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: run dir does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    hf_cache = os.environ.get("HF_HOME") or os.environ.get("HF_CACHE_DIR")

    evaluator = DialogPerplexityEvaluator(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        hf_cache_dir=hf_cache,
    )

    ppl_result = evaluator.evaluate_run(run_dir)

    print(f"[ppl] Results for {run_dir.name}:")
    for k, v in ppl_result.items():
        print(f"  {k}: {v}")

    metrics_path = run_dir / "metrics.json"
    metrics = _load_or_init_metrics(metrics_path)
    metrics.setdefault("summary", {})["perplexity"] = ppl_result
    _save_metrics(metrics_path, metrics)


if __name__ == "__main__":
    main()
