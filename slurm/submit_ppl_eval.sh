#!/bin/bash
# Submit a GPU array job that computes perplexity for every run under an
# experiment directory and amends each run's metrics.json with the score.
#
# Usage:
#   slurm/submit_ppl_eval.sh <experiment_dir> [max_concurrent]
#
# Optional env overrides:
#   DEPENDENCY       slurm dependency expression (e.g. "afterok:12345")
#   PPL_MODEL_NAME   HF model used for scoring (default: read from manifest.txt system_model, else Qwen/Qwen2.5-7B-Instruct)
#   PPL_BATCH_SIZE   batch size (default: 8)
#   PPL_MAX_LENGTH   max token length per utterance (default: 256)

set -euo pipefail

EXP_DIR="${1:?usage: $0 <experiment_dir> [max_concurrent]}"
MAX_CONCURRENT="${2:-4}"
DEPENDENCY="${DEPENDENCY:-}"

# Derive default PPL model from experiment manifest if not explicitly set
if [ -z "${PPL_MODEL_NAME:-}" ] && [ -f "$EXP_DIR/manifest.txt" ]; then
  PPL_MODEL_NAME="$(grep '^system_model:' "$EXP_DIR/manifest.txt" | awk '{print $2}')"
fi
PPL_MODEL_NAME="${PPL_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"

if [ ! -d "$EXP_DIR/runs" ]; then
  echo "ERROR: no runs/ subdir under $EXP_DIR"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARRAY_SCRIPT="$SCRIPT_DIR/workers/eval_ppl_array.slurm"

if [ ! -f "$ARRAY_SCRIPT" ]; then
  echo "ERROR: array script not found: $ARRAY_SCRIPT"
  exit 1
fi

MANIFEST_DIR="$EXP_DIR/logs/ppl_eval"
mkdir -p "$MANIFEST_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
MANIFEST="$MANIFEST_DIR/ppl_eval_runs-$STAMP.txt"

: > "$MANIFEST"
for run in "$EXP_DIR"/runs/*/; do
  run="${run%/}"
  if compgen -G "$run/dialogs/*/dialog_*.json" > /dev/null \
     || compgen -G "$run/dialog_*.json" > /dev/null; then
    echo "$run" >> "$MANIFEST"
  fi
done

N_RUNS="$(wc -l < "$MANIFEST")"
if [ "$N_RUNS" -eq 0 ]; then
  echo "ERROR: no runs with dialog json files under $EXP_DIR/runs"
  exit 1
fi

echo "Experiment dir : $EXP_DIR"
echo "Manifest       : $MANIFEST"
echo "Runs           : $N_RUNS"
echo "Max concurrent : $MAX_CONCURRENT"
echo "PPL model      : $PPL_MODEL_NAME"

_dep_args=()
if [ -n "$DEPENDENCY" ]; then
  _dep_args+=(--dependency="$DEPENDENCY")
  echo "Dependency     : $DEPENDENCY"
fi
echo ""

SLURM_LOG_DIR="$EXP_DIR/logs/ppl_eval/slurm"
mkdir -p "$SLURM_LOG_DIR"

ARRAY_JOB_ID="$(sbatch --parsable \
  --array="0-$((N_RUNS-1))%${MAX_CONCURRENT}" \
  --chdir="$PROJECT_ROOT" \
  --output="$SLURM_LOG_DIR/slurm-ppl-%A_%a.out" \
  "${_dep_args[@]}" \
  --export=ALL,RUN_MANIFEST="$MANIFEST",PPL_MODEL_NAME="$PPL_MODEL_NAME",PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-8}",PPL_MAX_LENGTH="${PPL_MAX_LENGTH:-256}" \
  "$ARRAY_SCRIPT")"

echo "Submitted PPL array job: $ARRAY_JOB_ID  ($N_RUNS tasks, ${MAX_CONCURRENT} concurrent)"
echo "Per-run logs: <run_dir>/logs/ppl_eval/log-<jobid>.out"
echo "Slurm stdout: $SLURM_LOG_DIR/slurm-ppl-${ARRAY_JOB_ID}_<task>.out"
