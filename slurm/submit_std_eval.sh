#!/bin/bash
# Submit non-LLM standard evaluation for every run×domain dir under an
# experiment. Computes BLEU, method/API accuracy, inform accuracy, BERTScore,
# dialog completion, successful-dialog rate.
#
# Results are saved to:
#   {OUTPUT_BASE_DIR}/{condition}/{run_label}/{domain}/metrics.json
#
# Usage:
#   slurm/submit_std_eval.sh <experiment_dir> <condition_label> [max_concurrent]
#
# Optional env override:
#   OUTPUT_BASE_DIR=/path/to/eval/dir slurm/submit_std_eval.sh ...
#
# Example (Qwen):
#   slurm/submit_std_eval.sh qwen/qwen-v2/qwen-user-steer-2026-05-25-15-49 us 8
# Example (Llama):
#   OUTPUT_BASE_DIR=$PWD/llama/evaluation slurm/submit_std_eval.sh llama/llama-user-steer-2026-05-27-07-43 us 8

set -euo pipefail

EXP_DIR="${1:?usage: $0 <experiment_dir> <condition_label> [max_concurrent]}"
CONDITION="${2:?usage: $0 <experiment_dir> <condition_label> [max_concurrent]}"
MAX_CONCURRENT="${3:-8}"
DATASET_TYPE="${DATASET_TYPE:-sgd}"

if [ ! -d "$EXP_DIR/runs" ]; then
  echo "ERROR: no runs/ subdir under $EXP_DIR"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARRAY_SCRIPT="$SCRIPT_DIR/workers/eval_traits_array.slurm"

# Allow caller to override the output root (e.g. for Llama vs Qwen)
_DEFAULT_OUT_ROOT="$PROJECT_ROOT/qwen/qwen-v2/evaluation"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$_DEFAULT_OUT_ROOT}/$CONDITION"
mkdir -p "$OUTPUT_BASE_DIR"

MANIFEST_DIR="$EXP_DIR/logs/std_eval"
mkdir -p "$MANIFEST_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
MANIFEST="$MANIFEST_DIR/std_eval_dirs-$STAMP.txt"

: > "$MANIFEST"
for run in "$EXP_DIR"/runs/*/; do
  run="${run%/}"
  for domain in "$run"/dialogs/*/; do
    domain="${domain%/}"
    if compgen -G "$domain/dialog_*.json" > /dev/null 2>&1; then
      echo "$domain" >> "$MANIFEST"
    fi
  done
done

N_DIRS="$(wc -l < "$MANIFEST")"
if [ "$N_DIRS" -eq 0 ]; then
  echo "ERROR: no domain dirs with dialog json files under $EXP_DIR/runs"
  exit 1
fi

echo "Experiment dir : $EXP_DIR"
echo "Condition      : $CONDITION"
echo "Output base    : $OUTPUT_BASE_DIR"
echo "Manifest       : $MANIFEST"
echo "Domain dirs    : $N_DIRS"
echo "Max concurrent : $MAX_CONCURRENT"
echo ""

SLURM_LOG_DIR="$EXP_DIR/logs/std_eval/slurm"
mkdir -p "$SLURM_LOG_DIR"

ARRAY_JOB_ID="$(sbatch --parsable \
  --array="0-$((N_DIRS-1))%${MAX_CONCURRENT}" \
  --chdir="$PROJECT_ROOT" \
  --output="$SLURM_LOG_DIR/slurm-std-%A_%a.out" \
  --export=ALL,RUN_MANIFEST="$MANIFEST",DATASET_TYPE="$DATASET_TYPE",EVAL_MODE="standard",OUTPUT_BASE_DIR="$OUTPUT_BASE_DIR" \
  "$ARRAY_SCRIPT")"

echo "Submitted array job: $ARRAY_JOB_ID  ($N_DIRS tasks, ${MAX_CONCURRENT} concurrent)"
echo "Results will appear in: $OUTPUT_BASE_DIR/{run_label}/{domain}/metrics.json"
echo "Slurm logs: $SLURM_LOG_DIR/slurm-std-${ARRAY_JOB_ID}_<task>.out"
