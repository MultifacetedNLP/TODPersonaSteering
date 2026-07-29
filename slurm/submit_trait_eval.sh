#!/bin/bash
# Submit a CPU array job that runs LLM-judged evaluation for every run under an
# experiment dir, amending scores into each run's metrics.json. The mode controls
# what gets computed:
#   traits  -> trait-pole scores (10 traits, per user turn)
#   quality -> inform / truthfulness / constraint sat / user satisfaction
#   both    -> traits + quality (same job)
#
# Usage:
#   slurm/submit_trait_eval.sh <experiment_dir> [max_concurrent] [eval_mode]
#
# Example (just quality):
#   EVAL_MODE=quality slurm/submit_trait_eval.sh \
#     /work/hdd/beto/balvesrodrigues/experiments-persona/qwen/user-steer-2026-05-13-16-26 8

set -euo pipefail

EXP_DIR="${1:?usage: $0 <experiment_dir> [max_concurrent] [eval_mode]}"
MAX_CONCURRENT="${2:-8}"
EVAL_MODE="${3:-${EVAL_MODE:-traits}}"   # traits | quality | both
DATASET_TYPE="${DATASET_TYPE:-sgd}"
# Optional slurm dependency expression (e.g. "afterok:12345") passed through
# to the underlying sbatch array submission via --dependency.
DEPENDENCY="${DEPENDENCY:-}"

case "$EVAL_MODE" in
  standard|traits|quality|both) ;;
  *) echo "ERROR: invalid eval_mode='$EVAL_MODE' (expected standard|traits|quality|both)"; exit 1 ;;
esac

if [ ! -d "$EXP_DIR/runs" ]; then
  echo "ERROR: no runs/ subdir under $EXP_DIR"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARRAY_SCRIPT="$SCRIPT_DIR/workers/eval_traits_array.slurm"

if [ ! -f "$ARRAY_SCRIPT" ]; then
  echo "ERROR: array script not found: $ARRAY_SCRIPT"
  exit 1
fi

MANIFEST_DIR="$EXP_DIR/logs/trait_eval"
mkdir -p "$MANIFEST_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
MANIFEST="$MANIFEST_DIR/trait_eval_runs-$STAMP.txt"

# One run dir per line. Skip dirs that don't contain dialog json files.
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
echo "Dataset type   : $DATASET_TYPE"
echo "Eval mode      : $EVAL_MODE"
echo ""

SLURM_LOG_DIR="$EXP_DIR/logs/trait_eval/slurm"
mkdir -p "$SLURM_LOG_DIR"

_dep_args=()
if [ -n "$DEPENDENCY" ]; then
  _dep_args+=(--dependency="$DEPENDENCY")
  echo "Dependency     : $DEPENDENCY"
fi

ARRAY_JOB_ID="$(sbatch --parsable \
  --array="0-$((N_RUNS-1))%${MAX_CONCURRENT}" \
  --chdir="$PROJECT_ROOT" \
  --output="$SLURM_LOG_DIR/slurm-${EVAL_MODE}-%A_%a.out" \
  "${_dep_args[@]}" \
  --export=ALL,RUN_MANIFEST="$MANIFEST",DATASET_TYPE="$DATASET_TYPE",EVAL_MODE="$EVAL_MODE" \
  "$ARRAY_SCRIPT")"

echo "Submitted array job: $ARRAY_JOB_ID  ($N_RUNS tasks, ${MAX_CONCURRENT} concurrent)"
echo "Per-run logs:        <run_dir>/logs/trait_eval/log-<jobid>.out"
echo "Slurm stdout/err:    $SLURM_LOG_DIR/slurm-trait-${ARRAY_JOB_ID}_<task>.out"
