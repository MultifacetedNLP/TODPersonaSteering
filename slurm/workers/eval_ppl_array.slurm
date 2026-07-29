#!/bin/bash
#SBATCH --job-name=activation-steered-personas-ppl-eval
#SBATCH --account=bbyl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

# GPU array job: computes perplexity over generated NL utterances for one run
# directory and amends that run's metrics.json.
#
# Required env vars (set by submit_ppl_eval.sh):
#   RUN_MANIFEST   path to a text file with one run-dir per line (0-indexed)
#
# Optional overrides:
#   PPL_MODEL_NAME   HF model for scoring  (default: Qwen/Qwen2.5-7B-Instruct)
#   PPL_BATCH_SIZE   batch size             (default: 8)
#   PPL_MAX_LENGTH   max token length       (default: 256)

set -euo pipefail

_slurm_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd 2>/dev/null)"
[ ! -f "$_slurm_dir/common_env.sh" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] && _slurm_dir="$SLURM_SUBMIT_DIR/slurm"
source "$_slurm_dir/common_env.sh"
unset _slurm_dir

: "${RUN_MANIFEST:?RUN_MANIFEST must be set by the submitter}"
: "${SLURM_ARRAY_TASK_ID:?must be run as a job array}"

RUN_DIR="$(awk -v n="$SLURM_ARRAY_TASK_ID" 'NR==n+1' "$RUN_MANIFEST")"
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
  echo "ERROR: invalid run dir for task $SLURM_ARRAY_TASK_ID from $RUN_MANIFEST: '$RUN_DIR'"
  exit 1
fi

LOG_DIR="$RUN_DIR/logs/ppl_eval"
mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/log-${SLURM_JOB_ID:-manual}.out" 2>&1

echo "Started perplexity evaluation at $(date)"
echo "Job ID:    ${SLURM_JOB_ID:-manual} (task ${SLURM_ARRAY_TASK_ID})"
echo "Run dir:   $RUN_DIR"
echo ""

module purge

HF_CACHE_DIR="${HF_CACHE_DIR:-$PROJECT_ROOT/hf_cache}"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_HUB_CACHE="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR"
export WANDB_MODE="${WANDB_MODE:-offline}"

export PYTHON_PATH="${PYTHON_PATH:-$(command -v python)}"
export PYTHONPATH="$PROJECT_ROOT"
echo "Python: $PYTHON_PATH"

cd "$PROJECT_ROOT" || exit 1

PPL_MODEL_NAME="${PPL_MODEL_NAME:-${PERSONA_TOD_SYSTEM_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-8}"
PPL_MAX_LENGTH="${PPL_MAX_LENGTH:-256}"

"$PYTHON_PATH" -u "$PROJECT_ROOT/entrypoints/evaluation/perplexity_eval.py" \
  --run-dir "$RUN_DIR" \
  --model-name "$PPL_MODEL_NAME" \
  --batch-size "$PPL_BATCH_SIZE" \
  --max-length "$PPL_MAX_LENGTH"

echo ""
echo "Perplexity evaluation done at $(date)"
