#!/bin/bash
#SBATCH --job-name=activation-steered-personas-traits-only
#SBATCH --account=bbyl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

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

LOG_DIR="$RUN_DIR/logs/trait_eval"
mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/log-${SLURM_JOB_ID:-manual}.out" 2>&1

echo "Started trait-only evaluation at $(date)"
echo "Job ID:    ${SLURM_JOB_ID:-manual} (task ${SLURM_ARRAY_TASK_ID})"
echo "Run dir:   $RUN_DIR"
echo ""

module purge
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /projects/bbyl/mshoaeinaeini/envs/personality

HF_CACHE_DIR="${HF_CACHE_DIR:-/projects/bbyl/mshoaeinaeini/hf_cache}"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_HUB_CACHE="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PYTHON_PATH="${PYTHON_PATH:-$(command -v python)}"
export PYTHONPATH="$PROJECT_ROOT"

cd "$PROJECT_ROOT" || exit 1

EVAL_MODE="${EVAL_MODE:-traits}"    # standard | traits | quality | both
EVAL_FLAGS=()
case "$EVAL_MODE" in
  standard) ;;  # no extra flags — runs BLEU, API accuracy, BERTScore, dialog completion
  traits)  EVAL_FLAGS+=(--only-traits) ;;
  quality) EVAL_FLAGS+=(--only-quality) ;;
  both)    EVAL_FLAGS+=(--only-traits --only-quality) ;;
  *) echo "ERROR: unknown EVAL_MODE=$EVAL_MODE (expected: standard|traits|quality|both)"; exit 1 ;;
esac

# If OUTPUT_BASE_DIR is set (used by submit_std_eval.sh), derive the output dir
# from the run dir structure: runs/{run_label}/dialogs/{domain} -> $OUTPUT_BASE_DIR/{run_label}/{domain}
OUTPUT_FLAG=()
if [ -n "${OUTPUT_BASE_DIR:-}" ]; then
  # Strip everything up to and including /runs/ to get "{run_label}/dialogs/{domain}"
  _rel="${RUN_DIR##*/runs/}"
  _run_label="${_rel%%/*}"
  _domain="${_rel##*/}"
  _out_dir="$OUTPUT_BASE_DIR/$_run_label/$_domain"
  mkdir -p "$_out_dir"
  OUTPUT_FLAG+=(--output-dir "$_out_dir")
  echo "Output dir : $_out_dir"
fi

"$PYTHON_PATH" -u "$PROJECT_ROOT/entrypoints/evaluation/evaluation.py" \
  --json-dir "$RUN_DIR" \
  --dataset-type "${DATASET_TYPE:-sgd}" \
  "${EVAL_FLAGS[@]}" \
  "${OUTPUT_FLAG[@]}"

echo ""
echo "Evaluation ($EVAL_MODE) done at $(date)"
