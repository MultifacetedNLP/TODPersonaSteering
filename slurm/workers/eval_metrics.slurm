#!/bin/bash
#SBATCH --job-name=activation-steered-personas-eval
#SBATCH --account=beto-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

_slurm_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd 2>/dev/null)"
[ ! -f "$_slurm_dir/common_env.sh" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] && _slurm_dir="$SLURM_SUBMIT_DIR/slurm"
source "$_slurm_dir/common_env.sh"
source "$_slurm_dir/output_helpers.sh"
unset _slurm_dir

: "${RUN_LABEL:?RUN_LABEL must be set by the submitter}"

persona_tod_init_run_identity "${RUN_NAME:-$RUN_LABEL}" "${RUN_DESCRIPTION:-TOD evaluation run}"

persona_tod_redirect_run_log
persona_tod_write_shell_metadata "evaluation_started"

echo ""
echo "Started TOD evaluation at $(date)"
echo "Job ID:    ${SLURM_JOB_ID:-manual}"
echo "Run name:  $PERSONA_TOD_RUN_NAME"
echo "Input:     $PERSONA_TOD_RUN_DIR"
echo ""

module purge

HF_CACHE_DIR="${HF_CACHE_DIR:-$PROJECT_ROOT/hf_cache}"
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

EVAL_ARGS=(--json-dir "$PERSONA_TOD_RUN_DIR" --dataset-type "${DATASET_TYPE:-sgd}")
if [ "${EVALUATE_TRAITS:-0}" = "1" ]; then
  EVAL_ARGS+=(--evaluate-traits)
fi

"$PYTHON_PATH" -u "$PROJECT_ROOT/entrypoints/evaluation/evaluation.py" "${EVAL_ARGS[@]}"

echo ""
echo "Evaluation done at $(date)"
persona_tod_write_shell_metadata "evaluation_finished"
