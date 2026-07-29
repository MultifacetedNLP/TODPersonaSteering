#!/bin/bash
#SBATCH --job-name=activation-steered-personas-sa
#SBATCH --account=bbyl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Array range 0-0: single task running all domains.
#   Task 0: Restaurants_2 + Movies_1 + Movies_3
# System: sa — env vars exported by the submitter.
# User: local (Qwen 2.5-7B, no vector).
# Submit via submit_sa.slurm, not directly.

set -euo pipefail

N_SHARDS="${PERSONA_TOD_NUM_DIALOG_SHARDS:-1}"
SHARD_ID="$SLURM_ARRAY_TASK_ID"
export PERSONA_TOD_NUM_DIALOG_SHARDS="$N_SHARDS"
export PERSONA_TOD_DIALOG_SHARD_ID="$SHARD_ID"

RUN_LABEL="sa"

_slurm_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd 2>/dev/null)"
[ ! -f "$_slurm_dir/common_env.sh" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] && _slurm_dir="$SLURM_SUBMIT_DIR/slurm"
source "$_slurm_dir/common_env.sh"
source "$_slurm_dir/output_helpers.sh"
unset _slurm_dir

: "${EXPERIMENT_DIR:?EXPERIMENT_DIR must be set by the submitter}"

export PERSONA_TOD_USER_PROVIDER="local"
export PERSONA_TOD_USER_MODEL_NAME="${PERSONA_TOD_USER_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"

persona_tod_init_run_identity "$RUN_LABEL" "Persona-flow TOD run"
RUN_OUT_DIR="$PERSONA_TOD_RUN_DIR"
persona_tod_redirect_run_log "$PERSONA_TOD_RUN_DIR/logs/log-shard-${SHARD_ID}.out"
persona_tod_write_shell_metadata "inference_started" "$PERSONA_TOD_RUN_DIR/metadata/shard-${SHARD_ID}.json"

echo "Started sa inference at $(date)"
echo "Array job:  $SLURM_ARRAY_JOB_ID  task: $SLURM_ARRAY_TASK_ID"
echo "Run label:  $RUN_LABEL  shard: $SHARD_ID / $N_SHARDS"
echo "Domains:    ${PERSONA_TOD_TEST_DOMAINS:-config default}"
echo "System:     ${PERSONA_TOD_SYSTEM_PROVIDER:-config default}  (layer=${PERSONA_TOD_SYSTEM_ANTHROPIC_LAYER:-?})"
echo "Output:     $RUN_OUT_DIR"
echo ""

module purge

HF_CACHE_DIR="${HF_CACHE_DIR:-$PROJECT_ROOT/hf_cache}"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_HUB_CACHE="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PYTHON_PATH="${PYTHON_PATH:-$(command -v python)}"
export PYTHONPATH="$PROJECT_ROOT"
echo "Python: $PYTHON_PATH"
"$PYTHON_PATH" -c "import numpy, pandas; print(f'numpy={numpy.__version__} pandas={pandas.__version__}')"
echo ""

cd "$PROJECT_ROOT" || exit 1

mkdir -p "$RUN_OUT_DIR"

CONFIG_NAME="${CONFIG_NAME:-${PERSONA_TOD_CONFIG_NAME:-tod_inference}}"

"$PYTHON_PATH" -u "$PROJECT_ROOT/entrypoints/inference/t5_tod_interactive_inference_modified.py" \
  --config-path "$PROJECT_ROOT/config/hydra/tod_runs" \
  --config-name "$CONFIG_NAME" \
  "hydra.run.dir=${RUN_OUT_DIR}/config"

echo ""
echo "Inference done for $RUN_LABEL shard $SHARD_ID at $(date)"
persona_tod_write_shell_metadata "inference_finished" "$PERSONA_TOD_RUN_DIR/metadata/shard-${SHARD_ID}.json"
