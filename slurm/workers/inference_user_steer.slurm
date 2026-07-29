#!/bin/bash
#SBATCH --job-name=activation-steered-personas-user-steer
#SBATCH --account=bbyl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Array range 0-39: 10 traits x 4 scalars.
# Submit via submit_user_steer.slurm, not directly.

set -euo pipefail

# ---------------------------------------------------------------------------
# Parameter mapping: task_id = trait_idx * 4 + scalar_idx
# ---------------------------------------------------------------------------
TRAITS=(calm careless compassionate consistent dependable inventive nervous outgoing self-interested solitary)
SCALARS=(-1 0 1 2)

N_SHARDS="${PERSONA_TOD_NUM_DIALOG_SHARDS:-1}"
COMBO_IDX=$(( SLURM_ARRAY_TASK_ID / N_SHARDS ))
SHARD_ID=$(( SLURM_ARRAY_TASK_ID % N_SHARDS ))
TRAIT_IDX=$(( COMBO_IDX / 4 ))
SCALAR_IDX=$(( COMBO_IDX % 4 ))
TRAIT="${TRAITS[$TRAIT_IDX]}"
SCALAR="${SCALARS[$SCALAR_IDX]}"
RUN_LABEL="${TRAIT}__s${SCALAR}"
export PERSONA_TOD_DIALOG_SHARD_ID="$SHARD_ID"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_slurm_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd 2>/dev/null)"
[ ! -f "$_slurm_dir/common_env.sh" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] && _slurm_dir="$SLURM_SUBMIT_DIR/slurm"
source "$_slurm_dir/common_env.sh"
source "$_slurm_dir/output_helpers.sh"
unset _slurm_dir

: "${EXPERIMENT_DIR:?EXPERIMENT_DIR must be set by the submitter}"

RUN_DESCRIPTION="User-steering TOD run for trait=$TRAIT scalar=$SCALAR"
persona_tod_init_run_identity "$RUN_LABEL" "$RUN_DESCRIPTION"
RUN_OUT_DIR="$PERSONA_TOD_RUN_DIR"
persona_tod_redirect_run_log "$PERSONA_TOD_RUN_DIR/logs/log-shard-${SHARD_ID}.out"
persona_tod_write_shell_metadata "inference_started" "$PERSONA_TOD_RUN_DIR/metadata/shard-${SHARD_ID}.json"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/slurm/persona_vector_paths.sh"
VECTOR_DIR="${VECTOR_DIR:-$(persona_default_vector_dir "meta-llama/Llama-3.1-8B-Instruct")}"
VECTOR_PATH="$VECTOR_DIR/${TRAIT}_response_avg_diff.pt"
USER_LAYER="${USER_LAYER:-20}"

echo "Started user-steer inference at $(date)"
echo "Array job:  $SLURM_ARRAY_JOB_ID  task: $SLURM_ARRAY_TASK_ID"
echo "Run label:  $RUN_LABEL  (trait=$TRAIT  scalar=$SCALAR  shard=$SHARD_ID/$N_SHARDS)"
echo "Vector:     $VECTOR_PATH"
echo "Layer:      $USER_LAYER"
echo "Output:     $RUN_OUT_DIR"
echo ""

if [ ! -f "$VECTOR_PATH" ]; then
  echo "ERROR: vector file not found: $VECTOR_PATH"
  exit 1
fi

module purge

# ---------------------------------------------------------------------------
# HF / WANDB
# ---------------------------------------------------------------------------
HF_CACHE_DIR="${HF_CACHE_DIR:-$PROJECT_ROOT/hf_cache}"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_HUB_CACHE="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR"
export WANDB_MODE=offline

export PERSONA_TOD_SYSTEM_PROVIDER="local"
export PERSONA_TOD_SYSTEM_MODEL_NAME="${PERSONA_TOD_SYSTEM_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"

export PERSONA_TOD_USER_PROVIDER="local_steer_anthropic"
export PERSONA_TOD_USER_MODEL_NAME="${PERSONA_TOD_USER_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
export PERSONA_TOD_USER_ANTHROPIC_VECTOR_PATH="$VECTOR_PATH"
export PERSONA_TOD_USER_ANTHROPIC_COEF="$SCALAR"
export PERSONA_TOD_USER_ANTHROPIC_LAYER="$USER_LAYER"

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
export PYTHON_PATH="${PYTHON_PATH:-$(command -v python)}"
export PYTHONPATH="$PROJECT_ROOT"
echo "Python: $PYTHON_PATH"
"$PYTHON_PATH" -c "import numpy, pandas, pathos; print(f'numpy={numpy.__version__} pandas={pandas.__version__} pathos={pathos.__version__}')"
echo ""

cd "$PROJECT_ROOT" || exit 1

mkdir -p "$RUN_OUT_DIR"

CONFIG_NAME="${CONFIG_NAME:-tod_inference}"
export PERSONA_TOD_TEST_DOMAINS="${PERSONA_TOD_TEST_DOMAINS:-[[\"Restaurants_2\"],[\"Movies_1\",\"Movies_3\"]]}"

"$PYTHON_PATH" -u "$PROJECT_ROOT/entrypoints/inference/t5_tod_interactive_inference_modified.py" \
  --config-path "$PROJECT_ROOT/config/hydra/tod_runs" \
  --config-name "$CONFIG_NAME" \
  "hydra.run.dir=${RUN_OUT_DIR}/config"

echo ""
echo "Inference done for $RUN_LABEL shard $SHARD_ID at $(date)"
persona_tod_write_shell_metadata "inference_finished" "$PERSONA_TOD_RUN_DIR/metadata/shard-${SHARD_ID}.json"
