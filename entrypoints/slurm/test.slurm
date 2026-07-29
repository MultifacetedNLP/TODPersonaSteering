#!/bin/bash
#SBATCH --job-name=activation-steered-personas-test
#SBATCH --account=beto-delta-gpu 
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

REPO_USERNAME="${PERSONA_TOD_REPO_USERNAME:-${USER:-}}"
if [ -z "$REPO_USERNAME" ]; then
  echo "PERSONA_TOD_REPO_USERNAME is required when USER is not set"
  exit 1
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/slurm/common_env.sh"
LOG_DIR="$PROJECT_ROOT/logs"
if [ ! -d "$PROJECT_ROOT" ]; then
  echo "Missing project root: $PROJECT_ROOT"
  exit 1
fi
mkdir -p "$LOG_DIR"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  exec >"$LOG_DIR/activation-steered-personas-${SLURM_JOB_ID}.out" 2>&1
fi

module purge
# module load anaconda3_gpu
# source activate "todbench"

echo "Started Activation-Steered-Personas In The Wild test at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi -L)"
echo ""

cd "$PROJECT_ROOT" || exit 1

mkdir -p "$PROJECT_ROOT/logs"

HF_CACHE_DIR="${HF_CACHE_DIR:-$PROJECT_ROOT/hf_cache}"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_HUB_CACHE="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR"
echo "HF cache: $HF_CACHE_DIR"

RUN_ID="${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}"
export PERSONA_TOD_RUN_ID="$RUN_ID"
echo "Run output: $(pwd)/storage/$RUN_ID"

# ---------------------------------------------------------------------------
# Config overrides via env vars (no need to edit YAML between jobs)
#
# You can either edit these exports directly, OR override at submit time:
#   sbatch --export=ALL,PERSONA_TOD_SYSTEM_ANTHROPIC_COEF=-2.0,PERSONA_TOD_SYSTEM_ANTHROPIC_VECTOR_PATH=/path/to.pt entrypoints/slurm/test.slurm
#
# Supported env vars in entrypoints/inference/t5_tod_interactive_inference_modified.py:
#   - PERSONA_TOD_SYSTEM_PROVIDER
#   - PERSONA_TOD_SYSTEM_MODEL_NAME
#   - PERSONA_TOD_SYSTEM_ANTHROPIC_VECTOR_PATH
#   - PERSONA_TOD_SYSTEM_ANTHROPIC_COEF
#   - PERSONA_TOD_SYSTEM_ANTHROPIC_LAYER
#
# Example values:
# export PERSONA_TOD_SYSTEM_PROVIDER="local_steer_anthropic"
# export PERSONA_TOD_SYSTEM_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export PERSONA_TOD_SYSTEM_ANTHROPIC_VECTOR_PATH="${PERSONA_TOD_SYSTEM_ANTHROPIC_VECTOR_PATH:-}"
export PERSONA_TOD_SYSTEM_ANTHROPIC_COEF="-1.0"
export PERSONA_TOD_SYSTEM_ANTHROPIC_LAYER="22"
# ---------------------------------------------------------------------------

if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
  PYTHON_PATH="$CONDA_PREFIX/bin/python"
else
  PYTHON_PATH="${PYTHON_PATH:-$(command -v python)}"
fi
echo "Using Python: $PYTHON_PATH"

ENTRYPOINT="$PROJECT_ROOT/entrypoints/inference/t5_tod_interactive_inference_modified.py"
CONFIG_DIR="$PROJECT_ROOT/config/hydra/tod_runs"

if [ -f "$ENTRYPOINT" ]; then
  echo "Running entrypoints/inference/t5_tod_interactive_inference_modified.py..."
  "$PYTHON_PATH" -u "$ENTRYPOINT" --config-path "$CONFIG_DIR" --config-name tod_inference hydra.run.dir="storage/${RUN_ID}/config"
else
  echo "entrypoints/inference/t5_tod_interactive_inference_modified.py not found at $ENTRYPOINT"
  exit 1
fi

echo "Activation-Steered-Personas In The Wild test finished at $(date)"

