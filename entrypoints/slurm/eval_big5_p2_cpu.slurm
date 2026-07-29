#!/bin/bash
#SBATCH --job-name=big5p2_eval_traits
#SBATCH --account=beto-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=logs/big5p2-eval-%j.out

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/slurm/common_env.sh"

module purge

# Required secrets for trait evaluation and HF access
# Expect these to be provided by the shell environment, an external secret manager,
# or exported by your SLURM submission wrapper.
: "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"
: "${HF_TOKEN:?HF_TOKEN is required}"

echo "Started Big5 p2 evaluation at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unset}"
echo ""

cd "$PROJECT_ROOT" || exit 1
mkdir -p logs

# Hugging Face cache.
HF_CACHE_DIR="${HF_CACHE_DIR:-$PROJECT_ROOT/hf_cache}"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_HUB_CACHE="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR"
echo "HF cache: $HF_CACHE_DIR"

# Avoid over-subscribing CPU threads when running many Python processes in parallel
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
  PYTHON_PATH="$CONDA_PREFIX/bin/python"
else
  PYTHON_PATH="${PYTHON_PATH:-$(command -v python)}"
fi
echo "Using Python: $PYTHON_PATH"
echo ""

EVAL_SCRIPT="$PROJECT_ROOT/entrypoints/evaluation/evaluation.py"
BASE_JSON_DIR="${BASE_JSON_DIR:-}"
: "${BASE_JSON_DIR:?BASE_JSON_DIR is required and must point to a directory with config.json files}"

if [ ! -f "$EVAL_SCRIPT" ]; then
  echo "Missing eval script: $EVAL_SCRIPT"
  exit 1
fi
if [ ! -d "$BASE_JSON_DIR" ]; then
  echo "Missing base json dir: $BASE_JSON_DIR"
  exit 1
fi

shopt -s globstar nullglob
declare -a EVAL_DIRS=()
for cfg in "$BASE_JSON_DIR"/**/config.json; do
  EVAL_DIRS+=("$(dirname "$cfg")")
done

if [ "${#EVAL_DIRS[@]}" -eq 0 ]; then
  echo "No config.json found under $BASE_JSON_DIR"
  exit 1
fi

# Dedupe and make a stable order
mapfile -t EVAL_DIRS < <(printf "%s\n" "${EVAL_DIRS[@]}" | sort -u)

echo "Found ${#EVAL_DIRS[@]} evaluation dirs under $BASE_JSON_DIR"
printf "%s\n" "${EVAL_DIRS[@]}"
echo ""

PARALLELISM="${SLURM_CPUS_PER_TASK:-1}"
echo "Running with parallelism = $PARALLELISM"
echo ""

set +e
export BASE_JSON_DIR
export PYTHON_PATH
export EVAL_SCRIPT
export JOB_ID="${SLURM_JOB_ID:-nojob}"

printf "%s\0" "${EVAL_DIRS[@]}" | xargs -0 -n 1 -P "$PARALLELISM" bash -lc '
set -euo pipefail
DIR="$1"

rel="${DIR#${BASE_JSON_DIR}/}"
if [ "$rel" = "$DIR" ]; then
  rel="$(basename "$DIR")"
fi

safe="$rel"
safe="${safe//\//_}"
safe="${safe//+/_plus_}"
safe="${safe//-/_minus_}"
safe="${safe//./p}"

out="logs/big5p2_eval_${JOB_ID}_${safe}.out"
echo "[$(date)] Evaluating: $DIR"

"$PYTHON_PATH" -u "$EVAL_SCRIPT" \
  --evaluate-traits \
  --json-dir "$DIR" \
  >"$out" 2>&1
' _
XARGS_STATUS=$?
set -e

if [ "$XARGS_STATUS" -ne 0 ]; then
  echo ""
  echo "One or more evaluations failed (xargs exit code: $XARGS_STATUS)."
  echo "Check per-folder logs: logs/big5p2_eval_${SLURM_JOB_ID}_*.out"
  exit "$XARGS_STATUS"
fi

echo ""
echo "Big5 p2 evaluation finished at $(date)"
