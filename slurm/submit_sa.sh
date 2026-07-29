#!/bin/bash

set -euo pipefail

REPO_USERNAME="${PERSONA_TOD_REPO_USERNAME:-${USER:-}}"
if [ -z "$REPO_USERNAME" ]; then
  echo "PERSONA_TOD_REPO_USERNAME is required when USER is not set"
  exit 1
fi

_src_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd 2>/dev/null)"
[ ! -f "$_src_dir/common_env.sh" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] && _src_dir="$SLURM_SUBMIT_DIR/slurm"
source "$_src_dir/common_env.sh"
source "$_src_dir/output_helpers.sh"
unset _src_dir

SCRIPT_DIR="$PROJECT_ROOT/slurm"
RUNNER_SCRIPT="$SCRIPT_DIR/workers/inference_sa.slurm"
POST_SCRIPT="$SCRIPT_DIR/workers/eval_metrics_and_traits.slurm"
N_DIALOG_SHARDS="${N_DIALOG_SHARDS:-5}"
MAX_CONCURRENT="${MAX_CONCURRENT:-5}"

cd "$PROJECT_ROOT" || exit 1

RUN_DATE="$(date +%Y-%m-%d)"
RUN_TIME="$(date +%H-%M)"
DEFAULT_EXPERIMENT_NAME="llama-sa-${RUN_DATE}-${RUN_TIME}"
export PERSONA_TOD_COMMAND="${PERSONA_TOD_COMMAND:-bash ${BASH_SOURCE[0]} ${1:-} ${2:-}}"
persona_tod_init_experiment_identity "$DEFAULT_EXPERIMENT_NAME" "Persona-flow TOD experiment" "${1:-}" "${2:-}"
unset RUN_NAME RUN_DESCRIPTION PERSONA_TOD_RUN_NAME PERSONA_TOD_RUN_DESCRIPTION
persona_tod_init_run_identity "main" "Persona-flow TOD run"
persona_tod_redirect_run_log "$PERSONA_TOD_EXPERIMENT_DIR/log.out"
persona_tod_write_experiment_metadata "submitter_started"
EXPERIMENT_ID="$PERSONA_TOD_EXPERIMENT_NAME"
EXPERIMENT_DIR="$PERSONA_TOD_EXPERIMENT_DIR"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/persona_vector_paths.sh"

echo "Started SA Evaluator at $(date)"

SYSTEM_PROVIDER="sa"
SYSTEM_MODEL_NAME="${PERSONA_TOD_SYSTEM_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
VECTOR_INPUT_DIR="$(persona_default_vector_input_dir "$SYSTEM_MODEL_NAME")"
ANTHROPIC_LAYER="${PERSONA_TOD_SYSTEM_ANTHROPIC_LAYER:-20}"

if [ ! -d "$VECTOR_INPUT_DIR" ]; then
  echo "Missing sa vector directory: $VECTOR_INPUT_DIR"
  exit 1
fi

echo "Persona flow model: $SYSTEM_MODEL_NAME"
echo "Persona flow vectors: $VECTOR_INPUT_DIR"
echo "Persona flow layer: $ANTHROPIC_LAYER"

echo "Submitting inference task for SA..."

unset RUN_NAME PERSONA_TOD_RUN_NAME PERSONA_TOD_RUN_DESCRIPTION
export EXPERIMENT_ID EXPERIMENT_DIR PROJECT_ROOT PERSONA_TOD_OUTPUT_ROOT
export PERSONA_TOD_EXPERIMENT_NAME PERSONA_TOD_EXPERIMENT_DESCRIPTION PERSONA_TOD_EXPERIMENT_DIR PERSONA_TOD_COMMAND
export CANONICAL_TRAIT="sa"
export SCALAR="dynamic"
export PERSONA_TOD_SYSTEM_PROVIDER="$SYSTEM_PROVIDER"
export PERSONA_TOD_SYSTEM_MODEL_NAME="$SYSTEM_MODEL_NAME"
export PERSONA_TOD_SYSTEM_ANTHROPIC_VECTOR_PATH="$VECTOR_INPUT_DIR"
export PERSONA_TOD_SYSTEM_ANTHROPIC_COEF="1.0"
export PERSONA_TOD_SYSTEM_ANTHROPIC_LAYER="$ANTHROPIC_LAYER"
export PERSONA_TOD_CONFIG_NAME="${PERSONA_TOD_CONFIG_NAME:-tod_inference}"
export PERSONA_TOD_TEST_DOMAINS="${PERSONA_TOD_TEST_DOMAINS:-[[\"Restaurants_2\"],[\"Movies_1\",\"Movies_3\"]]}"
export PERSONA_TOD_NUM_DIALOG_SHARDS="$N_DIALOG_SHARDS"

mkdir -p "$PERSONA_TOD_EXPERIMENT_DIR/logs/slurm"
ARRAY_JOB_ID="$(sbatch --parsable \
  --array="0-$((N_DIALOG_SHARDS-1))%${MAX_CONCURRENT}" \
  --chdir="$PROJECT_ROOT" \
  --output="$PERSONA_TOD_EXPERIMENT_DIR/logs/slurm/slurm-array-%A_%a.out" \
  --export=ALL \
  "$RUNNER_SCRIPT")"

echo "Submitted array job: $ARRAY_JOB_ID  ($N_DIALOG_SHARDS shard tasks)"
echo "Per-shard logs: $PERSONA_TOD_EXPERIMENT_DIR/runs/sa/log-shard-{0..$((N_DIALOG_SHARDS-1))}.out"
echo ""

echo "array_job_id: $ARRAY_JOB_ID" >> "$PERSONA_TOD_EXPERIMENT_DIR/manifest.txt" 2>/dev/null || true

export RUN_LABEL="sa"
export RUN_NAME="sa"
export RUN_DESCRIPTION="Persona-flow TOD eval"

_dep="afterok"
for s in $(seq 0 $((N_DIALOG_SHARDS-1))); do
  _dep+=":${ARRAY_JOB_ID}_${s}"
done

cpu_job_id="$(sbatch --parsable \
  --chdir="$PROJECT_ROOT" \
  --output="$PERSONA_TOD_EXPERIMENT_DIR/logs/slurm/slurm-eval-%j-sa.out" \
  --dependency="$_dep" \
  --export=ALL \
  "$POST_SCRIPT")"

echo "Submitted eval job $cpu_job_id (dependency: all $N_DIALOG_SHARDS inference shards)"

# Chain LLM-judged trait/quality eval after the CPU eval job
TRAIT_EVAL_SCRIPT="$SCRIPT_DIR/submit_trait_eval.sh"
TRAIT_ARRAY_JOB_ID=""
if [ -x "$TRAIT_EVAL_SCRIPT" ]; then
  TRAIT_ARRAY_JOB_ID="$(DEPENDENCY="afterok:$cpu_job_id" \
    "$TRAIT_EVAL_SCRIPT" "$PERSONA_TOD_EXPERIMENT_DIR" "${TRAIT_EVAL_MAX_CONCURRENT:-8}" "${TRAIT_EVAL_MODE:-both}" \
    | grep "^Submitted" | grep -oP '\d+(?=\s+\()' || true)"
  [ -n "$TRAIT_ARRAY_JOB_ID" ] || echo "WARN: could not parse trait eval job id (submission may still have succeeded)"
else
  echo "WARN: $TRAIT_EVAL_SCRIPT not executable; skipping trait eval chain"
fi

PPL_EVAL_SCRIPT="$SCRIPT_DIR/submit_ppl_eval.sh"
if [ -x "$PPL_EVAL_SCRIPT" ]; then
  _ppl_dep="afterok:$cpu_job_id"
  [ -n "$TRAIT_ARRAY_JOB_ID" ] && _ppl_dep="${_ppl_dep}:$TRAIT_ARRAY_JOB_ID"
  DEPENDENCY="$_ppl_dep" \
    "$PPL_EVAL_SCRIPT" "$PERSONA_TOD_EXPERIMENT_DIR" "${PPL_MAX_CONCURRENT:-2}" \
    || echo "WARN: PPL eval submission failed (non-fatal)"
else
  echo "WARN: $PPL_EVAL_SCRIPT not executable; skipping PPL eval chain"
fi

echo "All done! Monitor logs in $PERSONA_TOD_EXPERIMENT_DIR/log.out"
persona_tod_write_experiment_metadata "submitter_finished"
