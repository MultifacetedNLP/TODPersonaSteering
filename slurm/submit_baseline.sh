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
ARRAY_SCRIPT="$SCRIPT_DIR/workers/inference_baseline.slurm"
POST_SCRIPT="$SCRIPT_DIR/workers/eval_metrics_and_traits.slurm"
MAX_CONCURRENT="${MAX_CONCURRENT:-5}"
N_DIALOG_SHARDS="${N_DIALOG_SHARDS:-5}"

cd "$PROJECT_ROOT" || exit 1

RUN_DATE="$(date +%Y-%m-%d)"
RUN_TIME="$(date +%H-%M)"
DEFAULT_EXPERIMENT_NAME="llama-baseline-${RUN_DATE}-${RUN_TIME}"
export PERSONA_TOD_COMMAND="${PERSONA_TOD_COMMAND:-bash ${BASH_SOURCE[0]} ${1:-} ${2:-}}"
persona_tod_init_experiment_identity "$DEFAULT_EXPERIMENT_NAME" "Baseline TOD experiment" "${1:-}" "${2:-}"
persona_tod_redirect_run_log "$PERSONA_TOD_EXPERIMENT_DIR/log.out"
persona_tod_write_experiment_metadata "submitter_started"
EXPERIMENT_ID="$PERSONA_TOD_EXPERIMENT_NAME"
EXPERIMENT_DIR="$PERSONA_TOD_EXPERIMENT_DIR"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/persona_vector_paths.sh"

BASELINE_VECTOR_PATH="${PERSONA_TOD_SYSTEM_ANTHROPIC_VECTOR_PATH:-$(persona_big5_vector_path "$(persona_default_vector_dir "meta-llama/Llama-3.1-8B-Instruct")" agreeableness 0.0)}"

if [ ! -f "$ARRAY_SCRIPT" ]; then
  echo "ERROR: array script not found: $ARRAY_SCRIPT"
  exit 1
fi
if [ ! -f "$POST_SCRIPT" ]; then
  echo "ERROR: eval script not found: $POST_SCRIPT"
  exit 1
fi

SYSTEM_PROVIDER="local"
SYSTEM_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
ANTHROPIC_LAYER="20"
PERSONA_TOD_CONFIG_NAME="${PERSONA_TOD_CONFIG_NAME:-tod_inference}"

echo "Started Baseline Evaluator at $(date)"
echo "Experiment:   $EXPERIMENT_ID"
echo "Output root:  $PERSONA_TOD_OUTPUT_ROOT"
echo "Experiment:   $PERSONA_TOD_EXPERIMENT_DIR"
echo "Vector:       $BASELINE_VECTOR_PATH"
echo "Config:       $PERSONA_TOD_CONFIG_NAME"
echo "Dialog shards: $N_DIALOG_SHARDS"
echo "Max parallel: $MAX_CONCURRENT"
echo ""

# ---------------------------------------------------------------------------
# Write manifest
# ---------------------------------------------------------------------------
MANIFEST="$PERSONA_TOD_EXPERIMENT_DIR/manifest.txt"
{
  echo "experiment_id: $EXPERIMENT_ID"
  echo "experiment_dir: $PERSONA_TOD_EXPERIMENT_DIR"
  echo "output_root: $PERSONA_TOD_OUTPUT_ROOT"
  echo "system_model: $SYSTEM_MODEL_NAME"
  echo "config_name: $PERSONA_TOD_CONFIG_NAME"
  echo "vector_path: $BASELINE_VECTOR_PATH"
  echo "experiment_description: $PERSONA_TOD_EXPERIMENT_DESCRIPTION"
  echo "command: $PERSONA_TOD_COMMAND"
  echo "submitted_at: $(date -Iseconds)"
  echo ""
  echo "task_id | run_label"
  echo "--------|------------------------------"
  for s in $(seq 0 $((N_DIALOG_SHARDS-1))); do
    printf "%-7d | %s\n" "$s" "baseline  (shard $s/$N_DIALOG_SHARDS)"
  done
} > "$MANIFEST"
echo "Manifest written: $MANIFEST"
echo ""

# ---------------------------------------------------------------------------
# Submit the job array (tasks 0-1, up to MAX_CONCURRENT at a time)
# ---------------------------------------------------------------------------
unset RUN_NAME PERSONA_TOD_RUN_NAME PERSONA_TOD_RUN_DESCRIPTION
export EXPERIMENT_ID EXPERIMENT_DIR PROJECT_ROOT PERSONA_TOD_OUTPUT_ROOT
export PERSONA_TOD_EXPERIMENT_NAME PERSONA_TOD_EXPERIMENT_DESCRIPTION PERSONA_TOD_EXPERIMENT_DIR PERSONA_TOD_COMMAND
export PERSONA_TOD_CONFIG_NAME
export PERSONA_TOD_SYSTEM_PROVIDER="$SYSTEM_PROVIDER"
export PERSONA_TOD_SYSTEM_MODEL_NAME="$SYSTEM_MODEL_NAME"
export PERSONA_TOD_SYSTEM_ANTHROPIC_VECTOR_PATH="$BASELINE_VECTOR_PATH"
export PERSONA_TOD_SYSTEM_ANTHROPIC_COEF="0.0"
export PERSONA_TOD_SYSTEM_ANTHROPIC_LAYER="$ANTHROPIC_LAYER"
export CANONICAL_TRAIT="sa"
export SCALAR="0.0"
export RUN_DESCRIPTION="$PERSONA_TOD_EXPERIMENT_DESCRIPTION"
export PERSONA_TOD_NUM_DIALOG_SHARDS="$N_DIALOG_SHARDS"

mkdir -p "$PERSONA_TOD_EXPERIMENT_DIR/logs/slurm"
ARRAY_JOB_ID="$(sbatch --parsable \
  --array="0-$((N_DIALOG_SHARDS-1))%${MAX_CONCURRENT}" \
  --chdir="$PROJECT_ROOT" \
  --output="$PERSONA_TOD_EXPERIMENT_DIR/logs/slurm/slurm-array-%A_%a.out" \
  --export=ALL \
  "$ARRAY_SCRIPT")"

echo "Submitted array job: $ARRAY_JOB_ID  ($N_DIALOG_SHARDS shard tasks)"
echo "Per-shard logs: $PERSONA_TOD_EXPERIMENT_DIR/runs/baseline/log-shard-{0..$((N_DIALOG_SHARDS-1))}.out"
echo ""

echo "array_job_id: $ARRAY_JOB_ID" >> "$MANIFEST"

# ---------------------------------------------------------------------------
# Submit eval job after all shards finish
# ---------------------------------------------------------------------------
echo "Submitting eval job..."
export RUN_LABEL="baseline"
export RUN_NAME="baseline"
export RUN_DESCRIPTION="Baseline TOD eval"

_dep="afterok"
for s in $(seq 0 $((N_DIALOG_SHARDS-1))); do
  _dep+=":${ARRAY_JOB_ID}_${s}"
done

EVAL_JOB_ID="$(sbatch --parsable \
  --dependency="$_dep" \
  --chdir="$PROJECT_ROOT" \
  --output="$PERSONA_TOD_EXPERIMENT_DIR/logs/slurm/slurm-eval-%j-baseline.out" \
  --export=ALL \
  "$POST_SCRIPT")"
echo "Submitted eval job $EVAL_JOB_ID (dependency: all $N_DIALOG_SHARDS inference shards)"
echo ""

# Chain LLM-judged trait/quality eval after the CPU eval job
TRAIT_EVAL_SCRIPT="$SCRIPT_DIR/submit_trait_eval.sh"
TRAIT_ARRAY_JOB_ID=""
if [ -x "$TRAIT_EVAL_SCRIPT" ]; then
  TRAIT_ARRAY_JOB_ID="$(DEPENDENCY="afterok:$EVAL_JOB_ID" \
    "$TRAIT_EVAL_SCRIPT" "$PERSONA_TOD_EXPERIMENT_DIR" "${TRAIT_EVAL_MAX_CONCURRENT:-8}" "${TRAIT_EVAL_MODE:-both}" \
    | grep "^Submitted" | grep -oP '\d+(?=\s+\()' || true)"
  [ -n "$TRAIT_ARRAY_JOB_ID" ] || echo "WARN: could not parse trait eval job id (submission may still have succeeded)"
else
  echo "WARN: $TRAIT_EVAL_SCRIPT not executable; skipping trait eval chain"
fi

PPL_EVAL_SCRIPT="$SCRIPT_DIR/submit_ppl_eval.sh"
if [ -x "$PPL_EVAL_SCRIPT" ]; then
  _ppl_dep="afterok:$EVAL_JOB_ID"
  [ -n "$TRAIT_ARRAY_JOB_ID" ] && _ppl_dep="${_ppl_dep}:$TRAIT_ARRAY_JOB_ID"
  DEPENDENCY="$_ppl_dep" \
    "$PPL_EVAL_SCRIPT" "$PERSONA_TOD_EXPERIMENT_DIR" "${PPL_MAX_CONCURRENT:-2}" \
    || echo "WARN: PPL eval submission failed (non-fatal)"
else
  echo "WARN: $PPL_EVAL_SCRIPT not executable; skipping PPL eval chain"
fi
echo ""

echo "All done. Monitor with:"
echo "  squeue -j $ARRAY_JOB_ID"
echo "  tail -f $PERSONA_TOD_EXPERIMENT_DIR/runs/baseline/log.out"
echo ""
echo "Results will appear under:"
echo "  $PERSONA_TOD_EXPERIMENT_DIR/runs/baseline/"
echo ""
echo "Submitter finished at $(date)"
persona_tod_write_experiment_metadata "submitter_finished"
