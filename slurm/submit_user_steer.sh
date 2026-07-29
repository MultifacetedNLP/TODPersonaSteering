#!/bin/bash

set -euo pipefail

_src_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd 2>/dev/null)"
[ ! -f "$_src_dir/common_env.sh" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] && _src_dir="$SLURM_SUBMIT_DIR/slurm"
source "$_src_dir/common_env.sh"
source "$_src_dir/output_helpers.sh"
unset _src_dir

SCRIPT_DIR="$PROJECT_ROOT/slurm"
ARRAY_SCRIPT="$SCRIPT_DIR/workers/inference_user_steer.slurm"
SYSTEM_MODEL_NAME="${PERSONA_TOD_SYSTEM_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
USER_LAYER="${USER_LAYER:-20}"
CONFIG_NAME="${CONFIG_NAME:-tod_inference}"
PERSONA_TOD_TEST_DOMAINS="${PERSONA_TOD_TEST_DOMAINS:-[[\"Restaurants_2\"],[\"Movies_1\",\"Movies_3\"]]}"
# Max concurrent array tasks — raise if you have GPU budget
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
N_DIALOG_SHARDS="${N_DIALOG_SHARDS:-5}"

EVAL_SCRIPT="$SCRIPT_DIR/workers/eval_metrics.slurm"

RUN_DATE="$(date +%Y-%m-%d)"
RUN_TIME="$(date +%H-%M)"
DEFAULT_EXPERIMENT_NAME="llama-user-steer-${RUN_DATE}-${RUN_TIME}"
export PERSONA_TOD_COMMAND="${PERSONA_TOD_COMMAND:-bash ${BASH_SOURCE[0]} ${1:-} ${2:-}}"
persona_tod_init_experiment_identity "$DEFAULT_EXPERIMENT_NAME" "User-steering TOD sweep" "${1:-}" "${2:-}"
persona_tod_redirect_run_log "$PERSONA_TOD_EXPERIMENT_DIR/log.out"
persona_tod_write_experiment_metadata "submitter_started"
EXPERIMENT_ID="$PERSONA_TOD_EXPERIMENT_NAME"
EXPERIMENT_DIR="$PERSONA_TOD_EXPERIMENT_DIR"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/persona_vector_paths.sh"
VECTOR_DIR="$(persona_default_vector_dir "$SYSTEM_MODEL_NAME")"

if [ ! -d "$VECTOR_DIR" ]; then
  echo "ERROR: vector directory not found: $VECTOR_DIR"
  exit 1
fi
if [ ! -f "$ARRAY_SCRIPT" ]; then
  echo "ERROR: array script not found: $ARRAY_SCRIPT"
  exit 1
fi
if [ ! -f "$EVAL_SCRIPT" ]; then
  echo "ERROR: eval script not found: $EVAL_SCRIPT"
  exit 1
fi

echo "Started user-steer submitter at $(date)"
echo "Experiment:   $EXPERIMENT_ID"
echo "Output root:  $PERSONA_TOD_OUTPUT_ROOT"
echo "Experiment folder: $PERSONA_TOD_EXPERIMENT_DIR"
echo "Vectors:      $VECTOR_DIR"
echo "User layer:   $USER_LAYER"
echo "Config:       $CONFIG_NAME"
echo "Test domains: $PERSONA_TOD_TEST_DOMAINS"
echo "Max parallel: $MAX_CONCURRENT"
echo ""

TRAITS=(calm careless compassionate consistent dependable inventive nervous outgoing self-interested solitary)
SCALARS=(-1 0 1 2)

# ---------------------------------------------------------------------------
# Write a human-readable manifest so you can map task IDs to runs later
# ---------------------------------------------------------------------------
MANIFEST="$PERSONA_TOD_EXPERIMENT_DIR/manifest.txt"
{
  echo "experiment_id: $EXPERIMENT_ID"
  echo "experiment_dir: $PERSONA_TOD_EXPERIMENT_DIR"
  echo "output_root: $PERSONA_TOD_OUTPUT_ROOT"
  echo "vector_dir: $VECTOR_DIR"
  echo "user_layer: $USER_LAYER"
  echo "config_name: $CONFIG_NAME"
  echo "test_domains: $PERSONA_TOD_TEST_DOMAINS"
  echo "experiment_description: $PERSONA_TOD_EXPERIMENT_DESCRIPTION"
  echo "command: $PERSONA_TOD_COMMAND"
  echo "submitted_at: $(date -Iseconds)"
  echo ""
  echo "task_id | trait                | scalar | run_label"
  echo "--------|----------------------|--------|------------------------------"
  for i in $(seq 0 39); do
    t_idx=$(( i / 4 ))
    s_idx=$(( i % 4 ))
    trait="${TRAITS[$t_idx]}"
    scalar="${SCALARS[$s_idx]}"
    printf "%-7d | %-20s | %-6s | %s\n" "$i" "$trait" "$scalar" "${trait}__s${scalar}"
  done
} > "$MANIFEST"
echo "Manifest written: $MANIFEST"
echo ""

# ---------------------------------------------------------------------------
# Submit the job array (tasks 0–39, up to MAX_CONCURRENT at a time)
# ---------------------------------------------------------------------------
unset RUN_NAME PERSONA_TOD_RUN_NAME PERSONA_TOD_RUN_DESCRIPTION
export EXPERIMENT_ID EXPERIMENT_DIR PROJECT_ROOT VECTOR_DIR USER_LAYER CONFIG_NAME PERSONA_TOD_TEST_DOMAINS PERSONA_TOD_OUTPUT_ROOT
export PERSONA_TOD_EXPERIMENT_NAME PERSONA_TOD_EXPERIMENT_DESCRIPTION PERSONA_TOD_EXPERIMENT_DIR PERSONA_TOD_COMMAND
export RUN_DESCRIPTION="$PERSONA_TOD_EXPERIMENT_DESCRIPTION"
export PERSONA_TOD_NUM_DIALOG_SHARDS="$N_DIALOG_SHARDS"

mkdir -p "$PERSONA_TOD_EXPERIMENT_DIR/logs/slurm"
TOTAL_TASKS=$(( 40 * N_DIALOG_SHARDS ))
ARRAY_JOB_ID="$(sbatch --parsable \
  --array="0-$((TOTAL_TASKS-1))%${MAX_CONCURRENT}" \
  --chdir="$PROJECT_ROOT" \
  --output="$PERSONA_TOD_EXPERIMENT_DIR/logs/slurm/slurm-array-%A_%a.out" \
  --export=ALL \
  "$ARRAY_SCRIPT")"

echo "Submitted array job: $ARRAY_JOB_ID  ($TOTAL_TASKS tasks = 40 combos × $N_DIALOG_SHARDS shards, ${MAX_CONCURRENT} concurrent)"
echo "Per-shard logs: $PERSONA_TOD_EXPERIMENT_DIR/runs/<trait>__s<scalar>/log-shard-<n>.out"
echo ""

# Record array job ID in manifest
echo "array_job_id: $ARRAY_JOB_ID" >> "$MANIFEST"

# ---------------------------------------------------------------------------
# Submit one eval job per combo, after all shards for that combo finish
# ---------------------------------------------------------------------------
echo "Submitting eval jobs (one per combo, waits for all $N_DIALOG_SHARDS shards)..."
ALL_EVAL_JOB_IDS=()
for i in $(seq 0 39); do
  t_idx=$(( i / 4 ))
  s_idx=$(( i % 4 ))
  run_label="${TRAITS[$t_idx]}__s${SCALARS[$s_idx]}"
  export RUN_LABEL="$run_label"
  export RUN_DESCRIPTION="User-steering TOD eval for $run_label"

  _dep="afterok"
  for s in $(seq 0 $((N_DIALOG_SHARDS-1))); do
    _dep+=":${ARRAY_JOB_ID}_$((i*N_DIALOG_SHARDS+s))"
  done

  _eval_job_id="$(sbatch --parsable \
    --dependency="$_dep" \
    --chdir="$PROJECT_ROOT" \
    --output="$PERSONA_TOD_EXPERIMENT_DIR/logs/slurm/slurm-eval-%j-${run_label}.out" \
    --export=ALL \
    "$EVAL_SCRIPT")"
  ALL_EVAL_JOB_IDS+=("$_eval_job_id")
done
echo "Submitted 40 eval jobs (each waits for all $N_DIALOG_SHARDS shards of its combo)"
echo ""

# Chain LLM-judged trait/quality eval after ALL CPU eval jobs complete
TRAIT_EVAL_SCRIPT="$SCRIPT_DIR/submit_trait_eval.sh"
TRAIT_ARRAY_JOB_ID=""
if [ -x "$TRAIT_EVAL_SCRIPT" ] && [ "${#ALL_EVAL_JOB_IDS[@]}" -gt 0 ]; then
  _trait_dep="afterok"
  for _jid in "${ALL_EVAL_JOB_IDS[@]}"; do
    _trait_dep+=":$_jid"
  done
  TRAIT_ARRAY_JOB_ID="$(DEPENDENCY="$_trait_dep" \
    "$TRAIT_EVAL_SCRIPT" "$PERSONA_TOD_EXPERIMENT_DIR" "${TRAIT_EVAL_MAX_CONCURRENT:-8}" "${TRAIT_EVAL_MODE:-both}" \
    | grep "^Submitted" | grep -oP '\d+(?=\s+\()' || true)"
  [ -n "$TRAIT_ARRAY_JOB_ID" ] || echo "WARN: could not parse trait eval job id (submission may still have succeeded)"
else
  echo "WARN: trait eval chain skipped (script missing or no eval job ids)"
fi

PPL_EVAL_SCRIPT="$SCRIPT_DIR/submit_ppl_eval.sh"
if [ -x "$PPL_EVAL_SCRIPT" ] && [ "${#ALL_EVAL_JOB_IDS[@]}" -gt 0 ]; then
  _ppl_dep="afterok"
  for _jid in "${ALL_EVAL_JOB_IDS[@]}"; do
    _ppl_dep+=":$_jid"
  done
  [ -n "$TRAIT_ARRAY_JOB_ID" ] && _ppl_dep="${_ppl_dep}:$TRAIT_ARRAY_JOB_ID"
  DEPENDENCY="$_ppl_dep" \
    "$PPL_EVAL_SCRIPT" "$PERSONA_TOD_EXPERIMENT_DIR" "${PPL_MAX_CONCURRENT:-2}" \
    || echo "WARN: PPL eval submission failed (non-fatal)"
else
  echo "WARN: PPL eval chain skipped (script missing or no eval job ids)"
fi
echo ""

echo "All done. Monitor with:"
echo "  squeue -j $ARRAY_JOB_ID"
echo "  tail -f $PERSONA_TOD_EXPERIMENT_DIR/runs/calm__s-1/log.out"
echo ""
echo "Results will appear under:"
echo "  $PERSONA_TOD_EXPERIMENT_DIR/runs/<trait>__s<scalar>/"
echo "Metrics will appear at:"
echo "  $PERSONA_TOD_EXPERIMENT_DIR/runs/<trait>__s<scalar>/metrics.json"
echo ""
echo "Submitter finished at $(date)"
persona_tod_write_experiment_metadata "submitter_finished"
