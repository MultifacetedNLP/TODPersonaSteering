#!/bin/bash
# Submit all four Activation-Steered-Personas experiment conditions as independent Slurm jobs.
#
# The four conditions are:
#   1. baseline            — unsteered system, unsteered user
#   2. sa        — system uses dynamic sa steering
#   3. user-steer          — user simulator steered across 10 traits × 4 scalars
#   4. user-steer-sa       — same sweep but system also uses sa
#
# Each submitter script chains its own eval jobs (trait, quality, PPL) via
# Slurm dependencies, so no manual follow-up is required.
#
# Usage:
#   bash slurm/run_all_experiments.sh [--dry-run]
#
# Optional env overrides (export before calling):
#   PERSONA_TOD_OUTPUT_ROOT        parent directory for all outputs
#                             (default: <project_root>/output)
#   N_DIALOG_SHARDS           shards per run (default: 5 per script)
#   MAX_CONCURRENT            max parallel array tasks (default: per-script)
#   TRAIT_EVAL_MODE           trait eval mode: traits|quality|both (default: both)
#
# Example — shared output root:
#   PERSONA_TOD_OUTPUT_ROOT=/work/hdd/shared/exp-2026-05 \
#     bash slurm/run_all_experiments.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DRY_RUN=false
[ "${1:-}" = "--dry-run" ] && DRY_RUN=true

export PERSONA_TOD_OUTPUT_ROOT="${PERSONA_TOD_OUTPUT_ROOT:-$PROJECT_ROOT/output}"

SCRIPTS=(
  "$SCRIPT_DIR/submit_baseline.sh"
  "$SCRIPT_DIR/submit_sa.sh"
  "$SCRIPT_DIR/submit_user_steer.sh"
  "$SCRIPT_DIR/submit_user_steer_sa.sh"
)

LABELS=(
  "baseline"
  "sa"
  "user-steer"
  "user-steer-sa"
)

echo "Activation-Steered-Personas — Submit all experiments"
echo "Output root : $PERSONA_TOD_OUTPUT_ROOT"
$DRY_RUN && echo "[DRY-RUN — no jobs will be submitted]"
echo ""

JOB_IDS=()
for i in "${!SCRIPTS[@]}"; do
  script="${SCRIPTS[$i]}"
  label="${LABELS[$i]}"
  if [ ! -f "$script" ]; then
    echo "ERROR: script not found: $script"
    exit 1
  fi
  if $DRY_RUN; then
    echo "  [dry-run] sbatch --export=ALL --chdir=$PROJECT_ROOT $script"
  else
    job_id="$(sbatch --parsable --export=ALL --chdir="$PROJECT_ROOT" "$script")"
    JOB_IDS+=("$job_id")
    printf "  %-30s → job %s\n" "[$label]" "$job_id"
  fi
done

if ! $DRY_RUN; then
  echo ""
  echo "All four experiments submitted. Monitor with:"
  joined_ids="$(IFS=,; echo "${JOB_IDS[*]}")"
  echo "  squeue -u \$USER -j $joined_ids"
  echo ""
  echo "Output directories will appear under:"
  echo "  $PERSONA_TOD_OUTPUT_ROOT/"
fi
