#!/usr/bin/env bash
# Run pairwise comparison evaluation on experimental conditions.
#
# Usage:
#   bash entrypoints/evaluation/run_pairwise_comparison.sh <model_name>
#
# where <model_name> is "llama" or "qwen"
#
# This script automatically finds the baseline/sa and
# user-steer/user-steer+sa experiment directories and runs pairwise
# comparison for matching dialog IDs.
#
# Can be run from any working directory -- it cd's into the project root
# itself before invoking `python -m entrypoints.evaluation.pairwise_metrics`.

set -euo pipefail

MODEL="${1:-llama}"

# Find project root (this file lives at entrypoints/evaluation/, two levels
# below the project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================="
echo "Pairwise Comparison Evaluation"
echo "Model: $MODEL"
echo "========================================="
echo

# Model-specific paths
if [ "$MODEL" = "llama" ]; then
    BASE_DIR="$PROJECT_ROOT/llama"
    BASELINE_PATTERN="llama-baseline-*"
    SA_PATTERN="llama-sa-*"
    USER_STEER_PATTERN="llama-user-steer-2*"
    USER_STEER_SA_PATTERN="llama-user-steer-sa-both-*"
elif [ "$MODEL" = "qwen" ]; then
    BASE_DIR="$PROJECT_ROOT/qwen/qwen-v2"
    BASELINE_PATTERN="qwen-baseline-*"
    SA_PATTERN="qwen-sa-*"
    USER_STEER_PATTERN="qwen-user-steer-2*"
    USER_STEER_SA_PATTERN="qwen-user-steer-sa-both-*"
else
    echo "Error: MODEL must be 'llama' or 'qwen'"
    exit 1
fi

# Find experiment directories
find_latest_dir() {
    local pattern="$1"
    local base="$2"
    local found
    found=$(find "$base" -maxdepth 1 -type d -name "$pattern" 2>/dev/null | sort | tail -1)
    echo "$found"
}

BASELINE_DIR=$(find_latest_dir "$BASELINE_PATTERN" "$BASE_DIR")
SA_DIR=$(find_latest_dir "$SA_PATTERN" "$BASE_DIR")
USER_STEER_DIR=$(find_latest_dir "$USER_STEER_PATTERN" "$BASE_DIR")
USER_STEER_SA_DIR=$(find_latest_dir "$USER_STEER_SA_PATTERN" "$BASE_DIR")

# Check if directories exist
if [ -z "$BASELINE_DIR" ] || [ ! -d "$BASELINE_DIR" ]; then
    echo "Error: Baseline directory not found matching $BASELINE_PATTERN"
    exit 1
fi

if [ -z "$SA_DIR" ] || [ ! -d "$SA_DIR" ]; then
    echo "Error: Persona-flow directory not found matching $SA_PATTERN"
    exit 1
fi

if [ -z "$USER_STEER_DIR" ] || [ ! -d "$USER_STEER_DIR" ]; then
    echo "Error: User-steer directory not found matching $USER_STEER_PATTERN"
    exit 1
fi

if [ -z "$USER_STEER_SA_DIR" ] || [ ! -d "$USER_STEER_SA_DIR" ]; then
    echo "Error: User-steer+SA directory not found matching $USER_STEER_SA_PATTERN"
    exit 1
fi

echo "Found experiment directories:"
echo "  Baseline:        $BASELINE_DIR"
echo "  Persona-flow:    $SA_DIR"
echo "  User-steer:      $USER_STEER_DIR"
echo "  User-steer+SA:   $USER_STEER_SA_DIR"
echo

# Create output directory for pairwise results
PAIRWISE_OUTPUT_DIR="$BASE_DIR/pairwise_comparison_results"
mkdir -p "$PAIRWISE_OUTPUT_DIR"

# -------------------------------------------------------------------
# Comparison 1: Baseline vs Persona-Flow
# -------------------------------------------------------------------

echo "========================================="
echo "Comparison 1: Baseline vs Persona-Flow"
echo "========================================="
echo

# For baseline vs SA, compare the single runs
BASELINE_RUN="$BASELINE_DIR/runs/baseline/dialogs"
SA_RUN="$SA_DIR/runs/sa/dialogs"

if [ ! -d "$BASELINE_RUN" ]; then
    echo "Warning: Baseline run directory not found: $BASELINE_RUN"
    echo "Skipping Baseline vs Persona-Flow comparison"
    echo
else
    python -m entrypoints.evaluation.pairwise_metrics \
        --dir-a "$BASELINE_RUN" \
        --dir-b "$SA_RUN" \
        --condition-a "baseline" \
        --condition-b "sa" \
        --output "$PAIRWISE_OUTPUT_DIR/baseline_vs_sa.json" \
        --max-concurrent 30

    echo
fi

# -------------------------------------------------------------------
# Comparison 2: User-Steer vs User-Steer+SA (per trait+scalar)
# -------------------------------------------------------------------

echo "========================================="
echo "Comparison 2: User-Steer vs User-Steer+SA"
echo "========================================="
echo

# Find all run subdirectories in user-steer
USER_STEER_RUNS_DIR="$USER_STEER_DIR/runs"
USER_STEER_SA_RUNS_DIR="$USER_STEER_SA_DIR/runs"

if [ ! -d "$USER_STEER_RUNS_DIR" ] || [ ! -d "$USER_STEER_SA_RUNS_DIR" ]; then
    echo "Warning: User-steer runs directories not found"
    echo "Skipping User-Steer vs User-Steer+SA comparison"
    exit 0
fi

# Get list of run labels (e.g., calm__s-1, calm__s0, etc.)
RUN_LABELS=$(find "$USER_STEER_RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort)

# Track overall stats
TOTAL_PAIRS=0
TOTAL_US_WINS=0
TOTAL_USSA_WINS=0
TOTAL_TIES=0

# Compare each matching run
for RUN_LABEL in $RUN_LABELS; do
    US_DIALOGS="$USER_STEER_RUNS_DIR/$RUN_LABEL/dialogs"
    USSA_DIALOGS="$USER_STEER_SA_RUNS_DIR/$RUN_LABEL/dialogs"

    if [ ! -d "$US_DIALOGS" ]; then
        echo "Warning: User-steer run not found: $US_DIALOGS"
        continue
    fi

    if [ ! -d "$USSA_DIALOGS" ]; then
        echo "Warning: User-steer+SA run not found: $USSA_DIALOGS"
        continue
    fi

    echo "Comparing run: $RUN_LABEL"

    OUTPUT_FILE="$PAIRWISE_OUTPUT_DIR/user_steer_vs_sa__${RUN_LABEL}.json"

    python -m entrypoints.evaluation.pairwise_metrics \
        --dir-a "$US_DIALOGS" \
        --dir-b "$USSA_DIALOGS" \
        --condition-a "user-steer" \
        --condition-b "user-steer+sa" \
        --output "$OUTPUT_FILE" \
        --max-concurrent 30

    # Extract stats from the output JSON
    if [ -f "$OUTPUT_FILE" ]; then
        WINS_A=$(python -c "import json; print(json.load(open('$OUTPUT_FILE'))['summary'].get('wins_a', 0))" 2>/dev/null || echo 0)
        WINS_B=$(python -c "import json; print(json.load(open('$OUTPUT_FILE'))['summary'].get('wins_b', 0))" 2>/dev/null || echo 0)
        TIES=$(python -c "import json; print(json.load(open('$OUTPUT_FILE'))['summary'].get('ties', 0))" 2>/dev/null || echo 0)
        N_PAIRS=$(python -c "import json; print(json.load(open('$OUTPUT_FILE'))['summary'].get('n_pairs', 0))" 2>/dev/null || echo 0)

        TOTAL_PAIRS=$((TOTAL_PAIRS + N_PAIRS))
        TOTAL_US_WINS=$((TOTAL_US_WINS + WINS_A))
        TOTAL_USSA_WINS=$((TOTAL_USSA_WINS + WINS_B))
        TOTAL_TIES=$((TOTAL_TIES + TIES))
    fi

    echo
done

# -------------------------------------------------------------------
# Aggregate summary across all user-steer runs
# -------------------------------------------------------------------

echo "========================================="
echo "AGGREGATE SUMMARY: User-Steer vs User-Steer+SA"
echo "========================================="
echo "Total pairs evaluated: $TOTAL_PAIRS"
echo "User-Steer wins:       $TOTAL_US_WINS"
echo "User-Steer+SA wins:    $TOTAL_USSA_WINS"
echo "Ties:                  $TOTAL_TIES"

if [ "$TOTAL_PAIRS" -gt 0 ]; then
    US_WIN_RATE=$(python -c "print(f'{$TOTAL_US_WINS / $TOTAL_PAIRS:.1%}')")
    USSA_WIN_RATE=$(python -c "print(f'{$TOTAL_USSA_WINS / $TOTAL_PAIRS:.1%}')")
    TIE_RATE=$(python -c "print(f'{$TOTAL_TIES / $TOTAL_PAIRS:.1%}')")

    echo "User-Steer win rate:   $US_WIN_RATE"
    echo "User-Steer+SA win rate: $USSA_WIN_RATE"
    echo "Tie rate:              $TIE_RATE"
fi

echo
echo "All results saved to: $PAIRWISE_OUTPUT_DIR"
echo "Done!"
