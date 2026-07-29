#!/bin/bash
set -euo pipefail

# Resolve script directory and use repo-local slurm path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SLURM_SCRIPT:-$SCRIPT_DIR/test.slurm}"

if [ ! -f "$SLURM_SCRIPT" ]; then
  echo "Slurm script not found at $SLURM_SCRIPT"
  exit 1
fi

# Coefficient values to sweep
COEFS=(-2.0 -1.0 0.0 1.0 2.0)

echo "Submitting jobs for coef sweep: ${COEFS[*]}"
for coef in "${COEFS[@]}"; do
  echo "Submitting job with COEF=${coef}"
  sbatch --job-name="activation-steered-personas-coef-${coef}" --export=ALL,COEF="${coef}" "$SLURM_SCRIPT"
done

echo "All jobs submitted."


