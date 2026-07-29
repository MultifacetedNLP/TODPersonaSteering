#!/bin/bash

# Shared path and environment setup for SLURM/local wrappers.
# Source this from scripts under slurm/ after `set -euo pipefail`.

if [ -z "${PROJECT_ROOT:-}" ]; then
  _persona_tod_common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${_persona_tod_common_dir}/.." && pwd)"
  unset _persona_tod_common_dir
fi
export PROJECT_ROOT

if [ ! -d "$PROJECT_ROOT" ]; then
  echo "ERROR: project root not found: $PROJECT_ROOT"
  exit 1
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$PROJECT_ROOT/.." && pwd)}"
export REPO_ROOT

HF_CACHE_DIR="${HF_CACHE_DIR:-$PROJECT_ROOT/hf_cache}"
export HF_CACHE_DIR

PERSONA_REPO_ROOT="${PERSONA_REPO_ROOT:-$PROJECT_ROOT}"
export PERSONA_REPO_ROOT

# Load .env so PYTHON_PATH and other vars set there reach compute nodes
if [ -f "$PROJECT_ROOT/slurm/load_dotenv.sh" ]; then
  source "$PROJECT_ROOT/slurm/load_dotenv.sh"
fi

PYTHON_PATH="${PYTHON_PATH:-$(command -v python)}"
export PYTHON_PATH

