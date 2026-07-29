#!/bin/bash

# Shared helpers for self-contained E2E TOD experiment folders.
# Source after common_env.sh so PROJECT_ROOT is available.

persona_tod_slugify() {
  local value="${1:-}"
  value="$(printf "%s" "$value" | tr '[:upper:]' '[:lower:]')"
  value="$(printf "%s" "$value" | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')"
  if [ -z "$value" ]; then
    value="experiment-$(date +%Y%m%d-%H%M%S)"
  fi
  printf "%s" "$value"
}

persona_tod_init_experiment_identity() {
  local default_name="$1"
  local default_description="${2:-}"
  local cli_name="${3:-}"
  local cli_description="${4:-}"

  local raw_name="${EXPERIMENT_NAME:-${PERSONA_TOD_EXPERIMENT_NAME:-${RUN_NAME:-${PERSONA_TOD_RUN_NAME:-${cli_name:-}}}}}"
  local raw_description="${EXPERIMENT_DESCRIPTION:-${PERSONA_TOD_EXPERIMENT_DESCRIPTION:-${RUN_DESCRIPTION:-${PERSONA_TOD_RUN_DESCRIPTION:-${cli_description:-}}}}}"

  if [ -z "$raw_name" ] && [ -t 0 ]; then
    read -r -p "Experiment name [$default_name]: " raw_name
  fi
  if [ -z "$raw_description" ] && [ -t 0 ]; then
    read -r -p "Experiment description [$default_description]: " raw_description
  fi

  raw_name="${raw_name:-$default_name}"
  raw_description="${raw_description:-$default_description}"

  PERSONA_TOD_EXPERIMENT_NAME="$(persona_tod_slugify "$raw_name")"
  PERSONA_TOD_EXPERIMENT_DESCRIPTION="$raw_description"
  PERSONA_TOD_OUTPUT_ROOT="${PERSONA_TOD_OUTPUT_ROOT:-$PROJECT_ROOT/output}"
  PERSONA_TOD_EXPERIMENT_DIR="$PERSONA_TOD_OUTPUT_ROOT/$PERSONA_TOD_EXPERIMENT_NAME"

  export PERSONA_TOD_EXPERIMENT_NAME
  export PERSONA_TOD_EXPERIMENT_DESCRIPTION
  export PERSONA_TOD_OUTPUT_ROOT
  export PERSONA_TOD_EXPERIMENT_DIR
  mkdir -p "$PERSONA_TOD_EXPERIMENT_DIR/runs"
}

persona_tod_init_run_identity() {
  local default_name="${1:-main}"
  local description="${2:-}"
  local cli_name="${3:-}"
  local cli_description="${4:-}"

  if [ -z "${PERSONA_TOD_EXPERIMENT_NAME:-}" ]; then
    persona_tod_init_experiment_identity "${EXPERIMENT_ID:-experiment}" "${PERSONA_TOD_EXPERIMENT_DESCRIPTION:-}" "" ""
  fi

  local raw_run_name="${RUN_NAME:-${PERSONA_TOD_RUN_NAME:-${cli_name:-$default_name}}}"
  local raw_run_description="${RUN_DESCRIPTION:-${PERSONA_TOD_RUN_DESCRIPTION:-${cli_description:-$description}}}"

  PERSONA_TOD_RUN_NAME="$(persona_tod_slugify "$raw_run_name")"
  PERSONA_TOD_RUN_DESCRIPTION="$raw_run_description"
  PERSONA_TOD_RUN_DIR="$PERSONA_TOD_EXPERIMENT_DIR/runs/$PERSONA_TOD_RUN_NAME"

  export PERSONA_TOD_RUN_NAME
  export PERSONA_TOD_RUN_DESCRIPTION
  export PERSONA_TOD_RUN_DIR
  mkdir -p "$PERSONA_TOD_RUN_DIR"
}

persona_tod_redirect_run_log() {
  local log_path="${1:-$PERSONA_TOD_RUN_DIR/log.out}"
  mkdir -p "$(dirname "$log_path")"
  exec >>"$log_path" 2>&1
}

persona_tod_write_shell_metadata() {
  local status="${1:-submitted}"
  local path="${2:-$PERSONA_TOD_RUN_DIR/metadata.json}"
  python - "$path" "$status" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime

path, status = sys.argv[1], sys.argv[2]
try:
    with open(path, "r", encoding="utf-8") as f:
        existing = json.load(f)
except Exception:
    existing = {}

now = datetime.now().astimezone().isoformat()
payload = {
    **existing,
    "experiment_name": os.environ.get("PERSONA_TOD_EXPERIMENT_NAME"),
    "experiment_description": os.environ.get("PERSONA_TOD_EXPERIMENT_DESCRIPTION", ""),
    "run_name": os.environ.get("PERSONA_TOD_RUN_NAME"),
    "run_description": os.environ.get("PERSONA_TOD_RUN_DESCRIPTION", ""),
    "status": status,
    "updated_at": now,
    "command": os.environ.get("PERSONA_TOD_COMMAND"),
    "project_root": os.environ.get("PROJECT_ROOT"),
    "experiment_dir": os.environ.get("PERSONA_TOD_EXPERIMENT_DIR"),
    "output_dir": os.environ.get("PERSONA_TOD_RUN_DIR"),
    "hostname": socket.gethostname(),
    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "submit_dir": os.environ.get("SLURM_SUBMIT_DIR"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    },
    "environment": {
        "experiment_id": os.environ.get("EXPERIMENT_ID"),
        "run_label": os.environ.get("RUN_LABEL"),
        "domain": os.environ.get("DOMAIN"),
        "config_name": os.environ.get("CONFIG_NAME") or os.environ.get("PERSONA_TOD_CONFIG_NAME"),
    },
}
payload.setdefault("created_at", now)
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY
}

persona_tod_write_experiment_metadata() {
  local status="${1:-submitted}"
  (
    unset RUN_NAME RUN_DESCRIPTION PERSONA_TOD_RUN_NAME PERSONA_TOD_RUN_DESCRIPTION PERSONA_TOD_RUN_DIR
    persona_tod_write_shell_metadata "$status" "$PERSONA_TOD_EXPERIMENT_DIR/metadata.json"
  )
}
