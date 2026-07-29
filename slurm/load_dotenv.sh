# Repo dotenv loader for SLURM scripts. Expects PROJECT_ROOT to be set before sourcing.
[[ -z "${PROJECT_ROOT:-}" ]] && return 0

_env_file="${PERSONA_TOD_ENV_FILE:-${PROJECT_ROOT}/.env}"
if [[ ! -f "${_env_file}" ]]; then
  unset _env_file
  return 0
fi

set -a
# shellcheck disable=SC1090
source "${_env_file}"
set +a
unset _env_file
