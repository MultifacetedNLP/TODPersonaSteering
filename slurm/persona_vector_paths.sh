#!/bin/bash

# Resolve persona steering vector paths from the sibling persona repo.

persona_model_dir_name() {
  local model_name="$1"
  printf "%s" "${model_name##*/}"
}

persona_repo_root() {
  local raw="${PERSONA_REPO_ROOT:-$PROJECT_ROOT/../persona}"
  # Resolve any ../  so exported paths are clean absolute paths
  if [ -d "$raw" ]; then
    (cd "$raw" && pwd)
  else
    printf "%s" "$raw"
  fi
}

persona_default_vector_dir() {
  local model_name="$1"
  printf "%s/vectors/%s" "$(persona_repo_root)" "$(persona_model_dir_name "$model_name")"
}

persona_default_vector_input_dir() {
  persona_default_vector_dir "$1"
}

persona_big5_positive_pole() {
  case "$1" in
    agreeableness) printf "compassionate" ;;
    conscientiousness) printf "dependable" ;;
    extraversion) printf "outgoing" ;;
    neuroticism) printf "nervous" ;;
    openness) printf "inventive" ;;
    *) printf "%s" "$1" ;;
  esac
}

persona_big5_negative_pole() {
  case "$1" in
    agreeableness) printf "self-interested" ;;
    conscientiousness) printf "careless" ;;
    extraversion) printf "solitary" ;;
    neuroticism) printf "calm" ;;
    openness) printf "consistent" ;;
    *) printf "%s" "$1" ;;
  esac
}

persona_big5_vector_path() {
  local vector_dir="$1"
  local trait="$2"
  local scalar="${3:-1}"
  local pole

  if [[ "$scalar" == -* ]]; then
    pole="$(persona_big5_negative_pole "$trait")"
  else
    pole="$(persona_big5_positive_pole "$trait")"
  fi

  printf "%s/%s_response_avg_diff.pt" "$vector_dir" "$pole"
}
