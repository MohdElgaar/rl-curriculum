#!/bin/bash
# Local machine: load HF/W&B credentials before `vastai create instance --env`.
# Do not pass secrets via onstart.sh; Vast injects --env into the container.

load_local_rent_secrets() {
  if [ -z "${HF_TOKEN:-}" ]; then
    for candidate in \
      "${HF_TOKEN_PATH:-}" \
      "${HOME}/.cache/huggingface/token" \
      "/work/pi_hadi_amiri_uml_edu/mohamed/.cache/huggingface/token"; do
      if [ -n "${candidate}" ] && [ -f "${candidate}" ]; then
        HF_TOKEN="$(tr -d '[:space:]' < "${candidate}")"
        export HF_TOKEN
        break
      fi
    done
  fi

  if [ -z "${WANDB_API_KEY:-}" ]; then
    WANDB_API_KEY="$(
      python3 -c "import netrc; a=netrc.netrc().authenticators('api.wandb.ai'); print(a[2] if a else '')" 2>/dev/null || true
    )"
    export WANDB_API_KEY
  fi
}

require_local_rent_secrets() {
  load_local_rent_secrets
  if [ -z "${HF_TOKEN:-}" ] || [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Set HF_TOKEN and WANDB_API_KEY (or ~/.netrc for W&B) before renting."
    exit 1
  fi
}

# Usage: ENV_STR="$(build_vast_create_env_str REPO_BRANCH=vastai VAST_GPU_CONFIGS=...)"
# HF_TOKEN and WANDB_API_KEY are always passed via --env when set (training rents require them).
build_vast_create_env_str() {
  local env_str=""
  [ -n "${HF_TOKEN:-}" ] && env_str="${env_str} -e HF_TOKEN=${HF_TOKEN}"
  [ -n "${WANDB_API_KEY:-}" ] && env_str="${env_str} -e WANDB_API_KEY=${WANDB_API_KEY}"
  local kv key val
  for kv in "$@"; do
    key="${kv%%=*}"
    val="${kv#*=}"
    env_str="${env_str} -e ${key}=${val}"
  done
  # Trim leading space
  env_str="${env_str# }"
  printf '%s' "${env_str}"
}
