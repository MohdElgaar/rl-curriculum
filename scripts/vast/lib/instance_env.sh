#!/bin/bash
# On a running Vast instance: read HF/W&B keys injected at create time via --env.
# SSH sessions do not inherit them; PID 1 (container init) does.

load_vast_instance_env() {
  local key val
  for key in HF_TOKEN WANDB_API_KEY; do
    if [ -n "${!key:-}" ]; then
      continue
    fi
    if [ ! -r /proc/1/environ ]; then
      continue
    fi
    val="$(tr '\0' '\n' < /proc/1/environ | sed -n "s/^${key}=//p" | head -1)"
    if [ -n "${val}" ]; then
      export "${key}=${val}"
    fi
  done
}

require_vast_instance_env() {
  load_vast_instance_env
  if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY not found."
    echo "Keys must be set at rent time: vastai create instance ... --env \"-e HF_TOKEN=... -e WANDB_API_KEY=...\""
    exit 1
  fi
}
