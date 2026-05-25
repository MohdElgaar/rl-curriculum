#!/bin/bash
# Restart IF-RLVR on a Vast instance (reads HF/W&B from container env set at rent time).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-/workspace/rl-curriculum}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
export PATH="${HOME}/.local/bin:${PATH}"
export NLTK_DATA="${NLTK_DATA:-/scratch/nltk_data}"

# shellcheck source=lib/instance_env.sh
source "${SCRIPT_DIR}/lib/instance_env.sh"
require_vast_instance_env

GPU_CONFIG="${1:-configs/gpus_4.sh}"
DATA_CONFIG="${2:-configs/data_vastai.sh}"
MODEL_CONFIG="${3:-configs/model_gemma_4_e2b_rs.sh}"

mkdir -p "${LOG_DIR}"
for pid in $(pgrep -f "open_instruct.grpo_fast" || true); do
  kill -9 "${pid}" 2>/dev/null || true
done
ray stop --force 2>/dev/null || true
sleep 5

ts=$(date +%s)
if [ -f "${LOG_DIR}/training.log" ]; then
  mv "${LOG_DIR}/training.log" "${LOG_DIR}/training.log.restart-${ts}"
fi

cd "${WORKDIR}"
nohup env NLTK_DATA="${NLTK_DATA}" HF_TOKEN="${HF_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY}" \
  bash "${SCRIPT_DIR}/run_if_rlvr.sh" \
  "${GPU_CONFIG}" "${DATA_CONFIG}" "${MODEL_CONFIG}" \
  >> "${LOG_DIR}/training.log" 2>&1 &
echo $! > "${LOG_DIR}/training.pid"
echo "Training PID=$(cat "${LOG_DIR}/training.pid") -> ${LOG_DIR}/training.log"
