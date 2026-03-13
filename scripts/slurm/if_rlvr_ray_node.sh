#!/bin/bash
#
# Ray node setup for if_rlvr (Beaker-style).
# Run via srun with one task per node.
#
# Rank 0: start Ray head, then exec the command passed as args.
# Rank 1+: start Ray worker, then poll until head is gone.
#
# Required env: PROJECT_ROOT, RAY_HEAD_IP, RAY_HEAD_PORT, RAY_ADDRESS
# Usage: if_rlvr_ray_node.sh <command> [args...]
#   e.g. if_rlvr_ray_node.sh python -m open_instruct.grpo_fast --exp_name foo ...
#
set -euo pipefail

export PATH="${PROJECT_ROOT}/${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
export VLLM_ALLOW_INSECURE_SERIALIZATION VLLM_DISABLE_COMPILE_CACHE VLLM_USE_V1
export NCCL_CUMEM_ENABLE TRITON_CACHE_DIR PYTHONUNBUFFERED=1

# RAY_HEAD_IP=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" 2>/dev/null | head -1)
# RAY_HEAD_IP=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" 2>/dev/null | head -2 | tail -1)
# RAY_HEAD_IP=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" 2>/dev/null | head -3 | tail -1)


RAY_HEAD_IP="${SLURM_STEPMGR}"
RAY_ADDRESS="${RAY_HEAD_IP}:${RAY_HEAD_PORT}"

unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES

mkdir -p "${HOME}/.triton/autotune"
ray stop --force 2>/dev/null || true

# Ray needs --num-gpus to avoid placement-group GPU index errors with ray start --head
NUM_GPUS_NODE="${SLURM_GPUS_ON_NODE}"

if [ "${RAY_HEAD_IP}" = "$(hostname)" ]; then
  echo "[if-rlvr] This is the head node with rank ${SLURM_PROCID} and IP ${RAY_HEAD_IP}"
  echo "[if-rlvr] Starting Ray head on port ${RAY_HEAD_PORT}, dashboard on port ${RAY_DASHBOARD_PORT} (${NUM_GPUS_NODE} GPUs)"
  ray start --head --port="${RAY_HEAD_PORT}" --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}" --num-gpus="${NUM_GPUS_NODE}"
  echo "[if-rlvr] RAY dashboard is running at ${RAY_HEAD_IP}.unity.rc.umass.edu:${RAY_DASHBOARD_PORT}"
  echo "[if-rlvr] Rank 0: Running $*"
  exec "$@"
else
  echo "[if-rlvr] Rank ${SLURM_PROCID}: Starting Ray worker, connecting to ${RAY_ADDRESS} (${NUM_GPUS_NODE} GPUs)"
  ray start --address="${RAY_ADDRESS}" --dashboard-host=0.0.0.0 --num-gpus="${NUM_GPUS_NODE}"
  cleanup() {
    ray stop --force 2>/dev/null || true
    exit 0
  }
  trap cleanup TERM INT HUP EXIT
  echo "[if-rlvr] Rank ${SLURM_PROCID}: Monitoring Ray head at ${RAY_ADDRESS}"
  while true; do
    if ! ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
      echo "[if-rlvr] Rank ${SLURM_PROCID}: Head is unreachable. Stopping worker and exiting 0."
      cleanup
    fi
    
    sleep 30
  done
fi
