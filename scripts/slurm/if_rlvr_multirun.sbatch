#!/bin/bash
#SBATCH --job-name=if-rlvr-multirun
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gpus=8
#SBATCH --constraint="mpi&a100-80g"
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

log() {
  echo "[if-rlvr-multirun] $*"
}

setup_log_filters() {
  if [ "${FILTER_LOGS:-1}" != "1" ]; then
    return 0
  fi
  if ! command -v ansifilter >/dev/null 2>&1; then
    return 0
  fi
  if command -v stdbuf >/dev/null 2>&1; then
    exec > >(stdbuf -oL -eL ansifilter | stdbuf -oL -eL tr -d '\r')
    exec 2> >(stdbuf -oL -eL ansifilter | stdbuf -oL -eL tr -d '\r' >&2)
  else
    exec > >(ansifilter | tr -d '\r')
    exec 2> >(ansifilter | tr -d '\r' >&2)
  fi
}

resolve_config_path() {
  local candidate="$1"
  if [ -f "${candidate}" ]; then
    echo "${candidate}"
    return 0
  fi
  if [ -f "${PROJECT_ROOT}/${candidate}" ]; then
    echo "${PROJECT_ROOT}/${candidate}"
    return 0
  fi
  if [ -f "${PROJECT_ROOT}/configs/${candidate}" ]; then
    echo "${PROJECT_ROOT}/configs/${candidate}"
    return 0
  fi
  return 1
}

setup_log_filters

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
log "Project root: ${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

MULTIRUN_MANIFEST="${1:-configs/multirun/rs_alpha10_n_sweep.sh}"
if ! MULTIRUN_MANIFEST="$(resolve_config_path "${MULTIRUN_MANIFEST}")"; then
  echo "Multirun manifest not found: ${1:-configs/multirun/rs_alpha10_n_sweep.sh}"
  exit 1
fi
log "Sourcing multirun manifest: ${MULTIRUN_MANIFEST}"
# shellcheck source=/dev/null
source "${MULTIRUN_MANIFEST}"

if [ "${#MULTIRUN_MODEL_CONFIGS[@]}" -eq 0 ]; then
  echo "MULTIRUN_MODEL_CONFIGS is empty in ${MULTIRUN_MANIFEST}"
  exit 1
fi

TRAINING_SCRIPT="${PROJECT_ROOT}/scripts/slurm/if_rlvr.sbatch"
if [ ! -f "${TRAINING_SCRIPT}" ]; then
  echo "Training script not found: ${TRAINING_SCRIPT}"
  exit 1
fi

total_runs="${#MULTIRUN_MODEL_CONFIGS[@]}"
run_idx=0

for model_config in "${MULTIRUN_MODEL_CONFIGS[@]}"; do
  run_idx=$((run_idx + 1))
  log "=== Run ${run_idx}/${total_runs}: ${model_config} ==="

  # Avoid per-run requeue restarting the whole multirun from run 1.
  MAX_RESTARTS=0 bash "${TRAINING_SCRIPT}" \
    "${MULTIRUN_SHARED_CONFIGS[@]}" \
    "${model_config}"

  log "=== Finished run ${run_idx}/${total_runs}: ${model_config} ==="
done

log "All ${total_runs} runs completed successfully"
