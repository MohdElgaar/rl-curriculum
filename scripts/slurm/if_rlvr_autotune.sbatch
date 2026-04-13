#!/bin/bash
#SBATCH --job-name=if-rlvr-autotune
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gpus=8
#SBATCH --constraint="mpi&a100-80g"
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

log() {
  echo "[if-rlvr-autotune] $*"
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

setup_modules() {
  module load "${CUDA_MODULE}"
  if [ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME}/lib64" ]; then
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  fi
}

setup_paths() {
  mkdir -p "${PROJECT_ROOT}/logs"
  if [ ! -d "${SCRATCH_ROOT}" ]; then
    echo "Scratch workspace not found: ${SCRATCH_ROOT}"
    echo "Set SCRATCH_ROOT to your Unity workspace path."
    exit 1
  fi
  mkdir -p "${DATASET_LOCAL_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${AUTOTUNE_RESULT_ROOT}"
}

setup_vllm_env() {
  export VLLM_ALLOW_INSECURE_SERIALIZATION VLLM_DISABLE_COMPILE_CACHE VLLM_USE_V1 NCCL_CUMEM_ENABLE TRITON_CACHE_DIR
}

setup_log_filters

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
log "Project root: ${PROJECT_ROOT}"

ARCH="$(uname -m)"
if [ "${ARCH}" = "aarch64" ]; then
  export UV_PROJECT_ENVIRONMENT=".venv_arm"
  export CUDA_MODULE="cuda/12.4.1"
else
  export UV_PROJECT_ENVIRONMENT=".venv"
  export CUDA_MODULE="${CUDA_MODULE:-cuda/13.1}"
fi

CONFIG_FILES=()
if [ -n "${CONFIG_PATH:-}" ]; then
  CONFIG_FILES+=("${CONFIG_PATH}")
fi
if [ -n "${CONFIG_PATHS:-}" ]; then
  read -r -a CONFIG_PATHS_ARR <<< "${CONFIG_PATHS}"
  CONFIG_FILES+=("${CONFIG_PATHS_ARR[@]}")
fi
if [ "${#}" -ge 1 ]; then
  CONFIG_FILES+=("$@")
fi

RESOLVED_CONFIG_FILES=()
if [ "${#CONFIG_FILES[@]}" -gt 0 ]; then
  for config_path in "${CONFIG_FILES[@]}"; do
    if ! CONFIG_FILE="$(resolve_config_path "${config_path}")"; then
      echo "Config not found: ${config_path}"
      exit 1
    fi
    RESOLVED_CONFIG_FILES+=("${CONFIG_FILE}")
    log "Sourcing config: ${CONFIG_FILE}"
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"
  done
fi

: "${SCRATCH_ROOT:=/scratch4/workspace/mohamed_elgaar_student_uml_edu-rl-curriculum}"
: "${DATASET_LOCAL_CACHE_DIR:=${SCRATCH_ROOT}/data/open-instruct}"
: "${HF_HOME:=${SCRATCH_ROOT}/cache/huggingface}"
: "${OUTPUT_DIR:=${SCRATCH_ROOT}/outputs}"

: "${EXP_NAME:=if_rlvr_autotune}"
: "${MODEL_NAME:=allenai/Llama-3.1-Tulu-3-8B-DPO}"
: "${TRAIN_DATASET:=allenai/IF_multi_constraints_upto5}"
: "${TRAIN_SPLIT:=train}"
: "${TRAIN_DATASET_FRACTION:=1.0}"

: "${BETA:=0.01}"
: "${KL_ESTIMATOR:=2}"
: "${LEARNING_RATE:=5e-7}"
: "${TEMPERATURE:=1.0}"

: "${NUM_UNIQUE_PROMPTS:=48}"
: "${NUM_SAMPLES_PER_PROMPT:=16}"
: "${ASYNC_STEPS:=1}"
: "${PER_DEVICE_BATCH_SIZE:=1}"
: "${NUM_MINI_BATCHES:=2}"

: "${MAX_PROMPT_TOKEN_LENGTH:=2048}"
: "${RESPONSE_LENGTH:=2048}"
: "${PACK_LENGTH:=4096}"

: "${DEEPSPEED_STAGE:=2}"
: "${NUM_EPOCHS:=1}"
: "${LR_SCHEDULER_TYPE:=constant}"
: "${SEED:=1}"
: "${IFEVAL_REWARD_SHAPING:=False}"
: "${IFEVAL_REWARD_SHAPING_CURRICULUM:=False}"
: "${IFEVAL_COMPETENCE_C0:=0.1}"
: "${IFEVAL_COMPETENCE_ALPHA:=1.0}"
: "${IFEVAL_NUM_CURRICULUM_STEPS:=-1}"

: "${MATH_REWARD_SHAPING:=False}"
: "${MATH_REWARD_SHAPING_CURRICULUM:=False}"
: "${MATH_COMPETENCE_C0:=0.1}"
: "${MATH_COMPETENCE_ALPHA:=1.0}"
: "${MATH_NUM_CURRICULUM_STEPS:=-1}"

: "${GSM_REWARD_SHAPING:=False}"
: "${GSM_REWARD_SHAPING_CURRICULUM:=False}"
: "${GSM_COMPETENCE_C0:=0.1}"
: "${GSM_COMPETENCE_ALPHA:=1.0}"
: "${GSM_NUM_CURRICULUM_STEPS:=-1}"

: "${APPLY_VERIFIABLE_REWARD:=True}"
: "${NON_STOP_PENALTY:=True}"
: "${NON_STOP_PENALTY_VALUE:=0.0}"
: "${VLLM_NUM_ENGINES:=2}"
: "${VLLM_TP:=1}"
: "${GRADIENT_CHECKPOINTING:=1}"

: "${DATASET_MIXER_EVAL_LIST:=mohdelgaar/ifeval_rlvr 32 mohdelgaar/ifbench_rlvr 64 allenai/aime2024-25-rlvr 32 allenai/aime2024-25-rlvr 32 allenai/RLVR-MATH 32 allenai/RLVR-GSM 32 allenai/rlvr-code-data-python-r1-format-filtered 32}"
: "${DATASET_MIXER_EVAL_LIST_SPLITS:=train train test_2024 test_2025 train train train}"

: "${VLLM_ALLOW_INSECURE_SERIALIZATION:=1}"
: "${VLLM_DISABLE_COMPILE_CACHE:=1}"
: "${VLLM_USE_V1:=1}"
: "${NCCL_CUMEM_ENABLE:=0}"
: "${TRITON_CACHE_DIR:=${SCRATCH_ROOT}/cache/triton}"

: "${AUTOTUNE_PILOT_STEPS:=2}"
: "${AUTOTUNE_LR_STEPS:=64}"
: "${AUTOTUNE_METRIC_WINDOW:=5}"
: "${AUTOTUNE_BATCH_SIZES:=1,2,4,8,16,32}"
: "${AUTOTUNE_LEARNING_RATES:=1e-7,5e-7,1e-6}"
: "${AUTOTUNE_FIXED_LR:=5e-7}"
: "${AUTOTUNE_MAX_TRAINER_IDLE_RATIO:=0.35}"
: "${AUTOTUNE_MAX_GENERATION_IDLE_RATIO:=0.35}"
: "${AUTOTUNE_ENGINE_CANDIDATES:=}"
: "${AUTOTUNE_RUN_TIMEOUT_SECONDS:=0}"
: "${AUTOTUNE_INTER_RUN_SLEEP_SECONDS:=5}"
: "${AUTOTUNE_FINAL_TIME:=${SLURM_TIMELIMIT:-48:00:00}}"

MODEL_BASENAME="$(basename "${MODEL_NAME}")"
DATASET_BASENAME="$(basename "${TRAIN_DATASET}" .jsonl)"
: "${AUTOTUNE_NAME:=${MODEL_BASENAME}_${DATASET_BASENAME}_job${SLURM_JOB_ID}}"
: "${AUTOTUNE_RESULT_ROOT:=${OUTPUT_DIR}/autotune/${AUTOTUNE_NAME}}"

setup_modules
setup_paths
setup_vllm_env

export PATH="${PROJECT_ROOT}/${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE DATASET_LOCAL_CACHE_DIR
export AUTOTUNE_MAX_TRAINER_IDLE_RATIO AUTOTUNE_MAX_GENERATION_IDLE_RATIO

controller_args=(
  --project-root "${PROJECT_ROOT}"
  --result-root "${AUTOTUNE_RESULT_ROOT}"
  --pilot-steps "${AUTOTUNE_PILOT_STEPS}"
  --lr-steps "${AUTOTUNE_LR_STEPS}"
  --metric-window "${AUTOTUNE_METRIC_WINDOW}"
  --batch-sizes "${AUTOTUNE_BATCH_SIZES}"
  --learning-rates "${AUTOTUNE_LEARNING_RATES}"
  --fixed-lr "${AUTOTUNE_FIXED_LR}"
  --run-timeout-seconds "${AUTOTUNE_RUN_TIMEOUT_SECONDS}"
  --inter-run-sleep-seconds "${AUTOTUNE_INTER_RUN_SLEEP_SECONDS}"
  --recommended-time "${AUTOTUNE_FINAL_TIME}"
)

if [ -n "${AUTOTUNE_ENGINE_CANDIDATES}" ]; then
  controller_args+=(--engine-candidates "${AUTOTUNE_ENGINE_CANDIDATES}")
fi

for config_path in "${RESOLVED_CONFIG_FILES[@]}"; do
  controller_args+=(--resolved-config "${config_path}")
done

log "Autotune results root: ${AUTOTUNE_RESULT_ROOT}"
log "Launching autotune controller"
python "${PROJECT_ROOT}/scripts/tune/if_rlvr_autotune.py" "${controller_args[@]}"

log "Autotune finished. Summary: ${AUTOTUNE_RESULT_ROOT}/summary.json"
