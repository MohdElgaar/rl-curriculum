#!/bin/bash
#SBATCH --job-name=if-rlvr
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
  echo "[if-rlvr] $*"
}

handle_exit() {
  local exit_code="$1"
  local restart_count="${SLURM_RESTART_COUNT:-0}"
  local max_restarts="${MAX_RESTARTS:-5}"
  local job_name="${SLURM_JOB_NAME:-if-rlvr}"
  local log_dir="${PROJECT_ROOT}/logs"
  local out_log="${log_dir}/${job_name}-${SLURM_JOB_ID}.out"
  local err_log="${log_dir}/${job_name}-${SLURM_JOB_ID}.err"

  trap - EXIT
  set +e

  log "Job completed with exit ${exit_code} (restart_count=${restart_count})"

  if [ "${exit_code}" -eq 0 ]; then
    exit 0
  fi

  if [ "${restart_count}" -ge "${max_restarts}" ]; then
    log "Job ${SLURM_JOB_ID} reached max restarts=${max_restarts}; not requeueing"
    exit "${exit_code}"
  fi

  [ -f "${out_log}" ] && cp -f -- "${out_log}" "${out_log%.out}_fail_${restart_count}.out"
  [ -f "${err_log}" ] && cp -f -- "${err_log}" "${err_log%.err}_fail_${restart_count}.err"

  log "Requeuing job ${SLURM_JOB_ID}"
  if scontrol requeue "${SLURM_JOB_ID}"; then
    exit 0
  fi

  log "Failed to requeue job ${SLURM_JOB_ID}"
  exit "${exit_code}"
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

setup_log_filters

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
log "Project root: ${PROJECT_ROOT}"
trap 'handle_exit $?' EXIT

ARCH="$(uname -m)"
IS_ARM_ARCH=0
if [ "${ARCH}" = "aarch64" ]; then
  IS_ARM_ARCH=1
  log "Running on ARM architecture"
  export UV_PROJECT_ENVIRONMENT=".venv_arm"
  export CUDA_MODULE=cuda/12.4.1
else
  export UV_PROJECT_ENVIRONMENT=".venv"
fi

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

if [ "${#CONFIG_FILES[@]}" -gt 0 ]; then
  for config_path in "${CONFIG_FILES[@]}"; do
    if ! CONFIG_FILE="$(resolve_config_path "${config_path}")"; then
      echo "Config not found: ${config_path}"
      exit 1
    fi
    log "Sourcing config: ${CONFIG_FILE}"
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"
  done
fi

# -----------------------------------------------------------------------------
# Defaults (override via sourced config or exported env vars)
# -----------------------------------------------------------------------------
: "${CUDA_MODULE:=cuda/13.1}"

: "${SCRATCH_ROOT:=/scratch4/workspace/mohamed_elgaar_student_uml_edu-rl-curriculum}"
: "${DATASET_LOCAL_CACHE_DIR:=${SCRATCH_ROOT}/data/open-instruct}"
: "${HF_HOME:=${SCRATCH_ROOT}/cache/huggingface}"
OUTPUT_DIR_DEFAULT="${SCRATCH_ROOT}/outputs"
: "${OUTPUT_DIR:=${OUTPUT_DIR_DEFAULT}}"

: "${EXP_NAME:=valpy_if_multi_tulu3.1_8b_grpo}"
: "${MODEL_NAME:=allenai/Llama-3.1-Tulu-3-8B-DPO}"
: "${CHAT_TEMPLATE_NAME:=tulu}"

: "${TRAIN_DATASET:=allenai/IF_multi_constraints_upto5}"
: "${TRAIN_SPLIT:=train}"
: "${TRAIN_DATASET_FRACTION:=1.0}"
: "${EVAL_DATASET:=${TRAIN_DATASET}}"
: "${EVAL_SAMPLES:=16}"

: "${BETA:=0.01}"
: "${KL_ESTIMATOR:=2}"
: "${LEARNING_RATE:=5e-7}"
: "${TEMPERATURE:=1.0}"
: "${TOTAL_EPISODES:=2000000}"

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
: "${LOCAL_EVAL_EVERY:=25}"
: "${SAVE_FREQ:=10}"
: "${CHECKPOINT_STATE_FREQ:=25}"
: "${KEEP_LAST_N_CHECKPOINTS:=-1}"
: "${IFEVAL_REWARD_SHAPING:=False}"
: "${IFEVAL_REWARD_SHAPING_CURRICULUM:=False}"
: "${IFEVAL_COMPETENCE_C0:=0.1}"
: "${IFEVAL_COMPETENCE_ALPHA:=1.0}"
: "${IFEVAL_NUM_CURRICULUM_STEPS:=-1}"


: "${APPLY_VERIFIABLE_REWARD:=True}"
: "${NON_STOP_PENALTY:=True}"
: "${NON_STOP_PENALTY_VALUE:=0.0}"
: "${OE_EVAL_TASKS:=ifeval::tulu}"
: "${OE_EVAL_MAX_LENGTH:=2048}"

: "${VLLM_NUM_ENGINES:=2}"
: "${VLLM_TP:=1}"
: "${NUM_LEARNERS_PER_NODE:=6}"

: "${VLLM_ALLOW_INSECURE_SERIALIZATION:=1}"
: "${VLLM_DISABLE_COMPILE_CACHE:=1}"
: "${VLLM_USE_V1:=1}"
: "${NCCL_CUMEM_ENABLE:=0}"
: "${TRITON_CACHE_DIR:=${SCRATCH_ROOT}/cache/triton}"

: "${CHECKPOINT_STATE_DIR:=}"
: "${GRADIENT_CHECKPOINTING:=1}"

# -----------------------------------------------------------------------------
# Setup (runs on submit node before srun)
# -----------------------------------------------------------------------------
setup_modules() {
  module load "${CUDA_MODULE}"
  # Some CUDA modulefiles do not add runtime libraries to LD_LIBRARY_PATH.
  # vLLM's extension import requires libcudart to be discoverable at runtime.
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
  OUTPUT_DIR="${OUTPUT_DIR}/${EXP_NAME}"
  if [ -n "${CHECKPOINT_STATE_FREQ}" ] && [ "${CHECKPOINT_STATE_FREQ}" -gt 0 ] 2>/dev/null; then
    : "${CHECKPOINT_STATE_DIR:=${OUTPUT_DIR}}"
  else
    CHECKPOINT_STATE_DIR=""
  fi
  export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE DATASET_LOCAL_CACHE_DIR TRITON_CACHE_DIR
  mkdir -p "${DATASET_LOCAL_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${OUTPUT_DIR}"
}

setup_vllm_env() {
  export VLLM_ALLOW_INSECURE_SERIALIZATION VLLM_DISABLE_COMPILE_CACHE VLLM_USE_V1 NCCL_CUMEM_ENABLE TRITON_CACHE_DIR
}

build_num_learners_list() {
  if [[ "${NUM_GPUS}" == "1" ]]; then
    NUM_LEARNERS_LIST=1
    return
  fi

  local -a gpus_per_node=()

  local scontrol_out
  scontrol_out="$(scontrol -d show job "$SLURM_JOB_ID")"

  local gpus_per_node=()
  while IFS= read -r line; do
    node=$(echo "$line" | sed -n -E 's/.*Nodes=([^[:space:]]+).*/\1/p')
    gpus=$(echo "$line" | sed -n -E 's/.*GRES=[^:]+:[^:]+:([0-9]+).*/\1/p')
    if [[ -n "$node" && -n "$gpus" ]]; then
      gpus_per_node+=("$gpus")
    fi
  done < <(echo "$scontrol_out" | grep -E "^\s*Nodes=")
  log "Detected per-node GPU allocation: ${gpus_per_node[@]}"

  local -a learner_per_node=("${gpus_per_node[@]}")
  local vllm_gpus=$((${VLLM_NUM_ENGINES:-0} * ${VLLM_TP:-1}))
  local remaining="${vllm_gpus}"
  for ((i = ${#learner_per_node[@]} - 1; i >= 0 && remaining > 0; i--)); do
    local take=$((learner_per_node[i] < remaining ? learner_per_node[i] : remaining))
    learner_per_node[i]=$((${learner_per_node[i]} - take))
    remaining=$((remaining - take))
  done
  [[ ${remaining} -gt 0 ]] && { echo "vLLM needs ${vllm_gpus} GPUs but allocation is smaller."; exit 1; }

  NUM_LEARNERS_LIST=""
  local c
  for c in "${learner_per_node[@]}"; do
    [[ ${c} -gt 0 ]] && NUM_LEARNERS_LIST="${NUM_LEARNERS_LIST} ${c}"
  done
  NUM_LEARNERS_LIST="${NUM_LEARNERS_LIST# }"
  [[ -z "${NUM_LEARNERS_LIST}" ]] && { echo "No learner GPUs after vLLM reservation."; exit 1; }
  log "NUM_LEARNERS_LIST=${NUM_LEARNERS_LIST}"
}

# -----------------------------------------------------------------------------
# Ray + training: one srun, one task per node
# See scripts/slurm/if_rlvr_ray_node.sh for per-node logic.
# -----------------------------------------------------------------------------
run_ray_and_training() {
  # Avoid Ray's default port range; use job-specific port
  RAY_HEAD_PORT=${RAY_HEAD_PORT:=$((30000 + (${SLURM_JOB_ID:-0} % 20000)))}
  RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:=$((50000 + (${SLURM_JOB_ID:-0} % 10000)))}

  export PROJECT_ROOT RAY_HEAD_PORT RAY_DASHBOARD_PORT

  log "Preflight: per-node GPU allocation (SLURM_GPUS_ON_NODE)"
  srun -n "${SLURM_NNODES}" --ntasks-per-node=1 bash -c 'echo "  $(hostname): gpus on node=$SLURM_GPUS_ON_NODE"'

  # Build training args for rank 0
  local -a train_args=(
    python -m open_instruct.grpo_fast
    --exp_name "${EXP_NAME}"
    --beta "${BETA}"
    --num_unique_prompts_rollout "${NUM_UNIQUE_PROMPTS}"
    --num_samples_per_prompt_rollout "${NUM_SAMPLES_PER_PROMPT}"
    --kl_estimator "${KL_ESTIMATOR}"
    --learning_rate "${LEARNING_RATE}"
    --dataset_local_cache_dir "${DATASET_LOCAL_CACHE_DIR}"
    --dataset_mixer_list "${TRAIN_DATASET}" "${TRAIN_DATASET_FRACTION}"
    --dataset_mixer_list_splits "${TRAIN_SPLIT}"
    --dataset_mixer_eval_list mohdelgaar/ifeval_rlvr 32 mohdelgaar/ifbench_rlvr 64 allenai/aime2024-25-rlvr 32 allenai/aime2024-25-rlvr 32 allenai/RLVR-MATH 32 allenai/RLVR-GSM 32 allenai/rlvr-code-data-python-r1-format-filtered 32
    --dataset_mixer_eval_list_splits train train test_2024 test_2025 train train train
    --max_prompt_token_length "${MAX_PROMPT_TOKEN_LENGTH}"
    --response_length "${RESPONSE_LENGTH}"
    --pack_length "${PACK_LENGTH}"
    --model_name_or_path "${MODEL_NAME}"
    --apply_verifiable_reward "${APPLY_VERIFIABLE_REWARD}"
    --non_stop_penalty "${NON_STOP_PENALTY}"
    --non_stop_penalty_value "${NON_STOP_PENALTY_VALUE}"
    --temperature "${TEMPERATURE}"
    --total_episodes "${TOTAL_EPISODES}"
    --ifeval_reward_shaping "${IFEVAL_REWARD_SHAPING}"
    --ifeval_reward_shaping_curriculum "${IFEVAL_REWARD_SHAPING_CURRICULUM}"
    --ifeval_competence_c0 "${IFEVAL_COMPETENCE_C0}"
    --ifeval_competence_alpha "${IFEVAL_COMPETENCE_ALPHA}"
    --ifeval_num_curriculum_steps "${IFEVAL_NUM_CURRICULUM_STEPS}"
    --deepspeed_stage "${DEEPSPEED_STAGE}"
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
    --num_mini_batches "${NUM_MINI_BATCHES}"
    --num_learners_per_node ${NUM_LEARNERS_LIST}
    --num_epochs "${NUM_EPOCHS}"
    --vllm_tensor_parallel_size "${VLLM_TP}"
    --vllm_num_engines "${VLLM_NUM_ENGINES}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
    --async_steps "${ASYNC_STEPS}"
    --seed "${SEED}"
    --local_eval_every "${LOCAL_EVAL_EVERY}"
    --eval_on_step_0
    --save_freq "${SAVE_FREQ}"
    --keep_last_n_checkpoints "${KEEP_LAST_N_CHECKPOINTS}"
    --checkpoint_state_freq "${CHECKPOINT_STATE_FREQ}"
    --checkpoint_state_dir "${CHECKPOINT_STATE_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --inflight_updates True
    --code_pass_rate_reward_threshold 0.99
    --with_tracking
  )

  [ "${GRADIENT_CHECKPOINTING}" = "1" ] && train_args+=(--gradient_checkpointing)
  if [ -n "${CHECKPOINT_STATE_DIR}" ] && [ -n "${CHECKPOINT_STATE_FREQ}" ] && [ "${CHECKPOINT_STATE_FREQ}" -gt 0 ] 2>/dev/null; then
    train_args+=(--checkpoint_state_freq "${CHECKPOINT_STATE_FREQ}" --checkpoint_state_dir "${CHECKPOINT_STATE_DIR}")
  fi
  if [ -n "${ADDITIONAL_ARGS:-}" ]; then
    read -r -a extra_arr <<< "${ADDITIONAL_ARGS}"
    train_args+=("${extra_arr[@]}")
  fi
  if [ "${IS_ARM_ARCH}" = "1" ]; then
    train_args+=(--attn_implementation sdpa --single_gpu_mode --vllm_enforce_eager --vllm_gpu_memory_utilization 0.3)
  fi

  srun -n "${SLURM_NNODES}" --ntasks-per-node=1 "${PROJECT_ROOT}/scripts/slurm/if_rlvr_ray_node.sh" "${train_args[@]}"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
  log "Setting up modules"
  setup_modules
  log "Setting up paths"
  setup_paths
  log "Setting up vllm environment"
  setup_vllm_env
  log "Building num learners list (${SLURM_NNODES:-1} nodes)"
  build_num_learners_list
  log "IFEval reward shaping: ${IFEVAL_REWARD_SHAPING}"
  log "IFEval shaping curriculum: ${IFEVAL_REWARD_SHAPING_CURRICULUM} (c0=${IFEVAL_COMPETENCE_C0}, alpha=${IFEVAL_COMPETENCE_ALPHA}), num curriculum steps=${IFEVAL_NUM_CURRICULUM_STEPS}"
  log "Running Ray + training (srun)"
  run_ray_and_training
}

main "$@"
