#!/bin/bash
# Single-node IF-RLVR launcher for Vast.ai (no SLURM).
# Usage: bash scripts/vast/run_if_rlvr.sh configs/gpus_8.sh configs/data_vastai.sh configs/model_gemma_4_e2b_rs.sh
set -euo pipefail

log() { echo "[vast-if-rlvr] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"

# shellcheck source=lib/instance_env.sh
source "${SCRIPT_DIR}/lib/instance_env.sh"
load_vast_instance_env
require_vast_instance_env

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

CONFIG_FILES=("$@")
if [ "${#CONFIG_FILES[@]}" -eq 0 ]; then
  echo "Usage: $0 <config.sh> [config.sh ...]"
  exit 1
fi

for config_path in "${CONFIG_FILES[@]}"; do
  if ! CONFIG_FILE="$(resolve_config_path "${config_path}")"; then
    echo "Config not found: ${config_path}"
    exit 1
  fi
  log "Sourcing config: ${CONFIG_FILE}"
  # shellcheck source=/dev/null
  source "${CONFIG_FILE}"
done

: "${SCRATCH_ROOT:=/workspace/rl-curriculum}"
: "${CUDA_MODULE:=}"
: "${EXP_NAME:=vast_if_rlvr}"
: "${MODEL_NAME:=allenai/Llama-3.1-Tulu-3-8B-DPO}"
: "${TRAIN_DATASET:=allenai/IF_multi_constraints_upto5}"
: "${TRAIN_SPLIT:=train}"
: "${TRAIN_DATASET_FRACTION:=1.0}"
: "${BETA:=0.01}"
: "${KL_ESTIMATOR:=2}"
: "${LEARNING_RATE:=5e-7}"
: "${TEMPERATURE:=1.0}"
: "${TOTAL_EPISODES:=2000000}"
: "${NUM_TRAINING_STEPS:=1000}"
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
: "${IFEVAL_RANDOM_ZERO_REWARD:=False}"
: "${IFEVAL_COMPETENCE_C0:=0.1}"
: "${IFEVAL_COMPETENCE_ALPHA:=1.0}"
: "${IFEVAL_NUM_CURRICULUM_STEPS:=-1}"
: "${MATH_REWARD_SHAPING:=False}"
: "${MATH_REWARD_SHAPING_CURRICULUM:=False}"
: "${MATH_RANDOM_ZERO_REWARD:=False}"
: "${MATH_COMPETENCE_C0:=0.1}"
: "${MATH_COMPETENCE_ALPHA:=1.0}"
: "${MATH_NUM_CURRICULUM_STEPS:=-1}"
: "${GSM_REWARD_SHAPING:=False}"
: "${GSM_REWARD_SHAPING_CURRICULUM:=False}"
: "${GSM_RANDOM_ZERO_REWARD:=False}"
: "${GSM_COMPETENCE_C0:=0.1}"
: "${GSM_COMPETENCE_ALPHA:=1.0}"
: "${GSM_NUM_CURRICULUM_STEPS:=-1}"
: "${APPLY_VERIFIABLE_REWARD:=True}"
: "${NON_STOP_PENALTY:=True}"
: "${NON_STOP_PENALTY_VALUE:=0.0}"
: "${VLLM_NUM_ENGINES:=2}"
: "${VLLM_TP:=1}"
: "${NUM_GPUS:=8}"
: "${NUM_LEARNERS_PER_NODE:=6}"
: "${PUSH_TO_HUB:=False}"
: "${GRADIENT_CHECKPOINTING:=1}"
: "${OUTPUT_DIR:=${SCRATCH_ROOT}/outputs}"

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export NCCL_CUMEM_ENABLE=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1

if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN
elif [ -n "${HF_TOKEN_PATH:-}" ] && [ -f "${HF_TOKEN_PATH}" ]; then
  export HF_TOKEN_PATH
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_API_KEY
fi

mkdir -p "${DATASET_LOCAL_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${PROJECT_ROOT}/logs"
OUTPUT_DIR="${OUTPUT_DIR}/${EXP_NAME}"
CHECKPOINT_STATE_DIR="${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE DATASET_LOCAL_CACHE_DIR TRITON_CACHE_DIR OUTPUT_DIR

# IFEval verifiable rewards call nltk.word_tokenize; Ray workers need punkt_tab on disk.
export NLTK_DATA="${NLTK_DATA:-${SCRATCH_ROOT}/nltk_data}"
mkdir -p "${NLTK_DATA}"
if ! uv run python -c "from nltk.data import find; find('tokenizers/punkt_tab/english/')" 2>/dev/null; then
  log "Downloading NLTK data into ${NLTK_DATA}"
  uv run python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt_tab', 'punkt', 'stopwords')]"
fi
if ! uv run python -c "from nltk.data import find; find('tokenizers/punkt_tab/english/')"; then
  echo "ERROR: NLTK punkt_tab missing under ${NLTK_DATA}"
  exit 1
fi
log "NLTK data ready at ${NLTK_DATA}"

GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
if [ "${GPU_COUNT}" -lt 1 ]; then
  echo "No GPUs visible via nvidia-smi"
  exit 1
fi
vllm_gpus=$((${VLLM_NUM_ENGINES:-0} * ${VLLM_TP:-1}))
learners=$((GPU_COUNT - vllm_gpus))
if [ "${learners}" -lt 1 ]; then
  echo "vLLM needs ${vllm_gpus} GPUs but only ${GPU_COUNT} are visible."
  exit 1
fi
NUM_LEARNERS_PER_NODE="${learners}"

if ! command -v uv &>/dev/null; then
  echo "Error: uv is not installed (curl -LsSf https://astral.sh/uv/install.sh | sh)"
  exit 1
fi

log "GPUs=${GPU_COUNT} learners=${NUM_LEARNERS_PER_NODE} vllm_engines=${VLLM_NUM_ENGINES} exp=${EXP_NAME}"
log "Output: ${OUTPUT_DIR}"

optional_args=()
[ "${GRADIENT_CHECKPOINTING}" = "1" ] && optional_args+=(--gradient_checkpointing)
if [ -n "${ATTN_IMPLEMENTATION:-}" ]; then
  optional_args+=(--attn_implementation "${ATTN_IMPLEMENTATION}")
fi
if [ -n "${VLLM_ATTENTION_BACKEND:-}" ]; then
  optional_args+=(--vllm_attention_backend "${VLLM_ATTENTION_BACKEND}")
fi
if [ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]; then
  optional_args+=(--vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}")
fi

cd "${PROJECT_ROOT}"
uv run python -m open_instruct.grpo_fast \
  --exp_name "${EXP_NAME}" \
  --beta "${BETA}" \
  --num_unique_prompts_rollout "${NUM_UNIQUE_PROMPTS}" \
  --num_samples_per_prompt_rollout "${NUM_SAMPLES_PER_PROMPT}" \
  --kl_estimator "${KL_ESTIMATOR}" \
  --learning_rate "${LEARNING_RATE}" \
  --dataset_local_cache_dir "${DATASET_LOCAL_CACHE_DIR}" \
  --dataset_mixer_list "${TRAIN_DATASET}" "${TRAIN_DATASET_FRACTION}" \
  --dataset_mixer_list_splits "${TRAIN_SPLIT}" \
  --dataset_mixer_eval_list mohdelgaar/ifeval_rlvr 32 mohdelgaar/ifbench_rlvr 64 allenai/aime2024-25-rlvr 32 allenai/aime2024-25-rlvr 32 allenai/RLVR-MATH 32 allenai/RLVR-GSM 32 allenai/rlvr-code-data-python-r1-format-filtered 32 \
  --dataset_mixer_eval_list_splits train train test_2024 test_2025 train train train \
  --max_prompt_token_length "${MAX_PROMPT_TOKEN_LENGTH}" \
  --response_length "${RESPONSE_LENGTH}" \
  --pack_length "${PACK_LENGTH}" \
  --model_name_or_path "${MODEL_NAME}" \
  --apply_verifiable_reward "${APPLY_VERIFIABLE_REWARD}" \
  --non_stop_penalty "${NON_STOP_PENALTY}" \
  --non_stop_penalty_value "${NON_STOP_PENALTY_VALUE}" \
  --temperature "${TEMPERATURE}" \
  --total_episodes "${TOTAL_EPISODES}" \
  --num_training_steps "${NUM_TRAINING_STEPS}" \
  --ifeval_reward_shaping "${IFEVAL_REWARD_SHAPING}" \
  --ifeval_reward_shaping_curriculum "${IFEVAL_REWARD_SHAPING_CURRICULUM}" \
  --ifeval_random_zero_reward "${IFEVAL_RANDOM_ZERO_REWARD}" \
  --ifeval_competence_c0 "${IFEVAL_COMPETENCE_C0}" \
  --ifeval_competence_alpha "${IFEVAL_COMPETENCE_ALPHA}" \
  --ifeval_num_curriculum_steps "${IFEVAL_NUM_CURRICULUM_STEPS}" \
  --math_reward_shaping "${MATH_REWARD_SHAPING}" \
  --math_reward_shaping_curriculum "${MATH_REWARD_SHAPING_CURRICULUM}" \
  --math_random_zero_reward "${MATH_RANDOM_ZERO_REWARD}" \
  --math_competence_c0 "${MATH_COMPETENCE_C0}" \
  --math_competence_alpha "${MATH_COMPETENCE_ALPHA}" \
  --math_num_curriculum_steps "${MATH_NUM_CURRICULUM_STEPS}" \
  --gsm_reward_shaping "${GSM_REWARD_SHAPING}" \
  --gsm_reward_shaping_curriculum "${GSM_REWARD_SHAPING_CURRICULUM}" \
  --gsm_random_zero_reward "${GSM_RANDOM_ZERO_REWARD}" \
  --gsm_competence_c0 "${GSM_COMPETENCE_C0}" \
  --gsm_competence_alpha "${GSM_COMPETENCE_ALPHA}" \
  --gsm_num_curriculum_steps "${GSM_NUM_CURRICULUM_STEPS}" \
  --deepspeed_stage "${DEEPSPEED_STAGE}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --num_mini_batches "${NUM_MINI_BATCHES}" \
  --num_learners_per_node "${NUM_LEARNERS_PER_NODE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --vllm_tensor_parallel_size "${VLLM_TP}" \
  --vllm_num_engines "${VLLM_NUM_ENGINES}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --async_steps "${ASYNC_STEPS}" \
  --seed "${SEED}" \
  --local_eval_every "${LOCAL_EVAL_EVERY}" \
  --eval_on_step_0 \
  --save_freq "${SAVE_FREQ}" \
  --keep_last_n_checkpoints "${KEEP_LAST_N_CHECKPOINTS}" \
  --checkpoint_state_freq "${CHECKPOINT_STATE_FREQ}" \
  --checkpoint_state_dir "${CHECKPOINT_STATE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --push_to_hub "${PUSH_TO_HUB}" \
  --inflight_updates True \
  --code_api_url https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program \
  --code_pass_rate_reward_threshold 0.99 \
  --with_tracking \
  "${optional_args[@]}"