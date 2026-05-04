#!/bin/bash
#
# IF-RLVR Training Script for GRPO
# Based on open-instruct example: scripts/train/rlvr/valpy_if_grpo_fast.sh
# Adapted for external use (no Beaker)
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Determine the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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
    source "${CONFIG_FILE}"
  done
fi

# Experiment name - change this for different runs
EXP_NAME="${EXP_NAME:-if_rlvr_tulu3_8b_grpo}"

# Model configuration
MODEL_NAME="${MODEL_NAME:-allenai/Llama-3.1-Tulu-3-8B-DPO}"
# If set, passed through to --chat_template_name (empty = use tokenizer default, e.g. Qwen).
CHAT_TEMPLATE_NAME="${CHAT_TEMPLATE_NAME:-}"

# Dataset configuration
TRAIN_DATASET="${TRAIN_DATASET:-allenai/IF_multi_constraints_upto5}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_DATASET="${EVAL_DATASET:-allenai/IF_multi_constraints_upto5}"
TRAIN_DATASET_FRACTION="${TRAIN_DATASET_FRACTION:-1.0}"

# Hardware configuration
NUM_GPUS="${NUM_GPUS:-8}"
NUM_LEARNERS_PER_NODE="${NUM_LEARNERS_PER_NODE:-6}"
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-10}"

# Training hyperparameters
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
BETA="${BETA:-0.01}"
TOTAL_EPISODES="${TOTAL_EPISODES:-2000000}"
# 768000 episodes = 1000 steps (48 unique prompts × 16 samples per prompt)
NUM_TRAINING_STEPS="${NUM_TRAINING_STEPS:-2604}"
TEMPERATURE="${TEMPERATURE:-1.0}"
ASYNC_STEPS="${ASYNC_STEPS:-1}"

# Batch configuration
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
NUM_UNIQUE_PROMPTS="${NUM_UNIQUE_PROMPTS:-48}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-16}"
NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-2}"

# Token length configuration
MAX_PROMPT_TOKEN_LENGTH="${MAX_PROMPT_TOKEN_LENGTH:-2048}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-2048}"
PACK_LENGTH="${PACK_LENGTH:-4096}"

# Other settings
SEED="${SEED:-1}"
SAVE_FREQ="${SAVE_FREQ:-10}"
LOCAL_EVAL_EVERY="${LOCAL_EVAL_EVERY:-25}"
CHECKPOINT_STATE_FREQ="${CHECKPOINT_STATE_FREQ:-25}"
KEEP_LAST_N_CHECKPOINTS="${KEEP_LAST_N_CHECKPOINTS:--1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-False}"
IFEVAL_REWARD_SHAPING="${IFEVAL_REWARD_SHAPING:-False}"
IFEVAL_REWARD_SHAPING_CURRICULUM="${IFEVAL_REWARD_SHAPING_CURRICULUM:-False}"
IFEVAL_RANDOM_ZERO_REWARD="${IFEVAL_RANDOM_ZERO_REWARD:-False}"
IFEVAL_COMPETENCE_C0="${IFEVAL_COMPETENCE_C0:-0.1}"
IFEVAL_COMPETENCE_ALPHA="${IFEVAL_COMPETENCE_ALPHA:-1.0}"
IFEVAL_NUM_CURRICULUM_STEPS="${IFEVAL_NUM_CURRICULUM_STEPS:--1}"

MATH_REWARD_SHAPING="${MATH_REWARD_SHAPING:-False}"
MATH_REWARD_SHAPING_CURRICULUM="${MATH_REWARD_SHAPING_CURRICULUM:-False}"
MATH_RANDOM_ZERO_REWARD="${MATH_RANDOM_ZERO_REWARD:-False}"
MATH_COMPETENCE_C0="${MATH_COMPETENCE_C0:-0.1}"
MATH_COMPETENCE_ALPHA="${MATH_COMPETENCE_ALPHA:-1.0}"
MATH_NUM_CURRICULUM_STEPS="${MATH_NUM_CURRICULUM_STEPS:--1}"

GSM_REWARD_SHAPING="${GSM_REWARD_SHAPING:-False}"
GSM_REWARD_SHAPING_CURRICULUM="${GSM_REWARD_SHAPING_CURRICULUM:-False}"
GSM_RANDOM_ZERO_REWARD="${GSM_RANDOM_ZERO_REWARD:-False}"
GSM_COMPETENCE_C0="${GSM_COMPETENCE_C0:-0.1}"
GSM_COMPETENCE_ALPHA="${GSM_COMPETENCE_ALPHA:-1.0}"
GSM_NUM_CURRICULUM_STEPS="${GSM_NUM_CURRICULUM_STEPS:--1}"

# Dataset cache (HF datasets / preprocessing); avoids empty path on fresh envs (e.g. cloud VMs).
DATASET_LOCAL_CACHE_DIR="${DATASET_LOCAL_CACHE_DIR:-${PROJECT_ROOT}/.dataset_cache}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
OUTPUT_DIR="${OUTPUT_DIR}/${EXP_NAME}"
CHECKPOINT_STATE_DIR="${CHECKPOINT_STATE_DIR:-${OUTPUT_DIR}}"

# vLLM configuration
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

# ============================================================================
# Check requirements
# ============================================================================

echo "============================================"
echo "IF-RLVR Training with GRPO"
echo "============================================"
echo "Experiment: ${EXP_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${TRAIN_DATASET}"
echo "GPUs: ${NUM_GPUS}"
echo "IFEval reward shaping: ${IFEVAL_REWARD_SHAPING}"
echo "IFEval shaping curriculum: ${IFEVAL_REWARD_SHAPING_CURRICULUM} (c0=${IFEVAL_COMPETENCE_C0}, alpha=${IFEVAL_COMPETENCE_ALPHA}), num curriculum steps=${IFEVAL_NUM_CURRICULUM_STEPS}"
echo "IFEval random-zero reward: ${IFEVAL_RANDOM_ZERO_REWARD}"
echo "MATH reward shaping: ${MATH_REWARD_SHAPING} (curriculum=${MATH_REWARD_SHAPING_CURRICULUM}, random_zero=${MATH_RANDOM_ZERO_REWARD}, c0=${MATH_COMPETENCE_C0}, alpha=${MATH_COMPETENCE_ALPHA}, steps=${MATH_NUM_CURRICULUM_STEPS})"
echo "GSM reward shaping: ${GSM_REWARD_SHAPING} (curriculum=${GSM_REWARD_SHAPING_CURRICULUM}, random_zero=${GSM_RANDOM_ZERO_REWARD}, c0=${GSM_COMPETENCE_C0}, alpha=${GSM_COMPETENCE_ALPHA}, steps=${GSM_NUM_CURRICULUM_STEPS})"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"
echo ""

# Check if we're in the right directory structure
if [ ! -d "${PROJECT_ROOT}/open-instruct" ]; then
    echo "Error: open-instruct directory not found!"
    echo "Please run this script from the rl_curriculum repository."
    echo "Current project root: ${PROJECT_ROOT}"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed!"
    echo "Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Make sure CUDA is properly installed."
fi

# ============================================================================
# Run Training
# ============================================================================

echo "Starting training..."
echo "Project root: ${PROJECT_ROOT}"
echo ""

export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

extra_args=()
if [ -n "${CHAT_TEMPLATE_NAME}" ]; then
    extra_args+=(--chat_template_name "${CHAT_TEMPLATE_NAME}")
fi
if [ -n "${EXTRA_GRPO_ARGS:-}" ]; then
    read -r -a _extra_grpo_arr <<< "${EXTRA_GRPO_ARGS}"
    extra_args+=("${_extra_grpo_arr[@]}")
fi

eval_step_args=()
if [ "${EVAL_ON_STEP_0:-True}" = "True" ] || [ "${EVAL_ON_STEP_0:-True}" = "true" ]; then
    eval_step_args+=(--eval_on_step_0)
fi

uv run python -m open_instruct.grpo_fast \
    --exp_name "${EXP_NAME}" \
    --beta ${BETA} \
    --num_unique_prompts_rollout ${NUM_UNIQUE_PROMPTS} \
    --num_samples_per_prompt_rollout ${NUM_SAMPLES_PER_PROMPT} \
    --kl_estimator 2 \
    --learning_rate ${LEARNING_RATE} \
    --dataset_local_cache_dir "${DATASET_LOCAL_CACHE_DIR}" \
    --dataset_mixer_list ${TRAIN_DATASET} ${TRAIN_DATASET_FRACTION} \
    --dataset_mixer_list_splits ${TRAIN_SPLIT} \
    --dataset_mixer_eval_list mohdelgaar/ifeval_rlvr 32 mohdelgaar/ifbench_rlvr 64 allenai/aime2024-25-rlvr 32 allenai/aime2024-25-rlvr 32 allenai/RLVR-MATH 32 allenai/RLVR-GSM 32 allenai/rlvr-code-data-python-r1-format-filtered 32 \
    --dataset_mixer_eval_list_splits train train test_2024 test_2025 train train train \
    --max_prompt_token_length ${MAX_PROMPT_TOKEN_LENGTH} \
    --response_length ${RESPONSE_LENGTH} \
    --pack_length ${PACK_LENGTH} \
    --model_name_or_path ${MODEL_NAME} \
    --apply_verifiable_reward True \
    --non_stop_penalty True \
    --non_stop_penalty_value 0.0 \
    --temperature ${TEMPERATURE} \
    --total_episodes ${TOTAL_EPISODES} \
    --num_training_steps ${NUM_TRAINING_STEPS} \
    --ifeval_reward_shaping ${IFEVAL_REWARD_SHAPING} \
    --ifeval_reward_shaping_curriculum ${IFEVAL_REWARD_SHAPING_CURRICULUM} \
    --ifeval_random_zero_reward ${IFEVAL_RANDOM_ZERO_REWARD} \
    --ifeval_competence_c0 ${IFEVAL_COMPETENCE_C0} \
    --ifeval_competence_alpha ${IFEVAL_COMPETENCE_ALPHA} \
    --ifeval_num_curriculum_steps ${IFEVAL_NUM_CURRICULUM_STEPS} \
    --math_reward_shaping ${MATH_REWARD_SHAPING} \
    --math_reward_shaping_curriculum ${MATH_REWARD_SHAPING_CURRICULUM} \
    --math_random_zero_reward ${MATH_RANDOM_ZERO_REWARD} \
    --math_competence_c0 ${MATH_COMPETENCE_C0} \
    --math_competence_alpha ${MATH_COMPETENCE_ALPHA} \
    --math_num_curriculum_steps ${MATH_NUM_CURRICULUM_STEPS} \
    --gsm_reward_shaping ${GSM_REWARD_SHAPING} \
    --gsm_reward_shaping_curriculum ${GSM_REWARD_SHAPING_CURRICULUM} \
    --gsm_random_zero_reward ${GSM_RANDOM_ZERO_REWARD} \
    --gsm_competence_c0 ${GSM_COMPETENCE_C0} \
    --gsm_competence_alpha ${GSM_COMPETENCE_ALPHA} \
    --gsm_num_curriculum_steps ${GSM_NUM_CURRICULUM_STEPS} \
    --deepspeed_stage 2 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --num_mini_batches ${NUM_MINI_BATCHES} \
    --num_learners_per_node ${NUM_LEARNERS_PER_NODE} \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines ${VLLM_NUM_ENGINES} \
    --lr_scheduler_type constant \
    --seed ${SEED} \
    --local_eval_every ${LOCAL_EVAL_EVERY} \
    --save_freq ${SAVE_FREQ} \
    --checkpoint_state_freq ${CHECKPOINT_STATE_FREQ} \
    --checkpoint_state_dir "${CHECKPOINT_STATE_DIR}" \
    --keep_last_n_checkpoints "${KEEP_LAST_N_CHECKPOINTS}" \
    --async_steps ${ASYNC_STEPS} \
    --gradient_checkpointing \
    --trust_remote_code "${TRUST_REMOTE_CODE}" \
    --with_tracking \
    --inflight_updates True \
    --code_pass_rate_reward_threshold 0.99 \
    "${eval_step_args[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    "${extra_args[@]}"

echo ""
echo "============================================"
echo "Training completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================"

