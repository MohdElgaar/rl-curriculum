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

# Experiment name - change this for different runs
EXP_NAME="${EXP_NAME:-if_rlvr_tulu3_8b_grpo}"

# Model configuration
MODEL_NAME="${MODEL_NAME:-allenai/Llama-3.1-Tulu-3-8B-DPO}"
CHAT_TEMPLATE_NAME="${CHAT_TEMPLATE_NAME:-tulu}"

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
CHECKPOINT_STATE_FREQ="${CHECKPOINT_STATE_FREQ:-25}" # Save full training state every 25 steps

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-../outputs/${EXP_NAME}}"
CHECKPOINT_STATE_DIR="${OUTPUT_DIR}"

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
echo "Output: ${OUTPUT_DIR}"
echo "============================================"
echo ""

# Determine the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

python -m open_instruct.grpo_fast \
    --exp_name "${EXP_NAME}" \
    --beta ${BETA} \
    --num_unique_prompts_rollout ${NUM_UNIQUE_PROMPTS} \
    --num_samples_per_prompt_rollout ${NUM_SAMPLES_PER_PROMPT} \
    --kl_estimator kl3 \
    --learning_rate ${LEARNING_RATE} \
    --dataset_local_cache_dir /data/mohamed/data/open-instruct/ \
    --dataset_mixer_list ${TRAIN_DATASET} ${TRAIN_DATASET_FRACTION} \
    --dataset_mixer_list_splits ${TRAIN_SPLIT} \
    --dataset_mixer_eval_list ${EVAL_DATASET} 16 \
    --dataset_mixer_eval_list_splits ${TRAIN_SPLIT} \
    --max_prompt_token_length ${MAX_PROMPT_TOKEN_LENGTH} \
    --response_length ${RESPONSE_LENGTH} \
    --pack_length ${PACK_LENGTH} \
    --model_name_or_path ${MODEL_NAME} \
    --apply_verifiable_reward True \
    --non_stop_penalty True \
    --non_stop_penalty_value 0.0 \
    --temperature ${TEMPERATURE} \
    --chat_template_name ${CHAT_TEMPLATE_NAME} \
    --oe_eval_tasks ifeval::tulu \
    --oe_eval_max_length 2048 \
    --total_episodes ${TOTAL_EPISODES} \
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
    --async_steps ${ASYNC_STEPS} \
    --gradient_checkpointing \
    --with_tracking \
    --eval_on_step_0 \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "============================================"
echo "Training completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================"

