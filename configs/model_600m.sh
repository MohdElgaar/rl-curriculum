#!/bin/bash
#
# Model config: Qwen/Qwen3-0.6B
# Usage:
#   source configs/model_600m.sh
#   source configs/gpus_2.sh   # or configs/gpus_8.sh
#
# This file defines model- and training-related settings.
# GPU-specific settings live in configs/gpus_*.sh.

# ============================================================================
# Model Configuration
# ============================================================================

# Base model to fine-tune
# Options:
#   - allenai/Llama-3.1-Tulu-3-8B-DPO (default, good starting point)
#   - allenai/OLMo-2-1124-7B-DPO
#   - allenai/OLMo-2-1124-13B-DPO
#   - meta-llama/Llama-3.1-8B
#   - Qwen/Qwen2.5-7B (for smaller experiments)
export MODEL_NAME="Qwen/Qwen3-0.6B"

# ============================================================================
# Dataset Configuration
# ============================================================================

# Training dataset
# Default: allenai/IF_multi_constraints_upto5 (IF-RLVR training data)
export TRAIN_SPLIT="train"
export TRAIN_DATASET="allenai/IF_multi_constraints_upto5"
export TRAIN_DATASET_FRACTION="1.0"
export EVAL_DATASET=${TRAIN_DATASET}

# ============================================================================
# Experiment Identification
# ============================================================================

# Extract dataset name from path and create experiment name
DATASET_BASENAME=$(basename "${TRAIN_DATASET}" .jsonl)
export EXP_NAME="qwen3_0.6b_instruct_${DATASET_BASENAME}"  # Name for this experiment run

# ============================================================================
# Scratch / Cache / Output Configuration
# ============================================================================

export SCRATCH_ROOT="/scratch4/workspace/mohamed_elgaar_student_uml_edu-rl-curriculum"
export HF_HOME="${SCRATCH_ROOT}/cache/huggingface"
export DATASET_LOCAL_CACHE_DIR="${SCRATCH_ROOT}/data/open-instruct"

# ============================================================================
# Training Hyperparameters
# ============================================================================

# Learning rate
# Typical range: 1e-7 to 1e-6 for RLVR
export LEARNING_RATE=3e-7

# KL divergence coefficient (beta)
# Higher = stay closer to reference policy
# Typical range: 0.001 to 0.1
export BETA=0.01

# Total training episodes
# Full training: 2000000
# Quick test: 10000-100000
export TOTAL_EPISODES=768000

# Sampling temperature
# Higher = more diverse generations
export TEMPERATURE=1.0

# Number of steps ahead to generate responses
# Higher = more context for generation
export ASYNC_STEPS=1

# ============================================================================
# Batch Configuration
# ============================================================================

# Batch size per device
# Usually 1 for large models
export PER_DEVICE_BATCH_SIZE=8

# Number of unique prompts per rollout
# Total samples per batch = NUM_UNIQUE_PROMPTS × NUM_SAMPLES_PER_PROMPT
export NUM_UNIQUE_PROMPTS=48

# Samples generated per prompt
# More samples = better gradient estimates but slower
export NUM_SAMPLES_PER_PROMPT=16

# Number of mini batches
export NUM_MINI_BATCHES=2

# ============================================================================
# Token Length Configuration
# ============================================================================

# Maximum tokens in the prompt
export MAX_PROMPT_TOKEN_LENGTH=2048

# Maximum tokens in the response
export RESPONSE_LENGTH=2048

# Packing length for training efficiency
export PACK_LENGTH=4096

# ============================================================================
# Training Schedule
# ============================================================================

# Random seed for reproducibility
export SEED=1

# Save checkpoint every N episodes
export SAVE_FREQ=-1

# Run local evaluation every N episodes
export LOCAL_EVAL_EVERY=25

# ============================================================================
# Output Configuration
# ============================================================================

# Output directory for checkpoints and logs
export OUTPUT_DIR="${SCRATCH_ROOT}/outputs/${EXP_NAME}"

# ============================================================================
# Advanced Options (Optional)
# ============================================================================

# Chat template to use
# Options: tulu, llama3, simple, r1_simple_chat_postpend_think etc.
export CHAT_TEMPLATE_NAME="qwen3"

# Stop strings for generation
export STOP_STRINGS="</answer>"

# Ground truth key in dataset
export GROUND_TRUTHS_KEY="ground_truth"

# ============================================================================
# Notes
# ============================================================================

# Memory usage guide:
# - 8B model with batch_size=1 needs ~40GB GPU memory per learner
# - Reduce NUM_LEARNERS_PER_NODE if you get OOM errors
# - Reduce RESPONSE_LENGTH or MAX_TOKEN_LENGTH to save memory
#
# Training time guide:
# - 8 GPUs, 2M episodes: ~2-3 days
# - 1 GPU, 200 episodes (debug): ~30 minutes
#
# Performance tips:
# - Use more NUM_SAMPLES_PER_PROMPT for better gradients
# - Use higher LEARNING_RATE for base models, lower for already fine-tuned models
# - Adjust BETA if model diverges too much from reference policy
