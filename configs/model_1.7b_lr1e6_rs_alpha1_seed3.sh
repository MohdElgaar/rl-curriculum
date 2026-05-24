#!/bin/bash
#
# Usage:
#   source configs/model_1.7b_lr1e6_reward_shaping.sh
#   source configs/gpus_2.sh   # or configs/gpus_8.sh
#
# This matches the default 1.7B / 1e-6 setup, but enables partial credit for
# numeric/cardinality-style IFEval constraints.

# ============================================================================
# Model Configuration
# ============================================================================

export MODEL_NAME="Qwen/Qwen3-1.7B"

# ============================================================================
# Dataset Configuration
# ============================================================================

export TRAIN_SPLIT="train"
export TRAIN_DATASET="allenai/IF_multi_constraints_upto5"
export TRAIN_DATASET_FRACTION="1.0"

# ============================================================================
# Training Hyperparameters
# ============================================================================

export LEARNING_RATE=1e-6
export BETA=0.01
export TOTAL_EPISODES=768000
export NUM_TRAINING_STEPS=1000
export TEMPERATURE=1.0
export ASYNC_STEPS=1

# ============================================================================
# Batch Configuration
# ============================================================================

export PER_DEVICE_BATCH_SIZE=2
export NUM_UNIQUE_PROMPTS=48
export NUM_SAMPLES_PER_PROMPT=16
export NUM_MINI_BATCHES=2

# ============================================================================
# Token Length Configuration
# ============================================================================

export MAX_PROMPT_TOKEN_LENGTH=2048
export RESPONSE_LENGTH=2048
export PACK_LENGTH=4096

# ============================================================================
# Training Schedule
# ============================================================================

export SEED=3
export SAVE_FREQ=-1
export CHECKPOINT_STATE_FREQ=25
export KEEP_LAST_N_CHECKPOINTS=1
export LOCAL_EVAL_EVERY=10

# ============================================================================
# Advanced Options
# ============================================================================

export GROUND_TRUTHS_KEY="ground_truth"
export IFEVAL_REWARD_SHAPING=True
export IFEVAL_REWARD_SHAPING_CURRICULUM=True
export IFEVAL_COMPETENCE_C0=0.1
export IFEVAL_COMPETENCE_ALPHA=1.0
export IFEVAL_NUM_CURRICULUM_STEPS=200

# ============================================================================
# Experiment Identification
# ============================================================================

DATASET_BASENAME=$(basename "${TRAIN_DATASET}" .jsonl)
MODEL_BASENAME=$(basename "${MODEL_NAME}")
export EXP_NAME="${MODEL_BASENAME}_${DATASET_BASENAME}_${LEARNING_RATE}_alpha${IFEVAL_COMPETENCE_ALPHA}_n${IFEVAL_NUM_CURRICULUM_STEPS}_seed${SEED}"  # separate OUTPUT_DIR per seed
