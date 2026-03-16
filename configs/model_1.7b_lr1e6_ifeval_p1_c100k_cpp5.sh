#!/bin/bash
#
# Usage:
#   source configs/model_1.7b_lr1e6_ifeval_1p_100k_c5.sh
#   source configs/gpus_2.sh   # or configs/gpus_8.sh
#
# 1.7B model, lr=1e-6, trained on synthetic IFEval data:
#   1 prompt, 100k constraint combinations, max 5 constraints per example.
#   Dataset: data/ifeval_1p_100k_c5.jsonl

# ============================================================================
# Model Configuration
# ============================================================================

export MODEL_NAME="Qwen/Qwen3-1.7B"

# ============================================================================
# Dataset Configuration
# ============================================================================

export TRAIN_SPLIT="train"
export TRAIN_DATASET="data/ifeval_p1_c100k_cpp5.jsonl"
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

export SEED=1
export SAVE_FREQ=-1
export CHECKPOINT_STATE_FREQ=25
export LOCAL_EVAL_EVERY=10

# ============================================================================
# Advanced Options (Optional)
# ============================================================================

export GROUND_TRUTHS_KEY="ground_truth"

# ============================================================================
# Experiment Identification
# ============================================================================

DATASET_BASENAME=$(basename "${TRAIN_DATASET}" .jsonl)
MODEL_BASENAME=$(basename "${MODEL_NAME}")
export EXP_NAME="${MODEL_BASENAME}_${DATASET_BASENAME}_${LEARNING_RATE}"
