#!/bin/bash
#
# Usage:
#   source configs/model_1.7b.sh
#   source configs/gpus_2.sh   # or configs/gpus_8.sh
#
# This file defines model- and training-related settings.
# GPU-specific settings live in configs/gpus_*.sh.

# ============================================================================
# Model Configuration
# ============================================================================

# Base model to fine-tune
export MODEL_NAME="Qwen/Qwen3-0.6B"

# ============================================================================
# Dataset Configuration
# ============================================================================

# Training dataset
# Default: allenai/IF_multi_constraints_upto5 (IF-RLVR training data)
export TRAIN_SPLIT="train"
export TRAIN_DATASET="allenai/IF_multi_constraints_upto5"
export TRAIN_DATASET_FRACTION="1.0"

# ============================================================================
# Training Hyperparameters
# ============================================================================

# Learning rate
# Typical range: 1e-7 to 1e-6 for RLVR
export LEARNING_RATE=1e-5

# KL divergence coefficient (beta)
# Higher = stay closer to reference policy
# Typical range: 0.001 to 0.1
export BETA=0.1

# Total training episodes (768000 = 1000 steps with 48×16 batch)
# Full training: 2000000
# Quick test: 10000-100000
export TOTAL_EPISODES=768000

# Number of training steps (768000 episodes = 1000 steps)
export NUM_TRAINING_STEPS=500

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
export PER_DEVICE_BATCH_SIZE=1

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
export CHECKPOINT_STATE_FREQ=25
export KEEP_LAST_N_CHECKPOINTS=-1

# Run local evaluation every N episodes
export LOCAL_EVAL_EVERY=10

# ============================================================================
# Advanced Options (Optional)
# ============================================================================

# Ground truth key in dataset
export GROUND_TRUTHS_KEY="ground_truth"

# ============================================================================
# Experiment Identification
# ============================================================================

# Extract dataset name from path and create experiment name
DATASET_BASENAME=$(basename "${TRAIN_DATASET}" .jsonl)
MODEL_BASENAME=$(basename "${MODEL_NAME}")
export EXP_NAME="${MODEL_BASENAME}_${DATASET_BASENAME}_${LEARNING_RATE}_${BETA}"  # Name for this experiment run