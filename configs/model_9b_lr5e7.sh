#!/bin/bash
#
# Usage (Unity SLURM 6×GPU if_rlvr.sbatch reserves 2 GPUs for vLLM → four learner GPUs; see sbatch NUM_LEARNERS_LIST):
#   source configs/data_unity.sh   # optional
#   source configs/model_9b_lr5e7.sh
#
# Eight-GPU layouts (e.g. Vast): set NUM_GPUS=8, NUM_LEARNERS_PER_NODE=6, VLLM_NUM_ENGINES=2 in env or smoke onstart.
#
# PER_DEVICE_BATCH_SIZE=1 matches scripts/slurm/if_rlvr.sbatch default and IF-RLVR autotune’s batch sweep floor
# (there is no committed autotune summary in-repo); raise only after a measured OOM/throughput pass.
#
# Qwen3.5-9B IF-RLVR defaults (aligned with model_8b_lr5e7.sh).

# ============================================================================
# Model Configuration
# ============================================================================

export MODEL_NAME="Qwen/Qwen3.5-9B"

# ============================================================================
# Dataset Configuration
# ============================================================================

export TRAIN_SPLIT="train"
export TRAIN_DATASET="allenai/IF_multi_constraints_upto5"
export TRAIN_DATASET_FRACTION="1.0"

# ============================================================================
# Training Hyperparameters
# ============================================================================

export LEARNING_RATE=5e-7

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
export KEEP_LAST_N_CHECKPOINTS=1

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
