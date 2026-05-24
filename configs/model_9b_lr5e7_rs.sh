#!/bin/bash
#
# Qwen3.5-9B: IFEval reward shaping (no curriculum).
# Layout: 2 vLLM engines, six learner GPUs on 8×GPU node, per-device batch 2, lr 5e-7.
#
# Usage (Unity paths + 8-GPU layout come first; VLLM_NUM_ENGINES below overrides gpus_8.sh):
#   sbatch scripts/slurm/if_rlvr.sbatch configs/data_unity.sh configs/gpus_8.sh configs/model_9b_lr5e7_vllm2_bs2_rs.sh

export MODEL_NAME="Qwen/Qwen3.5-9B"

export TRAIN_SPLIT="train"
export TRAIN_DATASET="allenai/IF_multi_constraints_upto5"
export TRAIN_DATASET_FRACTION="1.0"

export LEARNING_RATE=5e-7
export BETA=0.01
export TOTAL_EPISODES=768000
export NUM_TRAINING_STEPS=1000
export TEMPERATURE=1.0
export ASYNC_STEPS=1

export VLLM_NUM_ENGINES=2
export PER_DEVICE_BATCH_SIZE=2
export NUM_UNIQUE_PROMPTS=48
export NUM_SAMPLES_PER_PROMPT=16
export NUM_MINI_BATCHES=2

export MAX_PROMPT_TOKEN_LENGTH=2048
export RESPONSE_LENGTH=2048
export PACK_LENGTH=4096

export SEED=1
export SAVE_FREQ=-1
export CHECKPOINT_STATE_FREQ=25
export KEEP_LAST_N_CHECKPOINTS=1
export LOCAL_EVAL_EVERY=10

export GROUND_TRUTHS_KEY="ground_truth"
export IFEVAL_REWARD_SHAPING=True
export IFEVAL_REWARD_SHAPING_CURRICULUM=False
export IFEVAL_COMPETENCE_C0=0.1
export IFEVAL_COMPETENCE_ALPHA=1.0
export IFEVAL_NUM_CURRICULUM_STEPS=-1

DATASET_BASENAME=$(basename "${TRAIN_DATASET}" .jsonl)
MODEL_BASENAME=$(basename "${MODEL_NAME}")
export EXP_NAME="${MODEL_BASENAME}_${DATASET_BASENAME}_rs_seed${SEED}"
