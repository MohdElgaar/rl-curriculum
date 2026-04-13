#!/bin/bash
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${D}/qwen_0.6b_base.sh"
export TRAIN_DATASET="allenai/RLVR-GSM"
export GSM_REWARD_SHAPING=True
export GSM_REWARD_SHAPING_CURRICULUM=True
export GSM_COMPETENCE_ALPHA=10.0
export GSM_NUM_CURRICULUM_STEPS=200
export EXP_NAME="Qwen3-0.6B_RLVR-GSM_seed1_rs_curr_a10_lr1e-6"
