#!/bin/bash
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${D}/qwen_0.6b_base.sh"
export TRAIN_DATASET="allenai/RLVR-MATH"
export MATH_REWARD_SHAPING=True
export MATH_REWARD_SHAPING_CURRICULUM=True
export MATH_COMPETENCE_ALPHA=10.0
export MATH_NUM_CURRICULUM_STEPS=200
export EXP_NAME="Qwen3-0.6B_RLVR-MATH_seed1_rs_curr_a10_lr1e-6"
