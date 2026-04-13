#!/bin/bash
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${D}/qwen_0.6b_base.sh"
export TRAIN_DATASET="allenai/RLVR-GSM"
export GSM_REWARD_SHAPING=True
export GSM_REWARD_SHAPING_CURRICULUM=False
export EXP_NAME="Qwen3-0.6B_RLVR-GSM_seed1_rs_lr1e-6"
