#!/bin/bash
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${D}/qwen_1.7b_base.sh"
export TRAIN_DATASET="allenai/RLVR-MATH"
export EXP_NAME="Qwen3-1.7B_RLVR-MATH_seed1_baseline_lr1e-6"
