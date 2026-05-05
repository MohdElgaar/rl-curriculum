#!/bin/bash
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${D}/qwen_0.6b_base.sh"
export TRAIN_DATASET="allenai/RLVR-MATH"
export EXP_NAME="Qwen3-0.6B_RLVR-MATH_seed1_baseline_lr1e-6"
