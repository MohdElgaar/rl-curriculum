#!/bin/bash
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${D}/model_0.6b_lr1e6.sh"

export IFEVAL_REWARD_SHAPING=False
export IFEVAL_REWARD_SHAPING_CURRICULUM=False
export IFEVAL_RANDOM_ZERO_REWARD=True

export EXP_NAME="${MODEL_BASENAME}_${DATASET_BASENAME}_${LEARNING_RATE}_random_zero"
