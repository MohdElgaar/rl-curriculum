#!/bin/bash
#
# GPU config: 8 GPUs
# Usage:
#   source configs/gpus_8.sh
#
# This file defines hardware-related settings only.

# ============================================================================
# Hardware Configuration
# ============================================================================

# Number of GPUs to use
export NUM_GPUS=6

# Number of learner processes per node
export NUM_LEARNERS_PER_NODE=2

# Number of vLLM inference engines
# More engines = faster generation but more memory
export VLLM_NUM_ENGINES=4
export VLLM_TP=1
