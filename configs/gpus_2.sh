#!/bin/bash
#
# GPU config: 2 GPUs
# Usage:
#   source configs/gpus_2.sh
#
# This file defines hardware-related settings only.

# ============================================================================
# Hardware Configuration
# ============================================================================

# Number of GPUs to use
export NUM_GPUS=2

# Number of learner processes per node
export NUM_LEARNERS_PER_NODE=1

# Number of vLLM inference engines
# More engines = faster generation but more memory
export VLLM_NUM_ENGINES=1
