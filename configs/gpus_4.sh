#!/bin/bash
#
# GPU config: 4 GPUs
# Usage:
#   source configs/gpus_4_13.sh
#
# This file defines hardware-related settings only.

# ============================================================================
# Hardware Configuration
# ============================================================================

# Number of GPUs to use
export NUM_GPUS=4

# Number of learner processes per node
export NUM_LEARNERS_PER_NODE=1

# Number of vLLM inference engines
export VLLM_NUM_ENGINES=3
