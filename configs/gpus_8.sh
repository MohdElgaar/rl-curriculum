#!/bin/bash
# Number of GPUs to use
export NUM_GPUS=8

# Number of learner processes per node
export NUM_LEARNERS_PER_NODE=3

# Number of vLLM inference engines
# More engines = faster generation but more memory
export VLLM_NUM_ENGINES=5
