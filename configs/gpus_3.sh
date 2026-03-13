#!/bin/bash

# Number of GPUs to use
export NUM_GPUS=3

# Number of learner processes per node
export NUM_LEARNERS_PER_NODE=2

# Number of vLLM inference engines
export VLLM_NUM_ENGINES=1
