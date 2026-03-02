#!/bin/bash

# Number of GPUs to use
export NUM_GPUS=4

# Number of learner processes per node
export NUM_LEARNERS_PER_NODE=3

# Number of vLLM inference engines
export VLLM_NUM_ENGINES=1
