#!/bin/bash
#
# Cache / output paths for Vast.ai instances (/workspace is persistent on the volume).

export SCRATCH_ROOT="/scratch"
export HF_HOME="${SCRATCH_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export DATASET_LOCAL_CACHE_DIR="${SCRATCH_ROOT}/data/open-instruct"
export OUTPUT_DIR="${SCRATCH_ROOT}/outputs"
export TRITON_CACHE_DIR="${SCRATCH_ROOT}/cache/triton"

# vLLM reserves KV cache from (1 - utilization) of free VRAM after the model load.
# 0.9 left too little headroom for inflight weight sync on 4×H100; 0.8 is Vast-only.
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
