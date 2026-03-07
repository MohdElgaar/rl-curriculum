#!/bin/bash

# ============================================================================
# Scratch / Cache / Output Configuration
# ============================================================================

export SCRATCH_ROOT="/scratch4/workspace/mohamed_elgaar_student_uml_edu-rl-curriculum"
export HF_HOME="${SCRATCH_ROOT}/cache/huggingface"
export DATASET_LOCAL_CACHE_DIR="${SCRATCH_ROOT}/data/open-instruct"
export OUTPUT_DIR="${SCRATCH_ROOT}/outputs/${EXP_NAME}"