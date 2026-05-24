#!/bin/bash
#
# Multirun manifest: shared configs + per-run model configs.
# Used by scripts/slurm/if_rlvr_multirun.sbatch
#
# Submit:
#   sbatch scripts/slurm/if_rlvr_multirun.sbatch
#   sbatch scripts/slurm/if_rlvr_multirun.sbatch configs/multirun/rs_alpha10_n_sweep.sh

MULTIRUN_SHARED_CONFIGS=(
  "configs/gpus_4.sh"
  "configs/data_unity.sh"
)

MULTIRUN_MODEL_CONFIGS=(
  "configs/model_1.7b_lr1e6_rs_alpha10_n400.sh"
  "configs/model_1.7b_lr1e6_rs_alpha10_n600.sh"
  "configs/model_1.7b_lr1e6_rs_alpha10_n800.sh"
)
