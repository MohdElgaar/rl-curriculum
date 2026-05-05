#!/bin/bash
# Vast.ai onstart: IF-RLVR smoke test (short steps, dense checkpoints, no reward shaping).
# Expects CUDA image; WANDB_* and optional HF_TOKEN injected by vastai --env.

set -euo pipefail
exec > >(tee -a /root/vast_if_rlvr_smoke.log) 2>&1

log() { echo "[vast-smoke] $(date -Is) $*"; }

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl wget ca-certificates build-essential pkg-config libssl-dev

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="/root/.local/bin:${PATH}"

log "Installing CPython 3.12 via uv (avoids Launchpad PPA flakiness on cloud hosts)"
uv python install 3.12

# Private GitHub repos (rl-curriculum submodule): authenticate HTTPS without printing the token.
if [ -n "${GH_TOKEN:-}${GITHUB_TOKEN:-}" ]; then
  _gh_t="${GH_TOKEN:-${GITHUB_TOKEN}}"
  git config --global url."https://x-access-token:${_gh_t}@github.com/".insteadOf "https://github.com/"
fi

WORKDIR="${WORKDIR:-/workspace}"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

RL_REPO_URL="${RL_CURRICULUM_REPO_URL:-https://github.com/MohdElgaar/rl-curriculum.git}"
# Token-based HTTPS is handled by git insteadOf above when GH_TOKEN is set.
RL_GIT_REF="${RL_CURRICULUM_GIT_REF:-cursor/if-rlvr-sbatch-gpus-6-default-13c0}"

if [ ! -d rl-curriculum/.git ]; then
  log "Cloning ${RL_REPO_URL}@${RL_GIT_REF}"
  git clone --depth 1 --branch "${RL_GIT_REF}" "${RL_REPO_URL}" rl-curriculum
fi
cd rl-curriculum

log "Initializing submodule open-instruct"
if ! git submodule update --init --depth 1 open-instruct 2>/dev/null; then
  log "Submodule shallow init failed; using fresh shallow clone of mohdelgaar branch"
  rm -rf open-instruct
  OI_URL="${OPEN_INSTRUCT_REPO_URL:-https://github.com/MohdElgaar/open-instruct.git}"
  git clone --depth 1 --branch mohdelgaar "${OI_URL}" open-instruct
fi

log "uv sync (--frozen, fallback --prerelease=allow)"
export UV_PYTHON=3.12
if ! uv sync --frozen; then
  log "Frozen lock install failed; resolving with prerelease allowance (matches local dev when lock drifts)"
  uv sync --prerelease=allow
fi

# `uv run` normally re-validates/syncs deps; workspace git sources (transformers/main, OLMo-core)
# can re-fetch and stall cloud smoke right as training starts—skip sync once the venv is installed.
export UV_NO_SYNC=1

# --- Training env (9B, 8 GPUs: 6 learners + 2 vLLM engines, lr 5e-7, no shaping).
# PER_DEVICE_BATCH_SIZE=1 matches SLURM if_rlvr defaults & autotune batch ladder start; safe on A100 40GB
# for 9B (autotune on Unity uses 6-GPU jobs → 4 learner GPUs after a 2-engine reserve; layout differs from this 8-GPU smoke).
export MODEL_NAME="Qwen/Qwen3.5-9B"
export TRAIN_SPLIT="train"
export TRAIN_DATASET="allenai/IF_multi_constraints_upto5"
export TRAIN_DATASET_FRACTION="1.0"
export LEARNING_RATE="5e-7"
export BETA="0.01"
export TEMPERATURE="1.0"
export ASYNC_STEPS="1"
export PER_DEVICE_BATCH_SIZE="1"
export NUM_UNIQUE_PROMPTS="48"
export NUM_SAMPLES_PER_PROMPT="16"
export NUM_MINI_BATCHES="2"
export MAX_PROMPT_TOKEN_LENGTH="2048"
export RESPONSE_LENGTH="2048"
export PACK_LENGTH="4096"
export SEED="1"
export NUM_GPUS="8"
export NUM_LEARNERS_PER_NODE="6"
export VLLM_NUM_ENGINES="2"
export IFEVAL_REWARD_SHAPING="False"
export IFEVAL_REWARD_SHAPING_CURRICULUM="False"
export MATH_REWARD_SHAPING="False"
export MATH_REWARD_SHAPING_CURRICULUM="False"
export GSM_REWARD_SHAPING="False"
export GSM_REWARD_SHAPING_CURRICULUM="False"
export TRUST_REMOTE_CODE="True"

export NUM_TRAINING_STEPS="${NUM_TRAINING_STEPS:-2}"
export SAVE_FREQ="${SAVE_FREQ:-1}"
export CHECKPOINT_STATE_FREQ="${CHECKPOINT_STATE_FREQ:-1}"
export LOCAL_EVAL_EVERY="${LOCAL_EVAL_EVERY:-1000}"
export TOTAL_EPISODES="${TOTAL_EPISODES:-768000}"

DATASET_BASENAME="$(basename "${TRAIN_DATASET}" .jsonl)"
MODEL_BASENAME="$(basename "${MODEL_NAME}")"
export EXP_NAME="${EXP_NAME:-vast_smoke_${MODEL_BASENAME}_${DATASET_BASENAME}_${LEARNING_RATE}}"

export OUTPUT_DIR="${OUTPUT_DIR:-${WORKDIR}/rl_outputs}"
export DATASET_LOCAL_CACHE_DIR="${DATASET_LOCAL_CACHE_DIR:-${WORKDIR}/dataset_cache}"
# checkpoint_state_dir defaults inside scripts/train_if_rlvr.sh (under output_dir/${EXP_NAME})

# Faster smoke: skip step-0 eval flood; disable hub push unless overridden
export EVAL_ON_STEP_0="${EVAL_ON_STEP_0:-False}"
export EXTRA_GRPO_ARGS="${EXTRA_GRPO_ARGS:---push_to_hub False}"

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

log "Starting training (timeout ${SMOKE_TIMEOUT:-90m})"
log "WANDB_PROJECT=${WANDB_PROJECT:-<unset>} entity=${WANDB_ENTITY:-<unset>}"

set +e
timeout "${SMOKE_TIMEOUT:-90m}" bash scripts/train_if_rlvr.sh
rc=$?
set -e

log "Train finished rc=${rc}; listing output tree"
find "${OUTPUT_DIR}" -maxdepth 4 -type f 2>/dev/null | head -200 || true

if command -v wandb >/dev/null 2>&1; then
  log "wandb CLI available"
else
  log "(uv run wandb disabled in PATH)"
fi

exit "${rc}"
