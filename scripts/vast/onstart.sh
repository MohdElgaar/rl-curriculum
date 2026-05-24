#!/bin/bash
# Vast.ai onstart: clone rl-curriculum (vastai branch), install deps, launch IF-RLVR training.
set -euo pipefail

LOG_DIR="/workspace/logs"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/onstart.log") 2>&1

log() { echo "[onstart] $(date -Is) $*"; }

REPO_URL="${REPO_URL:-https://github.com/MohdElgaar/rl-curriculum.git}"
REPO_BRANCH="${REPO_BRANCH:-vastai}"
WORKDIR="${WORKDIR:-/workspace/rl-curriculum}"
VAST_GPU_CONFIGS="${VAST_GPU_CONFIGS:-configs/gpus_8.sh}"
VAST_DATA_CONFIG="${VAST_DATA_CONFIG:-configs/data_vastai.sh}"
VAST_MODEL_CONFIG="${VAST_MODEL_CONFIG:-configs/model_gemma_4_e2b_rs.sh}"
# Set VAST_ONSTART_TEST_ONLY=1 to validate bootstrap (clone + uv sync) without launching training.
VAST_ONSTART_TEST_ONLY="${VAST_ONSTART_TEST_ONLY:-0}"

ensure_cmd() {
  command -v "$1" >/dev/null 2>&1
}

download_file() {
  local url="$1" dest="$2"
  if ensure_cmd curl; then
    curl -fsSL "${url}" -o "${dest}"
  elif ensure_cmd wget; then
    wget -q -O "${dest}" "${url}"
  elif ensure_cmd python3; then
    python3 - "${url}" "${dest}" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
  else
    log "ERROR: no curl, wget, or python3 available to download ${url}"
    return 1
  fi
}

fix_apt_sources() {
  if [ ! -f /etc/apt/sources.list ] && [ ! -d /etc/apt/sources.list.d ]; then
    return 0
  fi
  # PyTorch images sometimes ship with minimal/broken apt config on Vast hosts.
  if ensure_cmd lsb_release; then
    local codename
    codename="$(lsb_release -cs 2>/dev/null || true)"
    if [ -n "${codename}" ] && ! grep -rq "${codename}" /etc/apt/ 2>/dev/null; then
      log "Repairing apt sources for ${codename}"
      printf '%s\n' \
        "deb http://archive.ubuntu.com/ubuntu ${codename} main restricted universe multiverse" \
        "deb http://archive.ubuntu.com/ubuntu ${codename}-updates main restricted universe multiverse" \
        "deb http://archive.ubuntu.com/ubuntu ${codename}-security main restricted universe multiverse" \
        > /etc/apt/sources.list
    fi
  fi
}

install_system_packages() {
  local missing=()
  for cmd in git curl ca-certificates; do
    ensure_cmd "${cmd}" || missing+=("${cmd}")
  done
  if [ "${#missing[@]}" -eq 0 ]; then
    log "git, curl, and ca-certificates already available"
    return 0
  fi
  if ! ensure_cmd apt-get; then
    log "apt-get not found; missing: ${missing[*]}"
    return 0
  fi
  log "Installing system packages (missing: ${missing[*]})"
  export DEBIAN_FRONTEND=noninteractive
  fix_apt_sources
  apt-get update -qq || apt-get update || true
  apt-get install -y -qq git curl ca-certificates build-essential \
    || apt-get install -y git curl ca-certificates build-essential \
    || log "WARNING: apt install failed; will use tarball/python fallbacks"
}

install_uv() {
  if ensure_cmd uv; then
    log "uv already installed: $(command -v uv)"
    return 0
  fi
  export PATH="${HOME}/.local/bin:${PATH}"
  if ensure_cmd uv; then
    return 0
  fi
  log "Installing uv"
  if ensure_cmd curl; then
    curl -fsSL https://astral.sh/uv/install.sh | sh
  elif ensure_cmd wget; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  elif ensure_cmd python3; then
    python3 -m pip install --user uv
  else
    log "ERROR: cannot install uv (no curl/wget/python3 pip)"
    return 1
  fi
  export PATH="${HOME}/.local/bin:${PATH}"
  ensure_cmd uv
}

fetch_github_tarball() {
  local repo="$1" branch="$2" dest="$3"
  local tmp="/tmp/${repo//\//-}-${branch}.tar.gz"
  local parent
  parent="$(dirname "${dest}")"
  mkdir -p "${parent}"
  log "Fetching tarball ${repo}@${branch} -> ${dest}"
  download_file "https://github.com/${repo}/archive/refs/heads/${branch}.tar.gz" "${tmp}"
  tar -xzf "${tmp}" -C "${parent}"
  local extracted="${parent}/$(basename "${repo}")-${branch}"
  rm -rf "${dest}"
  mv "${extracted}" "${dest}"
  rm -f "${tmp}"
}

clone_or_update_repo() {
  if ensure_cmd git; then
    if [ -d "${WORKDIR}/.git" ]; then
      log "Updating existing repo at ${WORKDIR}"
      cd "${WORKDIR}"
      git fetch origin "${REPO_BRANCH}"
      git checkout "${REPO_BRANCH}"
      git pull --ff-only origin "${REPO_BRANCH}"
    else
      log "Cloning ${REPO_URL} (branch ${REPO_BRANCH})"
      git clone --branch "${REPO_BRANCH}" --recurse-submodules --depth 1 "${REPO_URL}" "${WORKDIR}"
      cd "${WORKDIR}"
    fi
    log "Syncing submodules"
    git submodule update --init --recursive
    return 0
  fi

  log "git unavailable; using GitHub tarballs"
  fetch_github_tarball "MohdElgaar/rl-curriculum" "${REPO_BRANCH}" "${WORKDIR}"
  cd "${WORKDIR}"
  fetch_github_tarball "MohdElgaar/open-instruct" "mohdelgaar" "${WORKDIR}/open-instruct"
  fetch_github_tarball "allenai/IFBench" "main" "${WORKDIR}/IFBench"
  fetch_github_tarball "MohdElgaar/lm-evaluation-harness" "main" "${WORKDIR}/lm-evaluation-harness"
}

run_onstart_self_test() {
  log "=== onstart self-test ==="
  log "git: $(command -v git 2>/dev/null || echo MISSING)"
  log "curl: $(command -v curl 2>/dev/null || echo MISSING)"
  log "uv: $(command -v uv 2>/dev/null || echo MISSING)"
  log "python3: $(command -v python3 2>/dev/null || echo MISSING)"
  if ensure_cmd nvidia-smi; then
    nvidia-smi -L || true
  else
    log "WARNING: nvidia-smi not found"
  fi
  if [ -f "${WORKDIR}/pyproject.toml" ]; then
    log "repo checkout OK: ${WORKDIR}/pyproject.toml exists"
  else
    log "ERROR: repo checkout missing pyproject.toml"
    return 1
  fi
  log "=== onstart self-test passed ==="
}

install_system_packages
install_uv
export PATH="${HOME}/.local/bin:${PATH}"

clone_or_update_repo

log "Installing Python dependencies (uv sync --frozen)"
if ! uv sync --frozen; then
  log "ERROR: uv sync --frozen failed"
  exit 1
fi
log "uv sync --frozen completed"

# uv.lock targets cu130 wheels; many Vast H100 hosts ship CUDA 12.6 drivers (e.g. 535.x).
realign_cuda_for_host_driver() {
  log "Checking torch CUDA against host driver"
  if ! uv run python -c "import torch; assert torch.cuda.is_available(), 'no cuda'" 2>/dev/null; then
    log "torch.cuda unavailable after uv sync; reinstalling cu126 torch stack"
    uv pip install --reinstall \
      "torch==2.6.0" "torchvision" "torchaudio" \
      --index-url https://download.pytorch.org/whl/cu126
    if ! uv run python -c "import torch; assert torch.cuda.is_available(), 'no cuda after cu126'" 2>/dev/null; then
      log "ERROR: torch still cannot use CUDA after cu126 realign"
      uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>/dev/null || true
      return 1
    fi
    log "cu126 torch realign OK: $(uv run python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
  else
    log "torch.cuda.is_available()=True"
  fi
}
realign_cuda_for_host_driver

install_nltk_data() {
  export NLTK_DATA="${NLTK_DATA:-/scratch/nltk_data}"
  mkdir -p "${NLTK_DATA}"
  log "Prefetching NLTK data into ${NLTK_DATA}"
  if ! uv run python -c "from nltk.data import find; find('tokenizers/punkt_tab/english/')" 2>/dev/null; then
    uv run python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt_tab', 'punkt', 'stopwords')]"
  fi
  uv run python -c "from nltk.data import find; find('tokenizers/punkt_tab/english/'); print('nltk punkt_tab OK')"
}
install_nltk_data
export NLTK_DATA="${NLTK_DATA:-/scratch/nltk_data}"

chmod +x scripts/vast/run_if_rlvr.sh scripts/vast/restart_training.sh

# HF_TOKEN / WANDB_API_KEY come from `vastai create instance --env` (see scripts/vast/rent_*.sh).
run_onstart_self_test

if [ "${VAST_ONSTART_TEST_ONLY}" = "1" ]; then
  log "VAST_ONSTART_TEST_ONLY=1; skipping training launch"
  exit 0
fi

TRAIN_LOG="${LOG_DIR}/training.log"
log "Launching training -> ${TRAIN_LOG}"
log "Configs: ${VAST_GPU_CONFIGS} ${VAST_DATA_CONFIG} ${VAST_MODEL_CONFIG}"

nohup env NLTK_DATA="${NLTK_DATA}" bash "${WORKDIR}/scripts/vast/run_if_rlvr.sh" \
  "${VAST_GPU_CONFIGS}" \
  "${VAST_DATA_CONFIG}" \
  "${VAST_MODEL_CONFIG}" \
  >> "${TRAIN_LOG}" 2>&1 &
echo $! > "${LOG_DIR}/training.pid"
log "Training PID=$(cat "${LOG_DIR}/training.pid")"
