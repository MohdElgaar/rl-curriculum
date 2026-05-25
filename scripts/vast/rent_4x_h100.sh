#!/bin/bash
# Rent cheapest available 4x H100 (interruptible) on Vast.ai under MAX_DPH/hr and run onstart training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ONSTART="${SCRIPT_DIR}/onstart.sh"
# shellcheck source=lib/secrets.sh
source "${SCRIPT_DIR}/lib/secrets.sh"

MAX_DPH="${MAX_DPH:-7.0}"
VAST_GPU_CONFIGS="${VAST_GPU_CONFIGS:-configs/gpus_4.sh}"
VAST_DATA_CONFIG="${VAST_DATA_CONFIG:-configs/data_vastai.sh}"
VAST_MODEL_CONFIG="${VAST_MODEL_CONFIG:-configs/model_gemma_4_e2b_rs.sh}"

require_local_rent_secrets

OFFER_ID=""
BID_PRICE=""
MODE="interruptible"

while [ "${#}" -gt 0 ]; do
  case "$1" in
    --on-demand) MODE="on-demand"; shift ;;
    --offer) OFFER_ID="$2"; shift 2 ;;
    --bid) BID_PRICE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

pick_offer() {
  vastai search offers -n 'num_gpus=4 rentable=true' -i --raw 2>/dev/null | python3 -c "
import json, sys
max_dph = float('${MAX_DPH}')
offers = json.load(sys.stdin)
candidates = [
    o for o in offers
    if 'H100' in str(o.get('gpu_name', ''))
    and o.get('rentable')
    and float(o.get('min_bid') or o.get('dph_total') or 1e9) < max_dph
]
if not candidates:
    sys.exit(1)
verified = [o for o in candidates if o.get('verification') == 'verified']
if not verified:
    sys.exit(2)  # hard: do not fall back to deverified for training
best = min(verified, key=lambda o: float(o.get('min_bid') or o.get('dph_total')))
print(best['id'])
print(best.get('min_bid', best.get('dph_total')))
"
}

if [ -n "${OFFER_ID}" ] && [ "${MODE}" = "interruptible" ] && [ -z "${BID_PRICE}" ]; then
  BID_PRICE="$(vastai search offers -n "id=${OFFER_ID}" -i --raw 2>/dev/null | python3 -c "
import json, sys
offers = json.load(sys.stdin)
if not offers:
    sys.exit(1)
o = offers[0]
print(o.get('min_bid', o.get('dph_total')))
" 2>/dev/null || true)"
fi

pick_on_demand_offer() {
  vastai search offers -n 'num_gpus=4 rentable=true' -d --raw 2>/dev/null | python3 -c "
import json, sys
max_dph = float('${MAX_DPH}')
offers = json.load(sys.stdin)
candidates = [
    o for o in offers
    if 'H100' in str(o.get('gpu_name', ''))
    and o.get('rentable')
    and float(o.get('dph_total') or 1e9) < max_dph
]
if not candidates:
    sys.exit(1)
verified = [o for o in candidates if o.get('verification') == 'verified']
pool = verified if verified else candidates
best = min(pool, key=lambda o: float(o.get('dph_total')))
print(best['id'])
print(best.get('dph_total'))
print(best.get('verification', ''))
"
}

if [ -z "${OFFER_ID}" ]; then
  if [ "${MODE}" = "on-demand" ]; then
    PICK="$(pick_on_demand_offer 2>/dev/null || true)"
    if [ -n "${PICK}" ]; then
      OFFER_ID="$(echo "${PICK}" | sed -n '1p')"
      echo "Selected dph: $(echo "${PICK}" | sed -n '2p') verification: $(echo "${PICK}" | sed -n '3p')"
    fi
  elif [ "${MODE}" = "interruptible" ]; then
    PICK="$(pick_offer 2>/dev/null || true)"
    if [ -n "${PICK}" ]; then
      OFFER_ID="$(echo "${PICK}" | sed -n '1p')"
      BID_PRICE="$(echo "${PICK}" | sed -n '2p')"
    fi
  fi
  if [ -z "${OFFER_ID}" ] && [ "${MODE}" != "on-demand" ]; then
    MODE="on-demand"
    OFFER_ID="$(vastai search offers -n 'num_gpus=4 rentable=true' -d --raw 2>/dev/null | python3 -c "
import json, sys
max_dph = float('${MAX_DPH}')
offers = json.load(sys.stdin)
candidates = [
    o for o in offers
    if 'H100' in str(o.get('gpu_name', ''))
    and o.get('rentable')
    and float(o.get('dph_total') or 1e9) < max_dph
]
verified = [o for o in candidates if o.get('verification') == 'verified']
if not verified:
    sys.exit(1)
best = min(verified, key=lambda o: float(o.get('dph_total')))
print(best['id'])
")"
    BID_PRICE=""
  fi
fi

if [ -z "${OFFER_ID}" ]; then
  echo "No rentable 4x H100 offer found under \$${MAX_DPH}/hr (interruptible or on-demand)."
  exit 1
fi

IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel}"
DISK_GB="${VAST_DISK_GB:-300}"
LABEL="${VAST_LABEL:-rl-curriculum-gemma4-e2b-rs-4gpu}"

ENV_STR="$(build_vast_create_env_str \
  "REPO_BRANCH=vastai" \
  "VAST_GPU_CONFIGS=${VAST_GPU_CONFIGS}" \
  "VAST_DATA_CONFIG=${VAST_DATA_CONFIG}" \
  "VAST_MODEL_CONFIG=${VAST_MODEL_CONFIG}")"

echo "Mode: ${MODE}"
echo "Offer ID: ${OFFER_ID}"
echo "Max \$/hr: ${MAX_DPH}"
echo "Configs: ${VAST_GPU_CONFIGS} ${VAST_DATA_CONFIG} ${VAST_MODEL_CONFIG}"
echo "Image: ${IMAGE}"
echo "Disk: ${DISK_GB} GB"
echo "Onstart: ${ONSTART}"
echo "Secrets: HF_TOKEN and WANDB_API_KEY via vastai create --env (not onstart)"

CREATE_ARGS=(
  create instance "${OFFER_ID}"
  --image "${IMAGE}"
  --disk "${DISK_GB}"
  --label "${LABEL}"
  --onstart "${ONSTART}"
  --env "${ENV_STR}"
  --ssh
  --direct
)

if [ "${MODE}" = "interruptible" ] && [ -n "${BID_PRICE}" ]; then
  CREATE_ARGS+=(--bid_price "${BID_PRICE}")
fi

set -x
RESULT="$(vastai "${CREATE_ARGS[@]}")"
set +x
echo "${RESULT}"

INSTANCE_ID="$(echo "${RESULT}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('new_contract', d.get('id','')))" 2>/dev/null || true)"
if [ -n "${INSTANCE_ID}" ]; then
  echo ""
  echo "Instance ID: ${INSTANCE_ID}"
  echo "Console: https://cloud.vast.ai/instances/"
  echo "After SSH is up: tail -f /workspace/logs/training.log"
fi
