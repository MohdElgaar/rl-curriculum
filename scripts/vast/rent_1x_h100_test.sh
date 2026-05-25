#!/bin/bash
# Rent cheapest 1x H100 on Vast.ai to test onstart/bootstrap (no training by default).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONSTART="${SCRIPT_DIR}/onstart.sh"
# shellcheck source=lib/secrets.sh
source "${SCRIPT_DIR}/lib/secrets.sh"

MAX_DPH="${MAX_DPH:-3.0}"
DISK_GB="${VAST_DISK_GB:-80}"
LABEL="${VAST_LABEL:-rl-curriculum-onstart-test}"
IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel}"
# 1 = clone + uv sync + self-test only; 0 = also launch training
VAST_ONSTART_TEST_ONLY="${VAST_ONSTART_TEST_ONLY:-1}"

OFFER_ID=""
BID_PRICE=""
MODE="interruptible"

while [ "${#}" -gt 0 ]; do
  case "$1" in
    --on-demand) MODE="on-demand"; shift ;;
    --offer) OFFER_ID="$2"; shift 2 ;;
    --bid) BID_PRICE="$2"; shift 2 ;;
    --train) VAST_ONSTART_TEST_ONLY=0; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

pick_offer() {
  local search_type="$1"
  vastai search offers -n 'num_gpus=1' "${search_type}" --raw 2>/dev/null | python3 -c "
import json, sys
max_dph = float('${MAX_DPH}')
offers = json.load(sys.stdin)
candidates = [
    o for o in offers
    if 'H100' in str(o.get('gpu_name', ''))
    and o.get('rentable')
]
price_key = 'min_bid' if '${search_type}' == '-i' else 'dph_total'
candidates = [o for o in candidates if float(o.get(price_key) or o.get('dph_total') or 1e9) <= max_dph]
if not candidates:
    sys.exit(1)
# Prefer verified hosts (deverified 1x H100 often fails CDI GPU injection on Vast).
verified = [o for o in candidates if o.get('verification') == 'verified']
pool = verified if verified else candidates
best = min(pool, key=lambda o: float(o.get(price_key) or o.get('dph_total')))
print(best['id'])
print(best.get('min_bid', best.get('dph_total')))
print(best.get('gpu_name', ''))
print(best.get('verification', ''))
"
}

if [ -z "${OFFER_ID}" ]; then
  if [ "${MODE}" = "interruptible" ]; then
    PICK="$(pick_offer -i 2>/dev/null || true)"
    if [ -n "${PICK}" ]; then
      OFFER_ID="$(echo "${PICK}" | sed -n '1p')"
      BID_PRICE="$(echo "${PICK}" | sed -n '2p')"
    fi
  fi
  if [ -z "${OFFER_ID}" ]; then
    MODE="on-demand"
    PICK="$(pick_offer -d 2>/dev/null || true)"
    if [ -n "${PICK}" ]; then
      OFFER_ID="$(echo "${PICK}" | sed -n '1p')"
      BID_PRICE=""
    fi
  fi
fi

if [ -z "${OFFER_ID}" ]; then
  echo "No rentable 1x H100 offer found under \$${MAX_DPH}/hr."
  exit 1
fi

if [ "${VAST_ONSTART_TEST_ONLY}" = "0" ]; then
  require_local_rent_secrets
else
  load_local_rent_secrets
fi

ENV_STR="$(build_vast_create_env_str \
  "REPO_BRANCH=vastai" \
  "VAST_ONSTART_TEST_ONLY=${VAST_ONSTART_TEST_ONLY}")"

echo "Mode: ${MODE}"
echo "Offer ID: ${OFFER_ID}"
echo "Max \$/hr: ${MAX_DPH}"
echo "Test only: ${VAST_ONSTART_TEST_ONLY}"
echo "Image: ${IMAGE}"
echo "Disk: ${DISK_GB} GB"
echo "Onstart: ${ONSTART}"

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

RESULT="$(vastai "${CREATE_ARGS[@]}")"
echo "${RESULT}"

INSTANCE_ID="$(echo "${RESULT}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('new_contract', d.get('id','')))" 2>/dev/null || true)"
if [ -n "${INSTANCE_ID}" ]; then
  echo ""
  echo "Instance ID: ${INSTANCE_ID}"
  echo "Monitor: vastai logs ${INSTANCE_ID} --tail 100 --filter onstart"
  echo "Console: https://cloud.vast.ai/instances/${INSTANCE_ID}"
fi
