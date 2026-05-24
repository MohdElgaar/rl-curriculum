#!/bin/bash
# Rent cheapest available 8x H100 on Vast.ai and run onstart training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ONSTART="${SCRIPT_DIR}/onstart.sh"
# shellcheck source=lib/secrets.sh
source "${SCRIPT_DIR}/lib/secrets.sh"

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

if [ -z "${OFFER_ID}" ]; then
  if [ "${MODE}" = "interruptible" ]; then
    OFFER_ID="$(vastai search offers -i 'num_gpus=8 gpu_name=H100_SXM rentable=true' -o 'min_bid-' --raw 2>/dev/null | python3 -c "
import json, sys
offers = json.load(sys.stdin)
if not offers:
    sys.exit(1)
print(offers[0]['id'])
")" || true
    BID_PRICE="$(vastai search offers -i "id=${OFFER_ID}" --raw 2>/dev/null | python3 -c "
import json, sys
o = json.load(sys.stdin)[0]
print(o.get('min_bid', o.get('dph_total')))
" 2>/dev/null || true)"
  fi
  if [ -z "${OFFER_ID}" ]; then
    MODE="on-demand"
    OFFER_ID="$(vastai search offers -d 'num_gpus=8 gpu_name=H100_SXM rentable=true' -o 'dph_total-' --raw 2>/dev/null | python3 -c "
import json, sys
offers = json.load(sys.stdin)
if not offers:
    sys.exit(1)
print(offers[0]['id'])
")"
    BID_PRICE=""
  fi
fi

IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel}"
DISK_GB="${VAST_DISK_GB:-300}"
LABEL="${VAST_LABEL:-rl-curriculum-gemma4-e2b-rs}"

ENV_STR="$(build_vast_create_env_str "REPO_BRANCH=vastai")"

echo "Mode: ${MODE}"
echo "Offer ID: ${OFFER_ID}"
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
