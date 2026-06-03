#!/usr/bin/env bash
# Upload swing_fold_towel part1-6 LeRobot datasets to Hugging Face.
# Requires: proxy on 127.0.0.1:7897, hf auth login

set -euo pipefail

unset HF_ENDPOINT
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export all_proxy=http://127.0.0.1:7897
export HF_HUB_ENABLE_HF_TRANSFER=0

LOG="/home/ruihao/PiperX-ToolKit/datasets/upload_swing_fold_towel_hf.log"
SCRIPT="/home/ruihao/PiperX-ToolKit/scripts/upload_swing_fold_towel_to_hf.py"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }

if ! curl -x http://127.0.0.1:7897 -sS -o /dev/null --connect-timeout 10 https://huggingface.co; then
  log "ERROR: proxy 7897 cannot reach huggingface.co"
  exit 1
fi

exec python3 "${SCRIPT}" 2>&1 | tee -a "${LOG}"
