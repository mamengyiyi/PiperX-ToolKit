#!/usr/bin/env bash
# Align merged_v21 norm_stats path for each checkpoint step (see deploy runbook §4).
set -euo pipefail

RUN_DIR="${1:-/home/ruihao/PiperX-ToolKit/train_checkpoint/fold_towel_parts123_horizon100_8gpu_bs64_150k_001}"

for step_dir in "$RUN_DIR"/*/; do
  step="$(basename "$step_dir")"
  [[ "$step" =~ ^[0-9]+$ ]] || continue
  src="${step_dir}/assets/ruio248/fold_towel_20260527_parts123_v21_rebuilt"
  dst="${step_dir}/assets/ruio248/fold_towel_20260527_merged_v21"
  if [[ ! -f "${src}/norm_stats.json" ]]; then
    echo "skip ${step}: missing ${src}/norm_stats.json"
    continue
  fi
  ln -sfn "$src" "$dst"
  echo "ok ${step}: merged_v21 -> parts123_v21_rebuilt"
done
