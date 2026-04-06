#!/usr/bin/env bash
set -e

unset LD_LIBRARY_PATH

cd "$(dirname "$0")"

mkdir -p nohup_logs

setsid nohup torchrun --standalone --nproc_per_node=4 scripts/finetune_policy.py \
  --pipeline baseline_adjacent_window \
  --data-config mano_18dim_baseline \
  --base-model-path nvidia/GR00T-N1.5-3B \
  --learning_rate 1e-4 \
  --window-length 5 \
  --action_horizon 10 \
  --output-dir /data1/yfl_data/DynaHOI/gr00t/checkpoints/adjacent_window/new_w5_h10 \
  "$@" > nohup_logs/adjacent_window.log 2>&1 < /dev/null &
