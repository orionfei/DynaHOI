#!/usr/bin/env bash
set -e

unset LD_LIBRARY_PATH

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" scripts/eval_policy.py \
  --pipeline baseline_adjacent_window \
  --data-config mano_18dim_baseline \
  --action-horizon 10 \
  --window-length 5 \
  --trajs $(seq 0 199) \
  --model-path /data1/yfl_data/DynaHOI/gr00t/checkpoints/adjacent_window/w5_h10/checkpoint-8000 \
  "$@"
