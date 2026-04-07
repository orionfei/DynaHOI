#!/usr/bin/env bash
set -e

unset LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" scripts/eval_policy.py \
  --pipeline Local \
  --data-config Local \
  --action-horizon 10 \
  --window-length 5 \
  --trajs $(seq 0 199) \
  --model-path /data1/yfl_data/DynaHOI/gr00t/checkpoints/adjacent_window/w5_h10/checkpoint-8000 \
  "$@"
