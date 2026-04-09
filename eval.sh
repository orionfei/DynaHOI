#!/usr/bin/env bash
set -e

unset LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" scripts/eval_policy.py \
  --pipeline Local \
  --data-config Local \
  --action-horizon 8 \
  --window-length 4 \
  --trajs $(seq 0 199) \
  --model-path /data1/yfl_data/DynaHOI/gr00t/checkpoints/Local/w4_h8/checkpoint-8000 \
  "$@"
