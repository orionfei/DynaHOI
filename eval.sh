#!/usr/bin/env bash
set -e

unset LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" scripts/eval_policy.py \
  --pipeline LoGo \
  --data-config LoGo \
  --action-horizon 10 \
  --trajs $(seq 0 199) \
  --window_length 5 \
  --observe_frame_offsets 5 4 3 2 1 \
  --motion_hint_ratio 0.3 \
  --model-path /data1/yfl_data/DynaHOI/gr00t/checkpoints/LoGo/0.3/v1/checkpoint-8000 \
  "$@"

# Example explicit offsets:
# ./eval.sh --observe-frame-offsets 10 5 3 2 1
