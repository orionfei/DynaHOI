#!/usr/bin/env bash
set -e

unset LD_LIBRARY_PATH

cd "$(dirname "$0")"

mkdir -p nohup_logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

setsid nohup torchrun --standalone --nproc_per_node=2 scripts/finetune_policy.py \
  --pipeline LoGo \
  --data-config LoGo \
  --base-model-path nvidia/GR00T-N1.5-3B \
  --learning-rate 1e-4 \
  --num_gpus 2 \
  --save-steps 8000 \
  --max-steps 8000 \
  --window_length 5 \
  --observe_frame_offsets 5 4 3 2 1 \
  --action-horizon 10 \
  --motion_hint_ratio 0.35 \
  --no-tune-diffusion-model \
  --output-dir /data1/yfl_data/DynaHOI/gr00t/checkpoints/LoGo/0.35 \
  "$@" > nohup_logs/LoGo_0.35.log 2>&1 < /dev/null &

# Example explicit offsets:
# ./finetune.sh --observe-frame-offsets 10 5 3 2 1
