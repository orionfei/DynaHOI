#!/usr/bin/env bash
set -e

unset LD_LIBRARY_PATH

cd "$(dirname "$0")"

mkdir -p nohup_logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

setsid nohup torchrun --standalone --nproc_per_node=4 scripts/finetune_policy.py \
  --pipeline Local \
  --data-config Local \
  --base-model-path nvidia/GR00T-N1.5-3B \
  --learning-rate 1e-4 \
  --num_gpus 2 \
  --save-steps 8000 \
  --max-steps 8000 \
  --window-length 5 \
  --action-horizon 12 \
  --no-tune-diffusion-model \
  --output-dir /data1/yfl_data/DynaHOI/gr00t/checkpoints/adjacent_window/w5_h10/lr1.5 \
  "$@" > nohup_logs/adjacent_window.log 2>&1 < /dev/null &
