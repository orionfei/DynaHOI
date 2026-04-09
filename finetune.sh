#!/usr/bin/env bash
set -e

unset LD_LIBRARY_PATH

cd "$(dirname "$0")"

mkdir -p nohup_logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"

setsid nohup torchrun --standalone --nproc_per_node=2 scripts/finetune_policy.py \
  --pipeline Local \
  --data-config Local \
  --base-model-path nvidia/GR00T-N1.5-3B \
  --learning-rate 1e-4 \
  --num_gpus 2 \
  --save-steps 8000 \
  --max-steps 8000 \
  --window-length 4 \
  --action-horizon 8 \
  --no-tune-diffusion-model \
  --output-dir /data1/yfl_data/DynaHOI/gr00t/checkpoints/Local/w4_h8 \
  "$@" > nohup_logs/Local_w4_h8.log 2>&1 < /dev/null &
