#!/bin/bash
#BSUB -n 4
#BSUB -R "select[ngpus>0] rusage[ngpus_shared=4] span[ptile=2]"
#BSUB -q gpuq
#BSUB -J gr00t_motion_hint_finetune_nf6
#BSUB -o %J.out
#BSUB -e %J.err

set -euo pipefail

REPO_DIR="/gpfsdata/home/liuyifei/DynaHOI"
SCRIPT_DIR="${REPO_DIR}/scripts"

cd "${REPO_DIR}"

module load gcc/12.1
conda activate gr00t

export PYTHON_BIN="$(command -v python)"
export GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
export EXPECTED_NNODES="${EXPECTED_NNODES:-2}"
export MASTER_PORT="${MASTER_PORT:-29500}"

TRAIN_ARGS=(
  --pipeline motion_hint_farneback
  --data-config mano_18dim_motion_hint
  --dataset-path /data1/yfl_data/Dyana_data/train
  --output-dir /data1/yfl_data/DynaHOI/gr00t/checkpoints/motion_hint_farneback/
  --num-gpus "${GPUS_PER_NODE}"
  --batch-size 8
  --max-steps 4000
  --save-steps 4000
  --action-horizon 10
  --action-dim 18
  --motion-hint-ratio 0.25
  --motion-hint-num-frames 6
  --learning-rate 1.5e-4
  --warmup-ratio 0.05
  --dataloader-num-workers 4
  --report-to wandb
)

bash "${SCRIPT_DIR}/gr00t_finetune.sh" "${TRAIN_ARGS[@]}"
