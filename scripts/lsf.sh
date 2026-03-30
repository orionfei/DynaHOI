#!/bin/bash
#BSUB -n 4
#BSUB -R "select[ngpus>0] rusage[ngpus_shared=4] span[ptile=2]"
#BSUB -q gpuq
#BSUB -J gr00t_motion_hint_finetune_nf6
#BSUB -o %J.out
#BSUB -e %J.err

set -euo pipefail
set -x

REPO_DIR="/gpfsdata/home/liuyifei/DynaHOI"
SCRIPT_DIR="${REPO_DIR}/scripts"

cd "${REPO_DIR}"

module load gcc/12.1

if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found" >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate gr00t

export PYTHON_BIN="$(command -v python)"
export GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export WANDB_PROJECT="${WANDB_PROJECT:-GR00T-N1.5-unity}"

pwd
hostname
which python
python -V
which torchrun
echo "LSB_DJOB_HOSTFILE=${LSB_DJOB_HOSTFILE:-}"
if [[ -n "${LSB_DJOB_HOSTFILE:-}" && -f "${LSB_DJOB_HOSTFILE}" ]]; then
    cat "${LSB_DJOB_HOSTFILE}"
fi
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "MASTER_PORT=${MASTER_PORT}"

TRAIN_ARGS=(
  --pipeline motion_hint_farneback
  --data-config mano_18dim_motion_hint
  --dataset-path /gpfsdata/home/liuyifei/Dyana_data/train
  --output-dir /gpfsdata/home/liuyifei/DynaHOI/gr00t/checkpoints/motion_hint_farneback
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
