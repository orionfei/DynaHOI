#!/usr/bin/env bash
set -e

unset LD_LIBRARY_PATH

cd "$(dirname "$0")"

mkdir -p nohup_logs

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_PATH="${DATASET_PATH:-/data1/yfl_data/Dyana_data/test}"
MOTION_HINT_RATIO="${MOTION_HINT_RATIO:-0.35}"
NUM_WORKERS="${NUM_WORKERS:-2}"
TRAJS="${TRAJS:-$(seq 0 999)}"
# TRAJS="${TRAJS:-}"
ARGS=(
  --dataset-path "$DATASET_PATH"
  --motion-hint-ratio "$MOTION_HINT_RATIO"
  --num-workers "$NUM_WORKERS"
)

if [[ -n "$TRAJS" ]]; then
  # TRAJS should be a space-separated list like: "0 1 2 3"
  # shellcheck disable=SC2206
  TRAJ_LIST=($TRAJS)
  ARGS+=(--trajs "${TRAJ_LIST[@]}")
fi

nohup "$PYTHON_BIN" scripts/precompute_motion_hints.py \
  "${ARGS[@]}" \
  "$@" > nohup_logs/precompute_motion_hints_0.35.log 2>&1 < /dev/null &

# Examples:
# ./precompute_motion_hints.sh
# ./precompute_motion_hints.sh --dataset-path /data1/yfl_data/Dyana_data/test
# ./precompute_motion_hints.sh --motion-hint-ratio 0.2 --overwrite
# TRAJS="0 1 2 3" ./precompute_motion_hints.sh --overwrite
