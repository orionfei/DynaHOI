#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

if [[ -n "${LSB_DJOB_HOSTFILE:-}" && -f "${LSB_DJOB_HOSTFILE}" ]]; then
    mapfile -t HOSTS < <(sort -u "${LSB_DJOB_HOSTFILE}")
    if (( ${#HOSTS[@]} > 1 )); then
        MASTER_ADDR="${MASTER_ADDR:-${HOSTS[0]}}"

        for NODE_RANK in "${!HOSTS[@]}"; do
            HOST="${HOSTS[$NODE_RANK]}"
            blaunch -z "${HOST}" bash -lc "
                cd ${REPO_DIR} &&
                export PYTHONPATH=${REPO_DIR}:\${PYTHONPATH:-} &&
                export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH:-} &&
                ${PYTHON_BIN} ${SCRIPT_DIR}/finetune_policy.py \
                    --nnodes ${#HOSTS[@]} \
                    --node-rank ${NODE_RANK} \
                    --master-addr ${MASTER_ADDR} \
                    --master-port ${MASTER_PORT} \
                    --num-gpus ${GPUS_PER_NODE} \
                    $*
            " &
        done

        wait
        exit 0
    fi
fi

${PYTHON_BIN} "${SCRIPT_DIR}/finetune_policy.py" --num-gpus "${GPUS_PER_NODE}" "$@"
