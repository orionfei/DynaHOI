#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

pwd
hostname
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "MASTER_PORT=${MASTER_PORT}"
which "${PYTHON_BIN}" || true
"${PYTHON_BIN}" -V || true
which blaunch || true

if [[ -n "${LSB_DJOB_HOSTFILE:-}" && -f "${LSB_DJOB_HOSTFILE}" ]]; then
    mapfile -t HOSTS < <(sort -u "${LSB_DJOB_HOSTFILE}")
    echo "Resolved hosts: ${HOSTS[*]}"

    if (( ${#HOSTS[@]} > 1 )); then
        MASTER_ADDR="${HOSTS[0]}"
        CURRENT_HOST="$(hostname)"
        LOCAL_NODE_RANK=""
        echo "Multi-node launch: nnodes=${#HOSTS[@]}, master_addr=${MASTER_ADDR}"

        USER_ARGS="$(printf ' %q' "$@")"
        USER_ARGS="${USER_ARGS:1}"

        for NODE_RANK in "${!HOSTS[@]}"; do
            HOST="${HOSTS[$NODE_RANK]}"
            if [[ "${HOST}" == "${CURRENT_HOST}" ]]; then
                LOCAL_NODE_RANK="${NODE_RANK}"
                continue
            fi

            echo "Launching remote host=${HOST} node_rank=${NODE_RANK}"
            blaunch -z "${HOST}" bash -lc "
                set -euo pipefail
                set -x
                cd ${REPO_DIR}
                export PYTHONPATH=${REPO_DIR}:\${PYTHONPATH:-}
                export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH:-}
                hostname
                which ${PYTHON_BIN}
                ${PYTHON_BIN} -V
                ${PYTHON_BIN} ${SCRIPT_DIR}/finetune_policy.py \
                    --nnodes ${#HOSTS[@]} \
                    --node-rank ${NODE_RANK} \
                    --master-addr ${MASTER_ADDR} \
                    --master-port ${MASTER_PORT} \
                    --num-gpus ${GPUS_PER_NODE} \
                    ${USER_ARGS}
            " &
        done

        if [[ -z "${LOCAL_NODE_RANK}" ]]; then
            echo "Current host ${CURRENT_HOST} not found in LSF host list: ${HOSTS[*]}" >&2
            exit 1
        fi

        echo "Launching local host=${CURRENT_HOST} node_rank=${LOCAL_NODE_RANK}"
        "${PYTHON_BIN}" "${SCRIPT_DIR}/finetune_policy.py" \
            --nnodes "${#HOSTS[@]}" \
            --node-rank "${LOCAL_NODE_RANK}" \
            --master-addr "${MASTER_ADDR}" \
            --master-port "${MASTER_PORT}" \
            --num-gpus "${GPUS_PER_NODE}" \
            "$@"

        wait
        exit 0
    fi
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/finetune_policy.py" --num-gpus "${GPUS_PER_NODE}" "$@"
