#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN_DIR="$(dirname "${PYTHON_BIN}")"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-${REPO_DIR}/scripts/lsf_remote_logs}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
mkdir -p "${REMOTE_LOG_DIR}"

resolve_master_addr() {
    local master_host="$1"
    local master_addr

    master_addr="$(getent ahostsv4 "${master_host}" | awk 'NR==1 {print $1}')"
    if [[ -z "${master_addr}" ]]; then
        master_addr="$(getent hosts "${master_host}" | awk 'NR==1 {print $1}')"
    fi
    if [[ -z "${master_addr}" ]]; then
        echo "Failed to resolve IPv4 address for master host ${master_host}" >&2
        exit 1
    fi

    printf '%s\n' "${master_addr}"
}

if [[ -n "${LSB_DJOB_HOSTFILE:-}" && -f "${LSB_DJOB_HOSTFILE}" ]]; then
    mapfile -t HOSTS < <(sort -u "${LSB_DJOB_HOSTFILE}")

    if (( ${#HOSTS[@]} > 1 )); then
        CURRENT_HOST="$(hostname -s)"
        LOCAL_NODE_RANK=""
        MASTER_HOST="${HOSTS[0]}"
        MASTER_ADDR="${MASTER_ADDR:-$(resolve_master_addr "${MASTER_HOST}")}"

        USER_ARGS="$(printf ' %q' "$@")"
        USER_ARGS="${USER_ARGS:1}"

        for NODE_RANK in "${!HOSTS[@]}"; do
            HOST="${HOSTS[$NODE_RANK]}"
            if [[ "${HOST}" == "${CURRENT_HOST}" ]]; then
                LOCAL_NODE_RANK="${NODE_RANK}"
                continue
            fi

            REMOTE_LOG_PATH="${REMOTE_LOG_DIR}/$(date +%Y%m%d_%H%M%S)_${HOST}_rank${NODE_RANK}.log"
            blaunch -z "${HOST}" bash -lc "
                set -euo pipefail
                cd ${REPO_DIR}
                export PYTHONPATH=${REPO_DIR}:\${PYTHONPATH:-}
                export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH:-}
                export PATH=${PYTHON_BIN_DIR}:\${PATH:-}
                ${PYTHON_BIN} ${SCRIPT_DIR}/finetune_policy.py \
                    --nnodes ${#HOSTS[@]} \
                    --node-rank ${NODE_RANK} \
                    --master-addr ${MASTER_ADDR} \
                    --master-port ${MASTER_PORT} \
                    --num-gpus ${GPUS_PER_NODE} \
                    ${USER_ARGS}
            " > "${REMOTE_LOG_PATH}" 2>&1 &
        done

        if [[ -z "${LOCAL_NODE_RANK}" ]]; then
            echo "Current host ${CURRENT_HOST} not found in LSF host list: ${HOSTS[*]}" >&2
            exit 1
        fi

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
