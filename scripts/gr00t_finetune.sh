#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
FINETUNE_SCRIPT="${SCRIPT_DIR}/finetune_policy.py"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
EXPECTED_NNODES="${EXPECTED_NNODES:-}"

has_flag() {
    local target="$1"
    shift
    local arg
    for arg in "$@"; do
        if [[ "${arg}" == "${target}" ]]; then
            return 0
        fi
    done
    return 1
}

shell_join() {
    local out=()
    local arg
    for arg in "$@"; do
        printf -v quoted "%q" "${arg}"
        out+=("${quoted}")
    done
    printf '%s ' "${out[@]}"
}

build_python_cmd() {
    local cmd=("${PYTHON_BIN}" "${FINETUNE_SCRIPT}")

    if ! has_flag "--num-gpus" "$@" && ! has_flag "--num_gpus" "$@"; then
        cmd+=("--num-gpus" "${GPUS_PER_NODE}")
    fi

    cmd+=("$@")
    printf '%s\n' "$(shell_join "${cmd[@]}")"
}

run_local() {
    local cmd_str
    cmd_str="$(build_python_cmd "$@")"

    echo "Running local finetune command:"
    echo "${cmd_str}"

    cd "${REPO_DIR}"
    export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
    eval "${cmd_str}"
}

run_lsf_multinode() {
    local user_args=("$@")

    if ! command -v blaunch >/dev/null 2>&1; then
        echo "blaunch is required for LSF multi-node launch but was not found." >&2
        exit 1
    fi

    if has_flag "--nnodes" "${user_args[@]}" || has_flag "--node-rank" "${user_args[@]}" || \
       has_flag "--master-addr" "${user_args[@]}" || has_flag "--master-port" "${user_args[@]}"; then
        echo "Do not pass --nnodes/--node-rank/--master-addr/--master-port to gr00t_finetune.sh in LSF multi-node mode." >&2
        exit 1
    fi

    mapfile -t HOSTS < <(sort -u "${LSB_DJOB_HOSTFILE}")
    local nnodes="${#HOSTS[@]}"
    local master_addr="${MASTER_ADDR:-${HOSTS[0]}}"

    if [[ -n "${EXPECTED_NNODES}" ]] && [[ "${nnodes}" != "${EXPECTED_NNODES}" ]]; then
        echo "Expected ${EXPECTED_NNODES} nodes from LSF, but got ${nnodes}: ${HOSTS[*]}" >&2
        exit 1
    fi

    echo "LSF multi-node launch detected"
    echo "Hosts: ${HOSTS[*]}"
    echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
    echo "MASTER_ADDR=${master_addr}"
    echo "MASTER_PORT=${MASTER_PORT}"

    local node_rank
    for node_rank in "${!HOSTS[@]}"; do
        local host="${HOSTS[$node_rank]}"
        local remote_cmd=(
            env
            "PYTHONPATH=${REPO_DIR}:${PYTHONPATH:-}"
            "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
            "WANDB_PROJECT=${WANDB_PROJECT:-GR00T-N1.5-unity}"
            "${PYTHON_BIN}"
            "${FINETUNE_SCRIPT}"
            "--nnodes" "${nnodes}"
            "--node-rank" "${node_rank}"
            "--master-addr" "${master_addr}"
            "--master-port" "${MASTER_PORT}"
        )

        if ! has_flag "--num-gpus" "${user_args[@]}" && ! has_flag "--num_gpus" "${user_args[@]}"; then
            remote_cmd+=("--num-gpus" "${GPUS_PER_NODE}")
        fi
        remote_cmd+=("${user_args[@]}")

        printf -v remote_bash_cmd 'cd %q && %s' "${REPO_DIR}" "$(shell_join "${remote_cmd[@]}")"

        echo "Launching node_rank=${node_rank} on host=${host}"
        blaunch -z "${host}" bash -lc "${remote_bash_cmd}" &
    done

    wait
}

main() {
    if [[ -n "${LSB_DJOB_HOSTFILE:-}" && -f "${LSB_DJOB_HOSTFILE}" ]]; then
        mapfile -t HOSTS < <(sort -u "${LSB_DJOB_HOSTFILE}")
        if (( ${#HOSTS[@]} > 1 )); then
            run_lsf_multinode "$@"
            return
        fi
    fi

    run_local "$@"
}

main "$@"
