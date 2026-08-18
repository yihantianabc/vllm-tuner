#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=data_disk_reproduction_env.sh
source "${SCRIPT_DIR}/data_disk_reproduction_env.sh"

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1

PYTHON="${REPO_DIR}/.venv/bin/python"
DEFAULT_CONFIG="${REPO_DIR}/experiments/long_context/v5/m3-apc-smoke.yaml"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}; run scripts/setup_data_disk_reproduction.sh first." >&2
  exit 1
fi

config_args=(--config "${DEFAULT_CONFIG}")
for argument in "$@"; do
  if [[ "${argument}" == "--config" || "${argument}" == "-c" || \
        "${argument}" == --config=* ]]; then
    config_args=()
    break
  fi
done

cd "${REPO_DIR}"
exec "${PYTHON}" -m vllm_tuner.longctx.m3_apc_cli "${config_args[@]}" "$@"
