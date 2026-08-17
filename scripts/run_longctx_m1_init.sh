#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=data_disk_reproduction_env.sh
source "${SCRIPT_DIR}/data_disk_reproduction_env.sh"

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1

CLI="${REPO_DIR}/.venv/bin/vllm-tuner"
CONFIG="${REPO_DIR}/experiments/long_context/v5/m1-initialization.yaml"
if [[ ! -x "${CLI}" ]]; then
  echo "Missing ${CLI}; run scripts/setup_data_disk_reproduction.sh first." >&2
  exit 1
fi

cd "${REPO_DIR}"
exec "${CLI}" longctx-m1-init --config "${CONFIG}" "$@"
