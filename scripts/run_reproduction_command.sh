#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=data_disk_reproduction_env.sh
source "${SCRIPT_DIR}/data_disk_reproduction_env.sh"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <vllm-tuner arguments...>" >&2
  echo "Example: $0 tune --config config/formal_3b_chat.yaml --study-name NAME --results-root PATH" >&2
  exit 2
fi

CLI="${SLOTUNE_REPO_DIR}/.venv/bin/vllm-tuner"
if [[ ! -x "${CLI}" ]]; then
  echo "Missing ${CLI}; run scripts/setup_data_disk_reproduction.sh first." >&2
  exit 1
fi

cd "${SLOTUNE_REPO_DIR}"
# Execute the installed entry point directly. Using 'uv run' here could resync the
# core lock after the GPU overlay and would make the formal process environment
# depend on the caller's shell.
exec "${CLI}" "$@"
