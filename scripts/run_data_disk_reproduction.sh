#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=data_disk_reproduction_env.sh
source "${SCRIPT_DIR}/data_disk_reproduction_env.sh"

STUDY_NAME="${1:-reproduction_smoke_$(date -u +%Y%m%d_%H%M%S)}"
cd "${SLOTUNE_REPO_DIR}"
exec "${SLOTUNE_REPO_DIR}/.venv/bin/vllm-tuner" tune \
  --config config/reproduction_smoke.yaml \
  --study-name "${STUDY_NAME}" \
  --results-root "${SLOTUNE_DATA_DIR}/vllm-tuner-output/slotune-results" \
  --no-baseline
