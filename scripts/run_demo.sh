#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /explicit/output/directory" >&2
  echo "Runs the deterministic CPU scheduler calibration/held-out demo." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

cd "${REPO_DIR}"
exec "${PYTHON_BIN}" scripts/run_scheduler_ablation.py \
  --output-dir "$1" \
  --fixed-budgets 512 1024 2048 4096 8192
