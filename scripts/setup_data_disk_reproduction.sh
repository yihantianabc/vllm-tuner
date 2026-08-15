#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=data_disk_reproduction_env.sh
source "${SCRIPT_DIR}/data_disk_reproduction_env.sh"

cd "${SLOTUNE_REPO_DIR}"

if [[ ! -x .venv/bin/python ]]; then
  uv venv --seed --python 3.12 .venv
fi

# The lock owns all shared project dependencies. The GPU stack remains a
# deliberately pinned overlay because uv must select a host-compatible Torch
# wheel. The lock and overlay pin the same Transformers, NumPy, and IDNA
# versions. Inexact sync retains an already-installed host overlay on reruns
# while frozen resolution still makes every locked project dependency immutable.
uv sync --extra dev --frozen --inexact
uv pip install \
  --python .venv/bin/python \
  --torch-backend=auto \
  -r requirements-reproduction.txt

uv pip check --python .venv/bin/python
echo "Environment ready: ${SLOTUNE_REPO_DIR}/.venv"
echo "Run smoke/formal commands through scripts/run_data_disk_reproduction.sh or"
echo "scripts/run_reproduction_command.sh so cache and temporary-directory"
echo "variables are inherited by vllm-tuner and every child server process."
