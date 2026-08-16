#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 /explicit/output/directory [/completed/formal/artifact]" >&2
  echo "Runs the CPU scheduler demo and optionally displays a pre-generated formal report." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
OUTPUT_DIR="$1"
FORMAL_ROOT="${2:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

cd "${REPO_DIR}"
"${PYTHON_BIN}" scripts/run_scheduler_ablation.py \
  --output-dir "${OUTPUT_DIR}" \
  --fixed-budgets 512 1024 2048 4096 8192

if [[ -z "${FORMAL_ROOT}" ]]; then
  echo
  echo "CPU scheduler artifact: ${OUTPUT_DIR}"
  echo "Pass a completed formal artifact as argument 2 to include the pre-generated GPU report."
  exit 0
fi

for relative_path in \
  report/report.md \
  report/report.html \
  report/capacity-curve.png \
  aggregate/candidate-validation.parquet; do
  if [[ ! -f "${FORMAL_ROOT}/${relative_path}" ]]; then
    echo "Missing formal evidence file: ${FORMAL_ROOT}/${relative_path}" >&2
    exit 1
  fi
done

echo
echo "Pre-generated formal report (no GPU experiment was launched):"
echo "  Markdown: ${FORMAL_ROOT}/report/report.md"
echo "  HTML: ${FORMAL_ROOT}/report/report.html"
echo "  Capacity: ${FORMAL_ROOT}/report/capacity-curve.png"
echo
sed -n '1,100p' "${FORMAL_ROOT}/report/report.md"

if [[ -f "${FORMAL_ROOT}/report/scheduler-negative-results.md" ]]; then
  echo
  sed -n '1,160p' "${FORMAL_ROOT}/report/scheduler-negative-results.md"
fi
