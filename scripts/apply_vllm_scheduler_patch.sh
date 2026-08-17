#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PATCH_FILE="${REPO_DIR}/patches/vllm-v0.16.0/0001-add-prefill-token-budget-hook.patch"
VLLM_PYTHON="${VLLM_PYTHON:-${REPO_DIR}/.venv/bin/python}"
UPSTREAM_SHA256="bb36be85a1054cdbfedb35c1f04ee02696d9f94a076f7829e9da0bb4f7987d07"
PATCHED_SHA256="ed1b8dc7816a48b69710631e67d929f6d4c3870ce868f422cd820389bd08731c"

if [[ ! -x "${VLLM_PYTHON}" ]]; then
  echo "vLLM Python is not executable: ${VLLM_PYTHON}" >&2
  exit 2
fi

VLLM_VERSION="$("${VLLM_PYTHON}" -c 'import vllm; print(vllm.__version__)')"
if [[ "${VLLM_VERSION}" != "0.16.0" ]]; then
  echo "Expected vLLM 0.16.0, found ${VLLM_VERSION}" >&2
  exit 2
fi

VLLM_PACKAGE_DIR="$("${VLLM_PYTHON}" -c 'from pathlib import Path; import vllm; print(Path(vllm.__file__).resolve().parent)')"
SITE_PACKAGES_DIR="$(dirname -- "${VLLM_PACKAGE_DIR}")"
SCHEDULER_FILE="${VLLM_PACKAGE_DIR}/v1/core/sched/scheduler.py"
CURRENT_SHA256="$(sha256sum "${SCHEDULER_FILE}" | awk '{print $1}')"

if [[ "${CURRENT_SHA256}" == "${PATCHED_SHA256}" ]]; then
  echo "SLOTune vLLM Scheduler patch is already applied."
  exit 0
fi
if [[ "${CURRENT_SHA256}" != "${UPSTREAM_SHA256}" ]]; then
  echo "Refusing to patch an unknown Scheduler source: ${CURRENT_SHA256}" >&2
  exit 2
fi

patch --batch --forward --directory "${SITE_PACKAGES_DIR}" -p1 < "${PATCH_FILE}"
CURRENT_SHA256="$(sha256sum "${SCHEDULER_FILE}" | awk '{print $1}')"
if [[ "${CURRENT_SHA256}" != "${PATCHED_SHA256}" ]]; then
  echo "Patched Scheduler checksum mismatch: ${CURRENT_SHA256}" >&2
  exit 2
fi

echo "Applied SLOTune Prefill-budget hook to vLLM ${VLLM_VERSION}."
