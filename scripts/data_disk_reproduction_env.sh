#!/usr/bin/env bash
# Shared environment for setup, smoke, and formal GPU reproduction commands.
# Source this file; scripts/run_reproduction_command.sh is the preferred entry point.

SLOTUNE_DATA_DIR="${SLOTUNE_DATA_DIR:-/root/autodl-tmp}"
SLOTUNE_REPO_DIR="${SLOTUNE_REPO_DIR:-${SLOTUNE_DATA_DIR}/vllm-tuner}"
export SLOTUNE_DATA_DIR SLOTUNE_REPO_DIR

export PATH="${SLOTUNE_REPO_DIR}/.venv/bin:${PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1

case ",${NO_PROXY:-}," in
  *,127.0.0.1,*) ;;
  *) export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,${NO_PROXY}}" ;;
esac
export no_proxy="${NO_PROXY}"

export UV_CACHE_DIR="${SLOTUNE_DATA_DIR}/uv-cache"
export PIP_CACHE_DIR="${SLOTUNE_DATA_DIR}/pip-cache"
export XDG_CACHE_HOME="${SLOTUNE_DATA_DIR}/cache"
export XDG_CONFIG_HOME="${SLOTUNE_DATA_DIR}/config"
export TMPDIR="${SLOTUNE_DATA_DIR}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"

export HF_HOME="${SLOTUNE_DATA_DIR}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_XET_CACHE="${HF_HOME}/xet"

export TORCH_HOME="${SLOTUNE_DATA_DIR}/torch"
export TORCH_EXTENSIONS_DIR="${SLOTUNE_DATA_DIR}/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="${SLOTUNE_DATA_DIR}/torchinductor"
export TRITON_CACHE_DIR="${SLOTUNE_DATA_DIR}/triton"
export CUDA_CACHE_PATH="${SLOTUNE_DATA_DIR}/cuda-cache"
export NUMBA_CACHE_DIR="${SLOTUNE_DATA_DIR}/numba-cache"
export VLLM_CACHE_ROOT="${SLOTUNE_DATA_DIR}/vllm-cache"
export FLASHINFER_WORKSPACE_BASE="${SLOTUNE_DATA_DIR}/flashinfer"

export VLLM_TUNER_STUDY_OUTPUT_DIR="${SLOTUNE_DATA_DIR}/vllm-tuner-output/studies"
export VLLM_TUNER_HTML_OUTPUT_DIR="${SLOTUNE_DATA_DIR}/vllm-tuner-output/reports"

mkdir -p \
  "${UV_CACHE_DIR}" \
  "${PIP_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${TMPDIR}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_XET_CACHE}" \
  "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" \
  "${NUMBA_CACHE_DIR}" \
  "${VLLM_CACHE_ROOT}" \
  "${FLASHINFER_WORKSPACE_BASE}" \
  "${VLLM_TUNER_STUDY_OUTPUT_DIR}" \
  "${VLLM_TUNER_HTML_OUTPUT_DIR}"
