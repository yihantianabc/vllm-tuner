# Installation

## Development environment

```bash
cd /root/autodl-tmp/vllm-tuner
uv venv --seed --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Install vLLM only when running GPU integration/smoke/formal experiments:

```bash
uv pip install vllm --torch-backend=auto
```

Verify the package and CLI:

```bash
python -c "from vllm_tuner.config.models import TuningConfig; print(TuningConfig().model)"
vllm-tuner --help
python scripts/run_scheduler_ablation.py --help
```

Unit tests and the deterministic scheduler demo do not require a GPU. Runtime integration requires
a CUDA-compatible NVIDIA GPU, working driver/NVML, vLLM, and enough disk for model and compilation
caches.

## Data-disk environment

The repository includes a machine-specific setup script for `/root/autodl-tmp`:

```bash
./scripts/setup_data_disk_reproduction.sh
```

It places the environment, Hugging Face, PyTorch, Triton, CUDA, vLLM, uv/pip, and temporary caches
on the data disk. Review the script before using it on another host. Because a setup subprocess
cannot export variables into its caller, use the command wrapper for formal runs:

```bash
./scripts/run_reproduction_command.sh tune --config config/formal_3b_chat.yaml \
  --study-name qwen25_3b_chat_001 --results-root /root/autodl-tmp/slotune-results
```

## Verification levels

CPU-only scheduler demo:

```bash
./scripts/run_demo.sh /root/autodl-tmp/slotune-demo/scheduler
```

Local Qwen3-0.6B GPU correctness smoke:

```bash
./scripts/run_data_disk_reproduction.sh slotune_smoke_001
```

The smoke is not a benchmark. Formal runs require the 3B model path in the selected validated
config and should use a unique explicit results root.

## Common installation checks

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import vllm; print(vllm.__version__)"
pytest -q tests/unit
```

Match PyTorch/CUDA/vLLM using their official compatibility guidance for the host rather than
copying a wheel URL from an unrelated environment.
