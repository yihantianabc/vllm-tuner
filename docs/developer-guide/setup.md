# Developer setup

```bash
cd /root/autodl-tmp/vllm-tuner
uv venv --seed --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Install vLLM only for GPU integration work:

```bash
uv pip install vllm --torch-backend=auto
```

Verify:

```bash
python -c "from vllm_tuner.config.models import TuningConfig; print(TuningConfig())"
pytest -q tests/unit
ruff check src tests scripts/run_scheduler_ablation.py
black --check src tests scripts/run_scheduler_ablation.py
mypy src
```

The deterministic scheduler tests and demo are pure Python. Runtime tests require an NVIDIA GPU,
NVML, a compatible vLLM/PyTorch/CUDA stack, and a local model. Keep large caches and generated
artifacts outside the repository, preferably under `/root/autodl-tmp` on the reference host.
