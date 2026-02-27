# Installation

This guide covers installing vLLM-Tuner and its dependencies.

## System Requirements

### Minimum Requirements

- **Python**: 3.10 or higher
- **RAM**: At least 8GB available
- **Disk**: ~2GB free space for dependencies
- **GPU**: NVIDIA GPU with CUDA support (recommended for running vLLM)

### Recommended for Development

- **Python**: 3.10, 3.11, or 3.12
- **RAM**: 16GB or more
- **GPU**: Recent NVIDIA GPU with at least 4GB VRAM
- **Disk**: 10GB or more free space

## Installation Steps

### Step 1: Clone or Download

```bash
# If you want to contribute, clone the repository
git clone https://github.com/your-org/vllm-tuner.git
cd vllm-tuner

# If using the repository locally:
cd /path/to/vllm-tuner
```

### Step 2: Install with Development Dependencies

```bash
# Create and activate uv environment
uv venv --seed --python 3.10
source .venv/bin/activate

# Install with all dev tools
pip install -e ".[dev]"

# Alternatively, install without dev tools
pip install -e .
```

The `[dev]` extra includes:
- **Testing**: pytest, pytest-asyncio, pytest-cov
- **Code Quality**: black, ruff, mypy
- **Type Stubs**: types-PyYAML

### Step 3: Install vLLM

vLLM is required to run tuning studies:

```bash
uv pip install vllm --torch-backend=auto
```

For specific CUDA versions (optional):

```bash
# CUDA 12.1
uv pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
uv pip install vllm --extra-index-url https://download.pydantic.org/whl/cu124
```

### Step 4: Verify Installation

```bash
python -c "
from src.config.models import TuningConfig
from src.tuner.optimizer import VLLMOptimizer
print('✓ vLLM-Tuner installation successful')
"

python -c "
import vllm
print(f'✓ vLLM {vllm.__version__} installed')
"
```

## Dependency Installation Issues

### Issue: No module named 'pynvml'

```bash
# Ubuntu/Debian
uv pip install pynvml

# Then reinstall
uv pip install -e .
```

### Issue: vLLM installation fails

```bash
# Check CUDA compatibility
nvidia-smi
nvcc --version

# Install PyTorch with correct CUDA version first
uv pip install torch

# Then install vLLM
uv pip install vllm --torch-backend=auto
```

## Verification

### Check Dependencies

```bash
python3 -c "
import sys
pkgs = ['pydantic', 'optuna', 'yaml', 'httpx', 'plotly', 'jinja2']
for pkg in pkgs:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError as e:
        print(f'✗ {pkg}: {e}')
"
```

### Run Health Check

```bash
# Check if CLI is available
vllm-tuner --help

# Run test import
python3 -c "
from src.cli.main import app
print('✓ CLI import successful')
"
```

## Post-Installation Steps

### 1. View Example Configurations

```bash
ls -la config/ examples/
cat config/default.yaml
```

### 2. Read AGENTS.md

See [AGENTS.md](../AGENTS.md) for:
- Build/lint/test commands
- Code style guidelines
- Development workflow

### 3. Test Installation

```bash
# List studies
vllm-tuner list-studies

# Should show empty list or existing studies
```

## Uninstall

```bash
pip uninstall vllm-tuner

# Remove venv (if used)
deactivate
rm -rf .venv
```

## See Also

- [Configuration](configuration.md) - Create your first config
- [CLI Commands](cli-commands.md) - Command reference
- [Examples](examples/) - Try ready-to-use examples
- [Troubleshooting](../troubleshooting/) - Common installation issues

---

**Next:** [Configuration Guide](configuration.md)