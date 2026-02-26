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
- **GPU**: Recent NVIDIA GPU with at least 8GB VRAM
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
pip install vllm
```

For specific CUDA versions (optional):

```bash
# CUDA 12.1
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install vllm --extra-index-url https://download.pydantic.org/whl/cu124
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

## Installation Methods

### Method 1: Pip Install (Recommended)

Install from local directory:

```bash
cd /path/to/vllm-tuner
pip install -e ".[dev]"
pip install vllm
```

### Method 2: Virtual Environment (Best Practice)

Create isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

pip install -e ".[dev]"
pip install vllm
```

### Method 3: Docker (For Production)

Create Dockerfile:

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y \
    python3.12-venv \
    python3.12-dev \
    cuda-runtime-12-1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install -e ".[dev]"
RUN pip install vllm

CMD ["bash"]
```

Build and run:

```bash
docker build -t vllm-tuner .
docker run --gpus all vllm-tuner
```

## Dependency Installation Issues

### Issue: No module named 'pynvml'

```bash
# Ubuntu/Debian
sudo apt-get install python3-pynvml

# Then reinstall
pip install -e ".[dev]"
```

### Issue: vLLM installation fails

```bash
# Check CUDA compatibility
nvidia-smi
nvcc --version

# Install PyTorch with correct CUDA version first
pip install torch

# Then install vLLM
pip install vllm
```

### Issue: Virtual environment creation fails

```bash
# Ubuntu/Debian
sudo apt install python3.12-venv
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

## Next Steps

- [Configuration Guide](configuration.md) - Create your first config file
- [CLI Commands](cli-commands.md) - Learn available commands
- [User Guide Overview](index.md) - Back to user guide
- [Developer Setup](../developer-guide/setup.md) - Development environment

## Troubleshooting

### Issue: "找不到模块 named" (Module not found)

**Solution:**
```bash
pip install -e ".[dev]"
```

This installs all dependencies including optuna, pydantic, etc.

### Issue: vLLM not found

**Solution:**
```bash
pip install vllm
```

### Issue: CUDA out of memory during install

**Solution:**
- Install minimal version first: `pip install vllm --no-deps`
- Then install dependencies separately
- Or use system package manager: `conda install -c nvidia vllm`

### Issue: Tests fail after installation

**Solution:**
```bash
# Reinstall dev dependencies
pip install -e ".[dev]"

# Check pytest version
pytest --version

# If pytest < 7.4.0, upgrade:
pip install pytest==7.4.0
```

See [Common Issues](../troubleshooting/common-issues.md) for more troubleshooting help.

## See Also

- [Configuration](configuration.md) - Create your first config
- [CLI Commands](cli-commands.md) - Command reference
- [Examples](examples/) - Try ready-to-use examples
- [Troubleshooting](../troubleshooting/) - Common installation issues

---

**Next:** [Configuration Guide](configuration.md)