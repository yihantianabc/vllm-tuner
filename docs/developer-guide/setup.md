# Developer Setup

## Prerequisites

- Python 3.10 or higher
- Git (optional, for contributing)
- NVIDIA GPU with CUDA (for running tests, optional for development)

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/vllm-tuner.git
cd vllm-tuner

# Install with dev dependencies
pip install -e ".[dev]"

# Install vLLM (for integration tests)
pip install vllm
```

## Verify Setup

```bash
# Test imports
python -c "from src.config.models import TuningConfig; from src.tuner.optimizer import VLLMOptimizer"

# Run unit tests
pytest tests/unit/ -v

# Check code quality
ruff check src/ tests/
black --check src/ tests/
mypy src/
```

## IDE Setup

### VS Code

Install extensions:
- Python
- Pylance
- Black Formatter
- Ruff
- Pyright

### PyCharm

Configure:
- Python interpreter to `.venv/bin/python`
- Enable Black formatting
- Enable Ruff linting
- Configure test runner

## See Also

- [Testing Guide](testing.md) - Testing procedures
- [AGENTS.md](../../AGENTS.md) - Code style guidelines
