# Testing Guide

## Running Tests

### All Unit Tests

```bash
pytest tests/unit/ -v
```

### Specific Test File

```bash
pytest tests/unit/test_config.py -v
```

### Single Test

```bash
pytest tests/unit/test_config.py::test_gpu_config_validation -v
```

### With Coverage

```bash
pytest tests/unit/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Quick Test Commands

```bash
# Format code
black src/ tests/

# Check linting
ruff check src/ tests/ --fix

# Type checking
mypy src/
```
