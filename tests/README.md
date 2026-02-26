# Tests Directory

This directory contains all unit and integration tests for vllm-tuner.

See [TESTING.md](../TESTING.md) for comprehensive testing documentation.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run all unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html
```

## Test Files

### Unit Tests (`tests/unit/`)

| File | Tests | Focus |
|------|-------|-------|
| `test_config.py` | 17 | Configuration validation |
| `test_search_space.py` | 7 | Parameter search space |
| `test_telemetry.py` | 9 | vLLM log parsing |
| `test_baseline.py` | 10 | Baseline generation |
| `test_html_report.py` | 13 | HTML report generation |
| `test_study_manager.py` | 8 | Study management |

**Total:** 64 unit tests

### Integration Tests (`tests/integration/`)

Currently empty. Integration tests require:
- vLLM server running
- GPU hardware access
- Model files downloaded

## Running Tests

### All Unit Tests
```bash
pytest tests/unit/ -v
```

### Specific Test File
```bash
pytest tests/unit/test_config.py -v
```

### Specific Test
```bash
pytest tests/unit/test_config.py::test_gpu_config_validation -v
```

### With Coverage
```bash
pytest tests/unit/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Detailed Output on Failure
```bash
pytest tests/unit/ -vv
```

## Coverage Goals

- **Config Models:** 95%
- **Search Space:** 90%
- **Telemetry:** 85%
- **Baseline Runner:** 80%
- **HTML Report:** 80%
- **Study Manager:** 70%
- **CLI:** 50%

**Overall Target:** 75%

## Adding Tests

1. Use descriptive test names: `test_function_name_scenario()`
2. Use pytest fixtures for shared setup
3. Mock external dependencies (vLLM, GPU)
4. Test both success and failure paths
5. Add type hints to test functions

Example:
```python
def test_baseline_metrics_with_samples():
    """Test BaselineMetrics.to_dict() with sample data."""
    metrics = BaselineMetrics(
        model="gpt2",
        timestamp="2024-02-26T00:00:00",
        vllm_params={},
        benchmark_params={},
    )
    metrics.memory_samples = [1000.0, 1500.0, 2000.0]

    result = metrics.to_dict()

    assert result["metrics"]["peak_memory_mb"] == 2000.0
    assert result["metrics"]["average_memory_mb"] == 1500.0
```

## CI/CD

See [TESTING.md](../TESTING.md) for GitHub Actions workflow example.