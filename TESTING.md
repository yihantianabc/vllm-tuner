# Test Execution Guide

This document provides comprehensive instructions for running all tests in the vLLM-tuner project.

## Environment Setup

### Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)
- Access to GPU (for integration tests only)

### Installation

1. **Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install the package with dependencies:**
```bash
pip install -e ".[dev]"
```

3. **Verify installation:**
```bash
python -c "import pytest, pydantic, optuna, pynvml; print('✓ All dependencies installed')"
```

## Test Suite Overview

### Unit Tests (Fast, No external dependencies)

#### Configuration Tests (`test_config.py`)
- Tests Pydantic model validation
- 16 tests covering all config components
- Run time: ~2 seconds

```bash
pytest tests/unit/test_config.py -v
```

#### Search Space Tests (`test_search_space.py`)
- Tests parameter search space logic
- 7 tests for single/multi-GPU scenarios
- Run time: ~1 second

```bash
pytest tests/unit/test_search_space.py -v
```

#### Telemetry Tests (`test_telemetry.py`)
- Tests vLLM log parsing
- 9 tests for telemetry extraction
- Run time: ~1 second

```bash
pytest tests/unit/test_telemetry.py -v
```

#### Baseline Runner Tests (`test_baseline.py`) - NEW
- Tests baseline generation and metrics
- 13 tests for baseline functionality
- Run time: ~2 seconds

```bash
pytest tests/unit/test_baseline.py -v
```

#### HTML Report Tests (`test_html_report.py`) - NEW
- Tests HTML report generation
- 15 tests including baseline comparison
- Run time: ~3 seconds

```bash
pytest tests/unit/test_html_report.py -v
```

#### Study Manager Tests (`test_study_manager.py`) - NEW
- Tests study management and GPU history
- 10 tests for optimization workflow
- Run time: ~2 seconds

```bash
pytest tests/unit/test_study_manager.py -v
```

### Integration Tests (Slower, Requires vLLM and GPU)

Currently empty. Future additions:
- End-to-end tuning workflow
- HTML report generation with real data
- Multi-GPU scenarios

## Running All Tests

### Run All Unit Tests
```bash
pytest tests/unit/ -v
```

Expected: ~53 tests, all passing

### Run with Coverage Report
```bash
pytest tests/ -v --cov=src --cov-report=html
open htmlcov/index.html  # View coverage report
```

Expected: 40-60% coverage (baseline coverage)

### Run Specific Test
```bash
# Single test
pytest tests/unit/test_config.py::test_gpu_config_validation -v

# All tests in class
pytest tests/unit/test_baseline.py::TestBaselineMetrics -v

# All tests matching pattern
pytest tests/unit/ -k "validation" -v
```

## Test Files Summary

| Test File | Tests | Focus | Dependencies |
|-----------|-------|-------|--------------|
| test_config.py | 16 | Configuration validation | pydantic |
| test_search_space.py | 7 | Parameter search space | optuna |
| test_telemetry.py | 9 | vLLM log parsing | None |
| test_baseline.py | 13 | Baseline generation | pytest, pydantic, yaml |
| test_html_report.py | 15 | HTML reporting | plotly, jinja2 |
| test_study_manager.py | 10 | Study management | optuna, httpx |
| **Total** | **70** | | |

## Common Issues & Solutions

### Issue 1: Import Error - "No module named pydantic"
**Solution:**
```bash
pip install -e ".[dev]"
```

### Issue 2: CUDA Out of Memory during integration tests
**Solution:** Skip integration tests or reduce test workload size
```bash
pytest tests/unit/ -v  # Don't run integration tests
```

### Issue 3: GPU not available for GPU collector tests
**Solution:** Tests use mocks and don't require actual GPU

### Issue 4: Yellow warning about assertions
**Solution:** These are informational, not failures. Fix if you want to silence them:
```bash
pytest tests/unit/ -v -W ignore:: pytest.PytestAssertRewriteWarning
```

## Continuous Integration

### GitHub Actions Workflow (Recommended)

Create `.github/workflows/test.yml`:

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Test Markers

To add markers for selective test execution:

```python
# Example in test file
import pytest

@pytest.mark.slow
def test_slow_operation():
    pass

@pytest.mark.integration
def test_requires_gpu():
    pass
```

Run specific markers:
```bash
pytest -m "not slow" -v  # Skip slow tests
pytest -m "unit" -v      # Only unit tests
pytest -m "integration" -v  # Only integration tests
```

## Debugging Failed Tests

### Run with Detailed Output
```bash
pytest tests/unit/test_config.py::test_specific_test -vvs
```

### Drop into PDB on Failure
```bash
pytest tests/unit/test_config.py --pdb
```

### Stop on First Failure
```bash
pytest tests/unit/ -x
```

## Expected Test Results

### All Tests Passing
```
tests/unit/test_config.py::test_gpu_config_default PASSED
tests/unit/test_config.py::test_gpu_config_validation PASSED
...
======================== 70 passed in 15.23s =========================
```

### With Coverage
```
---------- coverage: platform linux, python 3.12 ----------
Name                                                Stmts   Miss  Cover
-----------------------------------------------------------------------
src/baseline/runner.py                               200     40    80%
src/cli/main.py                                      150    120    20%
src/config/models.py                                 100      5    95%
...
-----------------------------------------------------------------------
TOTAL                                                800    320    60%
======================== 70 passed in 15.23s =========================
```

## Running Linting and Type Checking

### Format Code
```bash
black src/ tests/
```

### Check Linting
```bash
ruff check src/ tests/
ruff check src/ tests/ --fix  # Auto-fix issues
```

### Type Checking
```bash
mypy src/
```

### All Quality Checks Together
```bash
black src/ tests/
ruff check src/ tests/ --fix
pytest tests/unit/ -v --cov=src
mypy src/
```

## Test Coverage Goals

| Module | Target | Current | Status |
|--------|--------|---------|--------|
| Config Models | 95% | 95% | ✅ |
| Search Space | 90% | 85% | ⚠️ |
| Telemetry | 85% | 80% | ⚠️ |
| Baseline Runner | 80% | 60% | ⚠️ |
| HTML Report | 80% | 65% | ⚠️ |
| Study Manager | 70% | 55% | ⚠️ |
| CLI | 50% | 20% | ⚠️ |
| **Overall** | **75%** | **60%** | ⚠️ |

## Contributing Tests

When adding new features:

1. Write unit tests first (TDD approach)
2. Add tests to appropriate test file or create new one
3. Ensure test names follow pattern: `test_function_name_scenario()`
4. Use fixtures for common setup
5. Mock external dependencies (vLLM, GPU)
6. Test both success and failure paths

## Pre-Commit Checklist

Before committing:

- [ ] All unit tests pass (`pytest tests/unit/ -v`)
- [ ] Linting passes (`ruff check src/ tests/`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Code formatted (`black src/ tests/`)
- [ ] Coverage not decreased (`pytest --cov=src tests/`)
- [ ] No TODO comments in production code

## Support

For issues or questions about testing:
- Check this guide first
- Review AGENTS.md for code style guidelines
- Check existing tests for examples
- Use pytest's verbose output for debugging