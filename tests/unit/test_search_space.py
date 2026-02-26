"""Unit tests for search space."""

import pytest

from src.optimization.search_space import VLLMSearchSpace
from src.config.models import TuningConfig


def test_search_space_defaults():
    """Test default search space."""
    config = TuningConfig()
    search_space = VLLMSearchSpace(config, num_gpus=1)

    params = search_space.get_parameter_names()
    assert "batch_size" in params
    assert "max_num_seqs" in params
    assert "gpu_memory_utilization" in params

    assert "tensor_parallel_size" not in params
    assert "pipeline_parallel_size" not in params


def test_search_space_multi_gpu():
    """Test search space for multi-GPU."""
    config = TuningConfig()
    search_space = VLLMSearchSpace(config, num_gpus=4)

    params = search_space.get_parameter_names()
    assert "tensor_parallel_size" in params
    assert "pipeline_parallel_size" in params


def test_search_space_get_bounds():
    """Test getting parameter bounds."""
    config = TuningConfig()
    search_space = VLLMSearchSpace(config, num_gpus=1)

    batch_size_bounds = search_space.get_bounds("batch_size")
    assert batch_size_bounds is not None
    assert batch_size_bounds[0] >= 1
    assert batch_size_bounds[1] <= 256


def test_search_space_get_categories():
    """Test getting categorical parameters."""
    config = TuningConfig()
    search_space = VLLMSearchSpace(config, num_gpus=4)

    tp_sizes = search_space.get_categories("tensor_parallel_size")
    assert tp_sizes is not None
    assert 1 in tp_sizes
    assert 2 in tp_sizes


def test_search_space_validate_params():
    """Test parameter validation."""
    config = TuningConfig()
    search_space = VLLMSearchSpace(config, num_gpus=1)

    valid_params = {
        "batch_size": 64,
        "max_num_seqs": 256,
        "gpu_memory_utilization": 0.85,
    }

    assert search_space.validate_params(valid_params) is True


def test_search_space_invalidate_params():
    """Test invalid parameter validation."""
    config = TuningConfig()
    search_space = VLLMSearchSpace(config, num_gpus=1)

    invalid_params = {
        "batch_size": 1000,
    }

    assert search_space.validate_params(invalid_params) is False


def test_search_space_override():
    """Test search space overrides from config."""
    config = TuningConfig()
    config.search_space.batch_size = (10, 100)

    search_space = VLLMSearchSpace(config, num_gpus=1)
    batch_size_bounds = search_space.get_bounds("batch_size")

    assert batch_size_bounds == (10, 100)
