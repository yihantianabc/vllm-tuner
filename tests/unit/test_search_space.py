"""Unit tests for the effective single-GPU search space."""

import pytest

from vllm_tuner.config.models import SearchSpaceOverride, TuningConfig
from vllm_tuner.optimization.search_space import VLLMSearchSpace


def test_search_space_defaults_exclude_invalid_batch_size() -> None:
    space = VLLMSearchSpace(TuningConfig())
    assert space.get_parameter_names() == [
        "gpu_memory_utilization",
        "max_num_seqs",
        "max_num_batched_tokens",
    ]
    assert "batch_size" not in space.get_parameter_names()


def test_search_space_rejects_multi_gpu() -> None:
    with pytest.raises(ValueError, match="exactly one GPU"):
        VLLMSearchSpace(TuningConfig(), num_gpus=4)


def test_search_space_bounds_categories_and_fixed_values() -> None:
    space = VLLMSearchSpace(TuningConfig())
    assert space.get_bounds("gpu_memory_utilization") == (0.60, 0.95)
    assert space.get_categories("max_num_seqs") == [8, 16, 32, 64, 128]
    assert space.get_fixed_params() == {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }


def test_search_space_validate_params() -> None:
    space = VLLMSearchSpace(TuningConfig())
    valid = {
        "gpu_memory_utilization": 0.85,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 2048,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }
    assert space.validate_params(valid)
    assert not space.validate_params({**valid, "max_num_seqs": 256})
    assert not space.validate_params({**valid, "batch_size": 4})


def test_search_space_override_and_checksum_are_stable() -> None:
    config = TuningConfig(
        search_space=SearchSpaceOverride(
            gpu_memory_utilization=(0.7, 0.8),
            max_num_seqs=[8, 32],
            max_num_batched_tokens=[1024, 4096],
        )
    )
    first = VLLMSearchSpace(config)
    second = VLLMSearchSpace(config)
    assert first.get_bounds("gpu_memory_utilization") == (0.7, 0.8)
    assert first.checksum() == second.checksum()


def test_search_space_rejects_fixed_arg_collision() -> None:
    config = TuningConfig(vllm_args={"tensor-parallel-size": 1})
    with pytest.raises(ValueError, match="duplicates"):
        VLLMSearchSpace(config)
