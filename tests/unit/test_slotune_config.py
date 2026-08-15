"""Tests for SLO-first configuration and effective search parameters."""

import pytest
from pydantic import ValidationError

from vllm_tuner.config.models import (
    GPUConfig,
    SearchSpaceOverride,
    TuningConfig,
    WorkloadConfig,
)
from vllm_tuner.optimization.search_space import VLLMSearchSpace


def test_legacy_batch_size_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchSpaceOverride(batch_size=(1, 8))


def test_weighted_objectives_are_not_top_level_configuration() -> None:
    with pytest.raises(ValidationError):
        TuningConfig(objectives={"throughput": 60, "latency": 30, "memory": 10})


def test_core_rejects_multiple_gpus() -> None:
    with pytest.raises(ValidationError, match="one GPU"):
        TuningConfig(gpu=GPUConfig(device_ids=[0, 1]))


def test_workload_normalizes_legacy_concurrency() -> None:
    workload = WorkloadConfig(concurrent_requests=7)
    assert workload.max_concurrency == 7
    assert workload.concurrent_requests == 7


def test_capacity_sweep_rates_are_finite_positive_and_unique() -> None:
    workload = WorkloadConfig(capacity_request_rates=[1, 2.5, 8], capacity_repeats=3)
    assert workload.capacity_request_rates == [1.0, 2.5, 8.0]
    assert workload.capacity_repeats == 3

    for invalid in ([0.0], [-1.0], [float("inf")], [2.0, 2.0]):
        with pytest.raises(ValidationError):
            WorkloadConfig(capacity_request_rates=invalid)


def test_nested_configuration_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        WorkloadConfig(batch_size=8)


def test_search_space_contains_only_effective_tunable_parameters() -> None:
    space = VLLMSearchSpace(TuningConfig())
    assert space.get_parameter_names() == [
        "gpu_memory_utilization",
        "max_num_seqs",
        "max_num_batched_tokens",
    ]
    assert space.get_fixed_params() == {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }


def test_search_space_rejects_vllm_argument_collision() -> None:
    config = TuningConfig(vllm_args={"max-num-seqs": 16})
    with pytest.raises(ValueError, match="duplicates"):
        VLLMSearchSpace(config)
