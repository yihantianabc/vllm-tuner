"""Unit tests for SLOTune configuration models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from vllm_tuner.config.models import (
    AdaptivePrefillConfig,
    BaselineConfig,
    Constraints,
    GPUConfig,
    SLOConfig,
    SearchSpaceOverride,
    StudySettings,
    TunerSettings,
    TuningConfig,
    WeightedObjectives,
    WorkloadConfig,
)
from vllm_tuner.config.validation import load_yaml_config, validate_study_name


def test_gpu_config_default() -> None:
    config = GPUConfig()
    assert config.count == 1
    assert config.device_ids == []


def test_top_level_config_resolves_default_gpu() -> None:
    config = TuningConfig()
    assert config.gpu.device_ids == [0]
    assert config.gpu.count == 1


def test_gpu_config_rejects_duplicates() -> None:
    with pytest.raises(ValidationError, match="unique"):
        GPUConfig(device_ids=[0, 0])


def test_legacy_weighted_objectives_validate_but_are_not_accepted_by_tuning_config() -> None:
    with pytest.warns(DeprecationWarning):
        legacy = WeightedObjectives(throughput=60, latency=30, memory=10)
    assert legacy.throughput == 60
    with pytest.raises(ValidationError):
        WeightedObjectives(throughput=50, latency=50, memory=50)
    with pytest.raises(ValidationError):
        TuningConfig(objectives={"throughput": 60, "latency": 30, "memory": 10})


def test_slo_requires_at_least_one_threshold() -> None:
    with pytest.raises(ValidationError, match="At least one"):
        SLOConfig(ttft_ms=None, tpot_ms=None, e2e_ms=None)


def test_constraints_validation() -> None:
    constraints = Constraints(max_error_rate=0.01, max_memory_utilization=0.9)
    assert constraints.max_error_rate == 0.01
    with pytest.raises(ValidationError):
        Constraints(max_memory_utilization=1.5)


def test_search_space_has_no_batch_size() -> None:
    with pytest.raises(ValidationError):
        SearchSpaceOverride(batch_size=(1, 128))


def test_workload_normalizes_concurrency_and_output_length() -> None:
    config = WorkloadConfig(concurrent_requests=5, fixed_output_tokens=17)
    assert config.max_concurrency == 5
    assert config.max_tokens == 17


def test_study_settings_use_equal_trial_budget() -> None:
    settings = StudySettings()
    assert settings.trial_budget == 16
    assert settings.methods == ["default", "random", "tpe"]
    migrated = StudySettings(min_trials=3)
    assert migrated.trial_budget == 3


def test_baseline_can_be_disabled() -> None:
    assert BaselineConfig(enabled=False).enabled is False


def test_adaptive_prefill_config_validates_ordered_caps_and_waits() -> None:
    config = AdaptivePrefillConfig(
        low_prefill_cap=512,
        balanced_prefill_cap=1024,
        high_prefill_cap=2048,
        min_prefill_progress=256,
        oldest_prefill_wait_ms=100,
        max_wait_ms=500,
    )
    assert config.enabled is False
    assert config.decision_log_enabled is True
    assert config.fixed_prefill_cap is None
    with pytest.raises(ValidationError, match="prefill caps"):
        AdaptivePrefillConfig(low_prefill_cap=2048, balanced_prefill_cap=1024)
    with pytest.raises(ValidationError, match="min_prefill_progress"):
        AdaptivePrefillConfig(low_prefill_cap=128, min_prefill_progress=256)
    with pytest.raises(ValidationError, match="max_wait_ms"):
        AdaptivePrefillConfig(oldest_prefill_wait_ms=500, max_wait_ms=100)
    with pytest.raises(ValidationError, match="requires"):
        AdaptivePrefillConfig(enabled=False, fixed_prefill_cap=512)


def test_tuning_config_rejects_multi_gpu() -> None:
    with pytest.raises(ValidationError, match="one GPU"):
        TuningConfig(gpu=GPUConfig(device_ids=[0, 1]))


def test_load_yaml_config() -> None:
    config = load_yaml_config(Path("config/default.yaml"))
    assert config.model.endswith("Qwen2.5-3B-Instruct")
    assert config.slo.ttft_ms == 1000
    assert config.search_space.tensor_parallel_size == 1


def test_validate_study_name() -> None:
    assert validate_study_name("my_study") == "my_study"
    assert "/" not in validate_study_name("my/study")
    with pytest.raises(ValueError):
        validate_study_name("")


def test_tuner_settings() -> None:
    settings = TunerSettings()
    assert settings.log_level == "INFO"
    assert settings.study_output_dir
