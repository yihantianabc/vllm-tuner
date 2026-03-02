"""Unit tests for configuration models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from vllm_tuner.config.models import (
    GPUConfig,
    WeightedObjectives,
    Constraints,
    SearchSpaceOverride,
    WorkloadConfig,
    StudySettings,
    TuningConfig,
)
from vllm_tuner.config.validation import (
    load_yaml_config,
    validate_study_name,
    TunerSettings,
)


def test_gpu_config_default():
    """Test GPUConfig default values."""
    config = GPUConfig()
    assert config.count == 1
    assert config.device_ids == []


def test_gpu_config_validation():
    """Test GPUConfig validates device_ids."""
    config = GPUConfig(device_ids=[0, 1])
    assert config.count == 2
    assert config.device_ids == [0, 1]


def test_weighted_objectives_valid():
    """Test valid weighted objectives."""
    objectives = WeightedObjectives(
        throughput=60,
        latency=30,
        memory=10,
    )
    assert objectives.throughput == 60
    assert objectives.latency == 30
    assert objectives.memory == 10


def test_weighted_objectives_invalid_sum():
    """Test weighted objects must sum to 100."""
    with pytest.raises(ValidationError):
        WeightedObjectives(
            throughput=50,
            latency=50,
            memory=50,
        )


def test_weighted_objectives_single_objective():
    """Test single objective is valid if it sums to 100."""
    objectives = WeightedObjectives(
        throughput=100,
        latency=0,
        memory=0,
    )
    assert objectives.throughput == 100


def test_constraints_validation():
    """Test constraint validation."""
    constraints = Constraints(
        max_latency_ms=100,
        max_memory_utilization=0.9,
    )
    assert constraints.max_latency_ms == 100
    assert constraints.max_memory_utilization == 0.9


def test_constraints_invalid_memory():
    """Test memory utilization must be between 0 and 1."""
    with pytest.raises(ValidationError):
        Constraints(max_memory_utilization=1.5)


def test_search_space_override():
    """Test search space override."""
    override = SearchSpaceOverride(
        batch_size=(1, 128),
    )
    assert override.batch_size == (1, 128)
    assert override.max_num_seqs is None


def test_workload_config_defaults():
    """Test workload config defaults."""
    config = WorkloadConfig()
    assert config.name == "alpaca"
    assert config.dataset_name == "tatsu-lab/alpaca"
    assert config.sample_size == 100


def test_workload_config_validation():
    """Test workload config validation."""
    config = WorkloadConfig(
        sample_size=50,
        concurrent_requests=5,
    )
    assert config.sample_size == 50
    assert config.concurrent_requests == 5


def test_study_settings_defaults():
    """Test study settings defaults."""
    settings = StudySettings()
    assert settings.min_trials == 20
    assert settings.timeout_minutes == 60
    assert settings.prune_enabled is True


def test_tuning_config_defaults():
    """Test tuning config defaults."""
    config = TuningConfig()
    assert config.model == "gpt2"
    assert config.gpu.count == 1
    assert config.objectives.throughput == 60
    assert config.study.min_trials == 20


def test_tuning_config_gpu_validation():
    """Test GPU config implicit validation."""
    config = TuningConfig(
        gpu=GPUConfig(device_ids=[0, 1, 2]),
    )
    assert config.gpu.count == 3


def test_load_yaml_config():
    """Test loading YAML config."""
    config_path = Path("config/default.yaml")
    config = load_yaml_config(config_path)
    assert config.model == "Qwen/Qwen1.5-0.5B-Chat-AWQ"
    assert config.objectives.throughput == 60


def test_validate_study_name():
    """Test study name validation."""
    valid_name = validate_study_name("my_study")
    assert valid_name == "my_study"

    sanitized = validate_study_name("my/study")
    assert "/" not in sanitized


def test_validate_study_name_empty():
    """Test empty study name raises error."""
    with pytest.raises(ValueError):
        validate_study_name("")


def test_tuner_settings():
    """Test tuner settings from env."""
    settings = TunerSettings()
    assert settings.log_level == "INFO"
    assert settings.study_output_dir == "studies"
