"""Unit tests for study manager."""

from unittest.mock import Mock

import pytest

from vllm_tuner.config.models import TuningConfig, GPUConfig, WorkloadConfig
from vllm_tuner.tuner.study_manager import StudyManager


class TestStudyManager:
    """Tests for StudyManager class."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return TuningConfig(
            model="gpt2",
            gpu=GPUConfig(device_ids=[0], count=1),
            workload=WorkloadConfig(
                sample_size=10,
                concurrent_requests=5,
                warmup_requests=2,
            ),
        )

    @pytest.fixture
    def sample_study_description(self):
        """Create sample Optuna study description."""
        return """
        Study description:
        - Name: test_study
        - Direction: maximize
        - Sampler: TPESampler
        """

    def test_study_manager_initialization(self, sample_config, tmp_path):
        """Test StudyManager initialization."""
        manager = StudyManager(sample_config, "test_study", tmp_path)

        assert manager.config == sample_config
        assert manager.study_name == "test_study"
        assert manager.output_dir == tmp_path
        assert manager._prompts is None

    def test_average_memory_calculation(self, sample_config, tmp_path):
        """Test that average memory is calculated from GPU history."""
        manager = StudyManager(sample_config, "test_study", tmp_path)

        # Mock GPU collector with history data
        # Simulate 3 GPUs with memory samples
        gpu1_samples = [
            Mock(memory_used_mb=4000.0),
            Mock(memory_used_mb=4500.0),
            Mock(memory_used_mb=5000.0),
        ]
        gpu2_samples = [
            Mock(memory_used_mb=4200.0),
            Mock(memory_used_mb=4600.0),
            Mock(memory_used_mb=4800.0),
        ]

        # Set up mock history
        manager.gpu_collector.history = {
            0: gpu1_samples,
            1: gpu2_samples,
        }

        # Simulate the calculation done in _run_benchmark
        all_memory_samples = []
        for device_id, history in manager.gpu_collector.history.items():
            if history:
                for stats in history:
                    all_memory_samples.append(stats.memory_used_mb)

        average_memory = (
            sum(all_memory_samples) / len(all_memory_samples) if all_memory_samples else 0
        )

        expected_average = (4000 + 4500 + 5000 + 4200 + 4600 + 4800) / 6
        assert average_memory == expected_average
        assert average_memory == 4516.666666666667

    def test_average_memory_with_empty_history(self, sample_config, tmp_path):
        """Test average memory calculation with empty GPU history."""
        manager = StudyManager(sample_config, "test_study", tmp_path)

        # Empty history
        manager.gpu_collector.history = {}

        all_memory_samples = []
        for device_id, history in manager.gpu_collector.history.items():
            if history:
                for stats in history:
                    all_memory_samples.append(stats.memory_used_mb)

        average_memory = (
            sum(all_memory_samples) / len(all_memory_samples) if all_memory_samples else 0
        )

        assert average_memory == 0.0

    def test_average_memory_with_single_gpu(self, sample_config, tmp_path):
        """Test average memory calculation with single GPU."""
        manager = StudyManager(sample_config, "test_study", tmp_path)

        # Single GPU with samples
        gpu_samples = [
            Mock(memory_used_mb=3000.0),
            Mock(memory_used_mb=3500.0),
            Mock(memory_used_mb=4000.0),
        ]

        manager.gpu_collector.history = {0: gpu_samples}

        all_memory_samples = []
        for device_id, history in manager.gpu_collector.history.items():
            if history:
                for stats in history:
                    all_memory_samples.append(stats.memory_used_mb)

        average_memory = (
            sum(all_memory_samples) / len(all_memory_samples) if all_memory_samples else 0
        )

        expected_average = (3000 + 3500 + 4000) / 3
        assert average_memory == expected_average
        assert average_memory == 3500.0

    def test_gpu_history_processing_ignores_empty_samples(self, sample_config, tmp_path):
        """Test that empty GPU samples in history are handled correctly."""
        manager = StudyManager(sample_config, "test_study", tmp_path)

        # Mix of GPUs with and without samples
        gpu1_samples = [Mock(memory_used_mb=4000.0), Mock(memory_used_mb=5000.0)]
        gpu2_samples = []  # Empty
        gpu3_samples = [Mock(memory_used_mb=4500.0)]

        manager.gpu_collector.history = {
            0: gpu1_samples,
            1: gpu2_samples,
            2: gpu3_samples,
        }

        all_memory_samples = []
        for device_id, history in manager.gpu_collector.history.items():
            if history:  # This check prevents including empty lists
                for stats in history:
                    all_memory_samples.append(stats.memory_used_mb)

        average_memory = (
            sum(all_memory_samples) / len(all_memory_samples) if all_memory_samples else 0
        )

        # Should only include GPU 0 and GPU 2
        expected_average = (4000 + 5000 + 4500) / 3
        assert average_memory == expected_average
        assert len(all_memory_samples) == 3

    def test_get_study_summary(self, sample_config, tmp_path):
        """Test get_study_summary method."""
        manager = StudyManager(sample_config, "test_study", tmp_path)

        # Mock the optimizer study
        manager.optimizer.study = Mock()
        manager.optimizer.study.trials = []
        manager.optimizer.get_best_result = Mock(return_value={"test": "result"})
        manager.optimizer.get_top_n_results = Mock(return_value=[])

        summary = manager.get_study_summary()

        assert summary["study_name"] == "test_study"
        assert summary["num_trials"] == 0
        assert summary["best_trial"] == {"test": "result"}

    def test_get_best_config_with_results(self, sample_config, tmp_path):
        """Test get_best_config when results are available."""
        manager = StudyManager(sample_config, "test_study", tmp_path)

        manager.optimizer.get_best_result = Mock(
            return_value={
                "parameters": {"batch_size": 64},
                "metrics": {"throughput_requests_per_sec": 100.0},
            }
        )

        config = manager.get_best_config()

        assert config is not None
        assert config["model"] == "gpt2"
        assert config["vllm_params"] == {"batch_size": 64}
        assert "metrics" in config

    def test_get_best_config_without_results(self, sample_config, tmp_path):
        """Test get_best_config when no results are available."""
        manager = StudyManager(sample_config, "test_study", tmp_path)

        manager.optimizer.get_best_result = Mock(return_value={})

        config = manager.get_best_config()

        assert config is None


def test_combined_metrics_include_average_memory():
    """Test that combined metrics include average_memory_mb."""
    from vllm_tuner.config import TuningConfig, GPUConfig, WorkloadConfig
    from unittest.mock import Mock

    config = TuningConfig(
        model="gpt2",
        gpu=GPUConfig(device_ids=[0], count=1),
        workload=WorkloadConfig(),
    )

    # Mock the combined metrics generation
    metrics = {"throughput_requests_per_sec": 100.0}
    aggregate_stats = {"aggregate_memory_utilization": 0.85}
    telemetry = {}
    initial_memory = 2000.0
    final_memory = 5000.0
    params = {"batch_size": 64}

    # Simulate GPU history with samples
    gpu_samples = [Mock(memory_used_mb=4000.0), Mock(memory_used_mb=5000.0)]
    all_memory_samples = [s.memory_used_mb for s in gpu_samples]
    average_memory = sum(all_memory_samples) / len(all_memory_samples) if all_memory_samples else 0

    combined_metrics = {
        **metrics,
        **aggregate_stats,
        **telemetry,
        "initial_memory_mb": initial_memory,
        "peak_memory_mb": final_memory,
        "average_memory_mb": average_memory,  # This is the key field
        "memory_delta_mb": final_memory - initial_memory,
        "vllm_params": params,
    }

    assert "average_memory_mb" in combined_metrics
    assert combined_metrics["average_memory_mb"] == 4500.0
    assert combined_metrics["peak_memory_mb"] == 5000.0
    assert combined_metrics["memory_delta_mb"] == 3000.0
