"""Unit tests for baseline runner module."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from src.config.models import TuningConfig, GPUConfig, WorkloadConfig, BaselineConfig
from src.baseline.runner import BaselineMetrics, VLLMBaselineRunner


class TestBaselineMetrics:
    """Tests for BaselineMetrics dataclass."""

    def test_baseline_metrics_initialization(self):
        """Test BaselineMetrics initialization."""
        metrics = BaselineMetrics(
            model="gpt2",
            timestamp="2024-02-26T00:00:00",
            vllm_params={"batch_size": 64},
            benchmark_params={"num_requests": 100},
        )

        assert metrics.model == "gpt2"
        assert metrics.timestamp == "2024-02-26T00:00:00"
        assert metrics.vllm_params == {"batch_size": 64}
        assert len(metrics.memory_samples) == 0
        assert len(metrics.gpu_util_samples) == 0

    def test_baseline_metrics_to_dict_empty_samples(self):
        """Test to_dict with empty samples."""
        metrics = BaselineMetrics(
            model="gpt2",
            timestamp="2024-02-26T00:00:00",
            vllm_params={},
            benchmark_params={},
        )

        result = metrics.to_dict()

        assert result["model"] == "gpt2"
        assert result["metrics"]["peak_memory_mb"] == 0
        assert result["metrics"]["average_memory_mb"] == 0
        assert result["metrics"]["average_gpu_utilization"] == 0

    def test_baseline_metrics_to_dict_with_samples(self):
        """Test to_dict with samples."""
        metrics = BaselineMetrics(
            model="gpt2",
            timestamp="2024-02-26T00:00:00",
            vllm_params={},
            benchmark_params={},
        )

        metrics.memory_samples = [1000.0, 1500.0, 2000.0]
        metrics.gpu_util_samples = [0.5, 0.6, 0.7]

        result = metrics.to_dict()

        assert result["metrics"]["peak_memory_mb"] == 2000.0
        assert result["metrics"]["average_memory_mb"] == 1500.0
        assert result["metrics"]["average_gpu_utilization"] == 0.6


class TestVLLMBaselineRunner:
    """Tests for VLLMBaselineRunner class."""

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

    def test_vllm_params_full_construction(self, sample_config, tmp_path):
        """Test that vllm_params includes all required parameters."""
        runner = VLLMBaselineRunner(sample_config, tmp_path)

        # Verify all required parameters are present
        assert "gpu_memory_utilization" in runner.metrics.vllm_params
        assert "max_num_seqs" in runner.metrics.vllm_params
        assert "max_num_batched_tokens" in runner.metrics.vllm_params
        assert "tensor_parallel_size" in runner.metrics.vllm_params

        # Verify types
        assert isinstance(runner.metrics.vllm_params["gpu_memory_utilization"], str)
        assert isinstance(runner.metrics.vllm_params["max_num_seqs"], int)
        assert isinstance(runner.metrics.vllm_params["tensor_parallel_size"], int)

    def test_prompt_slicing_with_warmup(self, sample_config, tmp_path):
        """Test that warmup and main prompts don't overlap."""
        runner = VLLMBaselineRunner(sample_config, tmp_path)
        runner.warmup_requests = 5
        runner.num_requests = 10
        runner.dataset_name = "tatsu-lab/alpaca"

        # Mock dataset loader
        with patch("src.baseline.runner.load_dataset") as mock_load:
            # Create sample data
            mock_data = [{"instruction": f"prompt {i}"} for i in range(20)]
            mock_load.return_value = mock_data

            prompts = runner._load_prompts()

            # Total should be warmup + num_requests
            total_needed = runner.warmup_requests + runner.num_requests
            assert len(prompts) == total_needed

    def test_gpu_collector_initialization_check(self, sample_config, tmp_path):
        """Test that GPU collector initialization is checked before use."""
        runner = VLLMBaselineRunner(sample_config, tmp_path)

        # Mark collector as not initialized
        runner.gpu_collector.initialized = False

        # This should not raise an error
        runner._generate_outputs()

        # GPU info should be empty when not initialized
        assert runner.metrics.gpu_info == {}

    def test_division_by_zero_in_text_summary(self, sample_config, tmp_path):
        """Test that division by zero is handled in utilization calculation."""
        runner = VLLMBaselineRunner(sample_config, tmp_path)

        # Set GPU info with zero total memory
        runner.metrics.gpu_info = {"total_memory_mb": 0}
        runner.metrics.metrics = {
            "peak_memory_mb": 1000,
        }

        # This should not raise ZeroDivisionError
        runner._generate_text_summary()

        # Verify text summary file was created
        summary_file = tmp_path / "baseline_summary.txt"
        assert summary_file.exists()

    def test_yaml_output_structure(self, sample_config, tmp_path):
        """Test that YAML output has the correct structure."""
        runner = VLLMBaselineRunner(sample_config, tmp_path)
        runner.metrics.metrics = {
            "throughput_requests_per_sec": 100.0,
            "avg_latency_ms": 50.0,
            "p50_latency_ms": 45.0,
            "p95_latency_ms": 70.0,
            "p99_latency_ms": 90.0,
        }
        runner.metrics.memory_samples = [1000.0, 1500.0, 2000.0]
        runner.metrics.gpu_util_samples = [0.5, 0.6, 0.7]

        runner._generate_outputs()

        # Check YAML file
        yaml_file = tmp_path / "baseline_config.yaml"
        assert yaml_file.exists()

        with open(yaml_file) as f:
            import yaml

            data = yaml.safe_load(f)

        # Verify structure
        assert "baseline_metrics" in data
        assert data["baseline_metrics"]["throughput_requests_per_sec"] == 100.0
        assert data["baseline_metrics"]["avg_latency_ms"] == 50.0
        assert "model" in data
        assert "timestamp" in data
        assert "vllm_params" in data

    def test_json_output_structure(self, sample_config, tmp_path):
        """Test that JSON output has the correct structure."""
        runner = VLLMBaselineRunner(sample_config, tmp_path)
        runner._generate_outputs()

        # Check JSON file
        json_file = tmp_path / "baseline_metrics.json"
        assert json_file.exists()

        with open(json_file) as f:
            data = json.load(f)

        # Verify structure
        assert "model" in data
        assert "timestamp" in data
        assert "metrics" in data
        assert "gpu_info" in data
        assert "configuration" in data


def test_baseline_comparison_metrics_included():
    """Test that all metrics needed for comparison are included."""
    from src.baseline.runner import BaselineMetrics

    metrics = BaselineMetrics(
        model="gpt2",
        timestamp="2024-02-26T00:00:00",
        vllm_params={},
        benchmark_params={},
    )

    metrics.metrics = {
        "throughput_requests_per_sec": 100.0,
        "throughput_tokens_per_sec": 5000.0,
        "avg_latency_ms": 50.0,
        "p50_latency_ms": 45.0,
        "p95_latency_ms": 70.0,
        "p99_latency_ms": 90.0,
    }

    result = metrics.to_dict()

    # Verify all comparison metrics are present
    assert "throughput_requests_per_sec" in result["metrics"]
    assert "avg_latency_ms" in result["metrics"]
    assert "p95_latency_ms" in result["metrics"]
    assert "p99_latency_ms" in result["metrics"]
    assert "average_memory_mb" in result["metrics"]
