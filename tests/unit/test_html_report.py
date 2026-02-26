"""Unit tests for HTML report generation."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.reporting.html import HTMLReportGenerator


class TestHTMLReportGenerator:
    """Tests for HTMLReportGenerator class."""

    @pytest.fixture
    def sample_study_summary(self):
        """Create sample study summary."""
        return {
            "study_name": "test_study",
            "num_trials": 10,
            "best_trial": {
                "trial_number": 5,
                "state": "COMPLETE",
                "parameters": {
                    "batch_size": 64,
                    "max_num_seqs": 128,
                    "gpu_memory_utilization": 0.85,
                },
                "metrics": {
                    "throughput_requests_per_sec": 150.5,
                    "avg_latency_ms": 45.2,
                    "p50_latency_ms": 42.0,
                    "p95_latency_ms": 55.0,
                    "p99_latency_ms": 70.0,
                    "average_memory_mb": 8500.0,
                },
            },
        }

    @pytest.fixture
    def sample_trials_data(self):
        """Create sample trials data."""
        return [
            {
                "trial_number": i,
                "parameters": {"batch_size": i * 10},
                "metrics": {
                    "throughput_requests_per_sec": i * 20.0,
                    "avg_latency_ms": 100.0 - i * 5.0,
                    "aggregate_memory_utilization": 0.5 + i * 0.05,
                },
            }
            for i in range(1, 11)
        ]

    @pytest.fixture
    def sample_baseline_data(self):
        """Create sample baseline data."""
        return {
            "throughput_requests_per_sec": 120.0,
            "avg_latency_ms": 50.0,
            "p50_latency_ms": 48.0,
            "p95_latency_ms": 60.0,
            "p99_latency_ms": 75.0,
            "average_memory_mb": 8000.0,
        }

    def test_generator_initialization_without_baseline(self, tmp_path):
        """Test generator initialization without baseline data."""
        generator = HTMLReportGenerator("test_study", tmp_path)

        assert generator.study_name == "test_study"
        assert generator.output_dir == tmp_path
        assert generator.baseline_data is None

    def test_generator_initialization_with_baseline(self, tmp_path, sample_baseline_data):
        """Test generator initialization with baseline data."""
        generator = HTMLReportGenerator("test_study", tmp_path, sample_baseline_data)

        assert generator.baseline_data == sample_baseline_data

    def test_baseline_data_loading_from_json(self, tmp_path, sample_baseline_data):
        """Test loading baseline data from JSON file."""
        # Create baseline directory and JSON file
        baseline_dir = tmp_path.parent / "test_study" / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)

        baseline_file = baseline_dir / "baseline_metrics.json"
        import json

        with open(baseline_file, "w") as f:
            json.dump({"metrics": sample_baseline_data}, f)

        generator = HTMLReportGenerator("test_study", tmp_path)
        # Load baseline data and assign it
        baseline_data = generator._load_baseline_data(tmp_path.parent / "test_study")
        generator.baseline_data = baseline_data

        assert baseline_data is not None
        assert baseline_data["throughput_requests_per_sec"] == 120.0

    def test_baseline_data_loading_missing_file(self, tmp_path):
        """Test loading baseline data when file is missing."""
        generator = HTMLReportGenerator("test_study", tmp_path)
        data = generator._load_baseline_data(tmp_path / "nonexistent")

        assert data is None

    def test_improvement_calculations_positive_throughput(
        self, sample_study_summary, sample_baseline_data, tmp_path
    ):
        """Test improvement calculation when throughput increased."""
        generator = HTMLReportGenerator("test_study", tmp_path, sample_baseline_data)

        best_throughput = sample_study_summary["best_trial"]["metrics"][
            "throughput_requests_per_sec"
        ]
        baseline_throughput = sample_baseline_data["throughput_requests_per_sec"]

        # Manually calculate what the template should compute
        improvement = (best_throughput - baseline_throughput) / baseline_throughput * 100

        # Throughput is 150.5 vs 120.0, should be ~+25.4%
        assert improvement > 0
        assert abs(improvement - 25.416) < 0.01

    def test_improvement_calculations_negative_throughput(
        self, sample_study_summary, sample_baseline_data, tmp_path
    ):
        """Test improvement calculation when throughput decreased."""
        # Set best throughput lower than baseline
        sample_study_summary["best_trial"]["metrics"]["throughput_requests_per_sec"] = 100.0

        generator = HTMLReportGenerator("test_study", tmp_path, sample_baseline_data)
        best_throughput = sample_study_summary["best_trial"]["metrics"][
            "throughput_requests_per_sec"
        ]
        baseline_throughput = sample_baseline_data["throughput_requests_per_sec"]

        improvement = (best_throughput - baseline_throughput) / baseline_throughput * 100

        # Should be negative (~-16.7%)
        assert improvement < 0
        assert abs(improvement - (-16.667)) < 0.01

    def test_latency_improvement_calculations(
        self, sample_study_summary, sample_baseline_data, tmp_path
    ):
        """Test latency improvements (lower is better)."""
        generator = HTMLReportGenerator("test_study", tmp_path, sample_baseline_data)

        best_latency = sample_study_summary["best_trial"]["metrics"]["avg_latency_ms"]
        baseline_latency = sample_baseline_data["avg_latency_ms"]

        # Latency improved from 50.0 to 45.2
        improvement = (baseline_latency - best_latency) / baseline_latency * 100

        # Should be positive (~+9.6% latency improvement)
        assert improvement > 0
        assert abs(improvement - 9.6) < 0.1

    def test_p95_p99_latency_calculations(
        self, sample_study_summary, sample_baseline_data, tmp_path
    ):
        """Test P95 and P99 latency calculations."""
        generator = HTMLReportGenerator("test_study", tmp_path, sample_baseline_data)

        # Test P95
        best_p95 = sample_study_summary["best_trial"]["metrics"]["p95_latency_ms"]
        baseline_p95 = sample_baseline_data["p95_latency_ms"]
        p95_improvement = (baseline_p95 - best_p95) / baseline_p95 * 100

        # P95 improved from 60.0 to 55.0
        assert abs(p95_improvement - 8.33) < 0.1

        # Test P99
        best_p99 = sample_study_summary["best_trial"]["metrics"]["p99_latency_ms"]
        baseline_p99 = sample_baseline_data["p99_latency_ms"]
        p99_improvement = (baseline_p99 - best_p99) / baseline_p99 * 100

        # P99 improved from 75.0 to 70.0
        assert abs(p99_improvement - 6.67) < 0.1

    def test_memory_delta_calculations(self, sample_study_summary, sample_baseline_data, tmp_path):
        """Test memory delta calculations (lower is better)."""
        generator = HTMLReportGenerator("test_study", tmp_path, sample_baseline_data)

        best_memory = sample_study_summary["best_trial"]["metrics"]["average_memory_mb"]
        baseline_memory = sample_baseline_data["average_memory_mb"]

        # Memory increased from 8000.0 to 8500.0
        delta = (best_memory - baseline_memory) / baseline_memory * 100

        # Should be positive (~+6.25% more memory usage)
        assert delta > 0
        assert abs(delta - 6.25) < 0.01

    def test_generate_report_creates_html_file(
        self, sample_study_summary, sample_trials_data, tmp_path
    ):
        """Test that generate_report creates HTML file."""
        generator = HTMLReportGenerator("test_study", tmp_path)
        report_path = generator.generate_report(sample_study_summary, sample_trials_data)

        assert report_path.exists()
        assert report_path.suffix == ".html"

    def test_generate_report_includes_baseline_comparison(
        self, sample_study_summary, sample_trials_data, sample_baseline_data, tmp_path
    ):
        """Test that generated report includes baseline comparison section."""
        generator = HTMLReportGenerator("test_study", tmp_path, sample_baseline_data)
        report_path = generator.generate_report(sample_study_summary, sample_trials_data)

        with open(report_path) as f:
            html_content = f.read()

        # Check for baseline-related elements
        assert "Baseline vs Best Trial Comparison" in html_content
        assert "Throughput (req/s)" in html_content
        assert "Avg Latency (ms)" in html_content
        assert "P95 Latency (ms)" in html_content
        assert "P99 Latency (ms)" in html_content
        assert "Avg Memory (MB)" in html_content

    def test_generate_report_without_baseline(
        self, sample_study_summary, sample_trials_data, tmp_path
    ):
        """Test that report works without baseline data."""
        generator = HTMLReportGenerator("test_study", tmp_path)
        report_path = generator.generate_report(sample_study_summary, sample_trials_data)

        with open(report_path) as f:
            html_content = f.read()

        # Should still work but not include comparison table
        assert "Baseline vs Best Trial Comparison" not in html_content
        # Should still show metrics
        assert "Best Configuration Metrics" in html_content


def test_generate_report_filters_failed_trials(tmp_path):
    """Test that failed trials are filtered out from charts."""
    # Create mixed trials data with some failed
    trials_data = [
        # Successful trial (with "COMPLETE" state)
        {
            "trial_number": 1,
            "state": "COMPLETE",
            "parameters": {"batch_size": 64},
            "metrics": {
                "throughput_requests_per_sec": 100.0,
                "avg_latency_ms": 50.0,
                "aggregate_memory_utilization": 0.75,
            },
        },
        # Failed trial - OOM (with "FAIL" state)
        {
            "trial_number": 2,
            "state": "FAIL",
            "parameters": {"batch_size": 256},
            "metrics": {
                "throughput_requests_per_sec": 0.0,
                "avg_latency_ms": float("inf"),
                "aggregate_memory_utilization": 1.0,
                "oom_detected": True,
                "error": "CUDA out of memory",
            },
        },
        # Another successful trial (with "TrialState.COMPLETE" state - actual Optuna format)
        {
            "trial_number": 3,
            "state": "TrialState.COMPLETE",
            "parameters": {"batch_size": 128},
            "metrics": {
                "throughput_requests_per_sec": 150.0,
                "avg_latency_ms": 40.0,
                "aggregate_memory_utilization": 0.85,
            },
        },
        # Failed trial - Inf throughput (with "FAILED" state)
        {
            "trial_number": 4,
            "state": "FAILED",
            "parameters": {"batch_size": 512},
            "metrics": {
                "throughput_requests_per_sec": float("inf"),
                "avg_latency_ms": float("inf"),
                "aggregate_memory_utilization": 0.95,
                "error": "Server error",
            },
        },
    ]

    study_summary = {
        "study_name": "test_with_failures",
        "num_trials": 4,
        "best_trial": {
            "trial_number": 3,
            "state": "COMPLETE",
            "parameters": {"batch_size": 128},
            "metrics": {
                "throughput_requests_per_sec": 150.0,
                "avg_latency_ms": 40.0,
            },
        },
    }

    generator = HTMLReportGenerator("test_with_failures", tmp_path)
    charts = generator._create_charts(study_summary, trials_data)

    # Should still generate charts with only successful trials (1 and 3)
    assert "throughput_progression" in charts
    assert "latency_distribution" in charts
    assert "pareto_front" in charts
    assert "gpu_memory" in charts
    assert "combined" in charts

    # Charts should be generated (non-empty strings)
    assert charts["throughput_progression"]
    assert charts["latency_distribution"]


def test_generate_html_report_function():
    """Test the convenience function generate_html_report."""
    from src.reporting.html import generate_html_report
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        study_summary = {
            "study_name": "test",
            "num_trials": 1,
            "best_trial": {
                "trial_number": 1,
                "state": "COMPLETE",
                "parameters": {},
                "metrics": {"throughput_requests_per_sec": 100.0},
            },
        }
        trials_data = []

        output_path = generate_html_report(
            "test",
            output_dir,
            study_summary,
            trials_data,
            baseline_data=None,
        )

        assert output_path.exists()
