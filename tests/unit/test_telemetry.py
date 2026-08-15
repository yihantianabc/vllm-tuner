"""Unit tests for telemetry parsing."""

from pathlib import Path

import pytest

from vllm_tuner.vllm.telemetry import (
    VLLMTelemetryParser,
    parse_vllm_logs,
    parse_vllm_prometheus,
    detect_oom_from_logs,
)


def test_telemetry_parser_empty_logs():
    """Test parsing empty logs."""
    parser = VLLMTelemetryParser()
    parser.parse_log_content("")

    summary = parser.get_summary()
    assert summary["throughput_tokens_per_sec"] == 0.0
    assert summary["oom_detected"] is False


def test_telemetry_parser_throughput():
    """Test parsing throughput from logs."""
    parser = VLLMTelemetryParser()
    log_content = "Throughput: 100.5 tokens/sec"
    parser.parse_log_content(log_content)

    assert parser.metrics["throughput_tokens_per_sec"] == 100.5
    assert parser.metrics["source"] == "log_diagnostic_fallback"
    assert parser.metrics["authoritative"] is False


def test_telemetry_parser_decode_latency():
    """Test parsing decode latency."""
    parser = VLLMTelemetryParser()
    log_content = "decode time per token: 15.3 ms"
    parser.parse_log_content(log_content)

    assert parser.metrics["decode_latency_ms"] == 15.3


def test_telemetry_parser_prefill_latency():
    """Test parsing prefill latency."""
    parser = VLLMTelemetryParser()
    log_content = "prefill time per token: 5.2 ms"
    parser.parse_log_content(log_content)

    assert parser.metrics["prefill_latency_ms"] == 5.2


def test_telemetry_parser_oom_error():
    """Test detecting OOM error."""
    parser = VLLMTelemetryParser()
    log_content = "RuntimeError: CUDA out of memory"
    parser.parse_log_content(log_content)

    assert parser.metrics["oom_detected"] is True


def test_telemetry_parser_multiple_metrics():
    """Test parsing multiple metrics."""
    parser = VLLMTelemetryParser()
    log_content = """
    Throughput: 150.0 tokens/sec
    decode time per token: 12.5 ms
    prefill time per token: 3.2 ms
    """
    parser.parse_log_content(log_content)

    assert parser.metrics["throughput_tokens_per_sec"] == 150.0
    assert parser.metrics["decode_latency_ms"] == 12.5
    assert parser.metrics["prefill_latency_ms"] == 3.2


def test_telemetry_parser_reset():
    """Test resetting metrics."""
    parser = VLLMTelemetryParser()
    parser.parse_log_content("Throughput: 100.0 tokens/sec")

    assert parser.metrics["throughput_tokens_per_sec"] == 100.0

    parser.reset()
    assert parser.metrics["throughput_tokens_per_sec"] == 0.0


def test_parse_vllm_logs_convenience():
    """Test convenience function for parsing logs."""
    result = parse_vllm_logs(Path("/nonexistent.log"))

    assert result["throughput_tokens_per_sec"] == 0.0


def test_detect_oom_from_logs():
    """Test OOM detection function."""
    parser = VLLMTelemetryParser()
    parser.parse_log_content("RuntimeError: CUDA out of memory")

    assert detect_oom_from_logs(Path("/nonexistent.log")) is False

    assert parser.metrics["oom_detected"] is True


def test_prometheus_is_the_authoritative_telemetry_source():
    metrics = parse_vllm_prometheus("""
vllm:num_requests_running 3
vllm:kv_cache_usage_perc 0.75
vllm:num_preemptions_total 4
vllm:inter_token_latency_seconds_sum 0.6
vllm:inter_token_latency_seconds_count 3
""")

    assert metrics["source"] == "prometheus"
    assert metrics["authoritative"] is True
    assert metrics["engine"]["num_requests_running"] == 3.0
    assert metrics["kv_cache_utilization"] == 0.75
    assert metrics["decode_latency_ms"] == pytest.approx(200.0)


def test_generic_runtime_error_is_not_misclassified_as_oom():
    parser = VLLMTelemetryParser()

    parser.parse_log_content("RuntimeError: invalid model configuration")

    assert parser.metrics["runtime_error_detected"] is True
    assert parser.metrics["oom_detected"] is False
