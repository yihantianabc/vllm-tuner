"""Tests for Prometheus parsing and vLLM metric window semantics."""

from pathlib import Path

import pytest

from vllm_tuner.profiling.prometheus import (
    PrometheusCollector,
    PrometheusParser,
    PrometheusSnapshot,
    parse_vllm_metrics,
    summarize_prometheus_snapshots,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "prometheus"


def _snapshot(text: str, monotonic_ns: int) -> PrometheusSnapshot:
    return PrometheusSnapshot(
        monotonic_ns=monotonic_ns,
        wall_time=f"2026-08-15T00:00:0{monotonic_ns}+00:00",
        samples=tuple(PrometheusParser().parse(text)),
    )


def test_parser_supports_labels_types_timestamps_and_exemplars() -> None:
    text = r"""# TYPE example_counter counter
example_counter{model="a\"b",path="line\nnext"} 1.5 123 # {trace_id="abc"} 1
"""
    samples = PrometheusParser(strict=True).parse(text)

    assert len(samples) == 1
    assert samples[0].name == "example_counter"
    assert samples[0].labels == {"model": 'a"b', "path": "line\nnext"}
    assert samples[0].metric_type == "counter"
    assert samples[0].timestamp_ms == 123


def test_current_vllm_metrics_parse_gauges_counters_and_histograms() -> None:
    metrics = parse_vllm_metrics((FIXTURES / "vllm_current.prom").read_text())

    assert metrics["num_requests_running"] == 3.0
    assert metrics["num_requests_waiting"] == 2.0
    assert metrics["kv_cache_usage_perc"] == 0.75
    assert metrics["generation_tokens_total"] == 600.0
    assert metrics["time_to_first_token_seconds"]["count"] == 10.0
    assert metrics["time_to_first_token_seconds"]["mean"] == pytest.approx(0.24)


def test_legacy_aliases_map_to_stable_keys_and_missing_is_none() -> None:
    metrics = parse_vllm_metrics((FIXTURES / "vllm_legacy.prom").read_text())

    assert metrics["num_requests_waiting"] == 4.0
    assert metrics["kv_cache_usage_perc"] == 0.9
    assert metrics["num_preemptions_total"] == 3.0
    assert metrics["inter_token_latency_seconds"]["count"] == 5.0
    assert metrics["e2e_request_latency_seconds"] is None


def test_snapshot_summary_uses_counter_window_delta_not_process_total() -> None:
    before = """
vllm:prompt_tokens_total 100
vllm:generation_tokens_total 50
vllm:num_preemptions_total 7
vllm:num_requests_running 1
"""
    after = """
vllm:prompt_tokens_total 140
vllm:generation_tokens_total 72
vllm:num_preemptions_total 9
vllm:num_requests_running 3
"""
    summary = summarize_prometheus_snapshots([_snapshot(before, 1), _snapshot(after, 2)])

    assert summary["prompt_tokens_total"]["delta"] == 40.0
    assert summary["generation_tokens_total"]["delta"] == 22.0
    assert summary["num_preemptions_total"]["delta"] == 2.0
    assert summary["num_requests_running"]["peak"] == 3.0
    assert summary["prefix_cache_hits"]["delta"] is None


def test_histogram_summary_is_scoped_to_window() -> None:
    before = """
vllm:request_queue_time_seconds_bucket{le="0.1"} 10
vllm:request_queue_time_seconds_bucket{le="+Inf"} 20
vllm:request_queue_time_seconds_sum 2
vllm:request_queue_time_seconds_count 20
"""
    after = """
vllm:request_queue_time_seconds_bucket{le="0.1"} 14
vllm:request_queue_time_seconds_bucket{le="+Inf"} 25
vllm:request_queue_time_seconds_sum 3
vllm:request_queue_time_seconds_count 25
"""
    summary = summarize_prometheus_snapshots([_snapshot(before, 1), _snapshot(after, 2)])
    queue = summary["request_queue_time_seconds"]

    assert queue["count"] == 5.0
    assert queue["sum"] == 1.0
    assert queue["mean"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_collector_returns_explicit_unavailable_snapshot_on_fetch_error() -> None:
    async def fail() -> str:
        raise ConnectionError("server stopped")

    collector = PrometheusCollector("http://127.0.0.1:8000", fetcher=fail)
    snapshot = await collector.collect()

    assert collector.endpoint == "http://127.0.0.1:8000/metrics"
    assert snapshot.endpoint_available is False
    assert snapshot.metrics["num_requests_running"] is None
    assert "ConnectionError" in snapshot.error
