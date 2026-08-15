"""Tests for timestamp, aggregation, and JSONL helpers."""

import json
from datetime import datetime, timezone

import pytest

from vllm_tuner.profiling.timeseries import (
    capture_timestamp,
    counter_window_delta,
    integrate_power_samples,
    percentile,
    summarize_values,
    write_jsonl,
)


def test_capture_timestamp_preserves_both_clocks() -> None:
    timestamp = capture_timestamp(
        lambda: 123_000_000,
        lambda: datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc),
    )

    assert timestamp.monotonic_ns == 123_000_000
    assert timestamp.wall_time == "2026-08-15T12:00:00+00:00"


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)
    assert percentile([], 0.95) is None


def test_summarize_values_distinguishes_zero_from_unavailable() -> None:
    missing = summarize_values([None, None])
    measured_zero = summarize_values([0.0, 0.0])

    assert missing == {
        "available": False,
        "count": 0,
        "mean": None,
        "peak": None,
        "max": None,
        "p95": None,
    }
    assert measured_zero["available"] is True
    assert measured_zero["mean"] == 0.0
    assert measured_zero["peak"] == 0.0


def test_counter_window_delta_handles_reset() -> None:
    summary = counter_window_delta([100, 110, 3, 8])

    assert summary["available"] is True
    assert summary["delta"] == pytest.approx(18.0)
    assert summary["reset_count"] == 1
    assert counter_window_delta([100])["delta"] is None


def test_integrate_power_samples_uses_monotonic_time() -> None:
    # 100 W -> 200 W over 1 second, then 200 W for another second.
    energy = integrate_power_samples([(0, 100.0), (1_000_000_000, 200.0), (2_000_000_000, 200.0)])

    assert energy == pytest.approx(350.0)
    assert integrate_power_samples([(0, 100.0)]) is None


def test_write_jsonl_serializes_records_and_creates_parent(tmp_path) -> None:
    path = tmp_path / "nested" / "telemetry.jsonl"
    write_jsonl(path, [{"value": 1}, {"value": 2}])

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [{"value": 1}, {"value": 2}]
