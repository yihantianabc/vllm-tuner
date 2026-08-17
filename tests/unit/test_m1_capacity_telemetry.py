"""Tests for M1 arrival-window queue and client-dispatch evidence."""

from __future__ import annotations

import pytest

from vllm_tuner.longctx.m1_capacity_telemetry import (
    analyze_arrival_window_queue,
    summarize_dispatch_delay,
)


def _queue_row(timestamp_ns: int, waiting: float) -> dict[str, object]:
    return {
        "monotonic_ns": timestamp_ns,
        "available": True,
        "metrics": {"num_requests_waiting": waiting},
    }


def test_queue_slope_uses_tail_of_arrival_window_and_excludes_drain() -> None:
    start = 100_000_000_000
    rows = [_queue_row(start + second * 1_000_000_000, float(second)) for second in range(11)]
    rows.extend(
        _queue_row(start + second * 1_000_000_000, float(20 - second)) for second in range(11, 21)
    )

    evidence = analyze_arrival_window_queue(
        rows,
        measurement_start_monotonic_ns=start,
        arrival_window_seconds=10,
        tail_fraction=0.5,
        minimum_tail_samples=5,
        sample_interval_seconds=1,
    )

    assert evidence.available is True
    assert evidence.window_sample_count == 11
    assert evidence.tail_sample_count == 6
    assert evidence.peak_waiting_requests == 10
    assert evidence.tail_growth_requests == 5
    assert evidence.tail_slope_requests_per_second == pytest.approx(1.0)
    assert evidence.sample_coverage_fraction == pytest.approx(1.0)
    assert evidence.maximum_sample_gap_seconds == pytest.approx(1.0)
    assert evidence.end_coverage_lag_seconds == pytest.approx(0.0)


def test_queue_slope_is_explicitly_unavailable_with_insufficient_samples() -> None:
    evidence = analyze_arrival_window_queue(
        [_queue_row(1_000_000_000, 0)],
        measurement_start_monotonic_ns=1_000_000_000,
        arrival_window_seconds=10,
        minimum_tail_samples=3,
        sample_interval_seconds=1,
    )

    assert evidence.available is False
    assert evidence.tail_slope_requests_per_second is None
    assert evidence.reason == "insufficient tail queue samples: required 3, observed 0"


def test_queue_slope_rejects_nonpositive_or_collapsed_windows() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        analyze_arrival_window_queue(
            [],
            measurement_start_monotonic_ns=0,
            arrival_window_seconds=0,
        )

    collapsed = [
        _queue_row(1_000_000_000, 0),
        _queue_row(1_750_000_000, 0),
        _queue_row(1_750_000_000, 1),
    ]
    evidence = analyze_arrival_window_queue(
        collapsed,
        measurement_start_monotonic_ns=1_000_000_000,
        arrival_window_seconds=1,
        tail_fraction=0.5,
        minimum_tail_samples=2,
        sample_interval_seconds=0.5,
    )
    assert evidence.available is False
    assert evidence.reason == "queue samples do not span positive monotonic time"


def test_queue_slope_fails_closed_on_missing_coverage_or_baseline() -> None:
    start = 10_000_000_000
    sparse = [
        _queue_row(start, 0),
        _queue_row(start + 1_000_000_000, 1),
        _queue_row(start + 9_000_000_000, 9),
        _queue_row(start + 10_000_000_000, 10),
    ]
    coverage = analyze_arrival_window_queue(
        sparse,
        measurement_start_monotonic_ns=start,
        arrival_window_seconds=10,
        tail_fraction=1,
        minimum_tail_samples=2,
        sample_interval_seconds=1,
        minimum_sample_coverage_fraction=0.8,
    )
    assert coverage.available is False
    assert "sample coverage is below policy" in str(coverage.reason)

    missing_baseline = analyze_arrival_window_queue(
        [_queue_row(start + second * 1_000_000_000, second) for second in range(1, 11)],
        measurement_start_monotonic_ns=start,
        arrival_window_seconds=10,
        tail_fraction=1,
        minimum_tail_samples=2,
        sample_interval_seconds=1,
    )
    assert missing_baseline.available is False
    assert "no valid baseline" in str(missing_baseline.reason)


def test_dispatch_delay_reports_percentiles_and_ignores_invalid_rows() -> None:
    rows = [
        {"scheduled_at": 1_000_000_000, "sent_at": 1_000_000_000},
        {"scheduled_at": 1_000_000_000, "sent_at": 1_010_000_000},
        {"scheduled_at": 1_000_000_000, "sent_at": 1_020_000_000},
        {"scheduled_at": 1_000_000_000, "sent_at": None},
        {"scheduled_at": 2_000_000_000, "sent_at": 1_000_000_000},
    ]

    evidence = summarize_dispatch_delay(rows)

    assert evidence.available is True
    assert evidence.sample_count == 3
    assert evidence.p50_ms == pytest.approx(10)
    assert evidence.p95_ms == pytest.approx(19)
    assert evidence.p99_ms == pytest.approx(19.8)
    assert evidence.max_ms == pytest.approx(20)


def test_dispatch_delay_is_unavailable_without_timestamps() -> None:
    evidence = summarize_dispatch_delay([{"status": "failed"}])
    assert evidence.available is False
    assert evidence.sample_count == 0
