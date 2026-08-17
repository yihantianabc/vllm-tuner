"""Arrival-window queue growth and client-dispatch evidence for M1 capacity."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional, TypedDict

from pydantic import Field, model_validator

from vllm_tuner.profiling.timeseries import percentile

from .kv_capacity_planner import StrictFrozenModel


class _QueueEvidenceCommon(TypedDict):
    measurement_start_monotonic_ns: int
    arrival_window_end_monotonic_ns: int
    arrival_window_seconds: float
    tail_fraction: float
    window_sample_count: int
    tail_sample_count: int
    expected_window_sample_count: int


class ArrivalWindowQueueEvidence(StrictFrozenModel):
    available: bool
    reason: Optional[str] = None
    method: Literal["tail-ols-v1"] = "tail-ols-v1"
    measurement_start_monotonic_ns: int = Field(ge=0)
    arrival_window_end_monotonic_ns: int = Field(ge=0)
    arrival_window_seconds: float = Field(gt=0)
    tail_fraction: float = Field(gt=0, le=1)
    window_sample_count: int = Field(ge=0)
    tail_sample_count: int = Field(ge=0)
    expected_window_sample_count: int = Field(gt=0)
    sample_coverage_fraction: Optional[float] = Field(default=None, ge=0)
    covered_span_seconds: Optional[float] = Field(default=None, ge=0)
    maximum_sample_gap_seconds: Optional[float] = Field(default=None, ge=0)
    end_coverage_lag_seconds: Optional[float] = Field(default=None, ge=0)
    waiting_positive_sample_fraction: Optional[float] = Field(default=None, ge=0, le=1)
    peak_waiting_requests: Optional[float] = Field(default=None, ge=0)
    tail_start_waiting_requests: Optional[float] = Field(default=None, ge=0)
    tail_end_waiting_requests: Optional[float] = Field(default=None, ge=0)
    tail_growth_requests: Optional[float] = None
    tail_slope_requests_per_second: Optional[float] = None

    @model_validator(mode="after")
    def require_available_values(self) -> "ArrivalWindowQueueEvidence":
        values = (
            self.sample_coverage_fraction,
            self.covered_span_seconds,
            self.maximum_sample_gap_seconds,
            self.end_coverage_lag_seconds,
            self.waiting_positive_sample_fraction,
            self.peak_waiting_requests,
            self.tail_start_waiting_requests,
            self.tail_end_waiting_requests,
            self.tail_growth_requests,
            self.tail_slope_requests_per_second,
        )
        if self.available and (self.reason is not None or any(value is None for value in values)):
            raise ValueError("available queue evidence requires every derived value")
        if not self.available and not self.reason:
            raise ValueError("unavailable queue evidence requires a reason")
        return self


class DispatchDelayEvidence(StrictFrozenModel):
    available: bool
    reason: Optional[str] = None
    sample_count: int = Field(ge=0)
    p50_ms: Optional[float] = Field(default=None, ge=0)
    p95_ms: Optional[float] = Field(default=None, ge=0)
    p99_ms: Optional[float] = Field(default=None, ge=0)
    max_ms: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_available_values(self) -> "DispatchDelayEvidence":
        values = (self.p50_ms, self.p95_ms, self.p99_ms, self.max_ms)
        if self.available and (self.reason is not None or any(value is None for value in values)):
            raise ValueError("available dispatch evidence requires every percentile")
        if not self.available and not self.reason:
            raise ValueError("unavailable dispatch evidence requires a reason")
        return self


def _finite_number(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _queue_sample(row: Mapping[str, Any]) -> Optional[tuple[int, float]]:
    monotonic_ns = row.get("monotonic_ns")
    if isinstance(monotonic_ns, bool) or not isinstance(monotonic_ns, int):
        return None
    if row.get("available") is not True:
        return None
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    waiting = _finite_number(metrics.get("num_requests_waiting"))
    if waiting is None or waiting < 0:
        return None
    return monotonic_ns, waiting


def _ols_slope(samples: Sequence[tuple[int, float]]) -> float:
    origin = samples[0][0]
    x_values = [(timestamp - origin) / 1_000_000_000 for timestamp, _ in samples]
    y_values = [value for _, value in samples]
    x_mean = math.fsum(x_values) / len(x_values)
    y_mean = math.fsum(y_values) / len(y_values)
    denominator = math.fsum((value - x_mean) ** 2 for value in x_values)
    if denominator <= 0:
        raise ValueError("queue samples do not span positive monotonic time")
    numerator = math.fsum(
        (x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x_values, y_values)
    )
    return numerator / denominator


def analyze_arrival_window_queue(
    rows: Sequence[Mapping[str, Any]],
    *,
    measurement_start_monotonic_ns: int,
    arrival_window_seconds: float,
    tail_fraction: float = 0.5,
    minimum_tail_samples: int = 5,
    sample_interval_seconds: float = 0.2,
    minimum_sample_coverage_fraction: float = 0.8,
    maximum_gap_intervals: float = 5.0,
) -> ArrivalWindowQueueEvidence:
    """Fit queue growth only inside the offered-arrival window, excluding drain."""
    if measurement_start_monotonic_ns < 0:
        raise ValueError("measurement_start_monotonic_ns must be non-negative")
    if not math.isfinite(arrival_window_seconds) or arrival_window_seconds <= 0:
        raise ValueError("arrival_window_seconds must be finite and positive")
    if not math.isfinite(tail_fraction) or not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must satisfy 0 < value <= 1")
    if minimum_tail_samples < 2:
        raise ValueError("minimum_tail_samples must be at least two")
    if not math.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be finite and positive")
    if (
        not math.isfinite(minimum_sample_coverage_fraction)
        or not 0 < minimum_sample_coverage_fraction <= 1
    ):
        raise ValueError("minimum_sample_coverage_fraction must satisfy 0 < value <= 1")
    if not math.isfinite(maximum_gap_intervals) or maximum_gap_intervals < 1:
        raise ValueError("maximum_gap_intervals must be finite and at least one")

    end_ns = measurement_start_monotonic_ns + round(arrival_window_seconds * 1_000_000_000)
    tail_start_ns = measurement_start_monotonic_ns + round(
        arrival_window_seconds * (1 - tail_fraction) * 1_000_000_000
    )
    valid = sorted(
        (sample for row in rows if (sample := _queue_sample(row)) is not None),
        key=lambda item: item[0],
    )
    baselines = [sample for sample in valid if sample[0] <= measurement_start_monotonic_ns]
    inside = [sample for sample in valid if measurement_start_monotonic_ns < sample[0] <= end_ns]
    samples = [(measurement_start_monotonic_ns, baselines[-1][1]), *inside] if baselines else inside
    tail = [sample for sample in samples if sample[0] >= tail_start_ns]
    expected_samples = math.floor(arrival_window_seconds / sample_interval_seconds) + 1
    common: _QueueEvidenceCommon = {
        "measurement_start_monotonic_ns": measurement_start_monotonic_ns,
        "arrival_window_end_monotonic_ns": end_ns,
        "arrival_window_seconds": arrival_window_seconds,
        "tail_fraction": tail_fraction,
        "window_sample_count": len(samples),
        "tail_sample_count": len(tail),
        "expected_window_sample_count": expected_samples,
    }
    if not baselines:
        return ArrivalWindowQueueEvidence(
            available=False,
            reason="queue telemetry has no valid baseline at or before measurement start",
            **common,
        )
    if len(tail) < minimum_tail_samples:
        return ArrivalWindowQueueEvidence(
            available=False,
            reason=(
                "insufficient tail queue samples: "
                f"required {minimum_tail_samples}, observed {len(tail)}"
            ),
            **common,
        )
    coverage = len(samples) / expected_samples
    gaps = [
        (current[0] - previous[0]) / 1_000_000_000
        for previous, current in zip(samples, samples[1:])
    ]
    maximum_gap = max(gaps, default=0.0)
    end_lag = max(0.0, (end_ns - samples[-1][0]) / 1_000_000_000)
    maximum_allowed_gap = sample_interval_seconds * maximum_gap_intervals
    if coverage < minimum_sample_coverage_fraction:
        return ArrivalWindowQueueEvidence(
            available=False,
            reason=(
                "queue telemetry sample coverage is below policy: "
                f"required {minimum_sample_coverage_fraction:.3f}, observed {coverage:.3f}"
            ),
            **common,
        )
    if maximum_gap > maximum_allowed_gap or end_lag > maximum_allowed_gap:
        return ArrivalWindowQueueEvidence(
            available=False,
            reason=(
                "queue telemetry gap exceeds policy: "
                f"allowed {maximum_allowed_gap:.3f}s, "
                f"maximum gap {maximum_gap:.3f}s, end lag {end_lag:.3f}s"
            ),
            **common,
        )
    try:
        slope = _ols_slope(tail)
    except ValueError as error:
        return ArrivalWindowQueueEvidence(
            available=False,
            reason=str(error),
            **common,
        )
    return ArrivalWindowQueueEvidence(
        available=True,
        sample_coverage_fraction=coverage,
        covered_span_seconds=(samples[-1][0] - samples[0][0]) / 1_000_000_000,
        maximum_sample_gap_seconds=maximum_gap,
        end_coverage_lag_seconds=end_lag,
        waiting_positive_sample_fraction=(sum(value > 0 for _, value in samples) / len(samples)),
        peak_waiting_requests=max(value for _, value in samples),
        tail_start_waiting_requests=tail[0][1],
        tail_end_waiting_requests=tail[-1][1],
        tail_growth_requests=tail[-1][1] - tail[0][1],
        tail_slope_requests_per_second=slope,
        **common,
    )


def summarize_dispatch_delay(
    request_rows: Sequence[Mapping[str, Any]],
) -> DispatchDelayEvidence:
    """Summarize sent-minus-scheduled delay to expose client-side admission queues."""
    delays_ms: list[float] = []
    for row in request_rows:
        scheduled_at = row.get("scheduled_at")
        sent_at = row.get("sent_at")
        if (
            isinstance(scheduled_at, bool)
            or not isinstance(scheduled_at, int)
            or isinstance(sent_at, bool)
            or not isinstance(sent_at, int)
            or sent_at < scheduled_at
        ):
            continue
        delays_ms.append((sent_at - scheduled_at) / 1_000_000)
    if not delays_ms:
        return DispatchDelayEvidence(
            available=False,
            reason="no request exposes valid scheduled_at and sent_at timestamps",
            sample_count=0,
        )
    p50 = percentile(delays_ms, 0.50)
    p95 = percentile(delays_ms, 0.95)
    p99 = percentile(delays_ms, 0.99)
    assert p50 is not None
    assert p95 is not None
    assert p99 is not None
    return DispatchDelayEvidence(
        available=True,
        sample_count=len(delays_ms),
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        max_ms=max(delays_ms),
    )
