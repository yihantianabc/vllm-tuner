"""Correct per-request and aggregate benchmark metric calculations."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional, Sequence

import numpy as np

from .models import RequestResult, RequestStatus, SLOThresholds

NS_PER_MS = 1_000_000
NS_PER_SECOND = 1_000_000_000
DEFAULT_PERCENTILES = (50.0, 90.0, 95.0, 99.0)


def _milliseconds(value_ns: Optional[float]) -> Optional[float]:
    if value_ns is None:
        return None
    return value_ns / NS_PER_MS


def calculate_ttft_ms(result: RequestResult) -> Optional[float]:
    """Return request-to-first-non-empty-output latency in milliseconds."""

    return _milliseconds(result.ttft_ns)


def calculate_e2e_ms(result: RequestResult) -> Optional[float]:
    """Return independently measured request end-to-end latency."""

    return _milliseconds(result.e2e_ns)


def calculate_tpot_ms(result: RequestResult) -> Optional[float]:
    """Return per-request decode time per output token, excluding token one."""

    return _milliseconds(result.tpot_ns)


def calculate_itl_ms(result: RequestResult) -> list[float]:
    """Return token-level inter-token latency when token timestamps are valid."""

    return [value / NS_PER_MS for value in result.itl_ns]


def calculate_inter_event_latency_ms(result: RequestResult) -> list[float]:
    """Return gaps between adjacent non-empty SSE output events."""

    return [value / NS_PER_MS for value in result.inter_event_latency_ns]


# Short aliases make formulas pleasant to use in analysis notebooks.
ttft_ms = calculate_ttft_ms
e2e_ms = calculate_e2e_ms
tpot_ms = calculate_tpot_ms
itl_ms = calculate_itl_ms
inter_event_latency_ms = calculate_inter_event_latency_ms


def numpy_percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    """Calculate a percentile with NumPy's interpolated implementation."""

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _percentile_label(percentile: float) -> str:
    return str(int(percentile)) if float(percentile).is_integer() else str(percentile)


def summarize_distribution(
    values: Sequence[float], percentiles: Sequence[float] = DEFAULT_PERCENTILES
) -> dict[str, Optional[float]]:
    """Summarize a metric without inventing zeroes for missing measurements."""

    summary: dict[str, Optional[float]] = {"count": float(len(values))}
    if values:
        array = np.asarray(values, dtype=np.float64)
        summary.update(
            {
                "mean": float(np.mean(array)),
                "median": float(np.median(array)),
                "std": float(np.std(array)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
            }
        )
    else:
        summary.update({"mean": None, "median": None, "std": None, "min": None, "max": None})

    for percentile in percentiles:
        summary[f"p{_percentile_label(percentile)}"] = numpy_percentile(values, percentile)
    return summary


def request_meets_slo(result: RequestResult, slo: SLOThresholds) -> bool:
    """Return whether a successful request satisfies every configured SLO."""

    if result.status != RequestStatus.SUCCESS:
        return False

    ttft = calculate_ttft_ms(result)
    e2e = calculate_e2e_ms(result)
    tpot = calculate_tpot_ms(result)

    if slo.ttft_ms is not None and (ttft is None or ttft > slo.ttft_ms):
        return False
    if slo.e2e_ms is not None and (e2e is None or e2e > slo.e2e_ms):
        return False
    if slo.tpot_ms is not None:
        # vLLM defines TPOT as zero for a successful one-token response when
        # evaluating goodput because there is no post-first-token decode phase.
        value = 0.0 if result.output_tokens <= 1 else tpot
        if value is None or value > slo.tpot_ms:
            return False
    return True


def _flatten_distribution(
    aggregate: dict[str, object],
    metric_name: str,
    values: Sequence[float],
    percentiles: Sequence[float],
) -> None:
    distribution = summarize_distribution(values, percentiles)
    aggregate[f"{metric_name}_count"] = int(distribution["count"] or 0)
    for statistic in ("mean", "median", "std", "min", "max"):
        aggregate[f"{statistic}_{metric_name}_ms"] = distribution[statistic]
    for percentile in percentiles:
        label = _percentile_label(percentile)
        aggregate[f"p{label}_{metric_name}_ms"] = distribution[f"p{label}"]


def _measurement_bounds(
    results: Sequence[RequestResult],
    started_at: Optional[int],
    finished_at: Optional[int],
) -> tuple[Optional[int], Optional[int]]:
    start_candidates = [result.sent_at for result in results if result.sent_at is not None]
    finish_candidates = [result.finished_at for result in results if result.finished_at is not None]
    start = (
        started_at
        if started_at is not None
        else (min(start_candidates) if start_candidates else None)
    )
    finish = (
        finished_at
        if finished_at is not None
        else (max(finish_candidates) if finish_candidates else None)
    )
    return start, finish


def aggregate_request_results(
    results: Iterable[RequestResult],
    *,
    started_at: Optional[int] = None,
    finished_at: Optional[int] = None,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
    slo: Optional[SLOThresholds] = None,
    include_request_results: bool = True,
) -> dict[str, object]:
    """Reduce measured requests while excluding warmup samples.

    Successful request latency distributions never include failures or missing
    measurements. Failed tasks remain visible through ``failed``, ``error_types``
    and the optional raw request list.
    """

    measured = [result for result in results if not result.warmup]
    successful = [result for result in measured if result.status == RequestStatus.SUCCESS]
    failed = [result for result in measured if result.status != RequestStatus.SUCCESS]

    start, finish = _measurement_bounds(measured, started_at, finished_at)
    duration_s: Optional[float]
    if start is None or finish is None or finish < start:
        duration_s = None
    else:
        duration_s = (finish - start) / NS_PER_SECOND

    completed = len(successful)
    total_input_tokens = sum(result.input_tokens for result in successful)
    total_output_tokens = sum(result.output_tokens for result in successful)
    good_completed = (
        sum(request_meets_slo(result, slo) for result in successful) if slo is not None else None
    )

    aggregate: dict[str, object] = {
        "num_requests": len(measured),
        "completed": completed,
        "failed": len(failed),
        "error_rate": len(failed) / len(measured) if measured else 0.0,
        "error_types": dict(Counter(result.error_type or "unknown" for result in failed)),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "duration": duration_s,
        "request_throughput": (completed / duration_s if duration_s and duration_s > 0 else None),
        "output_throughput": (
            total_output_tokens / duration_s if duration_s and duration_s > 0 else None
        ),
        "total_token_throughput": (
            (total_input_tokens + total_output_tokens) / duration_s
            if duration_s and duration_s > 0
            else None
        ),
        "good_completed": good_completed,
        "request_goodput": (
            good_completed / duration_s
            if good_completed is not None and duration_s and duration_s > 0
            else None
        ),
        "measurement_started_at": start,
        "measurement_finished_at": finish,
    }

    ttfts = [value for result in successful if (value := calculate_ttft_ms(result)) is not None]
    tpots = [value for result in successful if (value := calculate_tpot_ms(result)) is not None]
    e2es = [value for result in successful if (value := calculate_e2e_ms(result)) is not None]
    itls = [value for result in successful for value in calculate_itl_ms(result)]
    event_latencies = [
        value for result in successful for value in calculate_inter_event_latency_ms(result)
    ]

    _flatten_distribution(aggregate, "ttft", ttfts, percentiles)
    _flatten_distribution(aggregate, "tpot", tpots, percentiles)
    _flatten_distribution(aggregate, "itl", itls, percentiles)
    _flatten_distribution(aggregate, "inter_event_latency", event_latencies, percentiles)
    _flatten_distribution(aggregate, "e2e", e2es, percentiles)

    if include_request_results:
        aggregate["request_results"] = [result.to_dict() for result in measured]
    return aggregate


# Compatibility names used by callers that treat reduction as calculation.
aggregate_results = aggregate_request_results
calculate_metrics = aggregate_request_results
