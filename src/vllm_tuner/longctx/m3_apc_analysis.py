"""Typed records and paired analysis for long-context v5 M3 APC."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal, Optional

from pydantic import Field

from .kv_capacity_planner import StrictFrozenModel


class M3LatencyPercentiles(StrictFrozenModel):
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    p99_ms: float = Field(ge=0)


class M3KVUsage(StrictFrozenModel):
    sample_count: int = Field(ge=1)
    minimum: float = Field(ge=0)
    median: float = Field(ge=0)
    p95: float = Field(ge=0)
    maximum: float = Field(ge=0)


class M3ReuseMetrics(StrictFrozenModel):
    reuse_percent: Literal[0, 50, 100]
    request_count: int = Field(gt=0)
    shared_request_count: int = Field(ge=0)
    cached_tokens: int = Field(ge=0)
    expected_cached_tokens: int = Field(ge=0)
    hit_ratio: float = Field(ge=0, le=1)
    achieved_requests_per_second: float = Field(ge=0)
    goodput_requests_per_second: float = Field(ge=0)
    slo_satisfied_fraction: float = Field(ge=0, le=1)
    ttft: M3LatencyPercentiles
    tpot: M3LatencyPercentiles
    end_to_end: M3LatencyPercentiles


class M3APCTrialRecord(StrictFrozenModel):
    record_kind: Literal["core"] = "core"
    trial_id: str
    profile_id: Literal["apc-off", "apc-on"]
    apc_enabled: bool
    cache_state: Literal["target-prefix-cold", "target-prefix-warm"]
    prefix_tokens: Literal[2048, 4096]
    repeat_index: int = Field(ge=0, le=2)
    trace_id: str
    warmup_trace_id: str
    request_count: int = Field(gt=0)
    prefix_cache_queries: int = Field(ge=0)
    prefix_cache_hits: int = Field(ge=0)
    expected_prefix_cache_hits: int = Field(ge=0)
    hit_ratio: float = Field(ge=0, le=1)
    exact_hit_tokens: bool
    completion_fraction: float = Field(ge=0, le=1)
    achieved_requests_per_second: float = Field(ge=0)
    goodput_requests_per_second: float = Field(ge=0)
    preemption_count: int = Field(ge=0)
    oom_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)
    peak_vram_mb: float = Field(ge=0)
    kv_usage: M3KVUsage
    reuse: tuple[M3ReuseMetrics, ...]


class M3BoundaryRecord(StrictFrozenModel):
    record_kind: Literal["boundary"] = "boundary"
    trial_id: str
    profile_id: Literal["apc-on"] = "apc-on"
    pool_size: int = Field(gt=0)
    prefix_tokens: Literal[4096]
    input_tokens: int = Field(gt=4096)
    trace_id: str
    warmup_trace_id: str
    request_count: int = Field(gt=0)
    prefix_cache_queries: int = Field(ge=0)
    prefix_cache_hits: int = Field(ge=0)
    hit_ratio: float = Field(ge=0, le=1)
    full_hit_requests: int = Field(ge=0)
    partial_hit_requests: int = Field(ge=0)
    miss_requests: int = Field(ge=0)
    first_miss_probe_position: Optional[int] = Field(default=None, ge=0)
    cached_tokens_by_probe: tuple[int, ...]
    kv_usage: M3KVUsage
    predicted_resident_prompts: int = Field(gt=0)
    runtime_cached_token_capacity: int = Field(gt=0)
    preemption_count: int = Field(ge=0)
    oom_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)


def _metric(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("M3 metric aggregation requires at least one value")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("M3 metric aggregation requires finite values")
    return {
        "minimum": ordered[0],
        "median": float(statistics.median(ordered)),
        "maximum": ordered[-1],
    }


def _percent_change(candidate: float, baseline: float) -> Optional[float]:
    if baseline == 0:
        return None
    return (candidate - baseline) / baseline * 100.0


def analyze_m3_apc_records(
    records: Sequence[M3APCTrialRecord],
    boundaries: Sequence[M3BoundaryRecord],
) -> dict[str, object]:
    """Aggregate exact cells and paired directional evidence without selecting one run."""
    groups: dict[tuple[str, str, int, int], list[M3ReuseMetrics]] = defaultdict(list)
    for record in records:
        for reuse_metrics in record.reuse:
            groups[
                (
                    record.profile_id,
                    record.cache_state,
                    record.prefix_tokens,
                    reuse_metrics.reuse_percent,
                )
            ].append(reuse_metrics)
    grouped: list[dict[str, object]] = []
    for (profile, state, prefix, reuse_percent), group_rows in sorted(groups.items()):
        grouped.append(
            {
                "profile_id": profile,
                "cache_state": state,
                "prefix_tokens": prefix,
                "reuse_percent": reuse_percent,
                "repeat_count": len(group_rows),
                "hit_ratio": _metric([row.hit_ratio for row in group_rows]),
                "ttft_p50_ms": _metric([row.ttft.p50_ms for row in group_rows]),
                "ttft_p99_ms": _metric([row.ttft.p99_ms for row in group_rows]),
                "tpot_p50_ms": _metric([row.tpot.p50_ms for row in group_rows]),
                "goodput_requests_per_second": _metric(
                    [row.goodput_requests_per_second for row in group_rows]
                ),
                "slo_satisfied_fraction": _metric(
                    [row.slo_satisfied_fraction for row in group_rows]
                ),
            }
        )

    by_key = {
        (record.profile_id, record.cache_state, record.prefix_tokens, record.repeat_index): record
        for record in records
    }
    paired: list[dict[str, object]] = []
    for prefix in sorted({record.prefix_tokens for record in records}):
        for reuse_percent in (0, 50, 100):
            pair_rows: list[dict[str, float | int | bool | None]] = []
            for repeat in sorted({record.repeat_index for record in records}):
                baseline = by_key.get(("apc-off", "target-prefix-cold", prefix, repeat))
                cold = by_key.get(("apc-on", "target-prefix-cold", prefix, repeat))
                warm = by_key.get(("apc-on", "target-prefix-warm", prefix, repeat))
                if baseline is None or cold is None or warm is None:
                    continue
                baseline_reuse = next(
                    value for value in baseline.reuse if value.reuse_percent == reuse_percent
                )
                cold_reuse = next(
                    value for value in cold.reuse if value.reuse_percent == reuse_percent
                )
                warm_reuse = next(
                    value for value in warm.reuse if value.reuse_percent == reuse_percent
                )
                pair_rows.append(
                    {
                        "repeat_index": repeat,
                        "cold_hit_ratio": cold_reuse.hit_ratio,
                        "warm_hit_ratio": warm_reuse.hit_ratio,
                        "cold_ttft_change_percent": _percent_change(
                            cold_reuse.ttft.p50_ms, baseline_reuse.ttft.p50_ms
                        ),
                        "warm_ttft_change_percent": _percent_change(
                            warm_reuse.ttft.p50_ms, baseline_reuse.ttft.p50_ms
                        ),
                        "warm_goodput_change_percent": _percent_change(
                            warm_reuse.goodput_requests_per_second,
                            baseline_reuse.goodput_requests_per_second,
                        ),
                        "warm_ttft_improved": warm_reuse.ttft.p50_ms < baseline_reuse.ttft.p50_ms,
                        "warm_goodput_not_lower": warm_reuse.goodput_requests_per_second
                        >= baseline_reuse.goodput_requests_per_second,
                    }
                )
            paired.append(
                {
                    "prefix_tokens": prefix,
                    "reuse_percent": reuse_percent,
                    "pairs": pair_rows,
                    "warm_ttft_improved_repeats": sum(
                        row["warm_ttft_improved"] is True for row in pair_rows
                    ),
                    "warm_goodput_not_lower_repeats": sum(
                        row["warm_goodput_not_lower"] is True for row in pair_rows
                    ),
                    "repeat_count": len(pair_rows),
                }
            )

    boundary_rows = [
        record.model_dump(mode="json")
        for record in sorted(boundaries, key=lambda item: item.pool_size)
    ]
    boundary_bracketed = (
        len(boundaries) >= 2
        and sorted(boundaries, key=lambda item: item.pool_size)[0].hit_ratio
        > sorted(boundaries, key=lambda item: item.pool_size)[-1].hit_ratio
        and sorted(boundaries, key=lambda item: item.pool_size)[-1].miss_requests > 0
    )
    return {
        "schema_version": "longctx-m3-apc-analysis.v1",
        "record_count": len(records),
        "boundary_record_count": len(boundaries),
        "groups": grouped,
        "paired": paired,
        "boundary": {
            "records": boundary_rows,
            "bracketed": boundary_bracketed,
            "rule": "smaller preregistered pool retains a higher measured token-hit ratio and the larger pool exposes at least one full miss",
        },
        "single_run_selection_used": False,
    }


__all__ = [
    "M3APCTrialRecord",
    "M3BoundaryRecord",
    "M3KVUsage",
    "M3LatencyPercentiles",
    "M3ReuseMetrics",
    "analyze_m3_apc_records",
]
