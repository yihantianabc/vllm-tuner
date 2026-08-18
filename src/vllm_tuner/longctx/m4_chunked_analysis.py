"""Typed records and paired analysis for long-context v5 M4 Chunked Prefill."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal, Optional

from pydantic import Field

from .kv_capacity_planner import StrictFrozenModel


class M4LatencyPercentiles(StrictFrozenModel):
    sample_count: int = Field(ge=1)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    p99_ms: float = Field(ge=0)
    maximum_ms: float = Field(ge=0)


class M4ResourceUsage(StrictFrozenModel):
    sample_count: int = Field(ge=1)
    minimum: float = Field(ge=0)
    median: float = Field(ge=0)
    p95: float = Field(ge=0)
    maximum: float = Field(ge=0)


class M4WaitingUsage(M4ResourceUsage):
    positive_sample_fraction: float = Field(ge=0, le=1)


class M4PrefillWindow(StrictFrozenModel):
    request_id: str
    sent_at_ns: int = Field(ge=0)
    first_token_at_ns: int = Field(ge=0)
    duration_ms: float = Field(ge=0)


class M4TrialRecord(StrictFrozenModel):
    trial_id: str
    profile_id: Literal[
        "production-default",
        "native-chunk-1024",
        "native-chunk-512",
    ]
    production_default: bool
    max_num_batched_tokens: Literal[512, 1024, 2048]
    max_num_partial_prefills: Literal[1, 2]
    max_long_partial_prefills: Literal[1]
    long_prefill_token_threshold: Literal[0, 2048]
    long_prefill_tokens: Literal[4096, 8192]
    repeat_index: int = Field(ge=0, le=2)
    trace_id: str
    warmup_trace_id: str
    request_count: int = Field(gt=0)
    decode_request_count: int = Field(gt=0)
    long_prefill_request_count: int = Field(gt=0)
    completion_fraction: float = Field(ge=0, le=1)
    decode_slo_satisfied_fraction: float = Field(ge=0, le=1)
    decode_goodput_requests_per_second: float = Field(ge=0)
    overall_goodput_requests_per_second: float = Field(ge=0)
    decode_ttft: M4LatencyPercentiles
    decode_tpot: M4LatencyPercentiles
    decode_itl: M4LatencyPercentiles
    decode_end_to_end: M4LatencyPercentiles
    long_prefill_ttft: M4LatencyPercentiles
    long_prefill_tpot: M4LatencyPercentiles
    long_prefill_end_to_end: M4LatencyPercentiles
    decode_interference_itl: M4LatencyPercentiles
    decode_non_interference_itl: M4LatencyPercentiles
    decode_overlap_request_count: int = Field(gt=0)
    prefill_windows: tuple[M4PrefillWindow, ...]
    waiting: M4WaitingUsage
    kv_usage: M4ResourceUsage
    preemption_count: int = Field(ge=0)
    prefix_cache_queries: int = Field(ge=0)
    prefix_cache_hits: Literal[0]
    peak_vram_mb: float = Field(ge=0)
    oom_count: Literal[0]
    timeout_count: Literal[0]
    mechanism_evidence_passed: Literal[True]


def _metric(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("M4 metric aggregation requires at least one value")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("M4 metric aggregation requires finite values")
    return {
        "minimum": ordered[0],
        "median": float(statistics.median(ordered)),
        "maximum": ordered[-1],
    }


def _percent_change(candidate: float, baseline: float) -> Optional[float]:
    if baseline == 0:
        return None
    return (candidate - baseline) / baseline * 100.0


def analyze_m4_records(
    records: Sequence[M4TrialRecord],
    *,
    formal: bool,
) -> dict[str, object]:
    """Aggregate paired repeats and select by repeated evidence, never one run."""
    groups: dict[tuple[str, int], list[M4TrialRecord]] = defaultdict(list)
    for record in records:
        groups[(record.profile_id, record.long_prefill_tokens)].append(record)
    grouped: list[dict[str, object]] = []
    for (profile_id, long_tokens), rows in sorted(groups.items()):
        grouped.append(
            {
                "profile_id": profile_id,
                "long_prefill_tokens": long_tokens,
                "repeat_count": len(rows),
                "decode_tpot_p99_ms": _metric([row.decode_tpot.p99_ms for row in rows]),
                "decode_itl_p99_ms": _metric([row.decode_itl.p99_ms for row in rows]),
                "decode_interference_itl_p99_ms": _metric(
                    [row.decode_interference_itl.p99_ms for row in rows]
                ),
                "long_prefill_ttft_p99_ms": _metric([row.long_prefill_ttft.p99_ms for row in rows]),
                "decode_goodput_requests_per_second": _metric(
                    [row.decode_goodput_requests_per_second for row in rows]
                ),
                "overall_goodput_requests_per_second": _metric(
                    [row.overall_goodput_requests_per_second for row in rows]
                ),
                "peak_waiting_requests": _metric([row.waiting.maximum for row in rows]),
                "kv_usage_p95": _metric([row.kv_usage.p95 for row in rows]),
                "preemption_count": _metric([float(row.preemption_count) for row in rows]),
            }
        )

    by_key: dict[tuple[str, int, int], M4TrialRecord] = {
        (record.profile_id, record.long_prefill_tokens, record.repeat_index): record
        for record in records
    }
    candidates = ("native-chunk-1024", "native-chunk-512")
    paired: list[dict[str, object]] = []
    candidate_pool_changes: dict[str, list[float]] = defaultdict(list)
    candidate_eligible: dict[str, bool] = {candidate: True for candidate in candidates}
    for candidate in candidates:
        for long_tokens in sorted({record.long_prefill_tokens for record in records}):
            pairs: list[dict[str, object]] = []
            for repeat in sorted({record.repeat_index for record in records}):
                baseline = by_key.get(("production-default", long_tokens, repeat))
                selected = by_key.get((candidate, long_tokens, repeat))
                if baseline is None or selected is None:
                    continue
                interference_change = _percent_change(
                    selected.decode_interference_itl.p99_ms,
                    baseline.decode_interference_itl.p99_ms,
                )
                if interference_change is not None:
                    candidate_pool_changes[candidate].append(interference_change)
                pairs.append(
                    {
                        "repeat_index": repeat,
                        "decode_interference_itl_p99_change_percent": interference_change,
                        "decode_tpot_p99_change_percent": _percent_change(
                            selected.decode_tpot.p99_ms, baseline.decode_tpot.p99_ms
                        ),
                        "long_prefill_ttft_p99_change_percent": _percent_change(
                            selected.long_prefill_ttft.p99_ms,
                            baseline.long_prefill_ttft.p99_ms,
                        ),
                        "decode_goodput_change_percent": _percent_change(
                            selected.decode_goodput_requests_per_second,
                            baseline.decode_goodput_requests_per_second,
                        ),
                        "overall_goodput_change_percent": _percent_change(
                            selected.overall_goodput_requests_per_second,
                            baseline.overall_goodput_requests_per_second,
                        ),
                        "interference_itl_improved": (
                            selected.decode_interference_itl.p99_ms
                            < baseline.decode_interference_itl.p99_ms
                        ),
                        "decode_goodput_not_lower": (
                            selected.decode_goodput_requests_per_second
                            >= baseline.decode_goodput_requests_per_second
                        ),
                        "preemptions_not_higher": (
                            selected.preemption_count <= baseline.preemption_count
                        ),
                    }
                )
            improved = sum(row["interference_itl_improved"] is True for row in pairs)
            goodput = sum(row["decode_goodput_not_lower"] is True for row in pairs)
            preemptions = sum(row["preemptions_not_higher"] is True for row in pairs)
            repeat_count = len(pairs)
            eligible_cell = (
                repeat_count == 3 and improved >= 2 and goodput >= 2 and preemptions >= 2
            )
            if formal and not eligible_cell:
                candidate_eligible[candidate] = False
            paired.append(
                {
                    "profile_id": candidate,
                    "long_prefill_tokens": long_tokens,
                    "repeat_count": repeat_count,
                    "pairs": pairs,
                    "interference_itl_improved_repeats": improved,
                    "decode_goodput_not_lower_repeats": goodput,
                    "preemptions_not_higher_repeats": preemptions,
                    "eligible_cell": eligible_cell,
                }
            )

    eligible = [candidate for candidate in candidates if candidate_eligible[candidate]]
    if not formal:
        selected_profile = "production-default"
        reason = "smoke validates the path and does not select a formal profile"
    elif eligible:
        selected_profile = min(
            eligible,
            key=lambda candidate: statistics.median(candidate_pool_changes[candidate]),
        )
        reason = (
            "candidate passed both 4K/8K majority guardrails and had the lowest pooled median "
            "decode-interference ITL p99 change"
        )
    else:
        selected_profile = "production-default"
        reason = (
            "no native candidate improved decode-interference ITL while preserving decode "
            "Goodput and preemption direction in at least two of three repeats at both lengths"
        )
    return {
        "schema_version": "longctx-m4-chunked-analysis.v1",
        "record_count": len(records),
        "groups": grouped,
        "paired": paired,
        "selection": {
            "profile_id": selected_profile,
            "eligible_native_profiles": eligible,
            "candidate_eligibility": candidate_eligible,
            "pooled_median_interference_itl_p99_change_percent": {
                candidate: (float(statistics.median(values)) if values else None)
                for candidate, values in candidate_pool_changes.items()
            },
            "rule": reason,
            "single_run_selection_used": False,
        },
    }


__all__ = [
    "M4LatencyPercentiles",
    "M4PrefillWindow",
    "M4ResourceUsage",
    "M4TrialRecord",
    "M4WaitingUsage",
    "analyze_m4_records",
]
