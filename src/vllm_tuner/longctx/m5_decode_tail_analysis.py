"""Typed records and preregistered paired analysis for v5 M5."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal, Optional

from pydantic import Field

from .kv_capacity_planner import StrictFrozenModel
from .m4_chunked_analysis import (
    M4LatencyPercentiles,
    M4PrefillWindow,
    M4ResourceUsage,
    M4WaitingUsage,
)

MIN_ITL_IMPROVEMENT_PERCENT = 25.0
MIN_MEDIAN_GOODPUT_CHANGE_PERCENT = -0.5
MIN_REPEAT_GOODPUT_CHANGE_PERCENT = -1.0
MAX_TTFT_DEGRADATION_PERCENT = 15.0
MAX_TPOT_DEGRADATION_PERCENT = 2.0


class M5TrialRecord(StrictFrozenModel):
    trial_id: str
    cohort_id: Literal["target", "held-out"]
    profile_id: Literal["production-default", "decode-tail-1024"]
    production_default: bool
    max_num_batched_tokens: Literal[2048]
    max_num_partial_prefills: Literal[1]
    max_long_partial_prefills: Literal[1]
    long_prefill_token_threshold: Literal[0, 1024]
    long_prefill_tokens: tuple[Literal[4096, 8192], ...]
    injection_offsets_seconds: tuple[float, ...]
    prompt_seed: int = Field(ge=0)
    arrival_seed: int = Field(ge=0)
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
    preemption_count: Literal[0]
    prefix_cache_queries: int = Field(ge=0)
    prefix_cache_hits: Literal[0]
    peak_vram_mb: float = Field(ge=0)
    oom_count: Literal[0]
    timeout_count: Literal[0]
    mechanism_evidence_passed: Literal[True]


def _metric(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("M5 metric aggregation requires at least one value")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("M5 metric aggregation requires finite values")
    return {
        "minimum": ordered[0],
        "median": float(statistics.median(ordered)),
        "maximum": ordered[-1],
    }


def _percent_change(candidate: float, baseline: float) -> Optional[float]:
    if baseline == 0:
        return None
    return (candidate - baseline) / baseline * 100.0


def _improvement(candidate: float, baseline: float) -> Optional[float]:
    change = _percent_change(candidate, baseline)
    return None if change is None else -change


def analyze_m5_records(
    records: Sequence[M5TrialRecord],
    *,
    formal: bool,
) -> dict[str, object]:
    """Apply the frozen target/held-out non-inferiority rules to paired repeats."""
    groups: dict[tuple[str, str], list[M5TrialRecord]] = defaultdict(list)
    for record in records:
        groups[(record.cohort_id, record.profile_id)].append(record)
    grouped: list[dict[str, object]] = []
    for (cohort_id, profile_id), rows in sorted(groups.items()):
        grouped.append(
            {
                "cohort_id": cohort_id,
                "profile_id": profile_id,
                "repeat_count": len(rows),
                "decode_interference_itl_p99_ms": _metric(
                    [row.decode_interference_itl.p99_ms for row in rows]
                ),
                "decode_goodput_requests_per_second": _metric(
                    [row.decode_goodput_requests_per_second for row in rows]
                ),
                "long_prefill_ttft_p99_ms": _metric([row.long_prefill_ttft.p99_ms for row in rows]),
                "decode_tpot_p99_ms": _metric([row.decode_tpot.p99_ms for row in rows]),
                "waiting_maximum": _metric([row.waiting.maximum for row in rows]),
                "kv_usage_maximum": _metric([row.kv_usage.maximum for row in rows]),
                "preemption_count": _metric([float(row.preemption_count) for row in rows]),
            }
        )

    by_key = {
        (record.cohort_id, record.profile_id, record.repeat_index): record for record in records
    }
    paired: list[dict[str, object]] = []
    cohort_acceptance: dict[str, dict[str, object]] = {}
    for cohort_id in ("target", "held-out"):
        pairs: list[dict[str, object]] = []
        for repeat in range(3 if formal else 1):
            baseline = by_key.get((cohort_id, "production-default", repeat))
            candidate = by_key.get((cohort_id, "decode-tail-1024", repeat))
            if baseline is None or candidate is None:
                continue
            pairs.append(
                {
                    "repeat_index": repeat,
                    "decode_interference_itl_p99_improvement_percent": _improvement(
                        candidate.decode_interference_itl.p99_ms,
                        baseline.decode_interference_itl.p99_ms,
                    ),
                    "decode_goodput_change_percent": _percent_change(
                        candidate.decode_goodput_requests_per_second,
                        baseline.decode_goodput_requests_per_second,
                    ),
                    "long_prefill_ttft_p99_degradation_percent": _percent_change(
                        candidate.long_prefill_ttft.p99_ms,
                        baseline.long_prefill_ttft.p99_ms,
                    ),
                    "decode_tpot_p99_degradation_percent": _percent_change(
                        candidate.decode_tpot.p99_ms,
                        baseline.decode_tpot.p99_ms,
                    ),
                    "waiting_maximum_delta": candidate.waiting.maximum - baseline.waiting.maximum,
                    "kv_usage_maximum_delta": (
                        candidate.kv_usage.maximum - baseline.kv_usage.maximum
                    ),
                    "baseline_preemption_count": baseline.preemption_count,
                    "candidate_preemption_count": candidate.preemption_count,
                    "baseline_oom_count": baseline.oom_count,
                    "candidate_oom_count": candidate.oom_count,
                    "baseline_timeout_count": baseline.timeout_count,
                    "candidate_timeout_count": candidate.timeout_count,
                }
            )
        expected_pairs = 3 if formal else 1
        if not pairs:
            continue
        numeric_fields = (
            "decode_interference_itl_p99_improvement_percent",
            "decode_goodput_change_percent",
            "long_prefill_ttft_p99_degradation_percent",
            "decode_tpot_p99_degradation_percent",
            "waiting_maximum_delta",
            "kv_usage_maximum_delta",
        )
        aggregates = {
            field: _metric(
                [float(row[field]) for row in pairs if isinstance(row.get(field), (int, float))]
            )
            for field in numeric_fields
        }
        checks = {
            "paired_repeat_count": len(pairs) == expected_pairs,
            "decode_interference_itl_p99_median_improvement_ge_25pct": (
                aggregates["decode_interference_itl_p99_improvement_percent"]["median"]
                >= MIN_ITL_IMPROVEMENT_PERCENT
            ),
            "decode_goodput_median_change_ge_minus_0_5pct": (
                aggregates["decode_goodput_change_percent"]["median"]
                >= MIN_MEDIAN_GOODPUT_CHANGE_PERCENT
            ),
            "decode_goodput_each_repeat_ge_minus_1pct": (
                aggregates["decode_goodput_change_percent"]["minimum"]
                >= MIN_REPEAT_GOODPUT_CHANGE_PERCENT
            ),
            "long_prefill_ttft_p99_median_degradation_le_15pct": (
                aggregates["long_prefill_ttft_p99_degradation_percent"]["median"]
                <= MAX_TTFT_DEGRADATION_PERCENT
            ),
            "decode_tpot_p99_median_degradation_le_2pct": (
                aggregates["decode_tpot_p99_degradation_percent"]["median"]
                <= MAX_TPOT_DEGRADATION_PERCENT
            ),
            "waiting_has_no_opposite_median_worsening": (
                aggregates["waiting_maximum_delta"]["median"] <= 0
            ),
            "kv_usage_has_no_opposite_median_worsening": (
                aggregates["kv_usage_maximum_delta"]["median"] <= 0
            ),
            "zero_oom_timeout_preemption": all(
                row[metric] == 0
                for row in pairs
                for metric in (
                    "baseline_preemption_count",
                    "candidate_preemption_count",
                    "baseline_oom_count",
                    "candidate_oom_count",
                    "baseline_timeout_count",
                    "candidate_timeout_count",
                )
            ),
        }
        passed = all(checks.values()) if formal else True
        cohort_acceptance[cohort_id] = {
            "passed": passed,
            "checks": checks,
            "failure_reasons": sorted(name for name, value in checks.items() if not value),
            "paired_metrics": aggregates,
        }
        paired.append(
            {
                "cohort_id": cohort_id,
                "repeat_count": len(pairs),
                "pairs": pairs,
                "aggregates": aggregates,
                "acceptance": cohort_acceptance[cohort_id],
            }
        )

    formal_cohorts_complete = set(cohort_acceptance) == {"target", "held-out"}
    deployment_passed = (
        formal
        and formal_cohorts_complete
        and all(value["passed"] is True for value in cohort_acceptance.values())
    )
    if deployment_passed:
        decision = {
            "profile_id": "decode-tail-1024",
            "positive_result": True,
            "wording": (
                "decode-tail-1024 passed the preregistered target and held-out Decode-tail "
                "non-inferiority validation"
            ),
        }
    elif formal:
        decision = {
            "profile_id": "production-default",
            "positive_result": False,
            "wording": (
                "M5 is a retained negative result; no final deployment-profile superiority "
                "claim is allowed"
            ),
        }
    else:
        decision = {
            "profile_id": "production-default",
            "positive_result": False,
            "wording": "smoke validates the execution path and does not make an M5 decision",
        }
    return {
        "schema_version": "longctx-m5-decode-tail-analysis.v1",
        "record_count": len(records),
        "groups": grouped,
        "paired": paired,
        "cohort_acceptance": cohort_acceptance,
        "acceptance": {
            "eligible": formal,
            "passed": deployment_passed if formal else True,
            "target_and_held_out_complete": formal_cohorts_complete if formal else None,
            "failure_reasons": (
                sorted(
                    f"{cohort}:{reason}"
                    for cohort, value in cohort_acceptance.items()
                    for reason in value["failure_reasons"]
                )
                if formal
                else []
            ),
        },
        "decision": decision,
        "preregistered_thresholds": {
            "decode_interference_itl_p99_median_improvement_percent": 25.0,
            "decode_goodput_median_change_percent": -0.5,
            "decode_goodput_each_repeat_change_percent": -1.0,
            "long_prefill_ttft_p99_median_degradation_percent": 15.0,
            "decode_tpot_p99_median_degradation_percent": 2.0,
            "waiting_and_kv_usage_rule": "paired median maximum delta must be non-positive",
            "oom_timeout_preemption": 0,
        },
    }


__all__ = ["M5TrialRecord", "analyze_m5_records"]
