"""Paired aggregation for long-context v5 M2 FP8 KV measurements."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal, cast

from pydantic import Field

from .kv_capacity_planner import StrictFrozenModel


class M2LatencyPercentiles(StrictFrozenModel):
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    p99_ms: float = Field(ge=0)


class M2FP8TrialRecord(StrictFrozenModel):
    """Semantically validated evidence from one independently sealed trial."""

    trial_id: str
    profile_id: Literal["bf16-auto", "fp8-dynamic", "fp8-unit-fallback", "fp8-e5m2"]
    context_id: str
    context_tokens: int = Field(gt=0)
    repeat_index: int = Field(ge=0, le=2)
    trace_id: str
    status: Literal["complete"]
    requested_kv_cache_dtype: Literal["auto", "fp8", "fp8_e5m2"]
    calculate_kv_scales: bool
    scale_source: Literal[
        "model-dtype", "dynamic-first-forward", "unit-fallback", "e5m2-unit-scale"
    ]
    attention_backend: Literal["FLASH_ATTN", "FLASHINFER"]
    backend_resolution: Literal["production-default", "automatic-fp8-fallback"]
    num_gpu_blocks: int = Field(gt=1)
    usable_num_gpu_blocks: int = Field(gt=0)
    block_size: int = Field(gt=0)
    cached_tokens: int = Field(gt=0)
    quality_probe_count: int = Field(gt=0)
    quality_pass_count: int = Field(ge=0)
    quality_passed: bool
    request_count: int = Field(gt=1)
    completion_fraction: float = Field(ge=0, le=1)
    achieved_requests_per_second: float = Field(ge=0)
    goodput_requests_per_second: float = Field(ge=0)
    slo_satisfied_fraction: float = Field(ge=0, le=1)
    preemption_count: int = Field(ge=0)
    oom_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)
    peak_vram_mb: float = Field(ge=0)
    ttft: M2LatencyPercentiles
    tpot: M2LatencyPercentiles
    itl: M2LatencyPercentiles
    end_to_end: M2LatencyPercentiles


def _metric(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("M2 aggregate metrics require finite non-empty samples")
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _optional_metric(values: Sequence[float | None]) -> dict[str, float | int | bool | None]:
    defined = [value for value in values if value is not None]
    if not defined:
        return {
            "available": False,
            "defined_count": 0,
            "undefined_count": len(values),
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    metric = _metric(defined)
    return {
        "available": True,
        "defined_count": len(defined),
        "undefined_count": len(values) - len(defined),
        **metric,
    }


def _percent_change(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return (candidate / baseline - 1.0) * 100.0


def analyze_m2_fp8_records(records: Sequence[M2FP8TrialRecord]) -> dict[str, object]:
    """Aggregate profiles and exact repeat pairs without selecting a best single run."""
    if not records:
        raise ValueError("M2 FP8 analysis requires at least one record")
    identities = {(record.profile_id, record.context_id, record.repeat_index) for record in records}
    if len(identities) != len(records):
        raise ValueError("M2 FP8 records contain duplicate profile/context/repeat identities")

    grouped: dict[tuple[str, str], list[M2FP8TrialRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.profile_id, record.context_id)].append(record)
    groups: list[dict[str, object]] = []
    for (profile_id, context_id), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: row.repeat_index)
        groups.append(
            {
                "profile_id": profile_id,
                "context_id": context_id,
                "repeat_indices": [row.repeat_index for row in ordered],
                "attention_backends": sorted({row.attention_backend for row in ordered}),
                "scale_sources": sorted({row.scale_source for row in ordered}),
                "quality_all_passed": all(row.quality_passed for row in ordered),
                "cached_tokens": _metric([float(row.cached_tokens) for row in ordered]),
                "num_gpu_blocks": _metric([float(row.num_gpu_blocks) for row in ordered]),
                "peak_vram_mb": _metric([row.peak_vram_mb for row in ordered]),
                "achieved_requests_per_second": _metric(
                    [row.achieved_requests_per_second for row in ordered]
                ),
                "goodput_requests_per_second": _metric(
                    [row.goodput_requests_per_second for row in ordered]
                ),
                "ttft_p50_ms": _metric([row.ttft.p50_ms for row in ordered]),
                "ttft_p99_ms": _metric([row.ttft.p99_ms for row in ordered]),
                "tpot_p50_ms": _metric([row.tpot.p50_ms for row in ordered]),
                "tpot_p99_ms": _metric([row.tpot.p99_ms for row in ordered]),
                "itl_p99_ms": _metric([row.itl.p99_ms for row in ordered]),
                "end_to_end_p99_ms": _metric([row.end_to_end.p99_ms for row in ordered]),
            }
        )

    by_identity = {
        (record.profile_id, record.context_id, record.repeat_index): record for record in records
    }
    contexts = sorted({record.context_id for record in records})
    paired: list[dict[str, object]] = []
    for context_id in contexts:
        repeat_indices = sorted(
            {
                record.repeat_index
                for record in records
                if record.context_id == context_id
                and record.profile_id in {"bf16-auto", "fp8-e5m2"}
            }
        )
        pairs: list[dict[str, object]] = []
        for repeat_index in repeat_indices:
            baseline = by_identity.get(("bf16-auto", context_id, repeat_index))
            fp8 = by_identity.get(("fp8-e5m2", context_id, repeat_index))
            if baseline is None or fp8 is None:
                continue
            pairs.append(
                {
                    "repeat_index": repeat_index,
                    "cached_tokens_ratio": fp8.cached_tokens / baseline.cached_tokens,
                    "achieved_change_percent": _percent_change(
                        fp8.achieved_requests_per_second,
                        baseline.achieved_requests_per_second,
                    ),
                    "goodput_change_percent": _percent_change(
                        fp8.goodput_requests_per_second,
                        baseline.goodput_requests_per_second,
                    ),
                    "ttft_p50_change_percent": _percent_change(
                        fp8.ttft.p50_ms, baseline.ttft.p50_ms
                    ),
                    "tpot_p50_change_percent": _percent_change(
                        fp8.tpot.p50_ms, baseline.tpot.p50_ms
                    ),
                }
            )
        if not pairs:
            continue
        paired.append(
            {
                "context_id": context_id,
                "pairs": pairs,
                "cached_tokens_ratio": _metric(
                    [cast(float, pair["cached_tokens_ratio"]) for pair in pairs]
                ),
                "achieved_change_percent": _optional_metric(
                    [cast(float, pair["achieved_change_percent"]) for pair in pairs]
                ),
                "goodput_change_percent": _optional_metric(
                    [cast(float, pair["goodput_change_percent"]) for pair in pairs]
                ),
                "ttft_p50_change_percent": _optional_metric(
                    [cast(float, pair["ttft_p50_change_percent"]) for pair in pairs]
                ),
                "tpot_p50_change_percent": _optional_metric(
                    [cast(float, pair["tpot_p50_change_percent"]) for pair in pairs]
                ),
            }
        )
    return {
        "schema_version": "longctx-m2-fp8-analysis.v1",
        "record_count": len(records),
        "groups": groups,
        "paired_fp8_vs_bf16": paired,
        "single_run_selection_used": False,
    }


__all__ = ["M2FP8TrialRecord", "M2LatencyPercentiles", "analyze_m2_fp8_records"]
