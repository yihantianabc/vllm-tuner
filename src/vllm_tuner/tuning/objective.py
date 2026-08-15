"""SLO goodput and hard-constraint evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from vllm_tuner.config.models import Constraints, SLOConfig


def _value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _milliseconds(record: Any, primary: str, seconds_name: str) -> Optional[float]:
    value = _value(record, primary)
    if value is not None:
        return float(value)
    stem = primary.removesuffix("_ms")
    value = _value(record, f"{stem}_ns")
    if value is not None:
        return float(value) / 1_000_000.0
    value = _value(record, seconds_name)
    return float(value) * 1000.0 if value is not None else None


def _successful(record: Any) -> bool:
    status = _value(record, "status", "")
    if hasattr(status, "value"):
        status = status.value
    success = _value(record, "success")
    if success is not None:
        return bool(success)
    return str(status).upper() in {"OK", "SUCCESS", "COMPLETE", "COMPLETED"}


def _request_reports_oom(record: Any) -> bool:
    """Recognize explicit allocator OOM evidence carried by a failed request."""
    evidence = " ".join(
        str(_value(record, field, "") or "") for field in ("error_type", "error_message")
    ).lower()
    markers = (
        "cuda out of memory",
        "outofmemoryerror",
        "out_of_memory",
        "cublas_status_alloc_failed",
        "hip out of memory",
    )
    return any(marker in evidence for marker in markers)


@dataclass(frozen=True)
class RequestSLOResult:
    """Per-request pass/fail decision with explicit reasons."""

    request_id: str
    good: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class ConstraintResult:
    """Hard trial feasibility separate from the maximization value."""

    feasible: bool
    violations: tuple[str, ...]
    values: dict[str, float]


@dataclass(frozen=True)
class ObjectiveResult:
    """Complete single-objective reduction for a measurement window."""

    goodput_requests_per_sec: float
    offered_requests_per_sec: float
    achieved_requests_per_sec: float
    completed_requests: int
    good_requests: int
    failed_requests: int
    total_input_tokens: int
    total_output_tokens: int
    request_slo: tuple[RequestSLOResult, ...]
    constraints: ConstraintResult


def evaluate_request_slo(request: Any, slo: SLOConfig) -> RequestSLOResult:
    """Apply TTFT, TPOT, and E2E thresholds to one successful request."""
    request_id = str(_value(request, "request_id", "unknown"))
    violations: list[str] = []
    if not _successful(request):
        violations.append("request_failed")

    ttft_ms = _milliseconds(request, "ttft_ms", "ttft_seconds")
    tpot_ms = _milliseconds(request, "tpot_ms", "tpot_seconds")
    if (
        tpot_ms is None
        and _successful(request)
        and int(_value(request, "output_tokens", 0) or 0) <= 1
    ):
        tpot_ms = 0.0
    e2e_ms = _milliseconds(request, "e2e_ms", "e2e_seconds")
    thresholds = (
        ("ttft", ttft_ms, slo.ttft_ms),
        ("tpot", tpot_ms, slo.tpot_ms),
        ("e2e", e2e_ms, slo.e2e_ms),
    )
    for name, observed, threshold in thresholds:
        if threshold is None:
            continue
        if observed is None:
            violations.append(f"missing_{name}")
        elif observed > threshold:
            violations.append(f"{name}_slo")
    return RequestSLOResult(request_id, not violations, tuple(violations))


def _percentile(values: list[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile, method="linear"))


def evaluate_constraints(
    requests: list[Any],
    *,
    measurement_seconds: float,
    slo: SLOConfig,
    constraints: Constraints,
    engine: Optional[Mapping[str, Any]] = None,
    gpu: Optional[Mapping[str, Any]] = None,
    server_alive: bool = True,
) -> ConstraintResult:
    """Evaluate errors, OOM, service exit, p99 SLO, and peak VRAM."""
    if measurement_seconds <= 0:
        raise ValueError("measurement_seconds must be positive")
    engine = engine or {}
    gpu = gpu or {}
    total = len(requests)
    failures = sum(not _successful(request) for request in requests)
    error_rate = failures / total if total else 1.0
    engine_oom_count = int(engine.get("oom_count", engine.get("oom_errors", 0)) or 0)
    request_oom_count = sum(_request_reports_oom(request) for request in requests)
    # Engine and client evidence can describe the same allocator failure, so a
    # maximum is a conservative count while either source remains a hard signal.
    oom_count = max(engine_oom_count, request_oom_count)
    oom_detected = bool(engine.get("oom_detected", False)) or oom_count > 0
    peak_vram_mb = gpu.get("peak_memory_mb")
    memory_utilization = gpu.get("peak_memory_utilization")

    values: dict[str, float] = {
        "error_rate": float(error_rate),
        "oom_count": float(oom_count),
        "engine_oom_count": float(engine_oom_count),
        "request_oom_count": float(request_oom_count),
        "achieved_requests_per_sec": float(
            sum(_successful(request) for request in requests) / measurement_seconds
        ),
    }
    if peak_vram_mb is not None:
        values["peak_vram_mb"] = float(peak_vram_mb)
    if memory_utilization is not None:
        values["peak_memory_utilization"] = float(memory_utilization)

    violations: list[str] = []
    if error_rate > constraints.max_error_rate:
        violations.append("error_rate")
    if constraints.require_no_oom and oom_detected:
        violations.append("oom")
    if constraints.require_server_alive and not server_alive:
        violations.append("server_exit")
    if constraints.max_peak_vram_mb is not None:
        if peak_vram_mb is None:
            violations.append("missing_peak_vram")
        elif float(peak_vram_mb) > constraints.max_peak_vram_mb:
            violations.append("peak_vram")
    if constraints.max_memory_utilization is not None:
        if memory_utilization is None:
            violations.append("missing_memory_utilization")
        elif float(memory_utilization) > constraints.max_memory_utilization:
            violations.append("memory_utilization")
    if (
        constraints.throughput_min is not None
        and values["achieved_requests_per_sec"] < constraints.throughput_min
    ):
        violations.append("throughput_min")

    latency_fields = (
        ("ttft", "ttft_ms", "ttft_seconds", slo.ttft_ms),
        ("tpot", "tpot_ms", "tpot_seconds", slo.tpot_ms),
        ("e2e", "e2e_ms", "e2e_seconds", slo.e2e_ms),
    )
    successful = [request for request in requests if _successful(request)]
    if constraints.max_latency_ms is not None:
        e2e_samples = [
            observed
            for request in successful
            if (observed := _milliseconds(request, "e2e_ms", "e2e_seconds")) is not None
        ]
        if not e2e_samples:
            violations.append("missing_max_latency")
        else:
            values["max_e2e_ms"] = max(e2e_samples)
            if values["max_e2e_ms"] > constraints.max_latency_ms:
                violations.append("max_latency")
    for name, primary, seconds, threshold in latency_fields:
        if threshold is None:
            continue
        samples = [
            observed
            for request in successful
            if (
                observed := (
                    0.0
                    if name == "tpot" and int(_value(request, "output_tokens", 0) or 0) <= 1
                    else _milliseconds(request, primary, seconds)
                )
            )
            is not None
        ]
        p99 = _percentile(samples, 99)
        if p99 is None:
            violations.append(f"missing_p99_{name}")
        else:
            values[f"p99_{name}_ms"] = p99
            if p99 > threshold:
                violations.append(f"p99_{name}_slo")

    return ConstraintResult(not violations, tuple(violations), values)


def compute_slo_goodput(
    requests: Iterable[Any],
    *,
    measurement_seconds: float,
    offered_requests: Optional[int] = None,
    offered_requests_per_second: Optional[float] = None,
    slo: SLOConfig,
    constraints: Constraints,
    engine: Optional[Mapping[str, Any]] = None,
    gpu: Optional[Mapping[str, Any]] = None,
    server_alive: bool = True,
) -> ObjectiveResult:
    """Reduce raw request results to the only optimization objective."""
    request_list = list(requests)
    if measurement_seconds <= 0:
        raise ValueError("measurement_seconds must be positive")
    if offered_requests_per_second is not None and (
        not math.isfinite(offered_requests_per_second) or offered_requests_per_second <= 0
    ):
        raise ValueError("offered_requests_per_second must be finite and positive")
    decisions = tuple(evaluate_request_slo(request, slo) for request in request_list)
    completed = sum(_successful(request) for request in request_list)
    good = sum(decision.good for decision in decisions)
    total_input_tokens = sum(
        int(_value(request, "input_tokens", 0) or 0)
        for request in request_list
        if _successful(request)
    )
    total_output_tokens = sum(
        int(_value(request, "output_tokens", 0) or 0)
        for request in request_list
        if _successful(request)
    )
    offered = offered_requests if offered_requests is not None else len(request_list)
    constraint_result = evaluate_constraints(
        request_list,
        measurement_seconds=measurement_seconds,
        slo=slo,
        constraints=constraints,
        engine=engine,
        gpu=gpu,
        server_alive=server_alive,
    )
    return ObjectiveResult(
        goodput_requests_per_sec=good / measurement_seconds,
        offered_requests_per_sec=(
            offered_requests_per_second
            if offered_requests_per_second is not None
            else offered / measurement_seconds
        ),
        achieved_requests_per_sec=completed / measurement_seconds,
        completed_requests=completed,
        good_requests=good,
        failed_requests=len(request_list) - completed,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        request_slo=decisions,
        constraints=constraint_result,
    )
