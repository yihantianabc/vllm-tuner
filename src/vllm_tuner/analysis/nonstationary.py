"""Phase-level metrics and a non-deployable fixed-policy Oracle."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from vllm_tuner.workloads.trace import TraceEntry, WorkloadTrace

LATENCY_METRICS = ("ttft_ms", "tpot_ms", "itl_ms", "e2e_ms", "client_wait_ms")
PERCENTILES = (50, 95, 99)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _percentile(values: Sequence[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _is_slo_good(row: Mapping[str, Any], slo: Mapping[str, float | None]) -> bool:
    if row.get("status") != "success":
        return False
    for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
        threshold = slo.get(metric)
        if threshold is None:
            continue
        value = _finite_number(row.get(metric))
        if value is None or value > threshold:
            return False
    return True


def _summarize_rows(
    entries: Sequence[TraceEntry],
    rows: Sequence[Mapping[str, Any]],
    slo: Mapping[str, float | None],
) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "success"]
    result: dict[str, Any] = {
        "requests": len(rows),
        "successful_requests": len(successful),
        "failed_requests": len(rows) - len(successful),
        "success_fraction": len(successful) / len(rows),
        "slo_good_requests": sum(_is_slo_good(row, slo) for row in rows),
        "slo_good_fraction": sum(_is_slo_good(row, slo) for row in rows) / len(rows),
        "input_tokens": sum(entry.input_tokens for entry in entries),
        "requested_output_tokens": sum(entry.output_tokens for entry in entries),
        "actual_output_tokens": sum(int(row.get("output_tokens") or 0) for row in successful),
    }
    latency_values: dict[str, list[float]] = defaultdict(list)
    for row in successful:
        for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
            value = _finite_number(row.get(metric))
            if value is not None:
                latency_values[metric].append(value)
        raw_itl = row.get("itl_ms")
        if isinstance(raw_itl, list):
            latency_values["itl_ms"].extend(
                value for raw_value in raw_itl if (value := _finite_number(raw_value)) is not None
            )
        scheduled_at = _finite_number(row.get("scheduled_at"))
        sent_at = _finite_number(row.get("sent_at"))
        if scheduled_at is not None and sent_at is not None:
            latency_values["client_wait_ms"].append(max(0.0, (sent_at - scheduled_at) / 1e6))
    for metric in LATENCY_METRICS:
        for percentile in PERCENTILES:
            result[f"p{percentile}_{metric}"] = _percentile(latency_values[metric], percentile)

    scheduled = [_finite_number(row.get("scheduled_at")) for row in rows]
    finished = [_finite_number(row.get("finished_at")) for row in successful]
    valid_scheduled = [value for value in scheduled if value is not None]
    valid_finished = [value for value in finished if value is not None]
    measurement_span_s = None
    if valid_scheduled and valid_finished:
        measurement_span_s = (max(valid_finished) - min(valid_scheduled)) / 1e9
        if measurement_span_s <= 0:
            measurement_span_s = None
    result["measurement_span_s"] = measurement_span_s
    result["achieved_requests_per_sec"] = (
        len(successful) / measurement_span_s if measurement_span_s else None
    )
    result["output_tokens_per_sec"] = (
        result["actual_output_tokens"] / measurement_span_s if measurement_span_s else None
    )
    return result


def summarize_labeled_requests(
    trace: WorkloadTrace,
    request_rows: Sequence[Mapping[str, Any]],
    *,
    slo: Mapping[str, float | None],
) -> dict[str, Any]:
    """Join one trial to offline phase labels and reject incomplete evidence."""
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for row in request_rows:
        request_id = row.get("request_id")
        if not isinstance(request_id, str):
            raise ValueError("request result is missing a string request_id")
        if request_id in rows_by_id:
            raise ValueError(f"duplicate request result: {request_id}")
        rows_by_id[request_id] = row
    trace_ids = {entry.request_id for entry in trace.entries}
    if set(rows_by_id) != trace_ids:
        missing = sorted(trace_ids.difference(rows_by_id))
        unexpected = sorted(set(rows_by_id).difference(trace_ids))
        raise ValueError(
            f"request evidence does not match trace; missing={missing}, unexpected={unexpected}"
        )

    phase_entries: dict[str, list[TraceEntry]] = defaultdict(list)
    for entry in trace.entries:
        phase_entries[entry.profile].append(entry)
    phases = {
        phase: _summarize_rows(
            entries,
            [rows_by_id[entry.request_id] for entry in entries],
            slo,
        )
        for phase, entries in phase_entries.items()
    }
    return {
        "overall": _summarize_rows(
            trace.entries,
            [rows_by_id[entry.request_id] for entry in trace.entries],
            slo,
        ),
        "phases": phases,
    }


def _median_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = set().union(*(row.keys() for row in rows))
    aggregate: dict[str, Any] = {"trials": len(rows)}
    for key in sorted(keys):
        values = [_finite_number(row.get(key)) for row in rows]
        numeric = [value for value in values if value is not None]
        if len(numeric) == len(rows):
            aggregate[key] = statistics.median(numeric)
    return aggregate


def aggregate_policy_trials(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Take per-metric medians without pooling requests across repetitions."""
    if not trials:
        raise ValueError("at least one complete policy trial is required")
    phase_names = list(trials[0]["phases"])
    if any(list(trial["phases"]) != phase_names for trial in trials):
        raise ValueError("policy trials do not share the same ordered phases")
    return {
        "trials": len(trials),
        "overall": _median_metrics([trial["overall"] for trial in trials]),
        "phases": {
            phase: _median_metrics([trial["phases"][phase] for trial in trials])
            for phase in phase_names
        },
    }


def _oracle_score(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    def lower_is_better(name: str) -> float:
        value = _finite_number(metrics.get(name))
        return -value if value is not None else -math.inf

    return (
        float(metrics.get("slo_good_fraction", 0.0)),
        float(metrics.get("achieved_requests_per_sec", 0.0)),
        lower_is_better("p99_e2e_ms"),
        lower_is_better("p99_ttft_ms"),
        lower_is_better("p99_tpot_ms"),
        lower_is_better("p99_itl_ms"),
    )


def select_phase_oracle(
    policies: Mapping[str, Mapping[str, Any]],
    *,
    eligible_policies: Sequence[str],
) -> dict[str, Any]:
    """Select a fixed policy after seeing each phase; this is analysis-only."""
    if not eligible_policies:
        raise ValueError("the Oracle requires at least one eligible fixed policy")
    missing = [name for name in eligible_policies if name not in policies]
    if missing:
        raise ValueError(f"Oracle policies are missing: {missing}")
    phase_names = list(policies[eligible_policies[0]]["phases"])
    selections: dict[str, Any] = {}
    for phase in phase_names:
        candidates = {name: policies[name]["phases"][phase] for name in eligible_policies}
        winner = max(eligible_policies, key=lambda name: _oracle_score(candidates[name]))
        selections[phase] = {
            "policy": winner,
            "metrics": candidates[winner],
            "candidate_scores": {
                name: list(_oracle_score(metrics)) for name, metrics in candidates.items()
            },
        }
    return {
        "deployable": False,
        "selection_rule": (
            "max SLO-good fraction, then achieved req/s, then lower p99 E2E/TTFT/TPOT/ITL"
        ),
        "phases": selections,
        "distinct_winners": sorted({value["policy"] for value in selections.values()}),
    }
