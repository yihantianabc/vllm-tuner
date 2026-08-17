"""Tests for formal M1 capacity aggregation and knee detection."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ValidationError

from vllm_tuner.longctx.m1_capacity_analysis import (
    CapacityKneePolicy,
    CapacityTrialMetrics,
    FailedCapacityTrial,
    LatencyPercentiles,
    analyze_capacity_sweep,
)
from vllm_tuner.longctx.m1_capacity_boundaries import derive_capacity_boundaries


def _policy(**overrides: object) -> CapacityKneePolicy:
    values: dict[str, object] = {
        "minimum_trace_duration_seconds": 300.0,
        "minimum_load_points": 3,
        "maximum_marginal_achieved_gain_ratio": 0.20,
        "minimum_queue_growth_slope_waiting_requests_per_second": 0.05,
        "minimum_peak_waiting_requests": 5,
        "minimum_completion_fraction": 0.98,
        "minimum_achieved_fraction_of_empirical_offered": 0.80,
        "minimum_goodput_fraction_of_empirical_offered": 0.85,
        "minimum_slo_satisfied_fraction": 0.90,
        "maximum_preemptions_for_stable": 0,
        "maximum_timeouts_for_stable": 0,
        "maximum_p99_dispatch_delay_ms": 50.0,
        "require_zero_oom_events": True,
        "minimum_joint_signal_repeats": 2,
    }
    values.update(overrides)
    return CapacityKneePolicy.model_validate(values)


def _latency(scale: float = 1.0) -> LatencyPercentiles:
    return LatencyPercentiles(
        p50_ms=10.0 * scale,
        p95_ms=20.0 * scale,
        p99_ms=30.0 * scale,
    )


def _trial(
    *,
    load_id: str,
    repeat: Literal[0, 1, 2],
    offered: float,
    achieved: float,
    completion: float = 1.0,
    goodput: float = 0.0,
    slo_fraction: float = 0.95,
    queue_slope: float = 0.0,
    peak_waiting: int = 0,
    duration: float = 600.0,
    preemptions: int = 0,
    ooms: int = 0,
    timeouts: int = 0,
) -> CapacityTrialMetrics:
    return CapacityTrialMetrics(
        evidence_kind="formal_capacity_sweep",
        trial_id=f"long-{load_id}-r{repeat}",
        context_id="long",
        context_tokens=32_768,
        load_id=load_id,
        repeat_index=repeat,
        trace_id=f"long-{load_id}-trace",
        planned_trace_duration_seconds=duration,
        target_offered_requests_per_second=offered,
        status="complete",
        observed_trace_duration_seconds=duration,
        empirical_offered_requests_per_second=offered,
        achieved_requests_per_second=achieved,
        p99_dispatch_delay_ms=1.0,
        completion_fraction=completion,
        goodput_requests_per_second=goodput,
        slo_satisfied_fraction=slo_fraction,
        queue_growth_slope_waiting_requests_per_second=queue_slope,
        peak_waiting_requests=peak_waiting,
        preemption_count=preemptions,
        oom_count=ooms,
        timeout_count=timeouts,
        ttft=_latency(1.0 + repeat / 10.0),
        tpot=_latency(0.5 + repeat / 20.0),
        itl=_latency(0.4 + repeat / 20.0),
        end_to_end=_latency(10.0 + repeat),
    )


def _valid_sweep() -> list[CapacityTrialMetrics]:
    trials: list[CapacityTrialMetrics] = []
    for repeat, achieved in enumerate((0.96, 0.98, 1.0)):
        trials.append(
            _trial(
                load_id="low",
                repeat=repeat,  # type: ignore[arg-type]
                offered=1.0,
                achieved=achieved,
                goodput=0.90,
            )
        )
    for repeat, achieved in enumerate((1.86, 1.90, 1.94)):
        trials.append(
            _trial(
                load_id="mid",
                repeat=repeat,  # type: ignore[arg-type]
                offered=2.0,
                achieved=achieved,
                goodput=1.80,
            )
        )
    for repeat, achieved in enumerate((1.91, 1.96, 2.0)):
        trials.append(
            _trial(
                load_id="high",
                repeat=repeat,  # type: ignore[arg-type]
                offered=3.0,
                achieved=achieved,
                completion=0.84,
                goodput=1.50,
                slo_fraction=0.70,
                queue_slope=0.20,
                peak_waiting=12,
                preemptions=1,
                timeouts=2,
            )
        )
    return trials


def test_capacity_sweep_reports_repeats_ranges_and_joint_knee() -> None:
    analysis = analyze_capacity_sweep(_valid_sweep(), _policy())

    assert analysis.passed is True
    assert analysis.initialization_evidence_used is False
    assert analysis.failure_reasons == ()
    context = analysis.contexts[0]
    assert context.knee.passed is True
    assert context.knee.last_stable_load_id == "mid"
    assert context.knee.first_bracketed_overload_load_id == "high"
    high = context.points[2]
    assert high.classification == "overloaded"
    assert high.metrics is not None
    assert tuple(trial.repeat_index for trial in high.trials) == (0, 1, 2)
    assert high.metrics.achieved_requests_per_second.median == pytest.approx(1.96)
    assert high.metrics.achieved_requests_per_second.minimum == pytest.approx(1.91)
    assert high.metrics.achieved_requests_per_second.maximum == pytest.approx(2.0)
    assert high.metrics.achieved_fraction_of_empirical_offered.median == pytest.approx(1.96 / 3.0)
    assert high.metrics.ttft_p99_ms.median == pytest.approx(33.0)
    assert high.signals.throughput_plateau is True
    assert high.signals.sustained_queue_growth is True
    assert high.signals.slo_goodput_breach is True
    assert high.signals.joint_overload_repeat_indices == (0, 1, 2)


def test_initialization_evidence_cannot_enter_capacity_analysis() -> None:
    payload = _trial(
        load_id="low",
        repeat=0,
        offered=1.0,
        achieved=0.9,
        goodput=0.9,
    ).model_dump()
    payload["evidence_kind"] = "planner_initialization"

    with pytest.raises(ValidationError, match="formal_capacity_sweep"):
        CapacityTrialMetrics.model_validate(payload)


def test_missing_repeat_fails_closed_with_explicit_reason() -> None:
    trials = [
        trial
        for trial in _valid_sweep()
        if not (trial.load_id == "mid" and trial.repeat_index == 2)
    ]

    analysis = analyze_capacity_sweep(trials, _policy())

    assert analysis.passed is False
    mid = next(point for point in analysis.contexts[0].points if point.load_id == "mid")
    assert mid.classification == "invalid"
    assert any(
        "requires_exactly_three_complete_repeats" in reason for reason in mid.validation_failures
    )
    assert any("load_mid" in reason for reason in analysis.failure_reasons)


def test_failed_trial_and_oom_are_preserved_without_fabricated_metrics() -> None:
    complete_trials = _valid_sweep()
    replaced = next(
        trial for trial in complete_trials if trial.load_id == "mid" and trial.repeat_index == 1
    )
    failure = FailedCapacityTrial(
        evidence_kind="formal_capacity_sweep",
        trial_id=replaced.trial_id,
        context_id=replaced.context_id,
        context_tokens=replaced.context_tokens,
        load_id=replaced.load_id,
        repeat_index=replaced.repeat_index,
        trace_id=replaced.trace_id,
        planned_trace_duration_seconds=replaced.planned_trace_duration_seconds,
        target_offered_requests_per_second=replaced.target_offered_requests_per_second,
        status="failed",
        failure_kind="oom",
        failure_reason="server exited during startup",
        observed_trace_duration_seconds=None,
        oom_observed=True,
        timeout_observed=False,
    )
    trials = [
        failure if trial.trial_id == replaced.trial_id else trial for trial in complete_trials
    ]

    analysis = analyze_capacity_sweep(trials, _policy())

    assert analysis.passed is False
    mid = next(point for point in analysis.contexts[0].points if point.load_id == "mid")
    assert mid.metrics is None
    assert isinstance(mid.trials[1], FailedCapacityTrial)
    assert any(reason.startswith("failed_trials:") for reason in mid.validation_failures)
    assert any(reason.startswith("oom_events_observed:") for reason in mid.validation_failures)


def test_three_signals_must_coincide_in_repeats() -> None:
    trials = _valid_sweep()
    for index, trial in enumerate(trials):
        if trial.load_id != "high":
            continue
        if trial.repeat_index == 0:
            trials[index] = trial.model_copy(
                update={
                    "completion_fraction": 1.0,
                    "slo_satisfied_fraction": 0.95,
                }
            )
        elif trial.repeat_index == 2:
            trials[index] = trial.model_copy(
                update={
                    "queue_growth_slope_waiting_requests_per_second": 0.0,
                    "peak_waiting_requests": 0,
                }
            )

    analysis = analyze_capacity_sweep(
        trials,
        _policy(
            minimum_achieved_fraction_of_empirical_offered=0.60,
            minimum_goodput_fraction_of_empirical_offered=0.40,
        ),
    )

    high = analysis.contexts[0].points[2]
    assert high.signals.throughput_plateau is True
    assert high.signals.sustained_queue_growth is True
    assert high.signals.slo_goodput_breach is True
    assert high.signals.joint_overload_repeat_indices == (1,)
    assert high.signals.joint_overload is False
    assert high.classification == "transitional"
    assert analysis.passed is False
    assert "no_joint_overload_point" in analysis.contexts[0].knee.failure_reasons


def test_policy_threshold_controls_plateau_instead_of_hidden_tuning() -> None:
    strict_plateau = _policy(maximum_marginal_achieved_gain_ratio=0.01)

    analysis = analyze_capacity_sweep(_valid_sweep(), strict_plateau)

    assert analysis.passed is False
    high = analysis.contexts[0].points[2]
    assert high.signals.throughput_plateau is False
    assert high.signals.joint_overload is False


def test_short_trace_and_unbracketed_curve_do_not_pass() -> None:
    trials = _valid_sweep()
    trials[0] = trials[0].model_copy(update={"observed_trace_duration_seconds": 60.0})
    short = analyze_capacity_sweep(trials, _policy())
    assert short.passed is False
    assert any("observed_trace_duration_below_policy" in reason for reason in short.failure_reasons)

    all_stable = [
        trial.model_copy(
            update={
                "achieved_requests_per_second": (
                    0.95 * trial.empirical_offered_requests_per_second
                ),
                "completion_fraction": 1.0,
                "goodput_requests_per_second": (0.9 * trial.empirical_offered_requests_per_second),
                "slo_satisfied_fraction": 0.95,
                "queue_growth_slope_waiting_requests_per_second": 0.0,
                "peak_waiting_requests": 0,
                "preemption_count": 0,
                "timeout_count": 0,
            }
        )
        for trial in _valid_sweep()
    ]
    unbracketed = analyze_capacity_sweep(all_stable, _policy())
    assert unbracketed.passed is False
    assert "no_joint_overload_point" in unbracketed.contexts[0].knee.failure_reasons


def test_client_dispatch_delay_invalidates_point_instead_of_faking_overload() -> None:
    trials = _valid_sweep()
    delayed = next(trial for trial in trials if trial.load_id == "high" and trial.repeat_index == 1)
    trials[trials.index(delayed)] = delayed.model_copy(update={"p99_dispatch_delay_ms": 75.0})

    analysis = analyze_capacity_sweep(trials, _policy())

    assert analysis.passed is False
    high = analysis.contexts[0].points[2]
    assert high.classification == "invalid"
    assert any(
        reason.startswith("p99_dispatch_delay_above_policy:") for reason in high.validation_failures
    )


def test_strict_schema_rejects_bad_latency_and_duplicate_repeat() -> None:
    with pytest.raises(ValidationError, match="p50 <= p95 <= p99"):
        LatencyPercentiles(p50_ms=20.0, p95_ms=10.0, p99_ms=30.0)

    failure_payload = {
        "evidence_kind": "formal_capacity_sweep",
        "trial_id": "failed-r0",
        "context_id": "long",
        "context_tokens": 32_768,
        "load_id": "high",
        "repeat_index": 0,
        "trace_id": "long-high-trace",
        "planned_trace_duration_seconds": 600.0,
        "target_offered_requests_per_second": 3.0,
        "status": "failed",
        "failure_kind": "oom",
        "failure_reason": "server exited",
        "observed_trace_duration_seconds": None,
        "oom_observed": False,
        "timeout_observed": False,
    }
    with pytest.raises(ValidationError, match="requires oom_observed"):
        FailedCapacityTrial.model_validate(failure_payload)

    trials = _valid_sweep()
    duplicate = trials[0].model_copy(update={"trial_id": "different-id"})
    with pytest.raises(ValueError, match="duplicate capacity repeat"):
        analyze_capacity_sweep([*trials, duplicate], _policy())


def test_v2_boundaries_preserve_a_joint_v1_knee() -> None:
    source = analyze_capacity_sweep(_valid_sweep(), _policy())

    derived = derive_capacity_boundaries(source)

    assert derived.accepted is True
    context = derived.contexts[0]
    assert context.slo_service_boundary.status == "bracketed"
    assert context.slo_service_boundary.last_stable is not None
    assert context.slo_service_boundary.last_stable.load_id == "mid"
    assert context.slo_service_boundary.first_slo_goodput_breach is not None
    assert context.slo_service_boundary.first_slo_goodput_breach.load_id == "high"
    assert context.joint_saturation_boundary.status == "bracketed"
    assert context.joint_saturation_boundary.last_pre_saturation is not None
    assert context.joint_saturation_boundary.last_pre_saturation.load_id == "mid"


def test_v2_separates_transitional_slo_breach_from_later_saturation() -> None:
    trials = _valid_sweep()
    trials = [
        (
            trial.model_copy(
                update={
                    "goodput_requests_per_second": 1.0,
                    "slo_satisfied_fraction": 0.50,
                }
            )
            if trial.load_id == "mid"
            else trial
        )
        for trial in trials
    ]
    source = analyze_capacity_sweep(trials, _policy())

    derived = derive_capacity_boundaries(source)

    assert source.passed is False
    assert source.contexts[0].points[1].classification == "transitional"
    assert "stable_and_first_overload_are_not_adjacent" in (source.contexts[0].knee.failure_reasons)
    context = derived.contexts[0]
    assert context.accepted is True
    assert context.slo_service_boundary.last_stable is not None
    assert context.slo_service_boundary.last_stable.load_id == "low"
    assert context.slo_service_boundary.first_slo_goodput_breach is not None
    assert context.slo_service_boundary.first_slo_goodput_breach.load_id == "mid"
    assert context.joint_saturation_boundary.last_pre_saturation is not None
    assert context.joint_saturation_boundary.last_pre_saturation.load_id == "mid"
    assert context.joint_saturation_boundary.first_joint_overload is not None
    assert context.joint_saturation_boundary.first_joint_overload.load_id == "high"


def test_v2_accepts_preregistered_left_censored_service_boundary() -> None:
    trials = _valid_sweep()
    trials = [
        (
            trial.model_copy(
                update={
                    "goodput_requests_per_second": 0.20,
                    "slo_satisfied_fraction": 0.50,
                }
            )
            if trial.load_id in {"low", "mid"}
            else trial
        )
        for trial in trials
    ]
    source = analyze_capacity_sweep(trials, _policy())

    derived = derive_capacity_boundaries(source)

    assert source.passed is False
    assert "no_stable_point_before_first_overload" in source.contexts[0].knee.failure_reasons
    context = derived.contexts[0]
    assert context.accepted is True
    assert context.slo_service_boundary.status == "left-censored-below-lowest-load"
    assert context.slo_service_boundary.last_stable is None
    assert context.slo_service_boundary.first_slo_goodput_breach is not None
    assert context.slo_service_boundary.first_slo_goodput_breach.load_id == "low"
    assert context.joint_saturation_boundary.status == "bracketed"


def test_v2_rejects_ineligible_capacity_evidence() -> None:
    trials = [
        trial
        for trial in _valid_sweep()
        if not (trial.load_id == "mid" and trial.repeat_index == 2)
    ]
    source = analyze_capacity_sweep(trials, _policy())

    derived = derive_capacity_boundaries(source)

    assert derived.accepted is False
    assert derived.contexts[0].all_points_eligible is False
    assert "not_all_capacity_points_are_eligible" in derived.contexts[0].failure_reasons
