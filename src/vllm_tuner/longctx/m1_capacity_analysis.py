"""Fail-closed aggregation and knee detection for formal M1 capacity sweeps."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Annotated, Literal, Optional

from pydantic import Field, field_validator, model_validator

from .kv_capacity_planner import StrictFrozenModel

CAPACITY_ANALYSIS_VERSION: Literal["longctx-m1-capacity.v1"] = "longctx-m1-capacity.v1"
REQUIRED_REPEAT_INDICES = (0, 1, 2)

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class LatencyPercentiles(StrictFrozenModel):
    """Required latency percentiles for one completed benchmark trial."""

    p50_ms: NonNegativeFiniteFloat
    p95_ms: NonNegativeFiniteFloat
    p99_ms: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def validate_percentile_order(self) -> "LatencyPercentiles":
        if not self.p50_ms <= self.p95_ms <= self.p99_ms:
            raise ValueError("latency percentiles must satisfy p50 <= p95 <= p99")
        return self


class CapacityTrialIdentity(StrictFrozenModel):
    """Identity and pre-registered load shared by complete and failed trials."""

    schema_version: Literal["longctx-m1-capacity.v1"] = CAPACITY_ANALYSIS_VERSION
    evidence_kind: Literal["formal_capacity_sweep"]
    trial_id: str
    context_id: str
    context_tokens: int = Field(gt=0)
    load_id: str
    repeat_index: Literal[0, 1, 2]
    trace_id: str
    planned_trace_duration_seconds: PositiveFiniteFloat
    target_offered_requests_per_second: PositiveFiniteFloat

    @field_validator("trial_id", "context_id", "load_id", "trace_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("capacity evidence identifiers must be non-empty and trimmed")
        return value


class CapacityTrialMetrics(CapacityTrialIdentity):
    """Strict metrics from one completed formal, long-trace capacity run.

    evidence_kind deliberately excludes Planner initialization probes. A caller
    cannot turn initialization concurrency into a capacity knee through this schema.
    """

    status: Literal["complete"]
    observed_trace_duration_seconds: PositiveFiniteFloat
    empirical_offered_requests_per_second: PositiveFiniteFloat
    achieved_requests_per_second: NonNegativeFiniteFloat
    p99_dispatch_delay_ms: NonNegativeFiniteFloat
    completion_fraction: Fraction
    goodput_requests_per_second: NonNegativeFiniteFloat
    slo_satisfied_fraction: Fraction

    queue_growth_slope_waiting_requests_per_second: FiniteFloat
    peak_waiting_requests: int = Field(ge=0)
    preemption_count: int = Field(ge=0)
    oom_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)

    ttft: LatencyPercentiles
    tpot: LatencyPercentiles
    itl: LatencyPercentiles
    end_to_end: LatencyPercentiles

    @model_validator(mode="after")
    def validate_metrics(self) -> "CapacityTrialMetrics":
        if self.goodput_requests_per_second > self.achieved_requests_per_second:
            raise ValueError("goodput_requests_per_second must not exceed achieved throughput")
        return self

    @property
    def goodput_fraction_of_empirical_offered(self) -> float:
        """Return SLO goodput divided by the empirically scheduled rate."""

        return self.goodput_requests_per_second / self.empirical_offered_requests_per_second

    @property
    def achieved_fraction_of_empirical_offered(self) -> float:
        """Return achieved throughput divided by the empirically scheduled rate."""

        return self.achieved_requests_per_second / self.empirical_offered_requests_per_second


class FailedCapacityTrial(CapacityTrialIdentity):
    """Strict failure evidence that never fabricates unavailable performance metrics."""

    status: Literal["failed"]
    failure_kind: Literal[
        "startup_failure",
        "runner_failure",
        "server_failure",
        "telemetry_failure",
        "cleanup_failure",
        "oom",
        "timeout",
    ]
    failure_reason: str = Field(min_length=1)
    observed_trace_duration_seconds: Optional[NonNegativeFiniteFloat] = None
    oom_observed: bool
    timeout_observed: bool

    @field_validator("failure_reason")
    @classmethod
    def validate_failure_reason(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("failure_reason must be trimmed")
        return value

    @model_validator(mode="after")
    def validate_failure_observation(self) -> "FailedCapacityTrial":
        if self.failure_kind == "oom" and not self.oom_observed:
            raise ValueError("oom failure_kind requires oom_observed")
        if self.failure_kind == "timeout" and not self.timeout_observed:
            raise ValueError("timeout failure_kind requires timeout_observed")
        return self


CapacityTrialRecord = Annotated[
    CapacityTrialMetrics | FailedCapacityTrial,
    Field(discriminator="status"),
]


class CapacityKneePolicy(StrictFrozenModel):
    """Pre-registered thresholds used to classify a capacity knee."""

    schema_version: Literal["longctx-m1-capacity.v1"] = CAPACITY_ANALYSIS_VERSION
    minimum_trace_duration_seconds: PositiveFiniteFloat
    minimum_load_points: int = Field(ge=2)
    maximum_marginal_achieved_gain_ratio: Fraction
    minimum_queue_growth_slope_waiting_requests_per_second: PositiveFiniteFloat
    minimum_peak_waiting_requests: int = Field(gt=0)
    minimum_completion_fraction: Fraction
    minimum_achieved_fraction_of_empirical_offered: Fraction
    minimum_goodput_fraction_of_empirical_offered: Fraction
    minimum_slo_satisfied_fraction: Fraction
    maximum_preemptions_for_stable: int = Field(ge=0)
    maximum_timeouts_for_stable: int = Field(ge=0)
    maximum_p99_dispatch_delay_ms: NonNegativeFiniteFloat
    require_zero_oom_events: Literal[True]
    minimum_joint_signal_repeats: Literal[2, 3]


class MetricRange(StrictFrozenModel):
    """Median and full repeat range for one scalar metric."""

    median: FiniteFloat
    minimum: FiniteFloat
    maximum: FiniteFloat

    @model_validator(mode="after")
    def validate_range(self) -> "MetricRange":
        if not self.minimum <= self.median <= self.maximum:
            raise ValueError("metric range must satisfy minimum <= median <= maximum")
        return self


class CapacityPointMetricSummaries(StrictFrozenModel):
    """Median and range for every required metric at one offered-load point."""

    planned_trace_duration_seconds: MetricRange
    observed_trace_duration_seconds: MetricRange
    target_offered_requests_per_second: MetricRange
    empirical_offered_requests_per_second: MetricRange
    achieved_requests_per_second: MetricRange
    achieved_fraction_of_empirical_offered: MetricRange
    p99_dispatch_delay_ms: MetricRange
    completion_fraction: MetricRange
    goodput_requests_per_second: MetricRange
    goodput_fraction_of_empirical_offered: MetricRange
    slo_satisfied_fraction: MetricRange
    queue_growth_slope_waiting_requests_per_second: MetricRange
    peak_waiting_requests: MetricRange
    preemption_count: MetricRange
    oom_count: MetricRange
    timeout_count: MetricRange
    ttft_p50_ms: MetricRange
    ttft_p95_ms: MetricRange
    ttft_p99_ms: MetricRange
    tpot_p50_ms: MetricRange
    tpot_p95_ms: MetricRange
    tpot_p99_ms: MetricRange
    itl_p50_ms: MetricRange
    itl_p95_ms: MetricRange
    itl_p99_ms: MetricRange
    end_to_end_p50_ms: MetricRange
    end_to_end_p95_ms: MetricRange
    end_to_end_p99_ms: MetricRange


class CapacityPointSignals(StrictFrozenModel):
    """Repeat-level evidence for the three jointly required overload signals."""

    marginal_achieved_gain_ratio: Optional[MetricRange]
    throughput_plateau_repeat_indices: tuple[int, ...]
    queue_growth_repeat_indices: tuple[int, ...]
    slo_goodput_breach_repeat_indices: tuple[int, ...]
    joint_overload_repeat_indices: tuple[int, ...]
    throughput_plateau: bool
    sustained_queue_growth: bool
    slo_goodput_breach: bool
    joint_overload: bool


class CapacityPointAnalysis(StrictFrozenModel):
    """All raw repeats, aggregates, and classification for one load point."""

    context_id: str
    context_tokens: int
    load_id: str
    trials: tuple[CapacityTrialRecord, ...]
    target_offered_requests_per_second: MetricRange
    metrics: Optional[CapacityPointMetricSummaries]
    signals: CapacityPointSignals
    eligible: bool
    validation_failures: tuple[str, ...]
    classification: Literal["stable", "overloaded", "transitional", "invalid"]


class CapacityKneeResult(StrictFrozenModel):
    """A stable-to-overload bracket, or explicit reasons it was not accepted."""

    passed: bool
    last_stable_load_id: Optional[str]
    last_stable_target_offered_requests_per_second: Optional[float]
    last_stable_empirical_offered_requests_per_second: Optional[float]
    first_bracketed_overload_load_id: Optional[str]
    first_bracketed_overload_target_offered_requests_per_second: Optional[float]
    first_bracketed_overload_empirical_offered_requests_per_second: Optional[float]
    failure_reasons: tuple[str, ...]


class ContextCapacityAnalysis(StrictFrozenModel):
    """Ordered capacity curve and knee result for one context profile."""

    context_id: str
    context_tokens: int
    points: tuple[CapacityPointAnalysis, ...]
    knee: CapacityKneeResult


class CapacitySweepAnalysis(StrictFrozenModel):
    """Fail-closed result across all context profiles in the supplied sweep."""

    schema_version: Literal["longctx-m1-capacity.v1"] = CAPACITY_ANALYSIS_VERSION
    evidence_kind: Literal["formal_capacity_sweep"] = "formal_capacity_sweep"
    initialization_evidence_used: Literal[False] = False
    policy: CapacityKneePolicy
    contexts: tuple[ContextCapacityAnalysis, ...]
    passed: bool
    failure_reasons: tuple[str, ...]


def _summary(values: Sequence[int | float]) -> MetricRange:
    if not values:
        raise ValueError("cannot summarize an empty metric sequence")
    numeric = [float(value) for value in values]
    return MetricRange(
        median=float(statistics.median(numeric)),
        minimum=float(min(numeric)),
        maximum=float(max(numeric)),
    )


def _metric_summaries(
    trials: Sequence[CapacityTrialMetrics],
) -> CapacityPointMetricSummaries:
    return CapacityPointMetricSummaries(
        planned_trace_duration_seconds=_summary(
            [trial.planned_trace_duration_seconds for trial in trials]
        ),
        observed_trace_duration_seconds=_summary(
            [trial.observed_trace_duration_seconds for trial in trials]
        ),
        target_offered_requests_per_second=_summary(
            [trial.target_offered_requests_per_second for trial in trials]
        ),
        empirical_offered_requests_per_second=_summary(
            [trial.empirical_offered_requests_per_second for trial in trials]
        ),
        achieved_requests_per_second=_summary(
            [trial.achieved_requests_per_second for trial in trials]
        ),
        achieved_fraction_of_empirical_offered=_summary(
            [trial.achieved_fraction_of_empirical_offered for trial in trials]
        ),
        p99_dispatch_delay_ms=_summary([trial.p99_dispatch_delay_ms for trial in trials]),
        completion_fraction=_summary([trial.completion_fraction for trial in trials]),
        goodput_requests_per_second=_summary(
            [trial.goodput_requests_per_second for trial in trials]
        ),
        goodput_fraction_of_empirical_offered=_summary(
            [trial.goodput_fraction_of_empirical_offered for trial in trials]
        ),
        slo_satisfied_fraction=_summary([trial.slo_satisfied_fraction for trial in trials]),
        queue_growth_slope_waiting_requests_per_second=_summary(
            [trial.queue_growth_slope_waiting_requests_per_second for trial in trials]
        ),
        peak_waiting_requests=_summary([trial.peak_waiting_requests for trial in trials]),
        preemption_count=_summary([trial.preemption_count for trial in trials]),
        oom_count=_summary([trial.oom_count for trial in trials]),
        timeout_count=_summary([trial.timeout_count for trial in trials]),
        ttft_p50_ms=_summary([trial.ttft.p50_ms for trial in trials]),
        ttft_p95_ms=_summary([trial.ttft.p95_ms for trial in trials]),
        ttft_p99_ms=_summary([trial.ttft.p99_ms for trial in trials]),
        tpot_p50_ms=_summary([trial.tpot.p50_ms for trial in trials]),
        tpot_p95_ms=_summary([trial.tpot.p95_ms for trial in trials]),
        tpot_p99_ms=_summary([trial.tpot.p99_ms for trial in trials]),
        itl_p50_ms=_summary([trial.itl.p50_ms for trial in trials]),
        itl_p95_ms=_summary([trial.itl.p95_ms for trial in trials]),
        itl_p99_ms=_summary([trial.itl.p99_ms for trial in trials]),
        end_to_end_p50_ms=_summary([trial.end_to_end.p50_ms for trial in trials]),
        end_to_end_p95_ms=_summary([trial.end_to_end.p95_ms for trial in trials]),
        end_to_end_p99_ms=_summary([trial.end_to_end.p99_ms for trial in trials]),
    )


def _point_failures(
    trials: Sequence[CapacityTrialRecord],
    policy: CapacityKneePolicy,
) -> tuple[str, ...]:
    failures: list[str] = []
    repeats = tuple(sorted(trial.repeat_index for trial in trials))
    if repeats != REQUIRED_REPEAT_INDICES:
        failures.append(
            "requires_exactly_repeats_0_1_2:"
            f"observed_{'_'.join(str(repeat) for repeat in repeats) or 'none'}"
        )
    complete = tuple(trial for trial in trials if isinstance(trial, CapacityTrialMetrics))
    complete_repeats = tuple(sorted(trial.repeat_index for trial in complete))
    if complete_repeats != REQUIRED_REPEAT_INDICES:
        failures.append(
            "requires_exactly_three_complete_repeats:"
            f"observed_{'_'.join(str(repeat) for repeat in complete_repeats) or 'none'}"
        )
    failed = tuple(trial for trial in trials if isinstance(trial, FailedCapacityTrial))
    if failed:
        details = ",".join(f"{trial.trial_id}({trial.failure_kind})" for trial in failed)
        failures.append(f"failed_trials:{details}")
    trace_ids = {trial.trace_id for trial in trials}
    if len(trace_ids) != 1:
        failures.append("repeat_trace_ids_do_not_match")
    targets = {trial.target_offered_requests_per_second for trial in trials}
    if len(targets) != 1:
        failures.append("repeat_target_offered_rates_do_not_match")
    short_planned_ids = tuple(
        trial.trial_id
        for trial in trials
        if trial.planned_trace_duration_seconds < policy.minimum_trace_duration_seconds
    )
    if short_planned_ids:
        failures.append(f"planned_trace_duration_below_policy:{','.join(short_planned_ids)}")
    short_observed_ids = tuple(
        trial.trial_id
        for trial in complete
        if trial.observed_trace_duration_seconds < policy.minimum_trace_duration_seconds
    )
    if short_observed_ids:
        failures.append(f"observed_trace_duration_below_policy:{','.join(short_observed_ids)}")
    dispatch_delay_ids = tuple(
        trial.trial_id
        for trial in complete
        if trial.p99_dispatch_delay_ms > policy.maximum_p99_dispatch_delay_ms
    )
    if dispatch_delay_ids:
        failures.append(f"p99_dispatch_delay_above_policy:{','.join(dispatch_delay_ids)}")
    if policy.require_zero_oom_events:
        oom_ids = [trial.trial_id for trial in complete if trial.oom_count > 0]
        oom_ids.extend(trial.trial_id for trial in failed if trial.oom_observed)
        if oom_ids:
            failures.append(f"oom_events_observed:{','.join(oom_ids)}")
    return tuple(failures)


def _signal_indices(
    trials: Sequence[CapacityTrialMetrics],
    policy: CapacityKneePolicy,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    queue = tuple(
        trial.repeat_index
        for trial in trials
        if trial.queue_growth_slope_waiting_requests_per_second
        >= policy.minimum_queue_growth_slope_waiting_requests_per_second
        and trial.peak_waiting_requests >= policy.minimum_peak_waiting_requests
    )
    slo = tuple(
        trial.repeat_index
        for trial in trials
        if trial.completion_fraction < policy.minimum_completion_fraction
        or trial.achieved_fraction_of_empirical_offered
        < policy.minimum_achieved_fraction_of_empirical_offered
        or trial.goodput_fraction_of_empirical_offered
        < policy.minimum_goodput_fraction_of_empirical_offered
        or trial.slo_satisfied_fraction < policy.minimum_slo_satisfied_fraction
    )
    return queue, slo


def _empty_signals(
    trials: Sequence[CapacityTrialMetrics],
    policy: CapacityKneePolicy,
) -> CapacityPointSignals:
    queue, slo = _signal_indices(trials, policy)
    required = policy.minimum_joint_signal_repeats
    return CapacityPointSignals(
        marginal_achieved_gain_ratio=None,
        throughput_plateau_repeat_indices=(),
        queue_growth_repeat_indices=queue,
        slo_goodput_breach_repeat_indices=slo,
        joint_overload_repeat_indices=(),
        throughput_plateau=False,
        sustained_queue_growth=len(queue) >= required,
        slo_goodput_breach=len(slo) >= required,
        joint_overload=False,
    )


def _paired_signals(
    previous: CapacityPointAnalysis,
    current_trials: Sequence[CapacityTrialMetrics],
    policy: CapacityKneePolicy,
) -> tuple[CapacityPointSignals, tuple[str, ...]]:
    if not previous.eligible:
        return _empty_signals(current_trials, policy), (
            "previous_load_point_ineligible_for_marginal_gain",
        )
    previous_by_repeat = {
        trial.repeat_index: trial
        for trial in previous.trials
        if isinstance(trial, CapacityTrialMetrics)
    }
    ratios: list[float] = []
    ratio_by_repeat: dict[int, float] = {}
    failures: list[str] = []
    for trial in current_trials:
        previous_trial = previous_by_repeat.get(trial.repeat_index)
        if previous_trial is None:
            failures.append(f"previous_load_missing_repeat_{trial.repeat_index}")
            continue
        offered_delta = (
            trial.empirical_offered_requests_per_second
            - previous_trial.empirical_offered_requests_per_second
        )
        if offered_delta <= 0.0:
            failures.append(f"non_increasing_offered_load_repeat_{trial.repeat_index}")
            continue
        achieved_delta = (
            trial.achieved_requests_per_second - previous_trial.achieved_requests_per_second
        )
        ratio = achieved_delta / offered_delta
        ratios.append(ratio)
        ratio_by_repeat[trial.repeat_index] = ratio

    queue, slo = _signal_indices(current_trials, policy)
    plateau = tuple(
        repeat
        for repeat, ratio in sorted(ratio_by_repeat.items())
        if ratio <= policy.maximum_marginal_achieved_gain_ratio
    )
    joint = tuple(sorted(set(plateau) & set(queue) & set(slo)))
    required = policy.minimum_joint_signal_repeats
    signals = CapacityPointSignals(
        marginal_achieved_gain_ratio=_summary(ratios) if ratios else None,
        throughput_plateau_repeat_indices=plateau,
        queue_growth_repeat_indices=queue,
        slo_goodput_breach_repeat_indices=slo,
        joint_overload_repeat_indices=joint,
        throughput_plateau=len(plateau) >= required,
        sustained_queue_growth=len(queue) >= required,
        slo_goodput_breach=len(slo) >= required,
        joint_overload=len(joint) >= required,
    )
    return signals, tuple(failures)


def _is_stable(
    trials: Sequence[CapacityTrialMetrics],
    signals: CapacityPointSignals,
    policy: CapacityKneePolicy,
) -> bool:
    return (
        not signals.queue_growth_repeat_indices
        and not signals.slo_goodput_breach_repeat_indices
        and all(
            trial.preemption_count <= policy.maximum_preemptions_for_stable
            and trial.timeout_count <= policy.maximum_timeouts_for_stable
            and trial.oom_count == 0
            for trial in trials
        )
    )


def _classify_point(
    *,
    eligible: bool,
    trials: Sequence[CapacityTrialMetrics],
    signals: CapacityPointSignals,
    policy: CapacityKneePolicy,
) -> Literal["stable", "overloaded", "transitional", "invalid"]:
    if not eligible:
        return "invalid"
    if signals.joint_overload:
        return "overloaded"
    if _is_stable(trials, signals, policy):
        return "stable"
    return "transitional"


def _analyze_context(
    *,
    context_id: str,
    context_tokens: int,
    load_groups: dict[str, list[CapacityTrialRecord]],
    policy: CapacityKneePolicy,
) -> ContextCapacityAnalysis:
    raw_points: list[CapacityPointAnalysis] = []
    for load_id, load_trials in load_groups.items():
        trials = tuple(sorted(load_trials, key=lambda trial: trial.repeat_index))
        complete = tuple(trial for trial in trials if isinstance(trial, CapacityTrialMetrics))
        complete_repeats = tuple(sorted(trial.repeat_index for trial in complete))
        failures = _point_failures(trials, policy)
        signals = _empty_signals(complete, policy)
        metrics = (
            _metric_summaries(complete) if complete_repeats == REQUIRED_REPEAT_INDICES else None
        )
        target_offered = _summary([trial.target_offered_requests_per_second for trial in trials])
        raw_points.append(
            CapacityPointAnalysis(
                context_id=context_id,
                context_tokens=context_tokens,
                load_id=load_id,
                trials=trials,
                target_offered_requests_per_second=target_offered,
                metrics=metrics,
                signals=signals,
                eligible=not failures and metrics is not None,
                validation_failures=failures,
                classification=_classify_point(
                    eligible=not failures and metrics is not None,
                    trials=complete,
                    signals=signals,
                    policy=policy,
                ),
            )
        )
    raw_points.sort(
        key=lambda point: (
            point.target_offered_requests_per_second.median,
            point.load_id,
        )
    )

    points: list[CapacityPointAnalysis] = []
    for index, raw_point in enumerate(raw_points):
        accumulated_failures = list(raw_point.validation_failures)
        complete = tuple(
            trial for trial in raw_point.trials if isinstance(trial, CapacityTrialMetrics)
        )
        if index == 0:
            signals = raw_point.signals
        else:
            signals, paired_failures = _paired_signals(points[index - 1], complete, policy)
            accumulated_failures.extend(paired_failures)
        eligible = not accumulated_failures and raw_point.metrics is not None
        points.append(
            raw_point.model_copy(
                update={
                    "signals": signals,
                    "eligible": eligible,
                    "validation_failures": tuple(accumulated_failures),
                    "classification": _classify_point(
                        eligible=eligible,
                        trials=complete,
                        signals=signals,
                        policy=policy,
                    ),
                }
            )
        )

    failure_reasons: list[str] = []
    if len(points) < policy.minimum_load_points:
        failure_reasons.append(
            f"requires_at_least_{policy.minimum_load_points}_load_points:observed_{len(points)}"
        )
    for point in points:
        failure_reasons.extend(
            f"load_{point.load_id}:{failure}" for failure in point.validation_failures
        )

    overload_indices = [
        index for index, point in enumerate(points) if point.classification == "overloaded"
    ]
    stable_index: Optional[int] = None
    overload_index: Optional[int] = None
    if not overload_indices:
        failure_reasons.append("no_joint_overload_point")
    else:
        overload_index = overload_indices[0]
        stable_indices = [
            index
            for index, point in enumerate(points[:overload_index])
            if point.classification == "stable"
        ]
        if not stable_indices:
            failure_reasons.append("no_stable_point_before_first_overload")
        else:
            stable_index = stable_indices[-1]
            if stable_index + 1 != overload_index:
                failure_reasons.append("stable_and_first_overload_are_not_adjacent")
        if any(point.classification == "stable" for point in points[overload_index + 1 :]):
            failure_reasons.append("stable_point_after_first_overload")

    bracketed = (
        stable_index is not None
        and overload_index is not None
        and stable_index + 1 == overload_index
    )
    passed = not failure_reasons and bracketed
    stable_point = points[stable_index] if bracketed and stable_index is not None else None
    overload_point = points[overload_index] if bracketed and overload_index is not None else None
    stable_metrics = stable_point.metrics if stable_point is not None else None
    overload_metrics = overload_point.metrics if overload_point is not None else None
    knee = CapacityKneeResult(
        passed=passed,
        last_stable_load_id=stable_point.load_id if stable_point is not None else None,
        last_stable_target_offered_requests_per_second=(
            stable_point.target_offered_requests_per_second.median
            if stable_point is not None
            else None
        ),
        last_stable_empirical_offered_requests_per_second=(
            stable_metrics.empirical_offered_requests_per_second.median
            if stable_metrics is not None
            else None
        ),
        first_bracketed_overload_load_id=(
            overload_point.load_id if overload_point is not None else None
        ),
        first_bracketed_overload_target_offered_requests_per_second=(
            overload_point.target_offered_requests_per_second.median
            if overload_point is not None
            else None
        ),
        first_bracketed_overload_empirical_offered_requests_per_second=(
            overload_metrics.empirical_offered_requests_per_second.median
            if overload_metrics is not None
            else None
        ),
        failure_reasons=tuple(failure_reasons),
    )
    return ContextCapacityAnalysis(
        context_id=context_id,
        context_tokens=context_tokens,
        points=tuple(points),
        knee=knee,
    )


def analyze_capacity_sweep(
    trials: Sequence[CapacityTrialRecord],
    policy: CapacityKneePolicy,
) -> CapacitySweepAnalysis:
    """Aggregate formal M1 repeats and locate a fail-closed knee per context.

    A passing knee has three repeat-supported overload signals at the same load:
    marginal-throughput plateau, sustained queue growth, and SLO/Goodput breach.
    Its immediately preceding point must be stable.  Every load point must contain
    exactly three complete repeats from a policy-compliant long trace.
    """

    trial_ids: set[str] = set()
    point_repeats: set[tuple[str, str, int]] = set()
    context_tokens_by_id: dict[str, int] = {}
    grouped: dict[tuple[str, int], dict[str, list[CapacityTrialRecord]]] = {}
    for trial in trials:
        if trial.trial_id in trial_ids:
            raise ValueError(f"duplicate capacity trial_id: {trial.trial_id}")
        trial_ids.add(trial.trial_id)
        repeat_key = (trial.context_id, trial.load_id, trial.repeat_index)
        if repeat_key in point_repeats:
            raise ValueError(
                "duplicate capacity repeat for "
                f"context={trial.context_id}, load={trial.load_id}, "
                f"repeat={trial.repeat_index}"
            )
        point_repeats.add(repeat_key)
        known_tokens = context_tokens_by_id.setdefault(trial.context_id, trial.context_tokens)
        if known_tokens != trial.context_tokens:
            raise ValueError(f"context_id {trial.context_id!r} has inconsistent context_tokens")
        grouped.setdefault((trial.context_id, trial.context_tokens), {}).setdefault(
            trial.load_id, []
        ).append(trial)

    contexts = tuple(
        _analyze_context(
            context_id=context_id,
            context_tokens=context_tokens,
            load_groups=load_groups,
            policy=policy,
        )
        for (context_id, context_tokens), load_groups in sorted(
            grouped.items(), key=lambda item: (item[0][1], item[0][0])
        )
    )
    failure_reasons: list[str] = []
    if not contexts:
        failure_reasons.append("no_formal_capacity_trials")
    for context in contexts:
        failure_reasons.extend(
            f"context_{context.context_id}:{reason}" for reason in context.knee.failure_reasons
        )
    return CapacitySweepAnalysis(
        policy=policy,
        contexts=contexts,
        passed=bool(contexts) and all(context.knee.passed for context in contexts),
        failure_reasons=tuple(failure_reasons),
    )
