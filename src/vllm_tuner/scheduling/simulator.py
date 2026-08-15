"""Pure-Python deterministic simulator for chunked-prefill scheduling.

This is an intentionally small mechanism model, not a cycle-accurate vLLM
emulator.  It captures the interactions needed for controlled M5 experiments:
arrivals, separate prefill/decode work, finite token budgets, sequence admission,
KV pressure, preemption, latency SLOs, and starvation/fairness accounting.
"""

from collections import deque
from dataclasses import asdict, dataclass, field
from math import ceil
from typing import Any, Iterable, Optional, Sequence

from .admission import (
    AdmissionCandidate,
    AdmissionConfig,
    AdmissionDecision,
    FairAdmissionController,
)
from .token_budget import (
    DEFAULT_FIXED_BUDGETS,
    AdaptiveBudgetConfig,
    AdaptiveTokenBudgetPolicy,
    FixedTokenBudgetPolicy,
    SchedulerSignals,
    TokenBudgetDecision,
    TokenBudgetPolicy,
)


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile (Hyndman-Fan type 7)."""

    if not 0.0 <= percentile_value <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass(frozen=True)
class SimulationRequest:
    """One immutable request in a repeatable workload trace."""

    request_id: str
    arrival_time: float
    prompt_tokens: int
    output_tokens: int
    priority: int = 0
    ttft_slo: Optional[float] = None
    tpot_slo: Optional[float] = None
    e2e_slo: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.arrival_time < 0:
            raise ValueError("arrival_time must be non-negative")
        if self.prompt_tokens < 1:
            raise ValueError("prompt_tokens must be positive")
        if self.output_tokens < 1:
            raise ValueError("output_tokens must be positive")
        slos = (self.ttft_slo, self.tpot_slo, self.e2e_slo)
        if any(value is not None and value <= 0 for value in slos):
            raise ValueError("request SLOs must be positive when provided")


# Friendly aliases for trace-building call sites.
Request = SimulationRequest
SimRequest = SimulationRequest


@dataclass(frozen=True)
class SimulationConfig:
    """Runtime and SLO constants for a simulation run."""

    step_duration: float = 0.01
    prefill_quantum: int = 128
    kv_capacity_tokens: int = 65536
    available_token_budget: Optional[int] = None
    signal_window_steps: int = 100
    max_steps: int = 1_000_000
    seed: int = 0
    ttft_slo: float = 1.0
    tpot_slo: float = 0.05
    e2e_slo: float = 10.0
    starvation_threshold: float = 2.0

    def __post_init__(self) -> None:
        if self.step_duration <= 0:
            raise ValueError("step_duration must be positive")
        if self.prefill_quantum < 1:
            raise ValueError("prefill_quantum must be positive")
        if self.kv_capacity_tokens < 1:
            raise ValueError("kv_capacity_tokens must be positive")
        if self.available_token_budget is not None and self.available_token_budget < 2:
            raise ValueError("available_token_budget must be at least two")
        if self.signal_window_steps < 1:
            raise ValueError("signal_window_steps must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.ttft_slo <= 0 or self.tpot_slo <= 0 or self.e2e_slo <= 0:
            raise ValueError("simulation SLOs must be positive")
        if self.starvation_threshold <= 0:
            raise ValueError("starvation_threshold must be positive")


@dataclass(frozen=True)
class RequestMetrics:
    """Per-request raw values retained for later metric verification."""

    request_id: str
    arrival_time: float
    first_service_time: float
    first_token_time: float
    finish_time: float
    prompt_tokens: int
    output_tokens: int
    token_timestamps: tuple[float, ...]
    queue_time: float
    ttft: float
    tpot: float
    e2e: float
    preemptions: int
    max_service_gap: float
    starved: bool
    good: bool

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable per-request metrics."""

        result = asdict(self)
        result["token_timestamps"] = list(self.token_timestamps)
        return result


@dataclass(frozen=True)
class StepRecord:
    """Audit record proving token conservation for one scheduler step."""

    step: int
    start_time: float
    end_time: float
    total_budget: int
    decode_budget: int
    prefill_budget: int
    decode_tokens: int
    prefill_tokens: int
    total_tokens: int
    decode_backlog: int
    prefill_backlog: int
    admitted_requests: int
    waiting_requests: int
    kv_pressure: float
    admitted_sequence_limit: int
    preemptions: int
    progressed_request_ids: tuple[str, ...]
    decision_reasons: tuple[str, ...]
    admission_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.total_tokens != self.decode_tokens + self.prefill_tokens:
            raise ValueError("step token counters are inconsistent")
        if self.total_tokens > self.total_budget:
            raise ValueError("step consumed more tokens than its budget")
        if self.decode_tokens < 0 or self.prefill_tokens < 0:
            raise ValueError("step token counters must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable decision/audit record."""

        result = asdict(self)
        result["progressed_request_ids"] = list(self.progressed_request_ids)
        result["decision_reasons"] = list(self.decision_reasons)
        result["admission_reasons"] = list(self.admission_reasons)
        return result


@dataclass(frozen=True)
class SimulationMetrics:
    """Aggregate latency, goodput, fairness, starvation, and preemption metrics."""

    completed_requests: int
    total_requests: int
    good_requests: int
    duration: float
    throughput: float
    goodput: float
    p50_queue_time: float
    p99_queue_time: float
    p50_ttft: float
    p99_ttft: float
    p50_tpot: float
    p99_tpot: float
    fairness_index: float
    starvation_count: int
    starvation_rate: float
    max_wait_observed: float
    preemption_count: int
    scheduled_prefill_tokens: int
    scheduled_decode_tokens: int

    @property
    def queue_time_p50(self) -> float:
        """Compatibility alias using metric-first naming."""

        return self.p50_queue_time

    @property
    def queue_time_p99(self) -> float:
        """Compatibility alias using metric-first naming."""

        return self.p99_queue_time

    @property
    def ttft_p50(self) -> float:
        """Compatibility alias using metric-first naming."""

        return self.p50_ttft

    @property
    def ttft_p99(self) -> float:
        """Compatibility alias using metric-first naming."""

        return self.p99_ttft

    @property
    def tpot_p50(self) -> float:
        """Compatibility alias using metric-first naming."""

        return self.p50_tpot

    @property
    def tpot_p99(self) -> float:
        """Compatibility alias using metric-first naming."""

        return self.p99_tpot

    @property
    def goodput_requests_per_second(self) -> float:
        """Return the request goodput with explicit units in the name."""

        return self.goodput

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate metrics with stable artifact keys."""

        return asdict(self)


@dataclass(frozen=True)
class SimulationResult:
    """Complete deterministic result, including raw requests and every step."""

    policy_name: str
    seed: int
    metrics: SimulationMetrics
    requests: tuple[RequestMetrics, ...]
    steps: tuple[StepRecord, ...]
    decisions: tuple[TokenBudgetDecision, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result artifact."""

        return {
            "policy_name": self.policy_name,
            "seed": self.seed,
            "metrics": self.metrics.to_dict(),
            "requests": [request.to_dict() for request in self.requests],
            "steps": [step.to_dict() for step in self.steps],
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


@dataclass
class _RequestState:
    request: SimulationRequest
    original_order: int
    remaining_prefill: int = field(init=False)
    remaining_decode: int = field(init=False)
    admitted: bool = False
    waiting_since: float = field(init=False)
    first_service_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    last_progress_time: Optional[float] = None
    max_service_gap: float = 0.0
    service_tokens: int = 0
    preemptions: int = 0
    token_timestamps: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.remaining_prefill = self.request.prompt_tokens
        self.remaining_decode = self.request.output_tokens
        self.waiting_since = self.request.arrival_time

    @property
    def complete(self) -> bool:
        return self.finish_time is not None

    @property
    def stage(self) -> str:
        return "prefill" if self.remaining_prefill > 0 else "decode"

    @property
    def processed_prefill(self) -> int:
        return self.request.prompt_tokens - self.remaining_prefill

    @property
    def generated_decode(self) -> int:
        return self.request.output_tokens - self.remaining_decode

    def candidate(self) -> AdmissionCandidate:
        return AdmissionCandidate(
            request_id=self.request.request_id,
            arrival_time=self.request.arrival_time,
            waiting_since=self.waiting_since,
            original_order=self.original_order,
            stage=self.stage,
            priority=self.request.priority,
            last_progress_time=self.last_progress_time,
            service_tokens=self.service_tokens,
        )


class SimulationLimitError(RuntimeError):
    """Raised when a policy cannot complete a trace within ``max_steps``."""


class DeterministicSchedulerSimulator:
    """Simulate one trace with deterministic admission and token scheduling."""

    def __init__(
        self,
        policy: TokenBudgetPolicy,
        config: Optional[SimulationConfig] = None,
        admission_controller: Optional[FairAdmissionController] = None,
    ) -> None:
        self.policy = policy
        self.config = config or SimulationConfig()
        self.admission_controller = admission_controller or FairAdmissionController()

    def _kv_pressure(self, states: Sequence[_RequestState]) -> float:
        cached_tokens = sum(
            state.processed_prefill + state.generated_decode
            for state in states
            if state.admitted and not state.complete
        )
        return min(1.0, cached_tokens / self.config.kv_capacity_tokens)

    def _mark_progress(self, state: _RequestState, now: float, end_time: float) -> None:
        anchor = state.last_progress_time
        if anchor is None:
            anchor = state.request.arrival_time
        state.max_service_gap = max(state.max_service_gap, end_time - anchor)
        if state.first_service_time is None:
            state.first_service_time = now
        state.last_progress_time = end_time

    def _make_signals(
        self,
        states: Sequence[_RequestState],
        now: float,
        step: int,
        ttft_samples: Sequence[float],
        tpot_samples: Sequence[float],
        preemption_window: Sequence[int],
    ) -> SchedulerSignals:
        arrived = [
            state for state in states if state.request.arrival_time <= now and not state.complete
        ]
        prefill = [state for state in arrived if state.remaining_prefill > 0]
        decode = [state for state in arrived if state.remaining_prefill == 0]
        oldest_prefill_age = max(
            (now - state.request.arrival_time for state in prefill),
            default=0.0,
        )
        return SchedulerSignals(
            step=step,
            decode_backlog=len(decode),
            prefill_backlog=len(prefill),
            oldest_prefill_age=oldest_prefill_age,
            kv_pressure=self._kv_pressure(states),
            recent_p99_ttft=percentile(ttft_samples, 99),
            recent_p99_tpot=percentile(tpot_samples, 99),
            recent_preemptions=sum(preemption_window),
            available_token_budget=self.config.available_token_budget,
        )

    def _apply_admission(
        self,
        states: Sequence[_RequestState],
        now: float,
        admitted_sequence_limit: int,
    ) -> AdmissionDecision:
        arrived = [
            state for state in states if state.request.arrival_time <= now and not state.complete
        ]
        waiting = [state.candidate() for state in arrived if not state.admitted]
        admitted = [state.candidate() for state in arrived if state.admitted]
        decision = self.admission_controller.decide(
            waiting=waiting,
            admitted=admitted,
            now=now,
            admitted_sequence_limit=admitted_sequence_limit,
        )
        by_id = {state.request.request_id: state for state in states}
        for request_id in decision.preempted_request_ids:
            state = by_id[request_id]
            state.admitted = False
            state.waiting_since = now
            state.preemptions += 1
        for request_id in decision.admitted_request_ids:
            state = by_id[request_id]
            state.admitted = True
        return decision

    def _schedule_decode(
        self,
        states: Sequence[_RequestState],
        now: float,
        end_time: float,
        budget: int,
        ttft_samples: deque[float],
        tpot_samples: deque[float],
    ) -> tuple[int, list[str]]:
        decode_states = [
            state
            for state in states
            if state.admitted and not state.complete and state.remaining_prefill == 0
        ]
        ranked = self.admission_controller.rank_for_service(
            [state.candidate() for state in decode_states], now
        )
        by_id = {state.request.request_id: state for state in decode_states}
        used = 0
        progressed: list[str] = []
        for candidate in ranked[:budget]:
            state = by_id[candidate.request_id]
            previous_token_time = state.token_timestamps[-1] if state.token_timestamps else None
            self._mark_progress(state, now, end_time)
            state.remaining_decode -= 1
            state.service_tokens += 1
            state.token_timestamps.append(end_time)
            used += 1
            progressed.append(state.request.request_id)
            if state.first_token_time is None:
                state.first_token_time = end_time
                ttft_samples.append(end_time - state.request.arrival_time)
            elif previous_token_time is not None:
                tpot_samples.append(end_time - previous_token_time)
            if state.remaining_decode == 0:
                state.finish_time = end_time
                state.admitted = False
        return used, progressed

    def _schedule_prefill(
        self,
        states: Sequence[_RequestState],
        now: float,
        end_time: float,
        budget: int,
    ) -> tuple[int, list[str]]:
        prefill_states = [
            state
            for state in states
            if state.admitted and not state.complete and state.remaining_prefill > 0
        ]
        ranked = self.admission_controller.rank_for_service(
            [state.candidate() for state in prefill_states], now
        )
        by_id = {state.request.request_id: state for state in prefill_states}
        ordered = [by_id[candidate.request_id] for candidate in ranked]
        remaining_budget = budget
        used = 0
        progressed: list[str] = []

        # Repeated bounded rounds model chunked prefill.  Every admitted request
        # gets one quantum before a request receives a second quantum.
        while remaining_budget > 0 and ordered:
            made_progress = False
            for state in ordered:
                if remaining_budget <= 0:
                    break
                if state.remaining_prefill <= 0:
                    continue
                chunk = min(
                    state.remaining_prefill,
                    self.config.prefill_quantum,
                    remaining_budget,
                )
                if chunk <= 0:
                    continue
                self._mark_progress(state, now, end_time)
                state.remaining_prefill -= chunk
                state.service_tokens += chunk
                remaining_budget -= chunk
                used += chunk
                made_progress = True
                if state.request.request_id not in progressed:
                    progressed.append(state.request.request_id)
            if not made_progress:
                break
        return used, progressed

    def _request_metrics(self, state: _RequestState) -> RequestMetrics:
        if (
            state.first_service_time is None
            or state.first_token_time is None
            or state.finish_time is None
        ):
            raise SimulationLimitError(f"request {state.request.request_id} did not complete")
        queue_time = state.first_service_time - state.request.arrival_time
        ttft = state.first_token_time - state.request.arrival_time
        if state.request.output_tokens > 1:
            tpot = (state.finish_time - state.first_token_time) / (state.request.output_tokens - 1)
        else:
            tpot = 0.0
        e2e = state.finish_time - state.request.arrival_time
        ttft_slo = state.request.ttft_slo or self.config.ttft_slo
        tpot_slo = state.request.tpot_slo or self.config.tpot_slo
        e2e_slo = state.request.e2e_slo or self.config.e2e_slo
        good = ttft <= ttft_slo and tpot <= tpot_slo and e2e <= e2e_slo
        starved = state.max_service_gap > self.config.starvation_threshold
        return RequestMetrics(
            request_id=state.request.request_id,
            arrival_time=state.request.arrival_time,
            first_service_time=state.first_service_time,
            first_token_time=state.first_token_time,
            finish_time=state.finish_time,
            prompt_tokens=state.request.prompt_tokens,
            output_tokens=state.request.output_tokens,
            token_timestamps=tuple(state.token_timestamps),
            queue_time=queue_time,
            ttft=ttft,
            tpot=tpot,
            e2e=e2e,
            preemptions=state.preemptions,
            max_service_gap=state.max_service_gap,
            starved=starved,
            good=good,
        )

    def _aggregate(
        self,
        request_metrics: Sequence[RequestMetrics],
        steps: Sequence[StepRecord],
    ) -> SimulationMetrics:
        if not request_metrics:
            return SimulationMetrics(
                completed_requests=0,
                total_requests=0,
                good_requests=0,
                duration=0.0,
                throughput=0.0,
                goodput=0.0,
                p50_queue_time=0.0,
                p99_queue_time=0.0,
                p50_ttft=0.0,
                p99_ttft=0.0,
                p50_tpot=0.0,
                p99_tpot=0.0,
                fairness_index=1.0,
                starvation_count=0,
                starvation_rate=0.0,
                max_wait_observed=0.0,
                preemption_count=0,
                scheduled_prefill_tokens=0,
                scheduled_decode_tokens=0,
            )

        start = min(request.arrival_time for request in request_metrics)
        finish = max(request.finish_time for request in request_metrics)
        duration = max(self.config.step_duration, finish - start)
        good_requests = sum(request.good for request in request_metrics)
        starved = sum(request.starved for request in request_metrics)

        max_budget = max((step.total_budget for step in steps), default=1)
        fairness_scores: list[float] = []
        for request in request_metrics:
            ideal_steps = ceil(request.prompt_tokens / max_budget) + request.output_tokens
            ideal_time = ideal_steps * self.config.step_duration
            fairness_scores.append(min(1.0, ideal_time / max(request.e2e, ideal_time)))
        score_sum = sum(fairness_scores)
        score_square_sum = sum(score * score for score in fairness_scores)
        fairness = (
            score_sum * score_sum / (len(fairness_scores) * score_square_sum)
            if score_square_sum
            else 1.0
        )
        fairness = min(1.0, max(0.0, fairness))

        queue_times = [request.queue_time for request in request_metrics]
        ttfts = [request.ttft for request in request_metrics]
        tpots = [request.tpot for request in request_metrics]
        return SimulationMetrics(
            completed_requests=len(request_metrics),
            total_requests=len(request_metrics),
            good_requests=good_requests,
            duration=duration,
            throughput=len(request_metrics) / duration,
            goodput=good_requests / duration,
            p50_queue_time=percentile(queue_times, 50),
            p99_queue_time=percentile(queue_times, 99),
            p50_ttft=percentile(ttfts, 50),
            p99_ttft=percentile(ttfts, 99),
            p50_tpot=percentile(tpots, 50),
            p99_tpot=percentile(tpots, 99),
            fairness_index=fairness,
            starvation_count=starved,
            starvation_rate=starved / len(request_metrics),
            max_wait_observed=max(request.max_service_gap for request in request_metrics),
            preemption_count=sum(request.preemptions for request in request_metrics),
            scheduled_prefill_tokens=sum(step.prefill_tokens for step in steps),
            scheduled_decode_tokens=sum(step.decode_tokens for step in steps),
        )

    def run(self, requests: Iterable[SimulationRequest]) -> SimulationResult:
        """Run a trace to completion or raise :class:`SimulationLimitError`."""

        trace = tuple(requests)
        request_ids = [request.request_id for request in trace]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request IDs must be unique within a trace")
        self.policy.reset()
        if not trace:
            metrics = self._aggregate((), ())
            return SimulationResult(
                policy_name=self.policy.name,
                seed=self.config.seed,
                metrics=metrics,
                requests=(),
                steps=(),
                decisions=(),
            )

        states = [
            _RequestState(request=request, original_order=index)
            for index, request in enumerate(trace)
        ]
        now = 0.0
        step = 0
        records: list[StepRecord] = []
        ttft_samples: deque[float] = deque(maxlen=self.config.signal_window_steps)
        tpot_samples: deque[float] = deque(maxlen=self.config.signal_window_steps)
        preemption_window: deque[int] = deque(maxlen=self.config.signal_window_steps)

        while not all(state.complete for state in states):
            if step >= self.config.max_steps:
                incomplete = [state.request.request_id for state in states if not state.complete]
                raise SimulationLimitError(
                    f"trace exceeded max_steps with incomplete requests: {incomplete}"
                )
            arrived = [
                state
                for state in states
                if state.request.arrival_time <= now and not state.complete
            ]
            if not arrived:
                next_arrival = min(
                    state.request.arrival_time for state in states if not state.complete
                )
                now = next_arrival
                continue

            signals = self._make_signals(
                states,
                now,
                step,
                ttft_samples,
                tpot_samples,
                preemption_window,
            )
            budget_decision = self.policy.decide(signals)
            admission_decision = self._apply_admission(
                states,
                now,
                budget_decision.admitted_sequence_limit,
            )
            preemptions = len(admission_decision.preempted_request_ids)
            preemption_window.append(preemptions)

            end_time = now + self.config.step_duration
            admitted_prefill_before = any(
                state.admitted and not state.complete and state.remaining_prefill > 0
                for state in states
            )
            decode_capacity = budget_decision.decode_budget
            if not admitted_prefill_before:
                decode_capacity += budget_decision.prefill_budget
            decode_used, decode_progress = self._schedule_decode(
                states,
                now,
                end_time,
                decode_capacity,
                ttft_samples,
                tpot_samples,
            )
            transferable_decode = max(0, budget_decision.decode_budget - decode_used)
            prefill_capacity = budget_decision.prefill_budget + transferable_decode
            # A stage allocation transferred to decode cannot be spent twice.
            if not admitted_prefill_before:
                prefill_capacity = 0
            prefill_used, prefill_progress = self._schedule_prefill(
                states,
                now,
                end_time,
                prefill_capacity,
            )
            total_used = decode_used + prefill_used
            if total_used > budget_decision.total_budget:
                raise RuntimeError("internal error: scheduler exceeded token budget")
            if total_used == 0:
                raise SimulationLimitError("policy/admission made no progress for arrived work")

            arrived_after = [
                state
                for state in states
                if state.request.arrival_time <= now and not state.complete
            ]
            record = StepRecord(
                step=step,
                start_time=now,
                end_time=end_time,
                total_budget=budget_decision.total_budget,
                decode_budget=budget_decision.decode_budget,
                prefill_budget=budget_decision.prefill_budget,
                decode_tokens=decode_used,
                prefill_tokens=prefill_used,
                total_tokens=total_used,
                decode_backlog=signals.decode_backlog,
                prefill_backlog=signals.prefill_backlog,
                admitted_requests=sum(state.admitted for state in arrived_after),
                waiting_requests=sum(not state.admitted for state in arrived_after),
                kv_pressure=self._kv_pressure(states),
                admitted_sequence_limit=budget_decision.admitted_sequence_limit,
                preemptions=preemptions,
                progressed_request_ids=tuple(dict.fromkeys(decode_progress + prefill_progress)),
                decision_reasons=budget_decision.reasons,
                admission_reasons=admission_decision.reasons,
            )
            records.append(record)
            now = end_time
            step += 1

        raw_metrics = tuple(self._request_metrics(state) for state in states)
        aggregate = self._aggregate(raw_metrics, records)
        return SimulationResult(
            policy_name=self.policy.name,
            seed=self.config.seed,
            metrics=aggregate,
            requests=raw_metrics,
            steps=tuple(records),
            decisions=self.policy.decision_log,
        )


DeterministicSimulator = DeterministicSchedulerSimulator
TokenBudgetSimulator = DeterministicSchedulerSimulator


@dataclass(frozen=True)
class NegativeGainCondition:
    """A machine-readable condition where adaptive has no benefit or regresses."""

    trace_name: str
    metric: str
    adaptive_value: float
    fixed_value: float
    fixed_budget: int
    relative_gain: float
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable negative/no-benefit observation."""

        return asdict(self)


@dataclass(frozen=True)
class TraceComparison:
    """Adaptive and fixed-budget results for the exact same trace."""

    trace_name: str
    fixed_baselines: dict[int, SimulationResult]
    adaptive: SimulationResult
    best_fixed_budget: int
    goodput_gain_vs_best: float
    negative_gain_conditions: tuple[NegativeGainCondition, ...]

    def __post_init__(self) -> None:
        if len(self.fixed_baselines) < 2:
            raise ValueError("a trace comparison requires at least two fixed baselines")
        if self.best_fixed_budget not in self.fixed_baselines:
            raise ValueError("best_fixed_budget must identify a baseline result")

    @property
    def has_negative_result(self) -> bool:
        """Whether any primary metric shows no benefit or a regression."""

        return bool(self.negative_gain_conditions)

    def to_dict(self) -> dict[str, Any]:
        """Return all controls, adaptive output, and analysis for one trace."""

        return {
            "trace_name": self.trace_name,
            "fixed_baselines": {
                str(budget): result.to_dict() for budget, result in self.fixed_baselines.items()
            },
            "adaptive": self.adaptive.to_dict(),
            "best_fixed_budget": self.best_fixed_budget,
            "goodput_gain_vs_best": self.goodput_gain_vs_best,
            "negative_gain_conditions": [
                condition.to_dict() for condition in self.negative_gain_conditions
            ],
        }


@dataclass(frozen=True)
class BudgetAblationReport:
    """Calibration and held-out comparisons for a complete M5 ablation."""

    calibration: TraceComparison
    held_out: TraceComparison

    @property
    def negative_gain_conditions(self) -> tuple[NegativeGainCondition, ...]:
        """Collect negative/no-benefit conditions across both traces."""

        return self.calibration.negative_gain_conditions + self.held_out.negative_gain_conditions

    @property
    def heldout(self) -> TraceComparison:
        """Compatibility alias without an underscore."""

        return self.held_out

    @property
    def has_negative_result(self) -> bool:
        """Whether either trace exposes a policy downside."""

        return bool(self.negative_gain_conditions)

    def to_dict(self) -> dict[str, Any]:
        """Return a self-contained calibration/held-out artifact."""

        return {
            "calibration": self.calibration.to_dict(),
            "held_out": self.held_out.to_dict(),
            "has_negative_result": self.has_negative_result,
            "negative_gain_conditions": [
                condition.to_dict() for condition in self.negative_gain_conditions
            ],
        }


AblationReport = BudgetAblationReport


def run_fixed_budget_baselines(
    requests: Iterable[SimulationRequest],
    budgets: Sequence[int] = DEFAULT_FIXED_BUDGETS,
    simulation_config: Optional[SimulationConfig] = None,
    admission_config: Optional[AdmissionConfig] = None,
    max_admitted_sequences: int = 64,
) -> dict[int, SimulationResult]:
    """Run at least two configurable fixed budgets against one frozen trace."""

    normalized_budgets = tuple(budgets)
    if len(normalized_budgets) < 2:
        raise ValueError("at least two fixed budgets are required for an ablation")
    if len(normalized_budgets) != len(set(normalized_budgets)):
        raise ValueError("fixed budgets must be unique")
    if any(budget < 2 for budget in normalized_budgets):
        raise ValueError("fixed budgets must be at least two")
    trace = tuple(requests)
    config = simulation_config or SimulationConfig()
    controller_config = admission_config or AdmissionConfig()
    results: dict[int, SimulationResult] = {}
    for budget in normalized_budgets:
        policy = FixedTokenBudgetPolicy(
            budget=budget,
            max_admitted_sequences=max_admitted_sequences,
            minimum_prefill_progress=controller_config.minimum_prefill_progress,
        )
        simulator = DeterministicSchedulerSimulator(
            policy=policy,
            config=config,
            admission_controller=FairAdmissionController(controller_config),
        )
        results[budget] = simulator.run(trace)
    return results


def analyze_negative_benefit(
    trace_name: str,
    fixed_baselines: dict[int, SimulationResult],
    adaptive: SimulationResult,
) -> tuple[NegativeGainCondition, ...]:
    """Explain primary metrics where adaptive fails to beat a fixed baseline."""

    if len(fixed_baselines) < 2:
        raise ValueError("negative-benefit analysis requires at least two baselines")
    conditions: list[NegativeGainCondition] = []

    best_goodput_budget, best_goodput_result = max(
        fixed_baselines.items(), key=lambda item: (item[1].metrics.goodput, -item[0])
    )
    fixed_goodput = best_goodput_result.metrics.goodput
    adaptive_goodput = adaptive.metrics.goodput
    if adaptive_goodput <= fixed_goodput:
        relative = (adaptive_goodput - fixed_goodput) / fixed_goodput if fixed_goodput > 0 else 0.0
        conditions.append(
            NegativeGainCondition(
                trace_name=trace_name,
                metric="goodput",
                adaptive_value=adaptive_goodput,
                fixed_value=fixed_goodput,
                fixed_budget=best_goodput_budget,
                relative_gain=relative,
                explanation=(
                    "Adaptive budget changes add no goodput benefit at this load; "
                    "the best fixed budget is equally good or better."
                ),
            )
        )

    latency_metrics = (
        ("p99_ttft", lambda result: result.metrics.p99_ttft),
        ("p99_tpot", lambda result: result.metrics.p99_tpot),
    )
    for metric_name, getter in latency_metrics:
        best_budget, best_result = min(
            fixed_baselines.items(), key=lambda item: (getter(item[1]), item[0])
        )
        fixed_latency = getter(best_result)
        adaptive_latency = getter(adaptive)
        if adaptive_latency >= fixed_latency:
            relative = (
                (fixed_latency - adaptive_latency) / fixed_latency if fixed_latency > 0 else 0.0
            )
            conditions.append(
                NegativeGainCondition(
                    trace_name=trace_name,
                    metric=metric_name,
                    adaptive_value=adaptive_latency,
                    fixed_value=fixed_latency,
                    fixed_budget=best_budget,
                    relative_gain=relative,
                    explanation=(
                        f"Adaptive hysteresis/admission overhead does not improve {metric_name} "
                        "for this trace."
                    ),
                )
            )
    return tuple(conditions)


def _compare_trace(
    trace_name: str,
    requests: Sequence[SimulationRequest],
    budgets: Sequence[int],
    adaptive_config: AdaptiveBudgetConfig,
    simulation_config: SimulationConfig,
    admission_config: AdmissionConfig,
) -> TraceComparison:
    fixed = run_fixed_budget_baselines(
        requests=requests,
        budgets=budgets,
        simulation_config=simulation_config,
        admission_config=admission_config,
        max_admitted_sequences=adaptive_config.max_admitted_sequences,
    )
    adaptive = DeterministicSchedulerSimulator(
        policy=AdaptiveTokenBudgetPolicy(adaptive_config),
        config=simulation_config,
        admission_controller=FairAdmissionController(admission_config),
    ).run(requests)
    best_budget, best_result = max(
        fixed.items(), key=lambda item: (item[1].metrics.goodput, -item[0])
    )
    denominator = best_result.metrics.goodput
    gain = (adaptive.metrics.goodput - denominator) / denominator if denominator > 0 else 0.0
    return TraceComparison(
        trace_name=trace_name,
        fixed_baselines=fixed,
        adaptive=adaptive,
        best_fixed_budget=best_budget,
        goodput_gain_vs_best=gain,
        negative_gain_conditions=analyze_negative_benefit(trace_name, fixed, adaptive),
    )


def run_budget_ablation(
    calibration_trace: Iterable[SimulationRequest],
    held_out_trace: Iterable[SimulationRequest],
    fixed_budgets: Sequence[int] = DEFAULT_FIXED_BUDGETS,
    adaptive_config: Optional[AdaptiveBudgetConfig] = None,
    simulation_config: Optional[SimulationConfig] = None,
    admission_config: Optional[AdmissionConfig] = None,
) -> BudgetAblationReport:
    """Compare adaptive against fixed budgets on calibration and held-out traces."""

    adaptive_settings = adaptive_config or AdaptiveBudgetConfig()
    simulation_settings = simulation_config or SimulationConfig()
    admission_settings = admission_config or AdmissionConfig(
        max_wait=adaptive_settings.max_wait,
        minimum_prefill_progress=adaptive_settings.minimum_prefill_progress,
    )
    calibration = tuple(calibration_trace)
    held_out = tuple(held_out_trace)
    return BudgetAblationReport(
        calibration=_compare_trace(
            "calibration",
            calibration,
            fixed_budgets,
            adaptive_settings,
            simulation_settings,
            admission_settings,
        ),
        held_out=_compare_trace(
            "held_out",
            held_out,
            fixed_budgets,
            adaptive_settings,
            simulation_settings,
            admission_settings,
        ),
    )


run_fixed_budget_ablation = run_budget_ablation
