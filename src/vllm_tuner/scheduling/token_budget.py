"""Deterministic token-budget policies for scheduler experiments.

The policies in this module deliberately do not depend on vLLM internals.  A
caller provides a snapshot of scheduler signals and receives an auditable split
of the next step's token budget between decode and prefill work.
"""

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Optional, Protocol

DEFAULT_FIXED_BUDGETS: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)


@dataclass(frozen=True)
class SchedulerSignals:
    """Signals observed immediately before a scheduling decision.

    Latency and age values are expressed in seconds.  ``available_token_budget``
    can model a temporary hardware/runtime cap; policies never exceed it.
    """

    step: int
    decode_backlog: int
    prefill_backlog: int
    oldest_prefill_age: float
    kv_pressure: float
    recent_p99_ttft: float
    recent_p99_tpot: float
    recent_preemptions: int
    available_token_budget: Optional[int] = None

    def __post_init__(self) -> None:
        integer_values = (
            self.step,
            self.decode_backlog,
            self.prefill_backlog,
            self.recent_preemptions,
        )
        if any(value < 0 for value in integer_values):
            raise ValueError("scheduler counters must be non-negative")
        if self.oldest_prefill_age < 0:
            raise ValueError("oldest_prefill_age must be non-negative")
        if not 0.0 <= self.kv_pressure <= 1.0:
            raise ValueError("kv_pressure must be between 0 and 1")
        if self.recent_p99_ttft < 0 or self.recent_p99_tpot < 0:
            raise ValueError("latency signals must be non-negative")
        if self.available_token_budget is not None and self.available_token_budget < 1:
            raise ValueError("available_token_budget must be positive when provided")


# The shorter name is convenient at call sites and keeps compatibility with early
# design notes that called this structure BudgetSignals.
BudgetSignals = SchedulerSignals


@dataclass(frozen=True)
class TokenBudgetDecision:
    """One policy decision and the exact signals that produced it."""

    step: int
    total_budget: int
    decode_budget: int
    prefill_budget: int
    admitted_sequence_limit: int
    reasons: tuple[str, ...]
    signals: SchedulerSignals
    changed: bool = False

    def __post_init__(self) -> None:
        if self.total_budget < 1:
            raise ValueError("total_budget must be positive")
        if self.decode_budget < 0 or self.prefill_budget < 0:
            raise ValueError("stage budgets must be non-negative")
        if self.decode_budget + self.prefill_budget != self.total_budget:
            raise ValueError("decode and prefill budgets must sum to total_budget")
        if self.admitted_sequence_limit < 1:
            raise ValueError("admitted_sequence_limit must be positive")
        available = self.signals.available_token_budget
        if available is not None and self.total_budget > available:
            raise ValueError("decision exceeds available_token_budget")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for experiment artifacts."""

        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


# A concise alias used by consumers that already know they are handling tokens.
BudgetDecision = TokenBudgetDecision


class TokenBudgetPolicy(Protocol):
    """Structural interface implemented by fixed and adaptive policies."""

    @property
    def name(self) -> str:
        """Return a stable name suitable for comparison artifacts."""

    @property
    def decision_log(self) -> tuple[TokenBudgetDecision, ...]:
        """Return decisions made since the last reset."""

    def decide(self, signals: SchedulerSignals) -> TokenBudgetDecision:
        """Choose the budget for one deterministic scheduler step."""

    def reset(self) -> None:
        """Reset state so a policy can be reused for an independent trace."""


def _split_budget(
    total_budget: int,
    signals: SchedulerSignals,
    decode_target: int,
    minimum_prefill_progress: int,
) -> tuple[int, int]:
    """Split a total while guaranteeing progress for every non-empty stage."""

    has_decode = signals.decode_backlog > 0
    has_prefill = signals.prefill_backlog > 0
    if not has_decode:
        return 0, total_budget
    if not has_prefill:
        return total_budget, 0

    # With a one-token external cap both stages cannot progress in the same step.
    # Favor the oldest prefill only after max-wait logic raised its target; normal
    # simulator configurations validate a budget of at least two tokens.
    if total_budget == 1:
        if minimum_prefill_progress > 0:
            return 0, 1
        return 1, 0

    prefill_floor = min(max(1, minimum_prefill_progress), total_budget - 1)
    decode = min(max(1, decode_target), total_budget - prefill_floor)
    return decode, total_budget - decode


class FixedTokenBudgetPolicy:
    """A configurable fixed-budget baseline.

    Decode requests require one token per engine iteration in the simulator, so
    the policy first reserves enough room for the current decode backlog and
    leaves at least ``minimum_prefill_progress`` tokens for prefill when both
    stages are non-empty.
    """

    def __init__(
        self,
        budget: int,
        max_admitted_sequences: int = 64,
        minimum_prefill_progress: int = 1,
    ) -> None:
        if budget < 2:
            raise ValueError("fixed budget must be at least two tokens")
        if max_admitted_sequences < 1:
            raise ValueError("max_admitted_sequences must be positive")
        if minimum_prefill_progress < 1:
            raise ValueError("minimum_prefill_progress must be positive")
        self.budget = budget
        self.max_admitted_sequences = max_admitted_sequences
        self.minimum_prefill_progress = minimum_prefill_progress
        self._decision_log: list[TokenBudgetDecision] = []

    @property
    def name(self) -> str:
        """Return the stable baseline label."""

        return f"fixed-{self.budget}"

    @property
    def decision_log(self) -> tuple[TokenBudgetDecision, ...]:
        """Return an immutable view of the decision log."""

        return tuple(self._decision_log)

    def reset(self) -> None:
        """Clear decisions from a previous simulation."""

        self._decision_log.clear()

    def decide(self, signals: SchedulerSignals) -> TokenBudgetDecision:
        """Return the same total budget with a backlog-aware stage split."""

        available = signals.available_token_budget
        total = self.budget if available is None else min(self.budget, available)
        decode, prefill = _split_budget(
            total,
            signals,
            decode_target=signals.decode_backlog,
            minimum_prefill_progress=self.minimum_prefill_progress,
        )
        decision = TokenBudgetDecision(
            step=signals.step,
            total_budget=total,
            decode_budget=decode,
            prefill_budget=prefill,
            admitted_sequence_limit=self.max_admitted_sequences,
            reasons=("fixed_budget",),
            signals=signals,
        )
        self._decision_log.append(decision)
        return decision

    __call__ = decide


FixedBudgetPolicy = FixedTokenBudgetPolicy


@dataclass(frozen=True)
class AdaptiveBudgetConfig:
    """Tunable but deterministic controls for :class:`AdaptiveTokenBudgetPolicy`."""

    min_budget: int = 512
    max_budget: int = 8192
    initial_budget: int = 2048
    budget_step: int = 512
    hysteresis_steps: int = 2
    decode_backlog_high: int = 16
    prefill_backlog_high: int = 4
    kv_pressure_high: float = 0.85
    kv_pressure_low: float = 0.60
    ttft_slo: float = 0.50
    tpot_slo: float = 0.05
    latency_guard_ratio: float = 0.90
    max_wait: float = 1.0
    minimum_prefill_progress: int = 32
    normal_prefill_share: float = 0.50
    pressure_decode_share: float = 0.75
    max_wait_prefill_share: float = 0.50
    min_admitted_sequences: int = 1
    max_admitted_sequences: int = 64
    admission_step: int = 4

    def __post_init__(self) -> None:
        if self.min_budget < 2:
            raise ValueError("min_budget must be at least two")
        if self.max_budget < self.min_budget:
            raise ValueError("max_budget must be >= min_budget")
        if not self.min_budget <= self.initial_budget <= self.max_budget:
            raise ValueError("initial_budget must be within budget bounds")
        if self.budget_step < 1:
            raise ValueError("budget_step must be positive")
        if self.hysteresis_steps < 1:
            raise ValueError("hysteresis_steps must be positive")
        if self.decode_backlog_high < 1 or self.prefill_backlog_high < 1:
            raise ValueError("backlog thresholds must be positive")
        if not 0.0 <= self.kv_pressure_low < self.kv_pressure_high <= 1.0:
            raise ValueError("KV thresholds must satisfy 0 <= low < high <= 1")
        if self.ttft_slo <= 0 or self.tpot_slo <= 0:
            raise ValueError("latency SLOs must be positive")
        if not 0.0 < self.latency_guard_ratio <= 1.0:
            raise ValueError("latency_guard_ratio must be in (0, 1]")
        if self.max_wait <= 0:
            raise ValueError("max_wait must be positive")
        if self.minimum_prefill_progress < 1:
            raise ValueError("minimum_prefill_progress must be positive")
        shares = (
            self.normal_prefill_share,
            self.pressure_decode_share,
            self.max_wait_prefill_share,
        )
        if any(not 0.0 < share < 1.0 for share in shares):
            raise ValueError("budget shares must be in (0, 1)")
        if self.min_admitted_sequences < 1:
            raise ValueError("min_admitted_sequences must be positive")
        if self.max_admitted_sequences < self.min_admitted_sequences:
            raise ValueError("max_admitted_sequences must be >= minimum")
        if self.admission_step < 1:
            raise ValueError("admission_step must be positive")


class AdaptiveTokenBudgetPolicy:
    """Stateful signal-driven token budget with hysteresis and guardrails."""

    def __init__(self, config: Optional[AdaptiveBudgetConfig] = None) -> None:
        self.config = config or AdaptiveBudgetConfig()
        self._decision_log: list[TokenBudgetDecision] = []
        self._current_budget = self.config.initial_budget
        self._current_admitted_limit = self.config.max_admitted_sequences
        self._pending_direction = 0
        self._pending_count = 0

    @property
    def name(self) -> str:
        """Return the stable adaptive policy label."""

        return "adaptive"

    @property
    def current_budget(self) -> int:
        """Return the uncapped stateful budget."""

        return self._current_budget

    @property
    def current_admitted_sequence_limit(self) -> int:
        """Return the current KV-pressure-aware sequence limit."""

        return self._current_admitted_limit

    @property
    def decision_log(self) -> tuple[TokenBudgetDecision, ...]:
        """Return an immutable view of all decisions."""

        return tuple(self._decision_log)

    def reset(self) -> None:
        """Restore the initial policy state and clear the audit log."""

        self._decision_log.clear()
        self._current_budget = self.config.initial_budget
        self._current_admitted_limit = self.config.max_admitted_sequences
        self._pending_direction = 0
        self._pending_count = 0

    def _desired_direction(self, signals: SchedulerSignals) -> tuple[int, list[str]]:
        reasons: list[str] = []
        decode_latency_pressure = (
            signals.decode_backlog >= self.config.decode_backlog_high
            or signals.recent_p99_tpot >= self.config.tpot_slo * self.config.latency_guard_ratio
        )
        kv_pressure = (
            signals.kv_pressure >= self.config.kv_pressure_high or signals.recent_preemptions > 0
        )
        ttft_pressure = (
            signals.prefill_backlog > 0
            and signals.recent_p99_ttft >= self.config.ttft_slo * self.config.latency_guard_ratio
        )

        if decode_latency_pressure:
            reasons.append("decode_or_tpot_pressure")
        if kv_pressure:
            reasons.append("kv_or_preemption_pressure")
        if ttft_pressure:
            reasons.append("ttft_pressure")

        # Decode/KV safety wins when signals conflict.  Old prefill requests are
        # protected by the split below rather than by expanding total work.
        if decode_latency_pressure or kv_pressure:
            return -1, reasons
        if ttft_pressure or signals.prefill_backlog >= self.config.prefill_backlog_high:
            reasons.append("prefill_throughput_demand")
            return 1, reasons
        return 0, reasons or ["steady_state"]

    def _apply_hysteresis(self, desired_direction: int) -> bool:
        if desired_direction == 0:
            self._pending_direction = 0
            self._pending_count = 0
            return False
        if desired_direction == self._pending_direction:
            self._pending_count += 1
        else:
            self._pending_direction = desired_direction
            self._pending_count = 1
        if self._pending_count < self.config.hysteresis_steps:
            return False

        previous = self._current_budget
        candidate = previous + desired_direction * self.config.budget_step
        self._current_budget = min(self.config.max_budget, max(self.config.min_budget, candidate))
        self._pending_direction = 0
        self._pending_count = 0
        return self._current_budget != previous

    def _update_admitted_limit(self, signals: SchedulerSignals) -> None:
        if signals.kv_pressure >= self.config.kv_pressure_high or signals.recent_preemptions > 0:
            self._current_admitted_limit = max(
                self.config.min_admitted_sequences,
                self._current_admitted_limit - self.config.admission_step,
            )
        elif signals.kv_pressure <= self.config.kv_pressure_low:
            self._current_admitted_limit = min(
                self.config.max_admitted_sequences,
                self._current_admitted_limit + self.config.admission_step,
            )

    def decide(self, signals: SchedulerSignals) -> TokenBudgetDecision:
        """Choose a bounded budget and stage allocation from all M5 signals."""

        direction, reasons = self._desired_direction(signals)
        changed = self._apply_hysteresis(direction)
        self._update_admitted_limit(signals)

        available = signals.available_token_budget
        total = self._current_budget if available is None else min(self._current_budget, available)
        overdue = signals.prefill_backlog > 0 and signals.oldest_prefill_age >= self.config.max_wait
        if overdue:
            reasons.append("max_wait_prefill_reservation")

        if signals.decode_backlog > 0 and signals.prefill_backlog > 0:
            if overdue:
                prefill_target = max(
                    self.config.minimum_prefill_progress,
                    ceil(total * self.config.max_wait_prefill_share),
                )
                decode_target = total - min(prefill_target, total - 1)
            elif direction < 0:
                decode_target = max(
                    signals.decode_backlog,
                    ceil(total * self.config.pressure_decode_share),
                )
            else:
                decode_target = max(
                    signals.decode_backlog,
                    ceil(total * (1.0 - self.config.normal_prefill_share)),
                )
        else:
            decode_target = signals.decode_backlog

        minimum_prefill = self.config.minimum_prefill_progress
        decode, prefill = _split_budget(
            total,
            signals,
            decode_target=decode_target,
            minimum_prefill_progress=minimum_prefill,
        )
        if changed:
            reasons.append("hysteresis_threshold_reached")

        decision = TokenBudgetDecision(
            step=signals.step,
            total_budget=total,
            decode_budget=decode,
            prefill_budget=prefill,
            admitted_sequence_limit=self._current_admitted_limit,
            reasons=tuple(dict.fromkeys(reasons)),
            signals=signals,
            changed=changed,
        )
        self._decision_log.append(decision)
        return decision

    __call__ = decide


AdaptiveBudgetPolicy = AdaptiveTokenBudgetPolicy
