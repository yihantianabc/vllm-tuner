"""Fair, deterministic admission and service ordering for scheduler simulation."""

from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class AdmissionConfig:
    """Fairness controls shared by admission and prefill service ordering."""

    max_wait: float = 1.0
    aging_weight: float = 1.0
    minimum_prefill_progress: int = 32
    allow_preemption: bool = True
    max_preemptions_per_step: int = 1

    def __post_init__(self) -> None:
        if self.max_wait <= 0:
            raise ValueError("max_wait must be positive")
        if self.aging_weight <= 0:
            raise ValueError("aging_weight must be positive")
        if self.minimum_prefill_progress < 1:
            raise ValueError("minimum_prefill_progress must be positive")
        if self.max_preemptions_per_step < 1:
            raise ValueError("max_preemptions_per_step must be positive")


@dataclass(frozen=True)
class AdmissionCandidate:
    """Read-only request view consumed by :class:`FairAdmissionController`."""

    request_id: str
    arrival_time: float
    waiting_since: float
    original_order: int
    stage: str = "prefill"
    priority: int = 0
    last_progress_time: Optional[float] = None
    service_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.arrival_time < 0 or self.waiting_since < 0:
            raise ValueError("candidate times must be non-negative")
        if self.waiting_since < self.arrival_time:
            raise ValueError("waiting_since cannot precede arrival_time")
        if self.original_order < 0:
            raise ValueError("original_order must be non-negative")
        if self.stage not in {"prefill", "decode"}:
            raise ValueError("stage must be 'prefill' or 'decode'")
        if self.last_progress_time is not None and self.last_progress_time < self.arrival_time:
            raise ValueError("last_progress_time cannot precede arrival_time")
        if self.service_tokens < 0:
            raise ValueError("service_tokens must be non-negative")


@dataclass(frozen=True)
class AdmissionDecision:
    """Requests to admit/preempt for one scheduling step."""

    admitted_request_ids: tuple[str, ...]
    preempted_request_ids: tuple[str, ...]
    oldest_wait: float
    starvation_prevented: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        result = asdict(self)
        result["admitted_request_ids"] = list(self.admitted_request_ids)
        result["preempted_request_ids"] = list(self.preempted_request_ids)
        result["reasons"] = list(self.reasons)
        return result


class FairAdmissionController:
    """Admission controller with aging, max-wait swaps, and stable tie breaks.

    The controller is stateless: all decisions are a pure function of the request
    views, time, and sequence limit.  This makes a trace reproducible and lets the
    simulator preserve work independently from admission status.
    """

    def __init__(self, config: Optional[AdmissionConfig] = None) -> None:
        self.config = config or AdmissionConfig()

    def _wait(self, candidate: AdmissionCandidate, now: float) -> float:
        return max(0.0, now - candidate.waiting_since)

    def _service_age(self, candidate: AdmissionCandidate, now: float) -> float:
        anchor = candidate.last_progress_time
        if anchor is None:
            anchor = candidate.waiting_since
        return max(0.0, now - anchor)

    def _waiting_key(
        self, candidate: AdmissionCandidate, now: float
    ) -> tuple[int, float, float, int, str]:
        wait = self._wait(candidate, now)
        overdue = wait >= self.config.max_wait
        age_score = self.config.aging_weight * wait / self.config.max_wait
        # Higher explicit priority and age should sort first.  Stable trace order
        # and request ID make exact ties deterministic across Python processes.
        score = candidate.priority + age_score
        return (
            0 if overdue else 1,
            -score,
            candidate.arrival_time,
            candidate.original_order,
            candidate.request_id,
        )

    def _service_key(
        self, candidate: AdmissionCandidate, now: float
    ) -> tuple[int, float, int, float, int, str]:
        service_age = self._service_age(candidate, now)
        overdue = service_age >= self.config.max_wait
        return (
            0 if overdue else 1,
            -service_age,
            candidate.service_tokens,
            candidate.arrival_time,
            candidate.original_order,
            candidate.request_id,
        )

    def _victim_key(
        self, candidate: AdmissionCandidate, now: float
    ) -> tuple[int, int, float, float, int, str]:
        # Never-progressed and long-unserved requests are the least desirable
        # victims.  Among safe victims choose low priority/new requests first.
        service_age = self._service_age(candidate, now)
        protected = service_age >= self.config.max_wait or candidate.service_tokens == 0
        return (
            1 if protected else 0,
            candidate.priority,
            service_age,
            -candidate.arrival_time,
            -candidate.original_order,
            candidate.request_id,
        )

    def rank_waiting(
        self, candidates: Sequence[AdmissionCandidate], now: float
    ) -> tuple[AdmissionCandidate, ...]:
        """Rank queued requests using max-wait first, then weighted aging."""

        if now < 0:
            raise ValueError("now must be non-negative")
        return tuple(sorted(candidates, key=lambda candidate: self._waiting_key(candidate, now)))

    def rank_for_service(
        self, candidates: Sequence[AdmissionCandidate], now: float
    ) -> tuple[AdmissionCandidate, ...]:
        """Rank admitted requests by time since last progress.

        The ordering is used by the simulator's round-robin prefill chunks and
        decode slots, so an admitted long request cannot be ignored forever.
        """

        if now < 0:
            raise ValueError("now must be non-negative")
        return tuple(sorted(candidates, key=lambda candidate: self._service_key(candidate, now)))

    def decide(
        self,
        waiting: Sequence[AdmissionCandidate],
        admitted: Sequence[AdmissionCandidate],
        now: float,
        admitted_sequence_limit: int,
    ) -> AdmissionDecision:
        """Select admissions and bounded preemptions for a scheduler step."""

        if now < 0:
            raise ValueError("now must be non-negative")
        if admitted_sequence_limit < 1:
            raise ValueError("admitted_sequence_limit must be positive")

        all_ids = [candidate.request_id for candidate in waiting]
        all_ids.extend(candidate.request_id for candidate in admitted)
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("candidate request IDs must be unique")

        ranked_waiting = list(self.rank_waiting(waiting, now))
        active = list(admitted)
        preempted: list[str] = []
        newly_admitted: list[str] = []
        reasons: list[str] = []

        excess = max(0, len(active) - admitted_sequence_limit)
        if excess:
            victims = sorted(active, key=lambda candidate: self._victim_key(candidate, now))
            for victim in victims[:excess]:
                preempted.append(victim.request_id)
                active.remove(victim)
            reasons.append("sequence_limit_reduced")

        free_slots = max(0, admitted_sequence_limit - len(active))
        selected = ranked_waiting[:free_slots]
        newly_admitted.extend(candidate.request_id for candidate in selected)
        remaining_waiting = ranked_waiting[free_slots:]
        if selected:
            reasons.append("capacity_available")

        # When capacity is full, an overdue request swaps with a request that has
        # recently made progress.  Work is suspended, not discarded, by the
        # simulator.  The per-step cap prevents preemption storms.
        overdue = [
            candidate
            for candidate in remaining_waiting
            if self._wait(candidate, now) >= self.config.max_wait
        ]
        swap_count = 0
        if overdue and active and self.config.allow_preemption:
            victims = sorted(active, key=lambda candidate: self._victim_key(candidate, now))
            swap_count = min(
                len(overdue),
                len(victims),
                self.config.max_preemptions_per_step,
            )
            for candidate, victim in zip(overdue[:swap_count], victims[:swap_count]):
                preempted.append(victim.request_id)
                newly_admitted.append(candidate.request_id)
            reasons.append("max_wait_swap")

        oldest_wait = max((self._wait(candidate, now) for candidate in waiting), default=0.0)
        starvation_prevented = swap_count > 0 or any(
            self._wait(candidate, now) >= self.config.max_wait for candidate in selected
        )
        if not reasons:
            reasons.append("no_change")
        return AdmissionDecision(
            admitted_request_ids=tuple(newly_admitted),
            preempted_request_ids=tuple(preempted),
            oldest_wait=oldest_wait,
            starvation_prevented=starvation_prevented,
            reasons=tuple(reasons),
        )


AdmissionController = FairAdmissionController
AgingAdmissionController = FairAdmissionController
