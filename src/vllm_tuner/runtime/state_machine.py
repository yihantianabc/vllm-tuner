"""Strict trial lifecycle with monotonic transition history."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from vllm_tuner.experiment.models import TrialStatus


class InvalidTransitionError(RuntimeError):
    """Raised when a controller skips or reverses a lifecycle phase."""


@dataclass(frozen=True)
class StateTransition:
    """One auditable state change."""

    previous: TrialStatus
    current: TrialStatus
    monotonic_ns: int
    reason: Optional[str] = None


NORMAL_TRANSITIONS: dict[TrialStatus, set[TrialStatus]] = {
    TrialStatus.CREATED: {TrialStatus.STARTING},
    TrialStatus.STARTING: {TrialStatus.READY},
    TrialStatus.READY: {TrialStatus.WARMING_UP},
    TrialStatus.WARMING_UP: {TrialStatus.MEASURING},
    TrialStatus.MEASURING: {TrialStatus.COLLECTING},
    TrialStatus.COLLECTING: {TrialStatus.STOPPING},
    TrialStatus.STOPPING: {TrialStatus.COMPLETE},
}
FAILURE_STATES = {TrialStatus.FAILED, TrialStatus.INFEASIBLE, TrialStatus.PRUNED}


class TrialStateMachine:
    """Validate transitions and notify an optional artifact callback."""

    def __init__(
        self,
        on_transition: Optional[Callable[[StateTransition], None]] = None,
    ) -> None:
        self.status = TrialStatus.CREATED
        self.history: list[StateTransition] = []
        self._on_transition = on_transition

    def transition(self, target: TrialStatus, reason: Optional[str] = None) -> StateTransition:
        """Move to an allowed next state or any terminal failure state."""
        if self.status.terminal:
            raise InvalidTransitionError(f"terminal state {self.status.value} cannot transition")
        allowed = NORMAL_TRANSITIONS.get(self.status, set())
        if target not in allowed and target not in FAILURE_STATES:
            raise InvalidTransitionError(
                f"invalid trial transition {self.status.value} -> {target.value}"
            )
        event = StateTransition(
            previous=self.status,
            current=target,
            monotonic_ns=time.perf_counter_ns(),
            reason=reason,
        )
        self.status = target
        self.history.append(event)
        if self._on_transition is not None:
            self._on_transition(event)
        return event

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible snapshot."""
        return {
            "status": self.status.value,
            "history": [
                {
                    "previous": event.previous.value,
                    "current": event.current.value,
                    "monotonic_ns": event.monotonic_ns,
                    "reason": event.reason,
                }
                for event in self.history
            ],
        }
