"""Pure three-state adaptive Prefill controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vllm_tuner.config.models import AdaptivePrefillConfig

from .instrumentation import SchedulerSignals


class ControllerState(str, Enum):
    """Stable controller modes with one explainable Prefill cap each."""

    PROTECT_DECODE = "PROTECT_DECODE"
    BALANCED = "BALANCED"
    DRAIN_PREFILL = "DRAIN_PREFILL"


@dataclass(frozen=True)
class ControllerDecision:
    """Pure controller output before vLLM capacity safety clamps."""

    state: ControllerState
    prefill_cap: int
    reason_code: str
    transitioned: bool
    max_wait_forced: bool


class AdaptivePrefillController:
    """Choose a Prefill cap from three signals with stable state transitions."""

    def __init__(self, config: AdaptivePrefillConfig) -> None:
        self.config = config
        self.state = ControllerState.BALANCED
        self.steps_in_state = 0
        self._pending_state: ControllerState | None = None
        self._pending_steps = 0

    def _target_state(self, signals: SchedulerSignals) -> tuple[ControllerState, str]:
        if signals.decode_backlog >= self.config.decode_backlog_high:
            return ControllerState.PROTECT_DECODE, "decode_backlog_high"
        if signals.kv_cache_usage >= self.config.kv_usage_high:
            return ControllerState.PROTECT_DECODE, "kv_usage_high"
        if signals.oldest_prefill_wait_ms >= self.config.oldest_prefill_wait_ms:
            return ControllerState.DRAIN_PREFILL, "prefill_wait_high"
        return ControllerState.BALANCED, "balanced"

    def _cap_for_state(self, state: ControllerState) -> int:
        if state is ControllerState.PROTECT_DECODE:
            return self.config.low_prefill_cap
        if state is ControllerState.DRAIN_PREFILL:
            return self.config.high_prefill_cap
        return self.config.balanced_prefill_cap

    def decide(self, signals: SchedulerSignals) -> ControllerDecision:
        """Return a deterministic decision and advance controller-only state."""
        target_state, trigger = self._target_state(signals)
        transitioned = False
        reason_code = trigger
        if target_state is self.state:
            self._pending_state = None
            self._pending_steps = 0
        elif self.steps_in_state < self.config.min_state_residency_steps:
            self._pending_state = None
            self._pending_steps = 0
            reason_code = f"hold_min_residency:{trigger}"
        else:
            if self._pending_state is target_state:
                self._pending_steps += 1
            else:
                self._pending_state = target_state
                self._pending_steps = 1
            if self._pending_steps >= self.config.hysteresis_steps:
                self.state = target_state
                self.steps_in_state = 0
                self._pending_state = None
                self._pending_steps = 0
                transitioned = True
                reason_code = f"transition:{trigger}"
            else:
                reason_code = f"hold_hysteresis:{trigger}"

        max_wait_forced = signals.oldest_prefill_wait_ms >= self.config.max_wait_ms
        prefill_cap = self._cap_for_state(self.state)
        if max_wait_forced:
            prefill_cap = max(prefill_cap, self.config.min_prefill_progress)
            reason_code = f"{reason_code};max_wait_progress"

        self.steps_in_state += 1
        return ControllerDecision(
            state=self.state,
            prefill_cap=prefill_cap,
            reason_code=reason_code,
            transitioned=transitioned,
            max_wait_forced=max_wait_forced,
        )
