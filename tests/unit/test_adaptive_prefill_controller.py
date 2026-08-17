"""State, hysteresis, residency, and starvation tests for the pure controller."""

from vllm_tuner.config.models import AdaptivePrefillConfig
from vllm_tuner.scheduler.controller import (
    AdaptivePrefillController,
    ControllerState,
)
from vllm_tuner.scheduler.instrumentation import SchedulerSignals


def make_signals(
    *,
    decode_backlog: int = 0,
    oldest_prefill_wait_ms: float = 0.0,
    kv_cache_usage: float = 0.0,
) -> SchedulerSignals:
    return SchedulerSignals(
        timestamp=1.0,
        decode_backlog=decode_backlog,
        oldest_prefill_wait_ms=oldest_prefill_wait_ms,
        kv_cache_usage=kv_cache_usage,
        running_requests=decode_backlog,
        waiting_requests=0,
        prefill_request_ids=frozenset(),
    )


def immediate_config(**overrides) -> AdaptivePrefillConfig:
    values = {
        "low_prefill_cap": 256,
        "balanced_prefill_cap": 512,
        "high_prefill_cap": 1024,
        "decode_backlog_high": 4,
        "oldest_prefill_wait_ms": 100.0,
        "kv_usage_high": 0.8,
        "min_prefill_progress": 128,
        "max_wait_ms": 500.0,
        "hysteresis_steps": 1,
        "min_state_residency_steps": 1,
        **overrides,
    }
    return AdaptivePrefillConfig(**values)


def test_controller_enters_all_three_states_and_returns_matching_caps() -> None:
    controller = AdaptivePrefillController(immediate_config())

    balanced = controller.decide(make_signals())
    protect = controller.decide(make_signals(decode_backlog=4))
    back_to_balanced = controller.decide(make_signals())
    drain = controller.decide(make_signals(oldest_prefill_wait_ms=100.0))

    assert (balanced.state, balanced.prefill_cap) == (ControllerState.BALANCED, 512)
    assert (protect.state, protect.prefill_cap) == (ControllerState.PROTECT_DECODE, 256)
    assert protect.transitioned is True
    assert back_to_balanced.state is ControllerState.BALANCED
    assert (drain.state, drain.prefill_cap) == (ControllerState.DRAIN_PREFILL, 1024)


def test_kv_pressure_has_protect_priority_over_prefill_wait() -> None:
    controller = AdaptivePrefillController(immediate_config())
    controller.decide(make_signals())

    decision = controller.decide(make_signals(oldest_prefill_wait_ms=200.0, kv_cache_usage=0.8))

    assert decision.state is ControllerState.PROTECT_DECODE
    assert "kv_usage_high" in decision.reason_code


def test_hysteresis_requires_consecutive_candidate_steps() -> None:
    controller = AdaptivePrefillController(immediate_config(hysteresis_steps=2))
    controller.decide(make_signals())

    first = controller.decide(make_signals(decode_backlog=4))
    reset = controller.decide(make_signals())
    second_first = controller.decide(make_signals(decode_backlog=4))
    second_second = controller.decide(make_signals(decode_backlog=4))

    assert first.state is ControllerState.BALANCED
    assert "hold_hysteresis" in first.reason_code
    assert reset.state is ControllerState.BALANCED
    assert second_first.state is ControllerState.BALANCED
    assert second_second.state is ControllerState.PROTECT_DECODE


def test_minimum_residency_delays_a_new_state() -> None:
    controller = AdaptivePrefillController(immediate_config(min_state_residency_steps=3))

    first = controller.decide(make_signals(decode_backlog=4))
    second = controller.decide(make_signals(decode_backlog=4))
    third = controller.decide(make_signals(decode_backlog=4))
    fourth = controller.decide(make_signals(decode_backlog=4))

    assert [first.state, second.state, third.state] == [
        ControllerState.BALANCED,
        ControllerState.BALANCED,
        ControllerState.BALANCED,
    ]
    assert "hold_min_residency" in first.reason_code
    assert fourth.state is ControllerState.PROTECT_DECODE


def test_max_wait_marks_forced_progress_without_overriding_protect_state() -> None:
    controller = AdaptivePrefillController(immediate_config())
    controller.decide(make_signals())

    decision = controller.decide(make_signals(decode_backlog=4, oldest_prefill_wait_ms=500.0))

    assert decision.state is ControllerState.PROTECT_DECODE
    assert decision.max_wait_forced is True
    assert decision.prefill_cap >= controller.config.min_prefill_progress
    assert "max_wait_progress" in decision.reason_code


def test_identical_signal_sequences_produce_identical_decisions() -> None:
    config = immediate_config(hysteresis_steps=2, min_state_residency_steps=2)
    controllers = [AdaptivePrefillController(config), AdaptivePrefillController(config)]
    signals = [
        make_signals(),
        make_signals(decode_backlog=4),
        make_signals(decode_backlog=4),
        make_signals(oldest_prefill_wait_ms=150.0),
        make_signals(oldest_prefill_wait_ms=600.0),
    ]

    sequences = [[controller.decide(signal) for signal in signals] for controller in controllers]

    assert sequences[0] == sequences[1]
