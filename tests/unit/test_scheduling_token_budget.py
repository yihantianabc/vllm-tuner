"""Unit tests for fixed and adaptive token-budget policies."""

from dataclasses import replace

import pytest

from vllm_tuner.scheduling.token_budget import (
    DEFAULT_FIXED_BUDGETS,
    AdaptiveBudgetConfig,
    AdaptiveTokenBudgetPolicy,
    FixedTokenBudgetPolicy,
    SchedulerSignals,
)


def make_signals(**overrides):
    """Build a valid signal snapshot with concise per-test overrides."""

    signals = SchedulerSignals(
        step=0,
        decode_backlog=1,
        prefill_backlog=1,
        oldest_prefill_age=0.0,
        kv_pressure=0.1,
        recent_p99_ttft=0.0,
        recent_p99_tpot=0.0,
        recent_preemptions=0,
    )
    return replace(signals, **overrides)


def test_default_fixed_budgets_cover_planned_baselines():
    """The project-plan baselines remain available without configuration."""

    assert DEFAULT_FIXED_BUDGETS == (512, 1024, 2048, 4096, 8192)


def test_fixed_policy_conserves_budget_and_progresses_both_stages():
    """A mixed backlog receives non-zero decode and prefill allocations."""

    policy = FixedTokenBudgetPolicy(
        budget=512,
        max_admitted_sequences=8,
        minimum_prefill_progress=32,
    )
    decision = policy.decide(make_signals(decode_backlog=20, prefill_backlog=3))

    assert decision.total_budget == 512
    assert decision.decode_budget >= 20
    assert decision.prefill_budget >= 32
    assert decision.decode_budget + decision.prefill_budget == decision.total_budget
    assert decision.admitted_sequence_limit == 8


def test_fixed_policy_honors_temporary_available_budget_cap():
    """Runtime availability can cap, but never be exceeded by, a baseline."""

    decision = FixedTokenBudgetPolicy(8192).decide(make_signals(available_token_budget=1024))

    assert decision.total_budget == 1024
    assert decision.decode_budget + decision.prefill_budget == 1024


def test_adaptive_policy_uses_hysteresis_before_reducing_budget():
    """A single pressure sample does not cause budget oscillation."""

    config = AdaptiveBudgetConfig(
        min_budget=512,
        max_budget=2048,
        initial_budget=1024,
        budget_step=512,
        hysteresis_steps=2,
        decode_backlog_high=2,
        minimum_prefill_progress=32,
    )
    policy = AdaptiveTokenBudgetPolicy(config)
    pressure = make_signals(decode_backlog=2, prefill_backlog=1)

    first = policy.decide(pressure)
    second = policy.decide(replace(pressure, step=1))

    assert first.total_budget == 1024
    assert not first.changed
    assert second.total_budget == 512
    assert second.changed
    assert second.prefill_budget >= config.minimum_prefill_progress
    assert "decode_or_tpot_pressure" in second.reasons


def test_adaptive_budget_increases_for_prefill_demand_but_stays_bounded():
    """Sustained prefill backlog moves toward, but not beyond, max_budget."""

    config = AdaptiveBudgetConfig(
        min_budget=256,
        max_budget=1024,
        initial_budget=512,
        budget_step=256,
        hysteresis_steps=1,
        prefill_backlog_high=2,
        minimum_prefill_progress=16,
    )
    policy = AdaptiveTokenBudgetPolicy(config)

    decisions = [
        policy.decide(
            make_signals(
                step=step,
                decode_backlog=0,
                prefill_backlog=4,
            )
        )
        for step in range(5)
    ]

    assert decisions[0].total_budget == 768
    assert decisions[-1].total_budget == config.max_budget
    assert all(decision.prefill_budget == decision.total_budget for decision in decisions)


def test_adaptive_policy_reserves_oldest_prefill_at_max_wait():
    """Old prefill work receives a material reservation even under decode load."""

    config = AdaptiveBudgetConfig(
        min_budget=512,
        max_budget=2048,
        initial_budget=1024,
        hysteresis_steps=1,
        decode_backlog_high=1,
        max_wait=0.25,
        minimum_prefill_progress=64,
        max_wait_prefill_share=0.5,
    )
    decision = AdaptiveTokenBudgetPolicy(config).decide(
        make_signals(
            decode_backlog=100,
            prefill_backlog=2,
            oldest_prefill_age=0.25,
        )
    )

    assert decision.prefill_budget >= decision.total_budget // 2
    assert decision.decode_budget > 0
    assert "max_wait_prefill_reservation" in decision.reasons


def test_kv_pressure_and_preemptions_reduce_admitted_sequence_limit():
    """KV and preemption signals directly close the admission valve."""

    config = AdaptiveBudgetConfig(
        min_budget=512,
        max_budget=2048,
        initial_budget=1024,
        hysteresis_steps=1,
        min_admitted_sequences=2,
        max_admitted_sequences=10,
        admission_step=3,
    )
    policy = AdaptiveTokenBudgetPolicy(config)
    decision = policy.decide(make_signals(kv_pressure=0.95, recent_preemptions=1))

    assert decision.admitted_sequence_limit == 7
    assert "kv_or_preemption_pressure" in decision.reasons
    assert decision.signals.recent_preemptions == 1


def test_ttft_and_tpot_signals_are_logged_with_decision():
    """Latency feedback is retained in the auditable decision record."""

    policy = AdaptiveTokenBudgetPolicy(
        AdaptiveBudgetConfig(hysteresis_steps=1, decode_backlog_high=100)
    )
    signals = make_signals(
        prefill_backlog=5,
        recent_p99_ttft=0.5,
        recent_p99_tpot=0.04,
    )
    decision = policy.decide(signals)

    assert decision.signals == signals
    assert "ttft_pressure" in decision.reasons
    assert policy.decision_log == (decision,)
    assert decision.to_dict()["signals"]["recent_p99_ttft"] == 0.5


def test_policy_reset_makes_repeated_signal_sequence_reproducible():
    """Resetting removes adaptive state and produces identical decisions."""

    policy = AdaptiveTokenBudgetPolicy(
        AdaptiveBudgetConfig(hysteresis_steps=1, prefill_backlog_high=1)
    )
    signals = make_signals(decode_backlog=0, prefill_backlog=2)
    first = policy.decide(signals)

    policy.reset()
    repeated = policy.decide(signals)

    assert first == repeated
    assert policy.decision_log == (repeated,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_budget": 1},
        {"min_budget": 1024, "max_budget": 512},
        {"initial_budget": 16384},
        {"hysteresis_steps": 0},
        {"kv_pressure_low": 0.9, "kv_pressure_high": 0.8},
        {"minimum_prefill_progress": 0},
    ],
)
def test_adaptive_config_rejects_invalid_guardrails(kwargs):
    """Invalid budget, hysteresis, pressure, and progress controls fail early."""

    with pytest.raises(ValueError):
        AdaptiveBudgetConfig(**kwargs)
