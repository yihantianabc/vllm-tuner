"""Unit tests for deterministic scheduling simulation and budget ablations."""

import pytest

from vllm_tuner.scheduling import (
    AdaptiveBudgetConfig,
    AdaptiveTokenBudgetPolicy,
    AdmissionConfig,
    DeterministicSimulator,
    FairAdmissionController,
    FixedTokenBudgetPolicy,
    SimulationConfig,
    SimulationRequest,
    analyze_negative_benefit,
    percentile,
    run_budget_ablation,
    run_fixed_budget_baselines,
)


def mixed_trace():
    """Return long-prefill/short-prefill requests that overlap decode."""

    return (
        SimulationRequest("short", 0.0, prompt_tokens=1, output_tokens=6),
        SimulationRequest("long", 0.0, prompt_tokens=1000, output_tokens=3),
        SimulationRequest("later", 0.02, prompt_tokens=80, output_tokens=4),
    )


def test_percentile_uses_linear_interpolation():
    """Percentiles do not use a biased nearest-index shortcut."""

    assert percentile([0.0, 10.0], 50) == 5.0
    assert percentile([0.0, 10.0], 99) == pytest.approx(9.9)
    assert percentile([], 99) == 0.0


def test_simple_trace_has_hand_calculated_ttft_tpot_and_e2e():
    """Prefill occupies one step and three decode tokens occupy three steps."""

    simulator = DeterministicSimulator(
        FixedTokenBudgetPolicy(32),
        SimulationConfig(step_duration=0.01, prefill_quantum=32),
    )
    result = simulator.run([SimulationRequest("one", 0.0, prompt_tokens=1, output_tokens=3)])
    request = result.requests[0]

    assert request.queue_time == 0.0
    assert request.ttft == pytest.approx(0.02)
    assert request.tpot == pytest.approx(0.01)
    assert request.e2e == pytest.approx(0.04)
    assert request.token_timestamps == pytest.approx((0.02, 0.03, 0.04))


def test_every_step_conserves_budget_and_both_stages_make_progress():
    """A mixed stage step advances decode and prefill without oversubscription."""

    result = DeterministicSimulator(
        FixedTokenBudgetPolicy(64, minimum_prefill_progress=16),
        SimulationConfig(step_duration=0.01, prefill_quantum=16),
    ).run(mixed_trace())

    assert all(step.total_tokens <= step.total_budget for step in result.steps)
    assert sum(step.prefill_tokens for step in result.steps) == 1081
    assert sum(step.decode_tokens for step in result.steps) == 13
    assert any(step.decode_tokens > 0 and step.prefill_tokens > 0 for step in result.steps)
    assert result.metrics.scheduled_prefill_tokens == 1081
    assert result.metrics.scheduled_decode_tokens == 13


def test_same_seed_and_trace_produce_identical_results_and_decisions():
    """Policy state is reset and no wall clock or hash ordering leaks into output."""

    config = SimulationConfig(seed=123, step_duration=0.01, prefill_quantum=32)
    policy = AdaptiveTokenBudgetPolicy(
        AdaptiveBudgetConfig(
            min_budget=64,
            max_budget=256,
            initial_budget=128,
            budget_step=64,
            hysteresis_steps=2,
            minimum_prefill_progress=16,
        )
    )
    simulator = DeterministicSimulator(policy, config)

    first = simulator.run(mixed_trace())
    repeated = simulator.run(mixed_trace())

    assert first == repeated
    assert first.seed == 123
    assert len(first.decisions) == len(first.steps)


def test_aging_max_wait_completes_all_requests_without_infinite_starvation():
    """One admitted slot rotates among long requests until every request finishes."""

    admission_config = AdmissionConfig(
        max_wait=0.03,
        minimum_prefill_progress=16,
        max_preemptions_per_step=1,
    )
    simulator = DeterministicSimulator(
        FixedTokenBudgetPolicy(
            64,
            max_admitted_sequences=1,
            minimum_prefill_progress=16,
        ),
        SimulationConfig(
            step_duration=0.01,
            prefill_quantum=16,
            starvation_threshold=0.30,
            max_steps=10_000,
        ),
        FairAdmissionController(admission_config),
    )
    trace = [
        SimulationRequest(str(index), 0.0, prompt_tokens=300, output_tokens=10)
        for index in range(3)
    ]
    result = simulator.run(trace)

    assert result.metrics.completed_requests == len(trace)
    assert result.metrics.starvation_count == 0
    assert result.metrics.max_wait_observed <= 0.06
    assert result.metrics.preemption_count > 0
    assert all(request.finish_time < 1.0 for request in result.requests)


def test_adaptive_kv_pressure_reduces_admission_and_records_preemption():
    """Synthetic KV saturation exercises the policy/controller feedback loop."""

    adaptive_config = AdaptiveBudgetConfig(
        min_budget=64,
        max_budget=256,
        initial_budget=128,
        budget_step=64,
        hysteresis_steps=1,
        decode_backlog_high=2,
        prefill_backlog_high=2,
        kv_pressure_high=0.5,
        kv_pressure_low=0.2,
        minimum_prefill_progress=16,
        min_admitted_sequences=1,
        max_admitted_sequences=4,
        admission_step=2,
    )
    trace = [
        SimulationRequest(str(index), 0.0, prompt_tokens=100, output_tokens=5) for index in range(4)
    ]
    result = DeterministicSimulator(
        AdaptiveTokenBudgetPolicy(adaptive_config),
        SimulationConfig(
            step_duration=0.01,
            prefill_quantum=32,
            kv_capacity_tokens=100,
        ),
        FairAdmissionController(AdmissionConfig(max_wait=1.0, minimum_prefill_progress=16)),
    ).run(trace)

    limits = [step.admitted_sequence_limit for step in result.steps]
    assert min(limits) < adaptive_config.max_admitted_sequences
    assert result.metrics.preemption_count > 0
    assert any(
        decision.signals.kv_pressure >= adaptive_config.kv_pressure_high
        for decision in result.decisions
    )


def test_aggregate_contains_all_required_m5_metrics():
    """Queue, latency, goodput, fairness, starvation, and preemption are exported."""

    result = DeterministicSimulator(
        FixedTokenBudgetPolicy(64),
        SimulationConfig(step_duration=0.01),
    ).run(mixed_trace())
    metrics = result.metrics.to_dict()

    required = {
        "p50_queue_time",
        "p99_queue_time",
        "p50_ttft",
        "p99_ttft",
        "p50_tpot",
        "p99_tpot",
        "goodput",
        "fairness_index",
        "starvation_count",
        "preemption_count",
    }
    assert required <= metrics.keys()
    assert 0.0 <= result.metrics.fairness_index <= 1.0
    assert result.to_dict()["requests"][0]["token_timestamps"]


def test_fixed_budget_baselines_are_configurable_and_require_two():
    """Ablations can choose budgets while preventing a one-baseline comparison."""

    config = SimulationConfig(step_duration=0.01, prefill_quantum=16)
    results = run_fixed_budget_baselines(mixed_trace(), budgets=(32, 128), simulation_config=config)

    assert set(results) == {32, 128}
    assert results[32].policy_name == "fixed-32"
    assert results[128].metrics.completed_requests == 3
    with pytest.raises(ValueError, match="at least two"):
        run_fixed_budget_baselines(mixed_trace(), budgets=(32,))


def test_ablation_includes_held_out_and_negative_benefit_analysis():
    """Calibration and unseen traces retain baseline and downside evidence."""

    adaptive_config = AdaptiveBudgetConfig(
        min_budget=32,
        max_budget=128,
        initial_budget=64,
        budget_step=32,
        hysteresis_steps=2,
        minimum_prefill_progress=8,
        max_admitted_sequences=8,
    )
    config = SimulationConfig(step_duration=0.01, prefill_quantum=16)
    held_out = (
        SimulationRequest("held-short", 0.0, 8, 4),
        SimulationRequest("held-long", 0.01, 400, 2),
    )
    report = run_budget_ablation(
        calibration_trace=mixed_trace(),
        held_out_trace=held_out,
        fixed_budgets=(32, 128),
        adaptive_config=adaptive_config,
        simulation_config=config,
    )

    assert set(report.calibration.fixed_baselines) == {32, 128}
    assert set(report.held_out.fixed_baselines) == {32, 128}
    assert report.heldout.trace_name == "held_out"
    assert report.held_out.adaptive.metrics.completed_requests == len(held_out)
    assert isinstance(report.negative_gain_conditions, tuple)
    assert report.negative_gain_conditions
    artifact = report.to_dict()
    assert artifact["held_out"]["fixed_baselines"].keys() == {"32", "128"}
    assert artifact["negative_gain_conditions"]


def test_negative_benefit_analysis_rejects_underpowered_comparison():
    """A claimed fixed-budget ablation must contain at least two controls."""

    result = DeterministicSimulator(FixedTokenBudgetPolicy(32)).run(
        [SimulationRequest("one", 0.0, 1, 1)]
    )
    with pytest.raises(ValueError, match="at least two"):
        analyze_negative_benefit("bad", {32: result}, result)


def test_empty_trace_and_duplicate_ids_are_handled_explicitly():
    """Empty experiments are valid, while ambiguous IDs fail fast."""

    simulator = DeterministicSimulator(FixedTokenBudgetPolicy(32))
    empty = simulator.run([])

    assert empty.metrics.total_requests == 0
    assert empty.metrics.fairness_index == 1.0
    with pytest.raises(ValueError, match="unique"):
        simulator.run(
            [
                SimulationRequest("same", 0.0, 1, 1),
                SimulationRequest("same", 0.0, 1, 1),
            ]
        )
