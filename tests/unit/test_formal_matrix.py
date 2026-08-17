"""Frozen M4 config and randomized job-order tests."""

from pathlib import Path

from vllm_tuner.experiment.formal import (
    build_formal_config,
    formal_job_order,
    load_formal_protocol,
)


def protocol() -> dict:
    repository = Path(__file__).resolve().parents[2]
    return load_formal_protocol(repository / "experiments/adaptive_prefill/m3_formal_protocol.yaml")


def test_formal_config_uses_frozen_policy_load_and_primary_slo() -> None:
    value = protocol()
    config = build_formal_config(
        value,
        load="main",
        policy="adaptive",
        resume=False,
    )

    assert config.workload.request_rate == 8.0
    assert config.workload.sample_size == 640
    assert config.workload.warmup_requests == 5
    assert config.study.repeat_count == 3
    assert config.slo.model_dump() == {
        "ttft_ms": 850.0,
        "tpot_ms": 37.0,
        "e2e_ms": 3170.0,
    }
    assert config.adaptive_prefill.decode_backlog_high == 28
    assert config.vllm_args["scheduler-cls"].endswith("AdaptivePrefillScheduler")


def test_fixed_config_reuses_frozen_controller_levels_but_bypasses_state_machine() -> None:
    config = build_formal_config(
        protocol(),
        load="low",
        policy="fixed_mid",
        resume=True,
    )

    assert config.study.resume is True
    assert config.adaptive_prefill.fixed_prefill_cap == 4096
    assert config.adaptive_prefill.low_prefill_cap == 1024
    assert config.adaptive_prefill.high_prefill_cap == 8192


def test_job_order_is_complete_deterministic_and_policy_randomized_per_block() -> None:
    value = protocol()
    first = formal_job_order(value, seed=2026)
    second = formal_job_order(value, seed=2026)

    assert first == second
    assert len(first) == 3 * 3 * 2 * 5
    identities = {(job.load, job.policy, job.trace_kind, job.repeat) for job in first}
    assert len(identities) == len(first)
    first_block = [
        job.policy
        for job in first
        if job.repeat == 0 and job.load == "low" and job.trace_kind == "calibration"
    ]
    assert set(first_block) == set(value["policies"])
    assert first_block != list(value["policies"])
