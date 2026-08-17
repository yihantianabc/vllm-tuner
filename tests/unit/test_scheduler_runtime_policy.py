"""Policy-selection tests for the vLLM runtime adapter."""

from dataclasses import dataclass

import pytest

pytest.importorskip("vllm")

from vllm_tuner.config.models import AdaptivePrefillConfig
from vllm_tuner.scheduler.controller import AdaptivePrefillController
from vllm_tuner.scheduler.instrumentation import SchedulerSignals
from vllm_tuner.scheduler.runtime import AdaptivePrefillScheduler


@dataclass
class FakeDecodeRequest:
    num_output_tokens: int = 1
    num_tokens_with_spec: int = 101
    num_output_placeholders: int = 0
    num_computed_tokens: int = 100


def make_runtime(config: AdaptivePrefillConfig) -> AdaptivePrefillScheduler:
    runtime = object.__new__(AdaptivePrefillScheduler)
    runtime.adaptive_prefill_config = config
    runtime.max_num_scheduled_tokens = 1024
    runtime.requests = {}
    runtime._controller = AdaptivePrefillController(config)
    return runtime


def signals() -> SchedulerSignals:
    return SchedulerSignals(
        timestamp=1.0,
        decode_backlog=0,
        oldest_prefill_wait_ms=0.0,
        kv_cache_usage=0.0,
        running_requests=0,
        waiting_requests=0,
        prefill_request_ids=frozenset(),
    )


def test_disabled_runtime_returns_the_unmodified_global_budget() -> None:
    runtime = make_runtime(AdaptivePrefillConfig(enabled=False))

    state, cap, reason, decision = runtime._controller_decision(signals())

    assert (state, cap, reason, decision) == (
        "DISABLED",
        1024,
        "controller_disabled",
        None,
    )


def test_fixed_runtime_reserves_decode_demand_without_running_state_machine() -> None:
    runtime = make_runtime(AdaptivePrefillConfig(enabled=True, fixed_prefill_cap=1024))
    runtime.requests = {"decode": FakeDecodeRequest()}

    state, cap, reason, decision = runtime._controller_decision(signals())

    assert state == "FIXED"
    assert cap == 1023
    assert reason == "fixed_prefill_cap;decode_reservation_limited"
    assert decision is None
