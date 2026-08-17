"""Pure tests for scheduler signal collection and JSONL instrumentation."""

import json
from dataclasses import dataclass

from vllm_tuner.scheduler.instrumentation import (
    SchedulerDecisionWriter,
    SchedulerStepRecord,
    collect_scheduler_signals,
    has_unfinished_prefill,
    split_scheduled_tokens,
)


@dataclass
class FakeRequest:
    request_id: str
    arrival_time: float
    num_prompt_tokens: int
    num_computed_tokens: int
    num_output_tokens: int


def test_collect_scheduler_signals_uses_only_request_state_and_kv_usage() -> None:
    requests = {
        "prefill-new": FakeRequest("prefill-new", 9.5, 100, 0, 0),
        "prefill-chunk": FakeRequest("prefill-chunk", 8.0, 1000, 256, 0),
        "decode": FakeRequest("decode", 7.0, 20, 22, 2),
    }

    signals = collect_scheduler_signals(
        requests,
        now=10.0,
        kv_cache_usage=0.75,
        running_requests=2,
        waiting_requests=1,
    )

    assert signals.decode_backlog == 1
    assert signals.oldest_prefill_wait_ms == 2000.0
    assert signals.kv_cache_usage == 0.75
    assert signals.running_requests == 2
    assert signals.waiting_requests == 1
    assert signals.prefill_request_ids == frozenset({"prefill-new", "prefill-chunk"})


def test_collect_scheduler_signals_clamps_clock_skew_and_kv_usage() -> None:
    request = FakeRequest("future", 20.0, 10, 0, 0)

    signals = collect_scheduler_signals(
        {request.request_id: request},
        now=10.0,
        kv_cache_usage=1.5,
        running_requests=0,
        waiting_requests=1,
    )

    assert signals.oldest_prefill_wait_ms == 0.0
    assert signals.kv_cache_usage == 1.0


def test_split_scheduled_tokens_conserves_total_budget() -> None:
    decode_tokens, prefill_tokens = split_scheduled_tokens(
        {"decode-a": 1, "prefill-a": 256, "decode-b": 2},
        frozenset({"prefill-a"}),
    )

    assert decode_tokens == 3
    assert prefill_tokens == 256
    assert decode_tokens + prefill_tokens == 259


def test_unfinished_prefill_distinguishes_a_short_final_chunk_from_starvation() -> None:
    requests = {
        "done": FakeRequest("done", 1.0, 1024, 1024, 0),
        "unfinished": FakeRequest("unfinished", 1.0, 1024, 992, 0),
    }

    assert has_unfinished_prefill(requests, frozenset({"done"})) is False
    assert has_unfinished_prefill(requests, frozenset({"done", "unfinished"})) is True


def test_scheduler_decision_writer_emits_one_complete_json_object(tmp_path) -> None:
    path = tmp_path / "scheduler-decisions.jsonl"
    writer = SchedulerDecisionWriter(path)
    record = SchedulerStepRecord(
        timestamp=1.25,
        step_index=0,
        controller_state="DISABLED",
        decode_backlog=2,
        oldest_prefill_wait_ms=10.0,
        kv_cache_usage=0.5,
        prefill_cap=2048,
        scheduled_decode_tokens=2,
        scheduled_prefill_tokens=128,
        total_scheduled_tokens=130,
        running_requests=2,
        waiting_requests=1,
        preemption_delta=0,
        scheduler_cpu_time_us=42.0,
        reason_code="controller_disabled",
    )

    writer.write(record)
    writer.close()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["controller_state"] == "DISABLED"
    assert payload["total_scheduled_tokens"] == 130
    assert payload["reason_code"] == "controller_disabled"
