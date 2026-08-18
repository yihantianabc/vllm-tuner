"""Focused tests for the frozen v5 M5 profiles, traces, and acceptance rules."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vllm_tuner.longctx.m4_chunked_analysis import (
    M4LatencyPercentiles,
    M4PrefillWindow,
    M4ResourceUsage,
    M4WaitingUsage,
)
from vllm_tuner.longctx.m5_decode_tail_analysis import M5TrialRecord, analyze_m5_records
from vllm_tuner.longctx.m5_decode_tail_config import (
    M5Cohort,
    M5DecodeTailProfile,
    M5Protocol,
)
from vllm_tuner.longctx.m5_decode_tail_workload import build_m5_trace


class CharacterTokenizer:
    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    @staticmethod
    def decode(token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def _smoke_protocol() -> M5Protocol:
    return M5Protocol(
        repeats=1,
        decode_requests=12,
        decode_input_tokens=256,
        decode_output_tokens=256,
        decode_interval_seconds=1.5,
        decode_arrival_jitter_fraction=0.1,
        long_output_tokens=32,
        warmup_requests=3,
        client_max_concurrency=16,
        request_timeout_seconds=600.0,
        ignore_eos=True,
        cohorts=(
            M5Cohort(
                cohort_id="target",
                prompt_seed=51,
                arrival_seed=53,
                warmup_seed=52,
                long_prefill_tokens=(4096, 8192),
                injection_offsets_seconds=(6.75, 11.25),
            ),
        ),
    )


def test_profiles_are_exactly_default_and_native_threshold_1024() -> None:
    default = M5DecodeTailProfile(profile_id="production-default")
    candidate = M5DecodeTailProfile(
        profile_id="decode-tail-1024", long_prefill_token_threshold=1024
    )
    assert default.vllm_args() == {}
    assert candidate.vllm_args() == {
        "enable-chunked-prefill": True,
        "long-prefill-token-threshold": 1024,
    }
    assert all(
        forbidden not in candidate.vllm_args()
        for forbidden in ("scheduler-cls", "kv-cache-dtype", "enable-prefix-caching")
    )
    with pytest.raises(ValidationError):
        M5DecodeTailProfile(profile_id="native-threshold-512", long_prefill_token_threshold=512)


def test_trace_mixes_4k_8k_uses_seeded_arrivals_and_isolates_apc() -> None:
    protocol = _smoke_protocol()
    cohort = protocol.cohorts[0]
    bundle = build_m5_trace(
        protocol=protocol,
        cohort=cohort,
        tokenizer=CharacterTokenizer(),
    )
    decode = [
        entry
        for entry in bundle.measured.entries
        if bundle.request_kind[entry.request_id] == "decode"
    ]
    long_prefill = [
        entry
        for entry in bundle.measured.entries
        if bundle.request_kind[entry.request_id] == "long-prefill"
    ]
    assert len(decode) == 12
    assert [entry.input_tokens for entry in long_prefill] == [4096, 8192]
    assert [entry.scheduled_offset_seconds for entry in long_prefill] == [6.75, 11.25]
    assert decode[1].scheduled_offset_seconds != 1.5
    assert bundle.prefix_isolation_proof["no_cacheable_shared_prefix"] is True
    assert bundle.prefix_isolation_proof["maximum_pairwise_lcp_tokens"] < 16


def _latency(value: float) -> M4LatencyPercentiles:
    return M4LatencyPercentiles(
        sample_count=8,
        p50_ms=value,
        p95_ms=value,
        p99_ms=value,
        maximum_ms=value,
    )


def _record(
    cohort: str,
    profile: str,
    repeat: int,
    *,
    interference: float,
    goodput: float,
) -> M5TrialRecord:
    usage = M4ResourceUsage(sample_count=2, minimum=0.0, median=0.1, p95=0.2, maximum=0.2)
    waiting = M4WaitingUsage(
        sample_count=2,
        minimum=0.0,
        median=0.0,
        p95=0.0,
        maximum=0.0,
        positive_sample_fraction=0.0,
    )
    candidate = profile == "decode-tail-1024"
    return M5TrialRecord(
        trial_id=f"{cohort}-{profile}-{repeat}",
        cohort_id=cohort,
        profile_id=profile,
        production_default=not candidate,
        max_num_batched_tokens=2048,
        max_num_partial_prefills=1,
        max_long_partial_prefills=1,
        long_prefill_token_threshold=1024 if candidate else 0,
        long_prefill_tokens=(4096, 8192, 4096),
        injection_offsets_seconds=(17.25, 35.25, 53.25),
        prompt_seed=51 if cohort == "target" else 151,
        arrival_seed=53 if cohort == "target" else 153,
        repeat_index=repeat,
        trace_id=f"trace-{cohort}",
        warmup_trace_id=f"warmup-{cohort}",
        request_count=51,
        decode_request_count=48,
        long_prefill_request_count=3,
        completion_fraction=1.0,
        decode_slo_satisfied_fraction=1.0,
        decode_goodput_requests_per_second=goodput,
        overall_goodput_requests_per_second=goodput,
        decode_ttft=_latency(10.0),
        decode_tpot=_latency(101.0 if candidate else 100.0),
        decode_itl=_latency(20.0),
        decode_end_to_end=_latency(1000.0),
        long_prefill_ttft=_latency(110.0 if candidate else 100.0),
        long_prefill_tpot=_latency(20.0),
        long_prefill_end_to_end=_latency(1000.0),
        decode_interference_itl=_latency(interference),
        decode_non_interference_itl=_latency(20.0),
        decode_overlap_request_count=4,
        prefill_windows=(
            M4PrefillWindow(
                request_id="long",
                sent_at_ns=1,
                first_token_at_ns=2,
                duration_ms=0.000001,
            ),
        ),
        waiting=waiting,
        kv_usage=usage,
        preemption_count=0,
        prefix_cache_queries=1024,
        prefix_cache_hits=0,
        peak_vram_mb=20_000.0,
        oom_count=0,
        timeout_count=0,
        mechanism_evidence_passed=True,
    )


def _formal_records() -> list[M5TrialRecord]:
    records: list[M5TrialRecord] = []
    for cohort in ("target", "held-out"):
        for repeat in range(3):
            records.append(
                _record(cohort, "production-default", repeat, interference=100.0, goodput=1.0)
            )
            records.append(
                _record(
                    cohort,
                    "decode-tail-1024",
                    repeat,
                    interference=70.0,
                    goodput=0.997,
                )
            )
    return records


def test_formal_acceptance_requires_both_cohorts_and_each_repeat_goodput() -> None:
    records = _formal_records()
    positive = analyze_m5_records(records, formal=True)
    assert positive["acceptance"]["passed"] is True
    assert positive["decision"]["profile_id"] == "decode-tail-1024"

    for index, record in enumerate(records):
        if (
            record.cohort_id == "held-out"
            and record.profile_id == "decode-tail-1024"
            and record.repeat_index == 0
        ):
            records[index] = record.model_copy(update={"decode_goodput_requests_per_second": 0.985})
            break
    negative = analyze_m5_records(records, formal=True)
    assert negative["acceptance"]["passed"] is False
    assert negative["decision"]["profile_id"] == "production-default"
    assert negative["decision"]["positive_result"] is False
