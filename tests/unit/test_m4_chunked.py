"""Focused tests for v5 M4 native-profile, trace, and paired-selection risks."""

from __future__ import annotations

from vllm_tuner.longctx.m4_chunked_analysis import (
    M4LatencyPercentiles,
    M4PrefillWindow,
    M4ResourceUsage,
    M4TrialRecord,
    M4WaitingUsage,
    analyze_m4_records,
)
from vllm_tuner.longctx.m4_chunked_config import M4ChunkedProfile, M4Protocol
from vllm_tuner.longctx.m4_chunked_workload import build_m4_trace


class CharacterTokenizer:
    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    @staticmethod
    def decode(token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def _protocol() -> M4Protocol:
    return M4Protocol(
        repeats=1,
        long_prefill_tokens=(4096,),
        decode_requests=12,
        decode_input_tokens=256,
        decode_output_tokens=256,
        decode_interval_seconds=1.5,
        injection_offsets_seconds=(6.75,),
        long_output_tokens=32,
        measurement_seed=51,
        warmup_seed=52,
        warmup_requests=2,
        client_max_concurrency=16,
        request_timeout_seconds=600.0,
        burstiness=1.0,
        ignore_eos=True,
    )


def test_profiles_are_minimal_native_chunked_prefill_only() -> None:
    default = M4ChunkedProfile(profile_id="production-default")
    one_k = M4ChunkedProfile(
        profile_id="native-threshold-1024",
        long_prefill_token_threshold=1024,
    )
    half_k = M4ChunkedProfile(
        profile_id="native-threshold-512",
        long_prefill_token_threshold=512,
    )
    assert default.vllm_args() == {}
    assert one_k.vllm_args() == {
        "enable-chunked-prefill": True,
        "long-prefill-token-threshold": 1024,
    }
    assert half_k.vllm_args()["long-prefill-token-threshold"] == 512
    assert all("scheduler" not in name and "kv-cache" not in name for name in one_k.vllm_args())


def test_trace_establishes_decode_before_long_prefill_and_isolates_apc() -> None:
    bundle = build_m4_trace(
        protocol=_protocol(),
        long_prefill_tokens=4096,
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
    assert len(long_prefill) == 1
    assert long_prefill[0].scheduled_offset_seconds == 6.75
    assert sum(entry.scheduled_offset_seconds < 6.75 for entry in decode) >= 4
    assert all(entry.input_tokens == 256 and entry.output_tokens == 256 for entry in decode)
    assert long_prefill[0].input_tokens == 4096
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
    profile: str,
    long_tokens: int,
    repeat: int,
    interference: float,
    goodput: float,
) -> M4TrialRecord:
    budget = 2048
    partial = 1
    threshold = {
        "production-default": 0,
        "native-threshold-1024": 1024,
        "native-threshold-512": 512,
    }[profile]
    usage = M4ResourceUsage(sample_count=2, minimum=0.0, median=0.1, p95=0.2, maximum=0.2)
    return M4TrialRecord(
        trial_id=f"{profile}-{long_tokens}-{repeat}",
        profile_id=profile,
        production_default=profile == "production-default",
        max_num_batched_tokens=budget,
        max_num_partial_prefills=partial,
        max_long_partial_prefills=1,
        long_prefill_token_threshold=threshold,
        long_prefill_tokens=long_tokens,
        repeat_index=repeat,
        trace_id=f"trace-{long_tokens}",
        warmup_trace_id=f"warmup-{long_tokens}",
        request_count=51,
        decode_request_count=48,
        long_prefill_request_count=3,
        completion_fraction=1.0,
        decode_slo_satisfied_fraction=1.0,
        decode_goodput_requests_per_second=goodput,
        overall_goodput_requests_per_second=goodput,
        decode_ttft=_latency(10.0),
        decode_tpot=_latency(20.0),
        decode_itl=_latency(20.0),
        decode_end_to_end=_latency(1000.0),
        long_prefill_ttft=_latency(500.0),
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
        waiting=M4WaitingUsage(**usage.model_dump(mode="json"), positive_sample_fraction=0.5),
        kv_usage=usage,
        preemption_count=0,
        prefix_cache_queries=1024,
        prefix_cache_hits=0,
        peak_vram_mb=20000.0,
        oom_count=0,
        timeout_count=0,
        mechanism_evidence_passed=True,
    )


def test_formal_selection_uses_majorities_not_best_single_run() -> None:
    records: list[M4TrialRecord] = []
    for long_tokens in (4096, 8192):
        for repeat in range(3):
            records.append(_record("production-default", long_tokens, repeat, 100.0, 0.60))
            records.append(_record("native-threshold-1024", long_tokens, repeat, 80.0, 0.60))
            # One excellent run cannot rescue two regressions.
            interference = 50.0 if repeat == 0 else 120.0
            goodput = 0.61 if repeat == 0 else 0.59
            records.append(
                _record("native-threshold-512", long_tokens, repeat, interference, goodput)
            )
    analysis = analyze_m4_records(records, formal=True)
    selection = analysis["selection"]
    assert selection["profile_id"] == "native-threshold-1024"
    assert selection["single_run_selection_used"] is False
    assert selection["candidate_eligibility"]["native-threshold-512"] is False
