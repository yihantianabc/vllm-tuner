"""Focused tests for the v5 M3 APC workload and hit-evidence risks."""

from __future__ import annotations

import json
from pathlib import Path

from vllm_tuner.benchmarks.models import RequestResult, RequestStatus
from vllm_tuner.longctx.m3_apc_config import M3APCProfile
from vllm_tuner.longctx.m3_apc_runner import (
    _command_evidence,
    _request_cached_tokens,
    _validate_counters,
)
from vllm_tuner.longctx.m3_apc_workload import (
    RAGCorpus,
    build_m3_boundary_trace,
    build_m3_core_trace,
    build_m3_core_warmup,
    expected_core_cached_tokens,
)


class CharacterTokenizer:
    """Minimal exact tokenizer protocol for deterministic prefix tests."""

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    @staticmethod
    def decode(token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def _corpus() -> RAGCorpus:
    text = (
        "Measured inference evidence records request timing, service objectives, cache geometry, "
        "and cleanup status. Failed trials remain visible and conclusions distinguish observation "
        "from inference. "
    ) * 20
    return RAGCorpus(documents=(("public-runbook.md", text),), sha256="a" * 64)


def test_apc_profiles_are_explicit_and_do_not_touch_fp8_or_m4() -> None:
    off = M3APCProfile(profile_id="apc-off", enable_prefix_caching=False)
    on = M3APCProfile(profile_id="apc-on", enable_prefix_caching=True)
    assert off.vllm_args() == {
        "enable-prompt-tokens-details": True,
        "no-enable-prefix-caching": True,
    }
    assert on.vllm_args() == {
        "enable-prompt-tokens-details": True,
        "enable-prefix-caching": True,
    }


def test_real_rag_trace_has_exact_reuse_and_cold_warm_hits() -> None:
    tokenizer = CharacterTokenizer()
    bundle = build_m3_core_trace(
        prefix_tokens=32,
        requests_per_reuse=4,
        input_tokens=96,
        output_tokens=8,
        offered_requests_per_second=1.0,
        burstiness=1.0,
        seed=41,
        tokenizer=tokenizer,
        corpus=_corpus(),
    )
    assert len(bundle.trace.entries) == 12
    assert all(entry.input_tokens == 96 for entry in bundle.trace.entries)
    assert list(bundle.reuse_by_request.values()).count(0) == 4
    assert list(bundle.reuse_by_request.values()).count(50) == 4
    assert list(bundle.reuse_by_request.values()).count(100) == 4
    assert (
        sum(
            bundle.shared_request[request_id]
            for request_id, reuse in bundle.reuse_by_request.items()
            if reuse == 50
        )
        == 2
    )
    assert bundle.prefix_proof["block_aligned_reuse_proved"] is True

    off = expected_core_cached_tokens(
        bundle,
        prefix_tokens=32,
        apc_enabled=False,
        cache_state="target-prefix-cold",
    )
    cold = expected_core_cached_tokens(
        bundle,
        prefix_tokens=32,
        apc_enabled=True,
        cache_state="target-prefix-cold",
    )
    warm = expected_core_cached_tokens(
        bundle,
        prefix_tokens=32,
        apc_enabled=True,
        cache_state="target-prefix-warm",
    )
    assert sum(off.values()) == 0
    assert sum(cold.values()) == 128
    assert sum(warm.values()) == 192
    assert all(
        warm[request_id] == 0 for request_id, reuse in bundle.reuse_by_request.items() if reuse == 0
    )

    cold_warmup = build_m3_core_warmup(
        bundle=bundle,
        cache_state="target-prefix-cold",
        prefix_tokens=32,
        input_tokens=96,
        output_tokens=8,
        seed=42,
        tokenizer=tokenizer,
        corpus=_corpus(),
    )
    warm_warmup = build_m3_core_warmup(
        bundle=bundle,
        cache_state="target-prefix-warm",
        prefix_tokens=32,
        input_tokens=96,
        output_tokens=8,
        seed=42,
        tokenizer=tokenizer,
        corpus=_corpus(),
    )
    measured_ids = {
        entry.shared_prefix_id for entry in bundle.trace.entries if entry.shared_prefix_id
    }
    assert measured_ids.isdisjoint({entry.shared_prefix_id for entry in cold_warmup.entries})
    assert {entry.shared_prefix_id for entry in warm_warmup.entries} == measured_ids


def test_boundary_trace_primes_each_real_prefix_and_probes_in_reverse() -> None:
    bundle = build_m3_boundary_trace(
        pool_size=4,
        prefix_tokens=32,
        tail_tokens=16,
        output_tokens=8,
        interval_seconds=1.0,
        seed=43,
        tokenizer=CharacterTokenizer(),
        corpus=_corpus(),
    )
    assert len(bundle.warmup.entries) == len(bundle.measured.entries) == 4
    assert [entry.shared_prefix_id for entry in bundle.measured.entries] == list(
        reversed([entry.shared_prefix_id for entry in bundle.warmup.entries])
    )
    assert bundle.prefix_proof["block_aligned_reuse_proved"] is True


def test_command_and_usage_evidence_fail_closed(tmp_path: Path) -> None:
    profile = M3APCProfile(profile_id="apc-on", enable_prefix_caching=True)
    argv = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
    ]
    (tmp_path / "server-command.json").write_text(json.dumps({"argv": argv}), encoding="utf-8")
    assert _command_evidence(tmp_path, profile)["passed"] is True
    row = {
        "input_tokens": 96,
        "metadata": {"usage": {"prompt_tokens_details": {"cached_tokens": 32}}},
    }
    assert _request_cached_tokens(row) == 32
    assert _request_cached_tokens({"input_tokens": 96, "metadata": {"usage": {}}}) == 0

    request = RequestResult(
        request_id="off-query",
        input_tokens=96,
        output_tokens=8,
        status=RequestStatus.SUCCESS,
    )
    counters = {
        name: {"available": True, "reset_count": 0, "delta": value}
        for name, value in {
            "prompt_tokens_total": 96,
            "generation_tokens_total": 8,
            "prefix_cache_queries": 0,
            "prefix_cache_hits": 0,
            "num_preemptions_total": 0,
        }.items()
    }
    queries, hits, preemptions = _validate_counters(
        counters=counters,
        requests=[request],
        cached_tokens=[0],
        apc_enabled=False,
    )
    assert (queries, hits, preemptions) == (0, 0, 0)
