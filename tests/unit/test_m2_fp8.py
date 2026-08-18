"""Focused tests for the v5 M2 FP8 compatibility and paired-analysis risks."""

from __future__ import annotations

import json
from pathlib import Path

from vllm_tuner.longctx.m2_fp8_analysis import (
    M2FP8TrialRecord,
    M2LatencyPercentiles,
    analyze_m2_fp8_records,
)
from vllm_tuner.longctx.m2_fp8_config import M2FP8Profile
from vllm_tuner.longctx.m2_fp8_runner import _command_evidence, _quality_trace


class CharacterTokenizer:
    """Minimal exact encode/decode protocol for deterministic prompt tests."""

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    @staticmethod
    def decode(token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def _profile(profile_id: str) -> M2FP8Profile:
    values = {
        "bf16-auto": {
            "kv_cache_dtype": "auto",
            "calculate_kv_scales": False,
            "scale_source": "model-dtype",
            "expected_attention_backend": "FLASH_ATTN",
            "backend_resolution": "production-default",
        },
        "fp8-dynamic": {
            "kv_cache_dtype": "fp8",
            "calculate_kv_scales": True,
            "scale_source": "dynamic-first-forward",
            "expected_attention_backend": "FLASHINFER",
            "backend_resolution": "automatic-fp8-fallback",
        },
        "fp8-unit-fallback": {
            "kv_cache_dtype": "fp8",
            "calculate_kv_scales": False,
            "scale_source": "unit-fallback",
            "expected_attention_backend": "FLASHINFER",
            "backend_resolution": "automatic-fp8-fallback",
        },
        "fp8-e5m2": {
            "kv_cache_dtype": "fp8_e5m2",
            "calculate_kv_scales": False,
            "scale_source": "e5m2-unit-scale",
            "expected_attention_backend": "FLASHINFER",
            "backend_resolution": "automatic-fp8-fallback",
        },
    }
    return M2FP8Profile(profile_id=profile_id, **values[profile_id])


def test_profiles_keep_dynamic_and_unit_fallback_explicit() -> None:
    assert _profile("bf16-auto").vllm_args() == {}
    assert _profile("fp8-dynamic").vllm_args() == {
        "kv-cache-dtype": "fp8",
        "calculate-kv-scales": True,
    }
    assert _profile("fp8-unit-fallback").vllm_args() == {"kv-cache-dtype": "fp8"}
    assert _profile("fp8-e5m2").vllm_args() == {"kv-cache-dtype": "fp8_e5m2"}


def test_command_evidence_rejects_silent_profile_changes(tmp_path: Path) -> None:
    argv = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--kv-cache-dtype",
        "fp8_e5m2",
    ]
    (tmp_path / "server-command.json").write_text(json.dumps({"argv": argv}), encoding="utf-8")
    evidence = _command_evidence(tmp_path, _profile("fp8-e5m2"))
    assert evidence["passed"] is True
    assert evidence["attention_backend_argument"] is None


def test_quality_trace_is_fixed_exact_length_and_scored() -> None:
    context = type(
        "Context", (), {"context_id": "context-test", "total_kv_tokens": 512, "input_tokens": 480}
    )()
    trace, markers = _quality_trace(
        context=context,
        count=2,
        output_tokens=16,
        seed=2026081831,
        prompt_offset=2_000_000,
        tokenizer=CharacterTokenizer(),
    )
    replay, replay_markers = _quality_trace(
        context=context,
        count=2,
        output_tokens=16,
        seed=2026081831,
        prompt_offset=2_000_000,
        tokenizer=CharacterTokenizer(),
    )
    assert trace.checksum() == replay.checksum()
    assert markers == replay_markers
    assert all(entry.input_tokens == 480 for entry in trace.entries)
    assert all(markers[entry.request_id] in entry.prompt for entry in trace.entries)
    assert all(entry.prompt.endswith("Answer:") for entry in trace.entries)


def _record(profile_id: str, cached_tokens: int, goodput: float) -> M2FP8TrialRecord:
    fp8 = profile_id == "fp8-e5m2"
    latency = M2LatencyPercentiles(p50_ms=10.0, p95_ms=12.0, p99_ms=14.0)
    return M2FP8TrialRecord(
        trial_id=f"{profile_id}-trial",
        profile_id=profile_id,
        context_id="context-8k",
        context_tokens=8192,
        repeat_index=0,
        trace_id="a" * 64,
        status="complete",
        requested_kv_cache_dtype="fp8_e5m2" if fp8 else "auto",
        calculate_kv_scales=False,
        scale_source="e5m2-unit-scale" if fp8 else "model-dtype",
        attention_backend="FLASHINFER" if fp8 else "FLASH_ATTN",
        backend_resolution="automatic-fp8-fallback" if fp8 else "production-default",
        num_gpu_blocks=20_000 if fp8 else 10_000,
        usable_num_gpu_blocks=19_999 if fp8 else 9_999,
        block_size=16,
        cached_tokens=cached_tokens,
        quality_probe_count=2,
        quality_pass_count=2,
        quality_passed=True,
        request_count=100,
        completion_fraction=1.0,
        achieved_requests_per_second=1.0,
        goodput_requests_per_second=goodput,
        slo_satisfied_fraction=goodput,
        preemption_count=0,
        oom_count=0,
        timeout_count=0,
        peak_vram_mb=30_000.0,
        ttft=latency,
        tpot=latency,
        itl=latency,
        end_to_end=latency,
    )


def test_analysis_uses_exact_pairs_and_reports_capacity_with_negative_goodput() -> None:
    analysis = analyze_m2_fp8_records(
        [_record("bf16-auto", 200_000, 1.0), _record("fp8-e5m2", 400_000, 0.9)]
    )
    paired = analysis["paired_fp8_vs_bf16"]
    assert isinstance(paired, list)
    assert paired[0]["cached_tokens_ratio"]["median"] == 2.0
    assert paired[0]["goodput_change_percent"]["median"] < 0
    assert analysis["single_run_selection_used"] is False
    zero = analyze_m2_fp8_records(
        [_record("bf16-auto", 200_000, 0.0), _record("fp8-e5m2", 400_000, 0.1)]
    )
    assert zero["paired_fp8_vs_bf16"][0]["goodput_change_percent"]["available"] is False
