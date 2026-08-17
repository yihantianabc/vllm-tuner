"""Tests for one-shot vLLM M1 capacity evidence extraction."""

from __future__ import annotations

import pytest

from vllm_tuner.longctx.capacity_evidence import (
    build_capacity_runtime_evidence,
    parse_cache_config_info,
)


def _metrics(*, blocks: int = 14618, utilization: str = "0.9", enabled: str = "True") -> str:
    return (
        "# TYPE vllm:cache_config_info gauge\n"
        "vllm:cache_config_info{"
        'block_size="16",cache_dtype="auto",calculate_kv_scales="False",'
        f'gpu_memory_utilization="{utilization}",num_gpu_blocks="{blocks}",'
        'kv_cache_memory_bytes="None",num_gpu_blocks_override="None",'
        f'enable_prefix_caching="{enabled}",is_attention_free="False",'
        'sliding_window="None",engine="0"'
        "} 1.0\n"
    )


def _server_log(*, tokens: int = 233888) -> str:
    return (
        "Resolved architecture: Qwen2ForCausalLM\n"
        "Using max model len 32768\n"
        "Using FLASH_ATTN attention backend\n"
        "config: dtype=torch.bfloat16, kv_cache_dtype=auto, "
        "enable_prefix_caching=True, enable_chunked_prefill=True\n"
        "Available KV cache memory: 12.49 GiB\n"
        f"GPU KV cache size: {tokens:,} tokens\n"
        "Maximum concurrency for 32,768 tokens per request: 7.14x\n"
    )


def test_parse_cache_config_info_preserves_exact_blocks_and_raw_line() -> None:
    evidence = parse_cache_config_info(_metrics())

    assert evidence.resolved_block_size == 16
    assert evidence.num_gpu_blocks == 14618
    assert evidence.usable_num_gpu_blocks == 14617
    assert evidence.requested_cache_dtype == "auto"
    assert evidence.gpu_memory_utilization_ppm == 900_000
    assert evidence.kv_cache_memory_bytes is None
    assert evidence.num_gpu_blocks_override is None
    assert evidence.enable_prefix_caching is True
    assert evidence.calculate_kv_scales is False
    assert evidence.is_attention_free is False
    assert evidence.sliding_window is None
    assert evidence.raw_metric_line.startswith("vllm:cache_config_info{")


def test_build_capacity_runtime_evidence_cross_checks_log_and_info_metric() -> None:
    evidence = build_capacity_runtime_evidence(
        run_id="m1-init",
        runtime_profile_sha256="profile",
        server_log_text=_server_log(),
        metrics_text=_metrics(),
        total_memory_bytes=32607 * (1 << 20),
        initial_free_memory_bytes=32607 * (1 << 20),
    )

    assert evidence.logged_capacity_consistent is True
    assert evidence.observation.num_blocks_source == "cache_config_info"
    assert evidence.startup_format.model_dtype == "torch.bfloat16"
    assert evidence.startup_format.attention_backend == "FLASH_ATTN"
    assert evidence.observation.cached_tokens == 233888
    assert evidence.observation.usable_num_gpu_blocks == 14617
    assert len(evidence.raw_capacity_log_lines) == 3


def test_capacity_evidence_rejects_metric_log_block_mismatch() -> None:
    with pytest.raises(ValueError, match="disagree"):
        build_capacity_runtime_evidence(
            run_id="bad",
            runtime_profile_sha256="profile",
            server_log_text=_server_log(tokens=233872),
            metrics_text=_metrics(blocks=14618),
            total_memory_bytes=32607 * (1 << 20),
            initial_free_memory_bytes=32607 * (1 << 20),
        )


def test_cache_config_info_rejects_missing_duplicate_or_invalid_labels() -> None:
    with pytest.raises(ValueError, match="no vLLM cache_config_info"):
        parse_cache_config_info("vllm:other 1\n")

    duplicate = _metrics() + _metrics()
    with pytest.raises(ValueError, match="exactly one"):
        parse_cache_config_info(duplicate)

    with pytest.raises(ValueError, match="represented exactly in ppm"):
        parse_cache_config_info(_metrics(utilization="0.9000001"))

    with pytest.raises(ValueError, match="not a boolean"):
        parse_cache_config_info(_metrics(enabled="yes"))
