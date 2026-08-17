"""Unit tests for the strict long-context KV Capacity Planner."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from vllm_tuner.longctx.kv_capacity_planner import (
    PPM,
    ByteEstimate,
    CacheLayoutSpec,
    ContextBin,
    ContextDistributionSpec,
    DeviceBudgetSpec,
    KVDType,
    KVCapacityPlannerInput,
    NonKVPredictionSpec,
    SafetyPolicy,
    ServingLimits,
    UniformFullAttentionModelSpec,
    VLLMInitializationObservation,
    calibrate_non_kv_from_runs,
    model_spec_from_hf_config,
    parse_vllm_initialization_observation,
    plan_kv_capacity,
    validate_kv_capacity_plan,
)

TOTAL_GPU_BYTES = 32607 * (1 << 20)
MODEL_WEIGHT_BYTES = 15_231_271_888
M0_NON_KV_UPPER_BYTES = 17_359_752_397
M0_RUNTIME_RESIDUAL_BYTES = M0_NON_KV_UPPER_BYTES - MODEL_WEIGHT_BYTES


def _qwen_model() -> UniformFullAttentionModelSpec:
    return UniformFullAttentionModelSpec(
        architecture="Qwen2ForCausalLM",
        attention_mode="uniform_full_attention",
        num_hidden_layers=28,
        num_attention_layers=28,
        hidden_size=3584,
        num_attention_heads=28,
        num_kv_heads=4,
        head_dim=128,
        max_position_embeddings=32768,
        model_dtype=KVDType.BFLOAT16,
    )


def _bf16_cache() -> CacheLayoutSpec:
    return CacheLayoutSpec(
        requested_dtype="auto",
        resolved_dtype=KVDType.BFLOAT16,
        block_size=16,
        reserved_null_blocks=1,
        inline_metadata_bytes_per_layer_block=0,
        page_padding_bytes_per_layer_block=0,
        format_evidence="vllm_full_attention_spec",
    )


def _estimate(
    point: int,
    upper: int | None = None,
    *,
    source: str = "structural",
    run_ids: tuple[str, ...] = (),
) -> ByteEstimate:
    return ByteEstimate(
        point_bytes=point,
        upper_bytes=point if upper is None else upper,
        source=source,
        calibration_run_ids=run_ids,
    )


def _non_kv(total_bytes: int = M0_NON_KV_UPPER_BYTES) -> NonKVPredictionSpec:
    residual = total_bytes - MODEL_WEIGHT_BYTES
    assert residual >= 0
    return NonKVPredictionSpec(
        weights=_estimate(MODEL_WEIGHT_BYTES),
        peak_activations=_estimate(0),
        runtime_non_torch=_estimate(0),
        post_profile_cuda_graph=_estimate(0),
        post_profile_persistent=_estimate(0),
        unattributed_runtime_residual=_estimate(
            residual,
            source="multi_run_calibration",
            run_ids=("calibration-0", "calibration-1"),
        ),
    )


def _distribution(*contexts: tuple[str, int, int, int]) -> ContextDistributionSpec:
    return ContextDistributionSpec(
        bins=tuple(
            ContextBin(
                name=name,
                weight_ppm=weight,
                prompt_tokens=prompt_tokens,
                reserved_output_tokens=output_tokens,
            )
            for name, weight, prompt_tokens, output_tokens in contexts
        ),
        confidence_ppm=990_000,
        iid_assumption=True,
        assume_no_prefix_reuse=True,
    )


def _input(
    *,
    model: UniformFullAttentionModelSpec | None = None,
    cache: CacheLayoutSpec | None = None,
    non_kv: NonKVPredictionSpec | None = None,
    safety_basis_points: int = 0,
    fixed_reserve: int = 0,
    residual_reserve: int = 0,
    max_model_len: int = 32768,
    max_num_seqs: int = 512,
    distribution: ContextDistributionSpec | None = None,
) -> KVCapacityPlannerInput:
    return KVCapacityPlannerInput(
        schema_version="longctx-m1.v1",
        model=model or _qwen_model(),
        cache=cache or _bf16_cache(),
        device=DeviceBudgetSpec(
            total_memory_bytes=TOTAL_GPU_BYTES,
            initial_free_memory_bytes=TOTAL_GPU_BYTES,
            gpu_memory_utilization_ppm=900_000,
        ),
        non_kv=non_kv or _non_kv(),
        safety=SafetyPolicy(
            fixed_operational_reserve_bytes=fixed_reserve,
            kv_reserve_basis_points=safety_basis_points,
            calibration_residual_upper_bytes=residual_reserve,
            source="test policy",
        ),
        serving=ServingLimits(
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        distribution=distribution or _distribution(("full", PPM, max_model_len - 128, 128)),
    )


def _observation(
    *,
    run_id: str,
    runtime_profile_sha256: str,
    utilization_ppm: int,
    non_kv_bytes: int,
    max_model_len: int = 32768,
) -> VLLMInitializationObservation:
    block_bytes = 917_504
    requested = (TOTAL_GPU_BYTES * utilization_ppm + PPM - 1) // PPM
    blocks = (requested - non_kv_bytes) // block_bytes
    concurrency = blocks / math.ceil(max_model_len / 16)
    return VLLMInitializationObservation(
        run_id=run_id,
        runtime_profile_sha256=runtime_profile_sha256,
        server_log_sha256=hashlib.sha256(run_id.encode()).hexdigest(),
        total_memory_bytes=TOTAL_GPU_BYTES,
        initial_free_memory_bytes=TOTAL_GPU_BYTES,
        gpu_memory_utilization_ppm=utilization_ppm,
        resolved_block_size=16,
        num_gpu_blocks=blocks,
        usable_num_gpu_blocks=blocks - 1,
        num_blocks_source="cache_config_info",
        cached_tokens=blocks * 16,
        max_model_len=max_model_len,
        reported_max_concurrency=round(concurrency, 2),
        reported_available_gib_text=f"{blocks * block_bytes / (1 << 30):.2f}",
    )


def test_qwen25_7b_bf16_golden_geometry_and_m0_capacity() -> None:
    plan = plan_kv_capacity(_input())

    assert plan.geometry.gqa_ratio == 7
    assert plan.geometry.payload_bytes_per_token_per_layer == 2048
    assert plan.geometry.payload_bytes_per_token_total == 57_344
    assert plan.geometry.payload_page_bytes_per_layer == 32_768
    assert plan.geometry.block_bytes_total == 917_504
    assert plan.capacity.raw_num_blocks == 14_618
    assert plan.capacity.raw_cached_tokens == 233_888
    assert plan.capacity.raw_usable_num_blocks == 14_617
    assert plan.capacity.raw_usable_cached_tokens == 233_872
    assert plan.capacity.raw_allocated_bytes == 13_412_073_472
    assert plan.contexts[0].raw_concurrency_ratio == pytest.approx(14_618 / 2048)
    assert plan.contexts[0].safe_integer_concurrency == 7


def test_fp8_payload_is_half_bf16_and_scales_are_non_kv() -> None:
    fp8_cache = CacheLayoutSpec(
        requested_dtype="fp8_e4m3",
        resolved_dtype=KVDType.FP8_E4M3,
        block_size=16,
        reserved_null_blocks=1,
        inline_metadata_bytes_per_layer_block=0,
        page_padding_bytes_per_layer_block=0,
        format_evidence="vllm_full_attention_spec",
    )
    non_kv = _non_kv(M0_NON_KV_UPPER_BYTES + 224)
    plan = plan_kv_capacity(_input(cache=fp8_cache, non_kv=non_kv))

    assert plan.geometry.payload_bytes_per_token_per_layer == 1024
    assert plan.geometry.payload_bytes_per_token_total == 28_672
    assert plan.geometry.block_bytes_total == 458_752
    assert plan.geometry.inline_metadata_bytes_per_block_total == 0
    assert plan.memory.non_kv_point_bytes == M0_NON_KV_UPPER_BYTES + 224


def test_mha_payload_is_seven_times_qwen_gqa_payload() -> None:
    mha = _qwen_model().model_copy(update={"num_kv_heads": 28})
    gqa_plan = plan_kv_capacity(_input())
    mha_plan = plan_kv_capacity(_input(model=mha))

    assert mha_plan.geometry.payload_bytes_per_token_total == (
        7 * gqa_plan.geometry.payload_bytes_per_token_total
    )


def test_metadata_padding_and_tail_rounding_are_separate() -> None:
    cache = CacheLayoutSpec(
        requested_dtype="auto",
        resolved_dtype=KVDType.BFLOAT16,
        block_size=16,
        reserved_null_blocks=1,
        inline_metadata_bytes_per_layer_block=8,
        page_padding_bytes_per_layer_block=24,
        format_evidence="measured_backend_spec",
    )
    distribution = _distribution(("seventeen", PPM, 16, 1))
    plan = plan_kv_capacity(_input(cache=cache, distribution=distribution, max_model_len=32))
    context = plan.contexts[0]

    assert plan.geometry.page_bytes_per_layer == 32_800
    assert context.blocks_per_sequence == 2
    assert context.tail_slots == 15
    assert context.tail_slot_waste_bytes == 15 * 57_344
    assert context.format_overhead_bytes == 2 * 28 * 32


def test_safety_uses_upper_non_kv_and_explicit_policy_without_double_counting() -> None:
    non_kv = _non_kv(M0_NON_KV_UPPER_BYTES)
    plan = plan_kv_capacity(
        _input(
            non_kv=non_kv,
            safety_basis_points=500,
            fixed_reserve=1_000_000,
            residual_reserve=2_000_000,
        )
    )

    expected_proportional = math.ceil(plan.memory.raw_available_kv_bytes * 500 / 10_000)
    assert plan.memory.proportional_kv_reserve_bytes == expected_proportional
    assert plan.memory.total_policy_reserve_bytes == expected_proportional + 3_000_000
    assert plan.capacity.safe_num_blocks < plan.capacity.raw_num_blocks


def test_distribution_reports_worst_expected_and_iid_quantile_separately() -> None:
    distribution = _distribution(
        ("short", 800_000, 112, 16),
        ("long", 200_000, 2032, 16),
    )
    plan = plan_kv_capacity(
        _input(
            distribution=distribution,
            max_model_len=2048,
            max_num_seqs=64,
            safety_basis_points=500,
        )
    )

    result = plan.distribution
    assert result.guaranteed_worst_case_concurrency <= result.iid_quantile_concurrency
    assert result.iid_quantile_concurrency <= result.expected_only_concurrency
    assert result.expected_only_concurrency <= 64
    assert result.iid_quantile_overflow_probability <= 0.01 + 1e-12


@pytest.mark.parametrize(
    "updates,error",
    [
        ({"num_kv_heads": 3}, "divisible"),
        ({"hidden_size": 3585}, "hidden_size must equal"),
        ({"num_attention_layers": 27}, "attention in every"),
    ],
)
def test_invalid_model_geometry_is_rejected(updates: dict[str, int], error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        _qwen_model().model_copy(update=updates).model_validate(
            {**_qwen_model().model_dump(), **updates}
        )


def test_auto_dtype_must_resolve_to_model_dtype() -> None:
    cache = _bf16_cache().model_copy(update={"resolved_dtype": KVDType.FLOAT16})
    with pytest.raises(ValidationError, match="auto KV dtype"):
        _input(cache=cache)


def test_device_initial_free_memory_must_cover_requested_budget() -> None:
    with pytest.raises(ValidationError, match="below the requested"):
        DeviceBudgetSpec(
            total_memory_bytes=1000,
            initial_free_memory_bytes=899,
            gpu_memory_utilization_ppm=900_000,
        )


def test_distribution_requires_exact_ppm_and_context_limit() -> None:
    with pytest.raises(ValidationError, match="sum to 1,000,000"):
        ContextDistributionSpec(
            bins=(
                ContextBin(
                    name="bad",
                    weight_ppm=999_999,
                    prompt_tokens=1,
                    reserved_output_tokens=1,
                ),
            ),
            confidence_ppm=990_000,
            iid_assumption=True,
            assume_no_prefix_reuse=True,
        )
    too_long = _distribution(("too-long", PPM, 32768, 1))
    with pytest.raises(ValidationError, match="exceed max_model_len"):
        _input(distribution=too_long)


def test_strict_models_reject_string_integers_and_bool_counts() -> None:
    with pytest.raises(ValidationError):
        ContextBin(
            name="strict",
            weight_ppm="1000000",
            prompt_tokens=1,
            reserved_output_tokens=1,
        )
    with pytest.raises(ValidationError):
        ContextBin(name="strict", weight_ppm=PPM, prompt_tokens=True, reserved_output_tokens=1)


def test_parse_m0_vllm_initialization_log() -> None:
    log = (
        "Available KV cache memory: 12.49 GiB\n"
        "GPU KV cache size: 233,888 tokens\n"
        "Maximum concurrency for 32,768 tokens per request: 7.14x\n"
    )
    observation = parse_vllm_initialization_observation(
        run_id="m0",
        runtime_profile_sha256="fingerprint",
        server_log_text=log,
        total_memory_bytes=TOTAL_GPU_BYTES,
        initial_free_memory_bytes=TOTAL_GPU_BYTES,
        gpu_memory_utilization_ppm=900_000,
    )

    assert observation.cached_tokens == 233_888
    assert observation.max_model_len == 32768
    assert observation.reported_max_concurrency == 7.14
    assert observation.reported_available_gib_text == "12.49"


def test_multi_run_calibration_intersects_block_intervals() -> None:
    observations = (
        _observation(
            run_id="util-90",
            runtime_profile_sha256="frozen-profile",
            utilization_ppm=900_000,
            non_kv_bytes=M0_NON_KV_UPPER_BYTES - 100_000,
        ),
        _observation(
            run_id="util-85",
            runtime_profile_sha256="frozen-profile",
            utilization_ppm=850_000,
            non_kv_bytes=M0_NON_KV_UPPER_BYTES - 100_000,
        ),
    )
    calibration = calibrate_non_kv_from_runs(
        observations=observations,
        model=_qwen_model(),
        cache=_bf16_cache(),
    )

    assert calibration.intersection_lower_exclusive_bytes < M0_NON_KV_UPPER_BYTES
    assert calibration.intersection_upper_inclusive_bytes >= M0_NON_KV_UPPER_BYTES - 100_000
    assert calibration.as_estimate().source == "multi_run_calibration"
    assert calibration.as_estimate().calibration_run_ids == ("util-90", "util-85")


def test_multi_run_calibration_rejects_duplicate_or_nonoverlapping_runs() -> None:
    first = _observation(
        run_id="same",
        runtime_profile_sha256="frozen",
        utilization_ppm=900_000,
        non_kv_bytes=M0_NON_KV_UPPER_BYTES,
    )
    with pytest.raises(ValueError, match="unique"):
        calibrate_non_kv_from_runs(
            observations=(first, first),
            model=_qwen_model(),
            cache=_bf16_cache(),
        )

    far = _observation(
        run_id="far",
        runtime_profile_sha256="frozen",
        utilization_ppm=900_000,
        non_kv_bytes=M0_NON_KV_UPPER_BYTES + 10_000_000,
    )
    with pytest.raises(ValueError, match="do not overlap"):
        calibrate_non_kv_from_runs(
            observations=(first, far),
            model=_qwen_model(),
            cache=_bf16_cache(),
        )


def test_m0_capacity_validation_is_exact_and_within_target() -> None:
    inputs = _input()
    plan = plan_kv_capacity(inputs)
    log = (
        "Available KV cache memory: 12.49 GiB\n"
        "GPU KV cache size: 233,888 tokens\n"
        "Maximum concurrency for 32,768 tokens per request: 7.14x\n"
    )
    observation = parse_vllm_initialization_observation(
        run_id="m0-formal",
        runtime_profile_sha256=plan.runtime_profile_sha256,
        server_log_text=log,
        total_memory_bytes=TOTAL_GPU_BYTES,
        initial_free_memory_bytes=TOTAL_GPU_BYTES,
        gpu_memory_utilization_ppm=900_000,
    )
    validation = validate_kv_capacity_plan(
        plan=plan,
        observation=observation,
        block_size=16,
    )

    assert validation.observed_num_blocks == 14_618
    assert validation.block_error == 0
    assert validation.cached_token_error_percent == 0
    assert validation.predicted_max_concurrency == pytest.approx(14_618 / 2048)
    assert validation.geometry_consistent_with_reported_interval is True
    assert validation.within_target is True


def test_validation_rejects_runtime_profile_sha256_mismatch() -> None:
    plan = plan_kv_capacity(_input())
    observation = _observation(
        run_id="wrong-config",
        runtime_profile_sha256="different",
        utilization_ppm=900_000,
        non_kv_bytes=M0_NON_KV_UPPER_BYTES,
    )
    with pytest.raises(ValueError, match="runtime_profile_sha256"):
        validate_kv_capacity_plan(plan=plan, observation=observation, block_size=16)


def test_model_spec_loader_reads_qwen_config(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "num_hidden_layers": 28,
                "hidden_size": 3584,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "max_position_embeddings": 32768,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )

    model = model_spec_from_hf_config(path)

    assert model.num_attention_layers == 28
    assert model.head_dim == 128
    assert model.num_kv_heads == 4
    assert model.model_dtype == KVDType.BFLOAT16
