"""Strict KV capacity planning for uniform full-attention decoder models."""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

GIB_BYTES = 1 << 30
PPM = 1_000_000
BASIS_POINTS = 10_000
PLANNER_VERSION: Literal["longctx-m1.v1"] = "longctx-m1.v1"


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class KVDType(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"

    @property
    def storage_bytes(self) -> int:
        return {
            KVDType.FLOAT32: 4,
            KVDType.FLOAT16: 2,
            KVDType.BFLOAT16: 2,
            KVDType.FP8_E4M3: 1,
            KVDType.FP8_E5M2: 1,
        }[self]


class UniformFullAttentionModelSpec(StrictFrozenModel):
    architecture: str = Field(min_length=1)
    attention_mode: Literal["uniform_full_attention"]
    num_hidden_layers: int = Field(gt=0)
    num_attention_layers: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    max_position_embeddings: int = Field(gt=0)
    model_dtype: KVDType

    @model_validator(mode="after")
    def validate_geometry(self) -> "UniformFullAttentionModelSpec":
        if self.num_attention_layers != self.num_hidden_layers:
            raise ValueError("planner v1 requires attention in every hidden layer")
        if self.num_kv_heads > self.num_attention_heads:
            raise ValueError("num_kv_heads must not exceed num_attention_heads")
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.model_dtype not in {KVDType.FLOAT32, KVDType.FLOAT16, KVDType.BFLOAT16}:
            raise ValueError("model_dtype must be a non-FP8 compute dtype")
        return self


class CacheLayoutSpec(StrictFrozenModel):
    requested_dtype: Literal[
        "auto",
        "float32",
        "float16",
        "bfloat16",
        "fp8_e4m3",
        "fp8_e5m2",
    ]
    resolved_dtype: KVDType
    block_size: Literal[1, 8, 16, 32]
    reserved_null_blocks: Literal[1]
    inline_metadata_bytes_per_layer_block: int = Field(ge=0)
    page_padding_bytes_per_layer_block: int = Field(ge=0)
    format_evidence: Literal["vllm_full_attention_spec", "measured_backend_spec"]

    def validate_resolution(self, model_dtype: KVDType) -> None:
        if self.requested_dtype == "auto":
            if self.resolved_dtype != model_dtype:
                raise ValueError("auto KV dtype must resolve to model_dtype")
        elif self.requested_dtype != self.resolved_dtype.value:
            raise ValueError("requested_dtype and resolved_dtype disagree")


class DeviceBudgetSpec(StrictFrozenModel):
    total_memory_bytes: int = Field(gt=0)
    initial_free_memory_bytes: int = Field(gt=0)
    gpu_memory_utilization_ppm: int = Field(gt=0, le=PPM)

    @property
    def requested_memory_bytes(self) -> int:
        return (self.total_memory_bytes * self.gpu_memory_utilization_ppm + PPM - 1) // PPM

    @model_validator(mode="after")
    def validate_startup_feasibility(self) -> "DeviceBudgetSpec":
        if self.initial_free_memory_bytes < self.requested_memory_bytes:
            raise ValueError("initial_free_memory_bytes is below the requested vLLM budget")
        return self


class ByteEstimate(StrictFrozenModel):
    point_bytes: int = Field(ge=0)
    upper_bytes: int = Field(ge=0)
    source: Literal["structural", "multi_run_calibration", "policy", "unavailable"]
    calibration_run_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_estimate(self) -> "ByteEstimate":
        if self.upper_bytes < self.point_bytes:
            raise ValueError("upper_bytes must be at least point_bytes")
        distinct_ids = {run_id for run_id in self.calibration_run_ids if run_id}
        if len(distinct_ids) != len(self.calibration_run_ids):
            raise ValueError("calibration_run_ids must be unique and non-empty")
        if self.source == "multi_run_calibration" and len(distinct_ids) < 2:
            raise ValueError("multi_run_calibration requires at least two run IDs")
        if self.source != "multi_run_calibration" and self.calibration_run_ids:
            raise ValueError("only multi_run_calibration may carry run IDs")
        return self


class NonKVPredictionSpec(StrictFrozenModel):
    weights: ByteEstimate
    peak_activations: ByteEstimate
    runtime_non_torch: ByteEstimate
    post_profile_cuda_graph: ByteEstimate
    post_profile_persistent: ByteEstimate
    unattributed_runtime_residual: ByteEstimate

    @property
    def point_bytes(self) -> int:
        return sum(component.point_bytes for component in self.components())

    @property
    def upper_bytes(self) -> int:
        return sum(component.upper_bytes for component in self.components())

    def components(self) -> tuple[ByteEstimate, ...]:
        return (
            self.weights,
            self.peak_activations,
            self.runtime_non_torch,
            self.post_profile_cuda_graph,
            self.post_profile_persistent,
            self.unattributed_runtime_residual,
        )


class SafetyPolicy(StrictFrozenModel):
    fixed_operational_reserve_bytes: int = Field(ge=0)
    kv_reserve_basis_points: int = Field(ge=0, lt=BASIS_POINTS)
    calibration_residual_upper_bytes: int = Field(ge=0)
    source: str = Field(min_length=1)


class ServingLimits(StrictFrozenModel):
    max_model_len: int = Field(gt=0)
    max_num_seqs: int = Field(gt=0)
    tensor_parallel_size: Literal[1]
    pipeline_parallel_size: Literal[1]


class ContextBin(StrictFrozenModel):
    name: str = Field(min_length=1)
    weight_ppm: int = Field(gt=0, le=PPM)
    prompt_tokens: int = Field(ge=0)
    reserved_output_tokens: int = Field(gt=0)

    @property
    def peak_kv_tokens(self) -> int:
        return self.prompt_tokens + self.reserved_output_tokens


class ContextDistributionSpec(StrictFrozenModel):
    bins: tuple[ContextBin, ...]
    confidence_ppm: int = Field(gt=0, le=PPM)
    iid_assumption: Literal[True]
    assume_no_prefix_reuse: Literal[True]

    @model_validator(mode="after")
    def validate_weights(self) -> "ContextDistributionSpec":
        if not self.bins:
            raise ValueError("context distribution must not be empty")
        if sum(context.weight_ppm for context in self.bins) != PPM:
            raise ValueError("context bin weights must sum to 1,000,000 ppm")
        names = [context.name for context in self.bins]
        if len(set(names)) != len(names):
            raise ValueError("context bin names must be unique")
        return self


class KVCapacityPlannerInput(StrictFrozenModel):
    schema_version: Literal["longctx-m1.v1"]
    model: UniformFullAttentionModelSpec
    cache: CacheLayoutSpec
    device: DeviceBudgetSpec
    non_kv: NonKVPredictionSpec
    safety: SafetyPolicy
    serving: ServingLimits
    distribution: ContextDistributionSpec

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "KVCapacityPlannerInput":
        self.cache.validate_resolution(self.model.model_dtype)
        if self.serving.max_model_len > self.model.max_position_embeddings:
            raise ValueError("max_model_len exceeds model max_position_embeddings")
        if any(
            context.peak_kv_tokens > self.serving.max_model_len
            for context in self.distribution.bins
        ):
            raise ValueError("context bin peak KV tokens exceed max_model_len")
        return self


class KVGeometryResult(StrictFrozenModel):
    gqa_ratio: int
    dtype_bytes: int
    payload_bytes_per_token_per_layer: int
    payload_bytes_per_token_total: int
    payload_page_bytes_per_layer: int
    page_bytes_per_layer: int
    block_bytes_total: int
    inline_metadata_bytes_per_block_total: int
    page_padding_bytes_per_block_total: int


class MemoryBudgetResult(StrictFrozenModel):
    requested_memory_bytes: int
    non_kv_point_bytes: int
    non_kv_upper_bytes: int
    raw_available_kv_bytes: int
    fixed_operational_reserve_bytes: int
    proportional_kv_reserve_bytes: int
    calibration_residual_upper_bytes: int
    total_policy_reserve_bytes: int
    safe_available_kv_bytes: int


class CapacityResult(StrictFrozenModel):
    raw_num_blocks: int
    raw_cached_tokens: int
    raw_usable_num_blocks: int
    raw_usable_cached_tokens: int
    raw_allocated_bytes: int
    raw_block_floor_remainder_bytes: int
    safe_num_blocks: int
    safe_cached_tokens: int
    safe_usable_num_blocks: int
    safe_usable_cached_tokens: int
    safe_allocated_bytes: int
    safe_block_floor_remainder_bytes: int


class ContextCapacityResult(StrictFrozenModel):
    name: str
    weight_ppm: int
    peak_kv_tokens: int
    blocks_per_sequence: int
    allocated_tokens_per_sequence: int
    tail_slots: int
    tail_slot_waste_bytes: int
    format_overhead_bytes: int
    allocated_bytes_per_sequence: int
    raw_concurrency_ratio: float
    usable_raw_concurrency_ratio: float
    safe_concurrency_ratio: float
    safe_integer_concurrency: int


class DistributionCapacityResult(StrictFrozenModel):
    expected_blocks_per_sequence: float
    expected_only_concurrency: int
    guaranteed_worst_case_concurrency: int
    iid_quantile_concurrency: int
    iid_quantile_overflow_probability: float
    confidence_ppm: int
    max_num_seqs_cap: int


class BudgetProvenance(StrictFrozenModel):
    component: str
    point_bytes: int
    upper_bytes: int
    source: str
    calibration_run_ids: tuple[str, ...]


class KVCapacityPlanOutput(StrictFrozenModel):
    planner_version: Literal["longctx-m1.v1"]
    input_sha256: str
    runtime_profile_sha256: str
    assumptions: tuple[str, ...]
    geometry: KVGeometryResult
    memory: MemoryBudgetResult
    capacity: CapacityResult
    contexts: tuple[ContextCapacityResult, ...]
    distribution: DistributionCapacityResult
    provenance: tuple[BudgetProvenance, ...]


class VLLMInitializationObservation(StrictFrozenModel):
    run_id: str = Field(min_length=1)
    runtime_profile_sha256: str = Field(min_length=1)
    server_log_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_memory_bytes: int = Field(gt=0)
    initial_free_memory_bytes: int = Field(gt=0)
    gpu_memory_utilization_ppm: int = Field(gt=0, le=PPM)
    resolved_block_size: int = Field(gt=0)
    num_gpu_blocks: int = Field(gt=1)
    usable_num_gpu_blocks: int = Field(gt=0)
    num_blocks_source: Literal["cache_config_info", "server_log_derived"]
    cached_tokens: int = Field(gt=0)
    max_model_len: int = Field(gt=0)
    reported_max_concurrency: float = Field(gt=0.0)
    reported_available_gib_text: str = Field(pattern=r"^[0-9]+(?:\.[0-9]+)?$")

    @field_validator("reported_max_concurrency")
    @classmethod
    def require_finite_concurrency(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("reported_max_concurrency must be finite")
        return value

    @model_validator(mode="after")
    def validate_block_evidence(self) -> "VLLMInitializationObservation":
        if self.cached_tokens != self.num_gpu_blocks * self.resolved_block_size:
            raise ValueError("cached_tokens disagree with num_gpu_blocks * resolved_block_size")
        if self.usable_num_gpu_blocks != self.num_gpu_blocks - 1:
            raise ValueError("usable_num_gpu_blocks must reserve exactly one null block")
        return self

    @property
    def requested_memory_bytes(self) -> int:
        return (self.total_memory_bytes * self.gpu_memory_utilization_ppm + PPM - 1) // PPM


class NonKVReserveInterval(StrictFrozenModel):
    run_id: str
    lower_exclusive_bytes: int
    upper_inclusive_bytes: int
    observed_num_blocks: int
    observed_allocated_kv_bytes: int


class MultiRunNonKVCalibration(StrictFrozenModel):
    run_ids: tuple[str, ...]
    intervals: tuple[NonKVReserveInterval, ...]
    intersection_lower_exclusive_bytes: int
    intersection_upper_inclusive_bytes: int
    point_bytes: int
    upper_bytes: int
    uncertainty_bytes: int

    def as_estimate(self) -> ByteEstimate:
        return ByteEstimate(
            point_bytes=self.point_bytes,
            upper_bytes=self.upper_bytes,
            source="multi_run_calibration",
            calibration_run_ids=self.run_ids,
        )


class KVCapacityValidationResult(StrictFrozenModel):
    run_id: str
    runtime_profile_sha256: str
    observed_num_blocks: int
    predicted_num_blocks: int
    observed_usable_num_blocks: int
    predicted_usable_num_blocks: int
    block_error: int
    block_error_percent: float
    observed_cached_tokens: int
    predicted_cached_tokens: int
    observed_usable_cached_tokens: int
    predicted_usable_cached_tokens: int
    cached_token_error_percent: float
    observed_max_concurrency: float
    predicted_max_concurrency: float
    max_concurrency_error_percent: float
    observed_allocated_kv_bytes: int
    predicted_allocated_kv_bytes: int
    allocated_capacity_delta_bytes: int
    predicted_block_floor_remainder_bytes: int
    reported_available_lower_bytes: int
    reported_available_upper_bytes: int
    geometry_consistent_with_reported_interval: bool
    target_error_percent: float
    within_target: bool


def _ceil_ratio(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _runtime_profile_sha256(inputs: KVCapacityPlannerInput) -> str:
    profile = {
        "model": inputs.model.model_dump(mode="json"),
        "cache": inputs.cache.model_dump(mode="json"),
        "device": {
            "total_memory_bytes": inputs.device.total_memory_bytes,
            "gpu_memory_utilization_ppm": inputs.device.gpu_memory_utilization_ppm,
        },
        "serving": inputs.serving.model_dump(mode="json"),
    }
    return _canonical_sha256(profile)


def _geometry(model: UniformFullAttentionModelSpec, cache: CacheLayoutSpec) -> KVGeometryResult:
    payload_per_layer = 2 * model.num_kv_heads * model.head_dim * cache.resolved_dtype.storage_bytes
    payload_total = model.num_attention_layers * payload_per_layer
    payload_page = cache.block_size * payload_per_layer
    page_bytes = (
        payload_page
        + cache.inline_metadata_bytes_per_layer_block
        + cache.page_padding_bytes_per_layer_block
    )
    return KVGeometryResult(
        gqa_ratio=model.num_attention_heads // model.num_kv_heads,
        dtype_bytes=cache.resolved_dtype.storage_bytes,
        payload_bytes_per_token_per_layer=payload_per_layer,
        payload_bytes_per_token_total=payload_total,
        payload_page_bytes_per_layer=payload_page,
        page_bytes_per_layer=page_bytes,
        block_bytes_total=model.num_attention_layers * page_bytes,
        inline_metadata_bytes_per_block_total=(
            model.num_attention_layers * cache.inline_metadata_bytes_per_layer_block
        ),
        page_padding_bytes_per_block_total=(
            model.num_attention_layers * cache.page_padding_bytes_per_layer_block
        ),
    )


def _iid_quantile_concurrency(
    *,
    block_distribution: tuple[tuple[int, int], ...],
    safe_blocks: int,
    max_num_seqs: int,
    confidence_ppm: int,
) -> tuple[int, float]:
    probabilities = {blocks: weight / PPM for blocks, weight in block_distribution}
    states: dict[int, float] = {0: 1.0}
    best_n = 0
    best_overflow = 0.0
    required_success = confidence_ppm / PPM
    for count in range(1, max_num_seqs + 1):
        next_states: dict[int, float] = {}
        for used_blocks, state_probability in states.items():
            for blocks, probability in probabilities.items():
                total = used_blocks + blocks
                if total <= safe_blocks:
                    next_states[total] = (
                        next_states.get(total, 0.0) + state_probability * probability
                    )
        success_probability = math.fsum(next_states.values())
        overflow_probability = max(0.0, 1.0 - success_probability)
        if success_probability + 1e-15 < required_success:
            break
        best_n = count
        best_overflow = overflow_probability
        states = next_states
    return best_n, best_overflow


def plan_kv_capacity(inputs: KVCapacityPlannerInput) -> KVCapacityPlanOutput:
    geometry = _geometry(inputs.model, inputs.cache)
    requested = inputs.device.requested_memory_bytes
    non_kv_point = inputs.non_kv.point_bytes
    non_kv_upper = inputs.non_kv.upper_bytes
    raw_available = requested - non_kv_point
    if raw_available <= 0:
        raise ValueError("point non-KV prediction exhausts requested GPU memory")
    proportional_reserve = _ceil_ratio(
        raw_available * inputs.safety.kv_reserve_basis_points,
        BASIS_POINTS,
    )
    total_policy_reserve = (
        inputs.safety.fixed_operational_reserve_bytes
        + proportional_reserve
        + inputs.safety.calibration_residual_upper_bytes
    )
    safe_available = requested - non_kv_upper - total_policy_reserve
    if safe_available <= 0:
        raise ValueError("upper non-KV prediction and policy reserve leave no KV memory")
    if geometry.block_bytes_total <= 0:
        raise ValueError("block_bytes_total must be positive")

    raw_blocks, raw_remainder = divmod(raw_available, geometry.block_bytes_total)
    safe_blocks, safe_remainder = divmod(safe_available, geometry.block_bytes_total)
    raw_usable_blocks = raw_blocks - inputs.cache.reserved_null_blocks
    safe_usable_blocks = safe_blocks - inputs.cache.reserved_null_blocks
    if raw_usable_blocks <= 0 or safe_usable_blocks <= 0:
        raise ValueError("capacity cannot hold a usable KV block after the null block")
    capacity = CapacityResult(
        raw_num_blocks=raw_blocks,
        raw_cached_tokens=raw_blocks * inputs.cache.block_size,
        raw_usable_num_blocks=raw_usable_blocks,
        raw_usable_cached_tokens=raw_usable_blocks * inputs.cache.block_size,
        raw_allocated_bytes=raw_blocks * geometry.block_bytes_total,
        raw_block_floor_remainder_bytes=raw_remainder,
        safe_num_blocks=safe_blocks,
        safe_cached_tokens=safe_blocks * inputs.cache.block_size,
        safe_usable_num_blocks=safe_usable_blocks,
        safe_usable_cached_tokens=safe_usable_blocks * inputs.cache.block_size,
        safe_allocated_bytes=safe_blocks * geometry.block_bytes_total,
        safe_block_floor_remainder_bytes=safe_remainder,
    )
    memory = MemoryBudgetResult(
        requested_memory_bytes=requested,
        non_kv_point_bytes=non_kv_point,
        non_kv_upper_bytes=non_kv_upper,
        raw_available_kv_bytes=raw_available,
        fixed_operational_reserve_bytes=inputs.safety.fixed_operational_reserve_bytes,
        proportional_kv_reserve_bytes=proportional_reserve,
        calibration_residual_upper_bytes=inputs.safety.calibration_residual_upper_bytes,
        total_policy_reserve_bytes=total_policy_reserve,
        safe_available_kv_bytes=safe_available,
    )

    contexts: list[ContextCapacityResult] = []
    weighted_blocks = 0.0
    block_distribution: list[tuple[int, int]] = []
    for context in inputs.distribution.bins:
        blocks = _ceil_ratio(context.peak_kv_tokens, inputs.cache.block_size)
        allocated_tokens = blocks * inputs.cache.block_size
        tail_slots = allocated_tokens - context.peak_kv_tokens
        raw_ratio = raw_blocks / blocks
        usable_raw_ratio = raw_usable_blocks / blocks
        safe_ratio = safe_usable_blocks / blocks
        contexts.append(
            ContextCapacityResult(
                name=context.name,
                weight_ppm=context.weight_ppm,
                peak_kv_tokens=context.peak_kv_tokens,
                blocks_per_sequence=blocks,
                allocated_tokens_per_sequence=allocated_tokens,
                tail_slots=tail_slots,
                tail_slot_waste_bytes=tail_slots * geometry.payload_bytes_per_token_total,
                format_overhead_bytes=(
                    blocks
                    * (
                        geometry.inline_metadata_bytes_per_block_total
                        + geometry.page_padding_bytes_per_block_total
                    )
                ),
                allocated_bytes_per_sequence=blocks * geometry.block_bytes_total,
                raw_concurrency_ratio=raw_ratio,
                usable_raw_concurrency_ratio=usable_raw_ratio,
                safe_concurrency_ratio=safe_ratio,
                safe_integer_concurrency=min(
                    inputs.serving.max_num_seqs, safe_usable_blocks // blocks
                ),
            )
        )
        weighted_blocks += context.weight_ppm / PPM * blocks
        block_distribution.append((blocks, context.weight_ppm))

    expected_only = min(
        inputs.serving.max_num_seqs,
        math.floor(safe_usable_blocks / weighted_blocks),
    )
    worst_blocks = max(blocks for blocks, _ in block_distribution)
    guaranteed = min(inputs.serving.max_num_seqs, safe_usable_blocks // worst_blocks)
    quantile_concurrency, overflow_probability = _iid_quantile_concurrency(
        block_distribution=tuple(block_distribution),
        safe_blocks=safe_usable_blocks,
        max_num_seqs=inputs.serving.max_num_seqs,
        confidence_ppm=inputs.distribution.confidence_ppm,
    )
    distribution = DistributionCapacityResult(
        expected_blocks_per_sequence=weighted_blocks,
        expected_only_concurrency=expected_only,
        guaranteed_worst_case_concurrency=guaranteed,
        iid_quantile_concurrency=quantile_concurrency,
        iid_quantile_overflow_probability=overflow_probability,
        confidence_ppm=inputs.distribution.confidence_ppm,
        max_num_seqs_cap=inputs.serving.max_num_seqs,
    )

    provenance: list[BudgetProvenance] = []
    for name in (
        "weights",
        "peak_activations",
        "runtime_non_torch",
        "post_profile_cuda_graph",
        "post_profile_persistent",
        "unattributed_runtime_residual",
    ):
        estimate = getattr(inputs.non_kv, name)
        component_name = "checkpoint_weight_bytes_proxy" if name == "weights" else name
        provenance.append(
            BudgetProvenance(
                component=component_name,
                point_bytes=estimate.point_bytes,
                upper_bytes=estimate.upper_bytes,
                source=estimate.source,
                calibration_run_ids=estimate.calibration_run_ids,
            )
        )
    provenance.append(
        BudgetProvenance(
            component="safety_policy",
            point_bytes=total_policy_reserve,
            upper_bytes=total_policy_reserve,
            source=inputs.safety.source,
            calibration_run_ids=(),
        )
    )
    assumptions = (
        "single GPU, TP=PP=1",
        "uniform full attention with one KV page per modeled attention layer",
        "no shared-prefix capacity credit; APC effects are measured separately in M3",
        "IID quantile concurrency is a memory bound, not a throughput capacity knee",
        "raw capacity uses point non-KV estimates; safe capacity uses upper estimates plus policy",
        "vLLM BlockPool reserves one allocated block as an unusable null block",
        "locked checkpoint bytes are a weights proxy; calibration residual absorbs load overhead",
    )
    return KVCapacityPlanOutput(
        planner_version=PLANNER_VERSION,
        input_sha256=_canonical_sha256(inputs.model_dump(mode="json")),
        runtime_profile_sha256=_runtime_profile_sha256(inputs),
        assumptions=assumptions,
        geometry=geometry,
        memory=memory,
        capacity=capacity,
        contexts=tuple(contexts),
        distribution=distribution,
        provenance=tuple(provenance),
    )


_AVAILABLE_RE = re.compile(r"Available KV cache memory:\s*([0-9]+(?:\.[0-9]+)?)\s*GiB")
_TOKENS_RE = re.compile(r"GPU KV cache size:\s*([0-9][0-9,]*)\s*tokens")
_CONCURRENCY_RE = re.compile(
    r"Maximum concurrency for\s*([0-9][0-9,]*)\s*tokens per request:\s*([0-9.]+)x"
)


def parse_vllm_initialization_observation(
    *,
    run_id: str,
    runtime_profile_sha256: str,
    server_log_text: str,
    total_memory_bytes: int,
    initial_free_memory_bytes: int,
    gpu_memory_utilization_ppm: int,
    resolved_block_size: int = 16,
    num_gpu_blocks_exact: int | None = None,
) -> VLLMInitializationObservation:
    memory_match = _AVAILABLE_RE.search(server_log_text)
    tokens_match = _TOKENS_RE.search(server_log_text)
    concurrency_match = _CONCURRENCY_RE.search(server_log_text)
    missing = [
        name
        for name, match in (
            ("available_kv_memory", memory_match),
            ("cached_tokens", tokens_match),
            ("maximum_concurrency", concurrency_match),
        )
        if match is None
    ]
    if missing:
        raise ValueError("vLLM initialization log is missing: " + ", ".join(missing))
    assert memory_match is not None
    assert tokens_match is not None
    assert concurrency_match is not None
    cached_tokens = int(tokens_match.group(1).replace(",", ""))
    if cached_tokens % resolved_block_size != 0:
        raise ValueError("logged cached_tokens are not divisible by resolved_block_size")
    derived_blocks = cached_tokens // resolved_block_size
    if num_gpu_blocks_exact is not None and num_gpu_blocks_exact != derived_blocks:
        raise ValueError("cache_config_info num_gpu_blocks disagree with logged cached_tokens")
    num_gpu_blocks = num_gpu_blocks_exact or derived_blocks
    return VLLMInitializationObservation(
        run_id=run_id,
        runtime_profile_sha256=runtime_profile_sha256,
        server_log_sha256=hashlib.sha256(server_log_text.encode("utf-8")).hexdigest(),
        total_memory_bytes=total_memory_bytes,
        initial_free_memory_bytes=initial_free_memory_bytes,
        gpu_memory_utilization_ppm=gpu_memory_utilization_ppm,
        resolved_block_size=resolved_block_size,
        num_gpu_blocks=num_gpu_blocks,
        usable_num_gpu_blocks=num_gpu_blocks - 1,
        num_blocks_source=(
            "cache_config_info" if num_gpu_blocks_exact is not None else "server_log_derived"
        ),
        cached_tokens=cached_tokens,
        max_model_len=int(concurrency_match.group(1).replace(",", "")),
        reported_max_concurrency=float(concurrency_match.group(2)),
        reported_available_gib_text=memory_match.group(1),
    )


def _reported_available_interval(gib_text: str) -> tuple[int, int]:
    value = Decimal(gib_text)
    decimals = len(gib_text.partition(".")[2])
    half_step = Decimal(10) ** -decimals * Decimal(GIB_BYTES) / 2
    center = value * Decimal(GIB_BYTES)
    return max(0, math.floor(center - half_step)), math.ceil(center + half_step)


def _reserve_interval(
    observation: VLLMInitializationObservation,
    geometry: KVGeometryResult,
    block_size: int,
) -> NonKVReserveInterval:
    if observation.resolved_block_size != block_size:
        raise ValueError("observation resolved_block_size does not match cache layout")
    blocks = observation.num_gpu_blocks
    allocated = blocks * geometry.block_bytes_total
    upper = observation.requested_memory_bytes - allocated
    if upper < 0:
        raise ValueError("observed KV allocation exceeds requested memory")
    lower_exclusive = upper - geometry.block_bytes_total
    return NonKVReserveInterval(
        run_id=observation.run_id,
        lower_exclusive_bytes=lower_exclusive,
        upper_inclusive_bytes=upper,
        observed_num_blocks=blocks,
        observed_allocated_kv_bytes=allocated,
    )


def calibrate_non_kv_from_runs(
    *,
    observations: tuple[VLLMInitializationObservation, ...],
    model: UniformFullAttentionModelSpec,
    cache: CacheLayoutSpec,
) -> MultiRunNonKVCalibration:
    """Intersect whole-block reserve intervals from at least two frozen runs."""
    if len(observations) < 2:
        raise ValueError("multi-run calibration requires at least two observations")
    run_ids = tuple(observation.run_id for observation in observations)
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("calibration observation run IDs must be unique")
    geometry = _geometry(model, cache)
    intervals = tuple(
        _reserve_interval(observation, geometry, cache.block_size) for observation in observations
    )
    lower = max(interval.lower_exclusive_bytes for interval in intervals)
    upper = min(interval.upper_inclusive_bytes for interval in intervals)
    if lower >= upper:
        raise ValueError("non-KV reserve intervals do not overlap; profile class is unstable")
    point = (lower + upper + 1) // 2
    return MultiRunNonKVCalibration(
        run_ids=run_ids,
        intervals=intervals,
        intersection_lower_exclusive_bytes=lower,
        intersection_upper_inclusive_bytes=upper,
        point_bytes=point,
        upper_bytes=upper,
        uncertainty_bytes=upper - lower - 1,
    )


def validate_kv_capacity_plan(
    *,
    plan: KVCapacityPlanOutput,
    observation: VLLMInitializationObservation,
    block_size: int,
    target_error_percent: float = 10.0,
) -> KVCapacityValidationResult:
    if observation.runtime_profile_sha256 != plan.runtime_profile_sha256:
        raise ValueError(
            "observation runtime_profile_sha256 does not match planner runtime profile"
        )
    if observation.resolved_block_size != block_size:
        raise ValueError("observation resolved_block_size does not match validation block_size")
    if not math.isfinite(target_error_percent) or target_error_percent < 0:
        raise ValueError("target_error_percent must be finite and non-negative")
    observed_blocks = observation.num_gpu_blocks
    predicted_blocks = plan.capacity.raw_num_blocks
    block_error = predicted_blocks - observed_blocks
    block_error_percent = block_error / observed_blocks * 100.0
    cached_error_percent = (
        (plan.capacity.raw_cached_tokens - observation.cached_tokens)
        / observation.cached_tokens
        * 100.0
    )
    blocks_per_request = _ceil_ratio(observation.max_model_len, block_size)
    predicted_concurrency = predicted_blocks / blocks_per_request
    concurrency_error = (
        (predicted_concurrency - observation.reported_max_concurrency)
        / observation.reported_max_concurrency
        * 100.0
    )
    observed_allocated = observed_blocks * plan.geometry.block_bytes_total
    lower, upper = _reported_available_interval(observation.reported_available_gib_text)
    geometry_consistent = (
        observed_allocated <= upper and observed_allocated + plan.geometry.block_bytes_total > lower
    )
    return KVCapacityValidationResult(
        run_id=observation.run_id,
        runtime_profile_sha256=observation.runtime_profile_sha256,
        observed_num_blocks=observed_blocks,
        predicted_num_blocks=predicted_blocks,
        observed_usable_num_blocks=observed_blocks - 1,
        predicted_usable_num_blocks=plan.capacity.raw_usable_num_blocks,
        block_error=block_error,
        block_error_percent=block_error_percent,
        observed_cached_tokens=observation.cached_tokens,
        predicted_cached_tokens=plan.capacity.raw_cached_tokens,
        observed_usable_cached_tokens=(observed_blocks - 1) * block_size,
        predicted_usable_cached_tokens=plan.capacity.raw_usable_cached_tokens,
        cached_token_error_percent=cached_error_percent,
        observed_max_concurrency=observation.reported_max_concurrency,
        predicted_max_concurrency=predicted_concurrency,
        max_concurrency_error_percent=concurrency_error,
        observed_allocated_kv_bytes=observed_allocated,
        predicted_allocated_kv_bytes=plan.capacity.raw_allocated_bytes,
        allocated_capacity_delta_bytes=plan.capacity.raw_allocated_bytes - observed_allocated,
        predicted_block_floor_remainder_bytes=plan.capacity.raw_block_floor_remainder_bytes,
        reported_available_lower_bytes=lower,
        reported_available_upper_bytes=upper,
        geometry_consistent_with_reported_interval=geometry_consistent,
        target_error_percent=target_error_percent,
        within_target=abs(block_error_percent) <= target_error_percent,
    )


def model_spec_from_hf_config(config_path: str | Path) -> UniformFullAttentionModelSpec:
    path = Path(config_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read model config {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("model config must be a JSON object")
    required = (
        "architectures",
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError("model config is missing: " + ", ".join(missing))
    architectures = payload["architectures"]
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError("planner requires exactly one model architecture")
    num_attention_heads = payload["num_attention_heads"]
    hidden_size = payload["hidden_size"]
    if isinstance(num_attention_heads, bool) or not isinstance(num_attention_heads, int):
        raise ValueError("num_attention_heads must be an integer")
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
        raise ValueError("hidden_size must be an integer")
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    dtype_text = str(payload.get("torch_dtype", "")).lower()
    dtype_aliases = {
        "float32": KVDType.FLOAT32,
        "float16": KVDType.FLOAT16,
        "bfloat16": KVDType.BFLOAT16,
    }
    if dtype_text not in dtype_aliases:
        raise ValueError(f"unsupported model torch_dtype: {dtype_text!r}")
    integer_values = {
        "num_hidden_layers": payload["num_hidden_layers"],
        "num_key_value_heads": payload.get("num_key_value_heads", num_attention_heads),
        "head_dim": payload.get("head_dim", hidden_size // num_attention_heads),
        "max_position_embeddings": payload["max_position_embeddings"],
    }
    for name, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    num_layers = integer_values["num_hidden_layers"]
    return UniformFullAttentionModelSpec(
        architecture=str(architectures[0]),
        attention_mode="uniform_full_attention",
        num_hidden_layers=num_layers,
        num_attention_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=integer_values["num_key_value_heads"],
        head_dim=integer_values["head_dim"],
        max_position_embeddings=integer_values["max_position_embeddings"],
        model_dtype=dtype_aliases[dtype_text],
    )
