"""One-shot vLLM cache-capacity evidence collection for M1 initialization probes."""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from typing import Literal, Optional

import httpx
from pydantic import ConfigDict, Field, model_validator

from vllm_tuner.profiling.prometheus import parse_prometheus_text

from .kv_capacity_planner import (
    PPM,
    StrictFrozenModel,
    VLLMInitializationObservation,
    parse_vllm_initialization_observation,
)

CACHE_INFO_METRIC_NAMES = frozenset({"vllm:cache_config_info", "vllm_cache_config_info"})


class CacheConfigInfoEvidence(StrictFrozenModel):
    metric_name: str
    raw_metric_line: str
    engine_index: int = Field(ge=0)
    resolved_block_size: int = Field(gt=0)
    num_gpu_blocks: int = Field(gt=1)
    usable_num_gpu_blocks: int = Field(gt=0)
    requested_cache_dtype: str
    gpu_memory_utilization_ppm: int = Field(gt=0, le=PPM)
    kv_cache_memory_bytes: Optional[int] = Field(default=None, ge=1)
    num_gpu_blocks_override: Optional[int] = Field(default=None, ge=1)
    enable_prefix_caching: bool
    calculate_kv_scales: bool
    is_attention_free: bool
    sliding_window: Optional[int] = None
    labels: dict[str, str]

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class StartupCacheFormatEvidence(StrictFrozenModel):
    architecture: str
    max_model_len: int = Field(gt=0)
    attention_backend: str
    model_dtype: str
    requested_kv_cache_dtype: str
    enable_prefix_caching: bool
    enable_chunked_prefill: bool


class DeviceMemoryEvidence(StrictFrozenModel):
    """Keep physical VRAM diagnostics separate from vLLM's CUDA budget."""

    device_index: int = Field(ge=0)
    physical_total_memory_bytes: int = Field(gt=0)
    physical_free_memory_bytes: int = Field(ge=0)
    cuda_allocatable_total_memory_bytes: int = Field(gt=0)
    cuda_free_memory_bytes: int = Field(gt=0)
    physical_minus_cuda_total_bytes: int = Field(ge=0)
    physical_source: Literal["nvidia-smi"] = "nvidia-smi"
    cuda_source: Literal["torch.cuda.mem_get_info"] = "torch.cuda.mem_get_info"
    cuda_probe_isolated_process: Literal[True] = True

    @model_validator(mode="after")
    def validate_memory_domains(self) -> "DeviceMemoryEvidence":
        if self.physical_free_memory_bytes > self.physical_total_memory_bytes:
            raise ValueError("physical free memory exceeds physical total memory")
        if self.cuda_free_memory_bytes > self.cuda_allocatable_total_memory_bytes:
            raise ValueError("CUDA free memory exceeds CUDA allocatable total memory")
        if self.physical_total_memory_bytes < self.cuda_allocatable_total_memory_bytes:
            raise ValueError("physical total memory is below CUDA allocatable total memory")
        expected_gap = self.physical_total_memory_bytes - self.cuda_allocatable_total_memory_bytes
        if self.physical_minus_cuda_total_bytes != expected_gap:
            raise ValueError("physical/CUDA total-memory gap is inconsistent")
        return self


class CapacityRuntimeEvidence(StrictFrozenModel):
    device_memory: DeviceMemoryEvidence
    cache_config: CacheConfigInfoEvidence
    startup_format: StartupCacheFormatEvidence
    observation: VLLMInitializationObservation
    logged_capacity_consistent: bool
    raw_capacity_log_lines: tuple[str, ...]

    @model_validator(mode="after")
    def require_cuda_budget_domain(self) -> "CapacityRuntimeEvidence":
        if (
            self.observation.total_memory_bytes
            != self.device_memory.cuda_allocatable_total_memory_bytes
        ):
            raise ValueError("Planner total memory must use CUDA allocatable memory")
        if self.observation.initial_free_memory_bytes != self.device_memory.cuda_free_memory_bytes:
            raise ValueError("Planner free memory must use the isolated CUDA reading")
        return self


def _required_label(labels: dict[str, str], name: str) -> str:
    value = labels.get(name)
    if value is None or not value:
        raise ValueError(f"cache_config_info is missing label {name!r}")
    return value


def _positive_int_label(labels: dict[str, str], name: str) -> int:
    text = _required_label(labels, name)
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(f"cache_config_info label {name!r} is not an integer: {text!r}") from error
    if value <= 0:
        raise ValueError(f"cache_config_info label {name!r} must be positive")
    return value


def _optional_positive_int_label(labels: dict[str, str], name: str) -> Optional[int]:
    text = _required_label(labels, name)
    if text == "None":
        return None
    value = _positive_int_label(labels, name)
    return value


def _bool_label(labels: dict[str, str], name: str) -> bool:
    text = _required_label(labels, name)
    if text == "True":
        return True
    if text == "False":
        return False
    raise ValueError(f"cache_config_info label {name!r} is not a boolean: {text!r}")


def _utilization_ppm(labels: dict[str, str]) -> int:
    text = _required_label(labels, "gpu_memory_utilization")
    try:
        scaled = Decimal(text) * PPM
    except InvalidOperation as error:
        raise ValueError(f"invalid gpu_memory_utilization label: {text!r}") from error
    if scaled != scaled.to_integral_value():
        raise ValueError("gpu_memory_utilization cannot be represented exactly in ppm")
    value = int(scaled)
    if not 0 < value <= PPM:
        raise ValueError("gpu_memory_utilization ppm must satisfy 0 < value <= 1,000,000")
    return value


def parse_cache_config_info(metrics_text: str) -> CacheConfigInfoEvidence:
    """Extract one single-engine cache_config_info sample from raw exposition."""
    candidates = [
        sample
        for sample in parse_prometheus_text(metrics_text, strict=True)
        if sample.name in CACHE_INFO_METRIC_NAMES and math.isclose(sample.value, 1.0)
    ]
    if not candidates:
        raise ValueError("/metrics contains no vLLM cache_config_info sample")
    engine_zero = [sample for sample in candidates if sample.labels.get("engine") in {None, "0"}]
    selected_candidates = engine_zero or candidates
    if len(selected_candidates) != 1:
        raise ValueError(
            "cache_config_info must resolve to exactly one engine-0 sample; "
            f"found {len(selected_candidates)}"
        )
    sample = selected_candidates[0]
    labels = dict(sample.labels)
    engine_text = labels.get("engine", "0")
    try:
        engine_index = int(engine_text)
    except ValueError as error:
        raise ValueError(f"cache_config_info engine is invalid: {engine_text!r}") from error
    block_size = _positive_int_label(labels, "block_size")
    num_gpu_blocks = _positive_int_label(labels, "num_gpu_blocks")
    raw_lines = [
        line.strip()
        for line in metrics_text.splitlines()
        if line.lstrip().startswith(tuple(CACHE_INFO_METRIC_NAMES))
        and f'engine="{engine_text}"' in line
    ]
    if len(raw_lines) != 1:
        raise ValueError("unable to preserve exactly one raw cache_config_info metric line")
    return CacheConfigInfoEvidence(
        metric_name=sample.name,
        raw_metric_line=raw_lines[0],
        engine_index=engine_index,
        resolved_block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        usable_num_gpu_blocks=num_gpu_blocks - 1,
        requested_cache_dtype=_required_label(labels, "cache_dtype"),
        gpu_memory_utilization_ppm=_utilization_ppm(labels),
        kv_cache_memory_bytes=_optional_positive_int_label(labels, "kv_cache_memory_bytes"),
        num_gpu_blocks_override=_optional_positive_int_label(labels, "num_gpu_blocks_override"),
        enable_prefix_caching=_bool_label(labels, "enable_prefix_caching"),
        calculate_kv_scales=_bool_label(labels, "calculate_kv_scales"),
        is_attention_free=_bool_label(labels, "is_attention_free"),
        sliding_window=_optional_positive_int_label(labels, "sliding_window"),
        labels=labels,
    )


def _startup_cache_format(server_log_text: str) -> StartupCacheFormatEvidence:
    patterns = {
        "architecture": r"Resolved architecture:\s*([A-Za-z0-9_]+)",
        "max_model_len": r"Using max model len\s+(\d+)",
        "attention_backend": r"Using\s+([A-Z0-9_]+)\s+attention backend",
        "model_dtype": r"\bdtype=([^,\s]+)",
        "requested_kv_cache_dtype": r"\bkv_cache_dtype=([^,\s]+)",
        "enable_prefix_caching": r"\benable_prefix_caching=(True|False)",
        "enable_chunked_prefill": r"\benable_chunked_prefill=(True|False)",
    }
    values: dict[str, str] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, server_log_text)
        if match is None:
            raise ValueError(f"server log is missing startup cache-format field {name}")
        values[name] = match.group(1)
    return StartupCacheFormatEvidence(
        architecture=values["architecture"],
        max_model_len=int(values["max_model_len"]),
        attention_backend=values["attention_backend"],
        model_dtype=values["model_dtype"],
        requested_kv_cache_dtype=values["requested_kv_cache_dtype"],
        enable_prefix_caching=values["enable_prefix_caching"] == "True",
        enable_chunked_prefill=values["enable_chunked_prefill"] == "True",
    )


def build_capacity_runtime_evidence(
    *,
    run_id: str,
    runtime_profile_sha256: str,
    server_log_text: str,
    metrics_text: str,
    device_memory: DeviceMemoryEvidence,
) -> CapacityRuntimeEvidence:
    cache_config = parse_cache_config_info(metrics_text)
    startup_format = _startup_cache_format(server_log_text)
    observation = parse_vllm_initialization_observation(
        run_id=run_id,
        runtime_profile_sha256=runtime_profile_sha256,
        server_log_text=server_log_text,
        total_memory_bytes=device_memory.cuda_allocatable_total_memory_bytes,
        initial_free_memory_bytes=device_memory.cuda_free_memory_bytes,
        gpu_memory_utilization_ppm=cache_config.gpu_memory_utilization_ppm,
        resolved_block_size=cache_config.resolved_block_size,
        num_gpu_blocks_exact=cache_config.num_gpu_blocks,
    )
    capacity_lines = tuple(
        line.strip()
        for line in server_log_text.splitlines()
        if any(
            marker in line
            for marker in (
                "Available KV cache memory:",
                "GPU KV cache size:",
                "Maximum concurrency for",
            )
        )
    )
    if len(capacity_lines) != 3:
        raise ValueError("server log must contain exactly three capacity evidence lines")
    return CapacityRuntimeEvidence(
        device_memory=device_memory,
        cache_config=cache_config,
        startup_format=startup_format,
        observation=observation,
        logged_capacity_consistent=(
            observation.cached_tokens
            == cache_config.num_gpu_blocks * cache_config.resolved_block_size
        ),
        raw_capacity_log_lines=capacity_lines,
    )


async def fetch_metrics_text(
    base_url: str,
    *,
    metrics_path: str = "/metrics",
    timeout_seconds: float = 30.0,
) -> str:
    """Fetch raw exposition once; callers persist only selected capacity evidence."""
    async with httpx.AsyncClient(timeout=timeout_seconds, trust_env=False) as client:
        response = await client.get(f"{base_url.rstrip('/')}{metrics_path}")
        response.raise_for_status()
        return response.text
