"""Validated configuration models for reproducible SLOTune experiments."""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SearchMethodName = Literal["default", "random", "tpe"]


def _default_search_methods() -> list[SearchMethodName]:
    return ["default", "random", "tpe"]


class GPUConfig(BaseModel):
    """GPU selection for the single-card core implementation."""

    device_ids: list[int] = Field(default_factory=list, description="CUDA device ID to use")
    count: int = Field(default=1, ge=1, description="Number of GPUs")

    @model_validator(mode="after")
    def validate_count(self) -> "GPUConfig":
        """Keep the explicit device list and count consistent."""
        if self.device_ids:
            if len(set(self.device_ids)) != len(self.device_ids):
                raise ValueError("GPU device_ids must be unique")
            if any(device_id < 0 for device_id in self.device_ids):
                raise ValueError("GPU device_ids must be non-negative")
            self.count = len(self.device_ids)
        return self

    model_config = ConfigDict(extra="forbid")


class WeightedObjectives(BaseModel):
    """Legacy model retained only to provide a clear migration warning."""

    throughput: int = 60
    latency: int = 30
    memory: int = 10

    @field_validator("throughput", "latency", "memory")
    @classmethod
    def validate_legacy_weight(cls, value: int) -> int:
        if not 0 <= value <= 100:
            raise ValueError("legacy weights must be between 0 and 100")
        return value

    @model_validator(mode="after")
    def reject_legacy_objective(self) -> "WeightedObjectives":
        if self.throughput + self.latency + self.memory != 100:
            raise ValueError("legacy weights must sum to 100")
        warnings.warn(
            "WeightedObjectives is deprecated; configure `slo` and maximize SLO goodput",
            DeprecationWarning,
            stacklevel=2,
        )
        return self

    model_config = ConfigDict(extra="forbid")


class SLOConfig(BaseModel):
    """Per-request service-level objectives, expressed in milliseconds."""

    ttft_ms: Optional[float] = Field(default=1000.0, gt=0)
    tpot_ms: Optional[float] = Field(default=100.0, gt=0)
    e2e_ms: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def require_threshold(self) -> "SLOConfig":
        """Require at least one latency SLO."""
        if self.ttft_ms is None and self.tpot_ms is None and self.e2e_ms is None:
            raise ValueError("At least one of ttft_ms, tpot_ms, or e2e_ms must be configured")
        return self

    model_config = ConfigDict(extra="forbid")


class Constraints(BaseModel):
    """Hard trial constraints; violations make a candidate infeasible."""

    max_error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    max_peak_vram_mb: Optional[float] = Field(default=None, gt=0)
    max_memory_utilization: Optional[float] = Field(default=0.95, gt=0.0, le=1.0)
    require_no_oom: bool = True
    require_server_alive: bool = True
    max_latency_ms: Optional[float] = Field(default=None, gt=0)
    throughput_min: Optional[float] = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid")


class SearchSpaceOverride(BaseModel):
    """Effective vLLM server search space for one GPU."""

    gpu_memory_utilization: tuple[float, float] = Field(default=(0.60, 0.95))
    max_num_seqs: list[int] = Field(default_factory=lambda: [8, 16, 32, 64, 128])
    max_num_batched_tokens: list[int] = Field(default_factory=lambda: [1024, 2048, 4096, 8192])
    tensor_parallel_size: Literal[1] = 1
    pipeline_parallel_size: Literal[1] = 1

    @field_validator("gpu_memory_utilization")
    @classmethod
    def validate_memory_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if not 0.0 < low <= high < 1.0:
            raise ValueError("gpu_memory_utilization must satisfy 0 < low <= high < 1")
        return value

    @field_validator("max_num_seqs", "max_num_batched_tokens")
    @classmethod
    def validate_choices(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("search-space choices must not be empty")
        if any(item <= 0 for item in value):
            raise ValueError("search-space choices must be positive")
        if len(set(value)) != len(value):
            raise ValueError("search-space choices must be unique")
        return sorted(value)

    model_config = ConfigDict(extra="forbid")


class WorkloadConfig(BaseModel):
    """Benchmark workload and open-loop arrival configuration."""

    name: str = Field(default="alpaca", min_length=1)
    dataset_name: str = Field(default="tatsu-lab/alpaca", min_length=1)
    sample_size: int = Field(default=100, ge=1)
    prompt_length_distribution: Literal["auto", "uniform", "weighted"] = "auto"
    warmup_requests: int = Field(default=5, ge=0)
    max_concurrency: int = Field(default=10, ge=1)
    concurrent_requests: Optional[int] = Field(default=None, ge=1)
    request_rate: Optional[float] = Field(default=None, gt=0)
    capacity_request_rates: list[float] = Field(
        default_factory=list,
        description="Finite offered rates for the vLLM-default capacity sweep",
    )
    capacity_repeats: int = Field(default=1, ge=1)
    burstiness: float = Field(default=1.0, gt=0)
    max_tokens: int = Field(default=256, ge=1)
    fixed_input_tokens: Optional[int] = Field(default=None, ge=1)
    fixed_output_tokens: Optional[int] = Field(default=None, ge=1)
    ignore_eos: bool = False
    seed: int = 2026
    request_timeout_seconds: float = Field(default=300.0, gt=0)
    benchmark_backend: Literal["official", "sse"] = "sse"

    @field_validator("capacity_request_rates")
    @classmethod
    def validate_capacity_request_rates(cls, value: list[float]) -> list[float]:
        rates = [float(rate) for rate in value]
        if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError("capacity_request_rates must contain finite positive rates")
        if len(set(rates)) != len(rates):
            raise ValueError("capacity_request_rates must not contain duplicates")
        return rates

    @model_validator(mode="after")
    def normalize_concurrency(self) -> "WorkloadConfig":
        """Accept the upstream name while exposing one canonical concurrency value."""
        if self.concurrent_requests is not None:
            self.max_concurrency = self.concurrent_requests
        else:
            self.concurrent_requests = self.max_concurrency
        if self.request_rate is not None and not math.isfinite(self.request_rate):
            self.request_rate = None
        if self.fixed_output_tokens is not None:
            self.max_tokens = self.fixed_output_tokens
        return self

    model_config = ConfigDict(extra="forbid")


class TelemetryConfig(BaseModel):
    """Cross-layer telemetry sampling settings."""

    enabled: bool = True
    interval_ms: int = Field(default=200, ge=100, le=5000)
    metrics_path: str = "/metrics"
    collect_nvml: bool = True
    collect_energy: bool = False

    model_config = ConfigDict(extra="forbid")


class StudySettings(BaseModel):
    """Equal-budget search, repetition, and holdout settings."""

    trial_budget: int = Field(default=16, ge=1)
    min_trials: Optional[int] = Field(default=None, ge=1)
    timeout_minutes: int = Field(default=60, ge=1)
    storage_backend: Optional[str] = Field(
        default=None,
        description="Legacy VLLMOptimizer storage; core resume replays immutable trial artifacts",
    )
    prune_enabled: bool = False
    n_startup_trials: int = Field(default=5, ge=0)
    seed: int = 2026
    methods: list[SearchMethodName] = Field(default_factory=_default_search_methods)
    repeat_count: int = Field(default=3, ge=1)
    top_candidates: int = Field(default=3, ge=1)
    holdout_enabled: bool = True
    holdout_min_goodput_ratio: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description="Minimum holdout/repeat median goodput ratio for final validation",
    )
    resume: bool = False

    @model_validator(mode="after")
    def migrate_trial_count(self) -> "StudySettings":
        """Map the old min_trials setting to the equal trial budget."""
        if self.min_trials is not None:
            self.trial_budget = self.min_trials
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("study methods must be unique")
        return self

    model_config = ConfigDict(extra="forbid")


class BaselineConfig(BaseModel):
    """Compatibility options for the legacy baseline command."""

    enabled: bool = True
    num_requests: int = Field(default=1000, ge=1)
    max_tokens: int = Field(default=256, ge=1)

    model_config = ConfigDict(extra="forbid")


class AdaptivePrefillConfig(BaseModel):
    """Runtime policy and instrumentation settings for the V1 Scheduler."""

    enabled: bool = False
    fixed_prefill_cap: Optional[int] = Field(default=None, ge=1)
    decision_log_enabled: bool = True
    low_prefill_cap: int = Field(default=1024, ge=1)
    balanced_prefill_cap: int = Field(default=4096, ge=1)
    high_prefill_cap: int = Field(default=8192, ge=1)
    decode_backlog_high: int = Field(default=32, ge=1)
    oldest_prefill_wait_ms: float = Field(default=200.0, gt=0)
    kv_usage_high: float = Field(default=0.90, gt=0.0, le=1.0)
    min_prefill_progress: int = Field(default=256, ge=1)
    max_wait_ms: float = Field(default=1000.0, gt=0)
    hysteresis_steps: int = Field(default=3, ge=1)
    min_state_residency_steps: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def validate_caps_and_waits(self) -> "AdaptivePrefillConfig":
        """Keep controller levels ordered and starvation thresholds coherent."""
        if not self.low_prefill_cap <= self.balanced_prefill_cap <= self.high_prefill_cap:
            raise ValueError(
                "prefill caps must satisfy low_prefill_cap <= balanced_prefill_cap "
                "<= high_prefill_cap"
            )
        if self.min_prefill_progress > self.low_prefill_cap:
            raise ValueError("min_prefill_progress must not exceed low_prefill_cap")
        if self.max_wait_ms < self.oldest_prefill_wait_ms:
            raise ValueError("max_wait_ms must be at least oldest_prefill_wait_ms")
        if not self.enabled and self.fixed_prefill_cap is not None:
            raise ValueError("fixed_prefill_cap requires adaptive_prefill.enabled=true")
        return self

    model_config = ConfigDict(extra="forbid")


class TuningConfig(BaseModel):
    """Top-level SLOTune experiment configuration."""

    model: str = Field(default="gpt2", min_length=1)
    model_revision: Optional[str] = None
    tokenizer: Optional[str] = None
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    slo: SLOConfig = Field(default_factory=SLOConfig)
    constraints: Constraints = Field(default_factory=Constraints)
    search_space: SearchSpaceOverride = Field(default_factory=SearchSpaceOverride)
    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    study: StudySettings = Field(default_factory=StudySettings)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    adaptive_prefill: AdaptivePrefillConfig = Field(default_factory=AdaptivePrefillConfig)
    vllm_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_gpu_config(self) -> "TuningConfig":
        """Resolve the default device and enforce the core single-GPU scope."""
        value = self.gpu
        if not value.device_ids:
            value.device_ids = [0]
            value.count = 1
        if value.count != 1 or len(value.device_ids) != 1:
            raise ValueError(
                "SLOTune core supports one GPU; tensor and pipeline parallel sizes are fixed to 1"
            )
        return self

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class TunerSettings(BaseSettings):
    """Runtime settings sourced from the environment."""

    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")
    study_output_dir: str = "studies"
    html_output_dir: str = "reports"

    model_config = SettingsConfigDict(env_prefix="VLLM_TUNER_", env_file=".env")
