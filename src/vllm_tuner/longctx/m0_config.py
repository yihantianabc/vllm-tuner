"""Strict configuration boundary for the long-context v5 M0 baseline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic import model_validator

from ..config.models import (
    AdaptivePrefillConfig,
    BaselineConfig,
    Constraints,
    GPUConfig,
    SLOConfig,
    StudySettings,
    TelemetryConfig,
    TuningConfig,
    WorkloadConfig,
)
from .model_identity import ModelIdentityLock, load_model_lock

PRIMARY_MIN_PARAMETERS = 7_000_000_000
PRIMARY_MAX_PARAMETERS = 8_999_999_999
FALLBACK_MIN_PARAMETERS = 2_500_000_000
FALLBACK_MAX_PARAMETERS = 3_999_999_999
LEGACY_ARTIFACT_MARKER = "slotune-results"


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every mapping depth."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _absolute_path(path: Path, field_name: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path")
    return path.resolve(strict=False)


class LongContextM0ModelConfig(BaseModel):
    """Local model bytes plus the only authoritative model identity lock."""

    local_path: Path
    lock_path: Path

    @field_validator("local_path")
    @classmethod
    def validate_local_path(cls, path: Path) -> Path:
        resolved = _absolute_path(path, "model.local_path")
        if not resolved.is_dir():
            raise ValueError("model.local_path must be an existing directory")
        return resolved

    @field_validator("lock_path")
    @classmethod
    def validate_lock_path(cls, path: Path) -> Path:
        resolved = _absolute_path(path, "model.lock_path")
        if not resolved.is_file():
            raise ValueError("model.lock_path must be an existing file")
        if resolved.suffix.casefold() not in {".yaml", ".yml"}:
            raise ValueError("model.lock_path must be a YAML file")
        return resolved

    def identity(self) -> ModelIdentityLock:
        """Load the authoritative repository, revision, parameter, and byte identity."""
        return load_model_lock(self.lock_path)

    model_config = ConfigDict(extra="forbid", frozen=True)


class LongContextM0ArtifactConfig(BaseModel):
    """Fresh artifact root reserved for long-context v5 evidence."""

    root: Path

    @field_validator("root")
    @classmethod
    def validate_root(cls, path: Path) -> Path:
        raw = path.as_posix().casefold()
        resolved = _absolute_path(path, "artifacts.root")
        if (
            LEGACY_ARTIFACT_MARKER in raw
            or LEGACY_ARTIFACT_MARKER in resolved.as_posix().casefold()
        ):
            raise ValueError("artifacts.root must not use the Legacy slotune-results tree")
        if resolved == Path(resolved.anchor):
            raise ValueError("artifacts.root must not be a filesystem root")
        if resolved.exists() and not resolved.is_dir():
            raise ValueError("artifacts.root must be a directory path")
        return resolved

    model_config = ConfigDict(extra="forbid", frozen=True)


class LongContextM0RuntimeConfig(BaseModel):
    """Immutable clean-upstream vLLM and host-runtime lock."""

    lock_path: Path

    @field_validator("lock_path")
    @classmethod
    def validate_lock_path(cls, path: Path) -> Path:
        resolved = _absolute_path(path, "runtime.lock_path")
        if not resolved.is_file():
            raise ValueError("runtime.lock_path must be an existing file")
        if resolved.suffix.casefold() not in {".yaml", ".yml"}:
            raise ValueError("runtime.lock_path must be a YAML file")
        return resolved

    model_config = ConfigDict(extra="forbid", frozen=True)


class LongContextM0GPUConfig(BaseModel):
    """Explicit single-card GPU selection."""

    device_ids: tuple[int, ...]
    count: Literal[1]

    @field_validator("device_ids")
    @classmethod
    def validate_device_ids(cls, device_ids: tuple[int, ...]) -> tuple[int, ...]:
        if len(device_ids) != 1:
            raise ValueError("longctx-v5 M0 requires exactly one GPU device_id")
        if isinstance(device_ids[0], bool) or device_ids[0] < 0:
            raise ValueError("GPU device_id must be a non-negative integer")
        return device_ids

    model_config = ConfigDict(extra="forbid", frozen=True)


class LongContextM0WorkloadConfig(BaseModel):
    """Fixed canary traffic used to establish the production baseline."""

    measured_requests: int = Field(ge=100)
    warmup_requests: int = Field(ge=1)
    fixed_input_tokens: int = Field(ge=1)
    fixed_output_tokens: int = Field(ge=1)
    request_rate: float = Field(gt=0)
    max_concurrency: int = Field(ge=1)
    request_timeout_seconds: float = Field(gt=0)
    seed: int = Field(ge=0)
    burstiness: float = Field(default=1.0, gt=0)
    ignore_eos: Literal[True] = True

    @field_validator("request_rate", "request_timeout_seconds", "burstiness")
    @classmethod
    def validate_finite_float(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("workload rate, timeout, and burstiness values must be finite")
        return value

    model_config = ConfigDict(extra="forbid", frozen=True)


class LongContextM0Config(BaseModel):
    """One fail-closed v5 M0 production-default or correctness-smoke run."""

    project_line: Literal["longctx-v5"]
    milestone: Literal["M0"]
    profile: Literal["production-default"]
    evidence_role: Literal["formal", "smoke"]
    model_tier: Literal["primary-7b-8b", "fallback-3b", "smoke"]
    fallback_reason: Optional[str] = None
    model: LongContextM0ModelConfig
    artifacts: LongContextM0ArtifactConfig
    runtime: LongContextM0RuntimeConfig
    gpu: LongContextM0GPUConfig
    workload: LongContextM0WorkloadConfig
    vllm_args: dict[str, object] = Field(default_factory=dict)

    @field_validator("fallback_reason")
    @classmethod
    def normalize_fallback_reason(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("fallback_reason must not be empty")
        return stripped

    @field_validator("vllm_args")
    @classmethod
    def require_upstream_defaults(cls, value: dict[str, object]) -> dict[str, object]:
        if value:
            raise ValueError("M0 production-default requires empty vllm_args")
        return {}

    @model_validator(mode="after")
    def validate_role_and_model_tier(self) -> "LongContextM0Config":
        model_identity = self.model.identity()
        parameter_count = model_identity.parameter_count
        if self.evidence_role == "smoke":
            if self.model_tier != "smoke":
                raise ValueError("smoke evidence_role requires model_tier=smoke")
            if self.fallback_reason is not None:
                raise ValueError("smoke configuration must not set fallback_reason")
            return self

        if self.model_tier == "smoke":
            raise ValueError("formal evidence_role cannot use model_tier=smoke")
        if self.model_tier == "primary-7b-8b":
            if not PRIMARY_MIN_PARAMETERS <= parameter_count <= PRIMARY_MAX_PARAMETERS:
                raise ValueError("primary-7b-8b model lock must contain 7B/8B parameters")
            if self.fallback_reason is not None:
                raise ValueError("primary model must not set fallback_reason")
        else:
            if not FALLBACK_MIN_PARAMETERS <= parameter_count <= FALLBACK_MAX_PARAMETERS:
                raise ValueError("fallback-3b model lock must contain a 3B-class parameter count")
            if self.fallback_reason is None:
                raise ValueError("fallback-3b requires an explicit fallback_reason")
        return self

    def to_tuning_config(self) -> TuningConfig:
        """Adapt frozen M0 identity and workload to the existing TrialController."""
        model_identity = self.model.identity()
        timeout_ms = self.workload.request_timeout_seconds * 1000.0
        measurement_span = (self.workload.measured_requests - 1) / self.workload.request_rate
        timeout_minutes = max(
            20,
            math.ceil(15 + (measurement_span + self.workload.request_timeout_seconds) / 60),
        )
        return TuningConfig(
            model=str(self.model.local_path),
            model_revision=model_identity.revision,
            tokenizer=str(self.model.local_path),
            gpu=GPUConfig(device_ids=list(self.gpu.device_ids), count=1),
            slo=SLOConfig(ttft_ms=timeout_ms, tpot_ms=timeout_ms, e2e_ms=timeout_ms),
            constraints=Constraints(
                max_error_rate=0.0,
                max_peak_vram_mb=None,
                max_memory_utilization=1.0,
                require_no_oom=True,
                require_server_alive=True,
            ),
            workload=WorkloadConfig(
                name="longctx-v5-m0-production-default",
                dataset_name=model_identity.repository_id,
                sample_size=self.workload.measured_requests,
                prompt_length_distribution="uniform",
                warmup_requests=self.workload.warmup_requests,
                max_concurrency=self.workload.max_concurrency,
                concurrent_requests=self.workload.max_concurrency,
                request_rate=self.workload.request_rate,
                burstiness=self.workload.burstiness,
                max_tokens=self.workload.fixed_output_tokens,
                fixed_input_tokens=self.workload.fixed_input_tokens,
                fixed_output_tokens=self.workload.fixed_output_tokens,
                ignore_eos=True,
                seed=self.workload.seed,
                request_timeout_seconds=self.workload.request_timeout_seconds,
                benchmark_backend="sse",
            ),
            telemetry=TelemetryConfig(
                enabled=True,
                interval_ms=200,
                metrics_path="/metrics",
                collect_nvml=True,
                collect_energy=False,
            ),
            study=StudySettings(
                trial_budget=1,
                timeout_minutes=timeout_minutes,
                prune_enabled=False,
                n_startup_trials=0,
                seed=self.workload.seed,
                methods=["default"],
                repeat_count=1,
                top_candidates=1,
                holdout_enabled=False,
                resume=False,
            ),
            baseline=BaselineConfig(
                enabled=False,
                num_requests=self.workload.measured_requests,
                max_tokens=self.workload.fixed_output_tokens,
            ),
            adaptive_prefill=AdaptivePrefillConfig(
                enabled=False,
                decision_log_enabled=False,
            ),
            vllm_args={},
        )

    model_config = ConfigDict(extra="forbid", frozen=True)


def load_longctx_m0_config(config_path: str | Path) -> LongContextM0Config:
    """Load one duplicate-key-free long-context v5 M0 YAML file."""
    path = Path(config_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"longctx-v5 M0 config file not found: {path}")
    if path.suffix.casefold() not in {".yaml", ".yml"}:
        raise ValueError("longctx-v5 M0 config must use a .yaml or .yml suffix")
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"unable to read longctx-v5 M0 config {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("longctx-v5 M0 YAML root must be a mapping")
    if any(not isinstance(key, str) for key in payload):
        raise ValueError("longctx-v5 M0 YAML root keys must be strings")
    try:
        return LongContextM0Config.model_validate(payload)
    except ValidationError as error:
        raise ValueError(f"invalid longctx-v5 M0 configuration: {error}") from error
