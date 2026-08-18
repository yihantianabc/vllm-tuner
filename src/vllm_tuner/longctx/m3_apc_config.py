"""Strict configuration for long-context v5 M3 APC experiments."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

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
from .kv_capacity_planner import StrictFrozenModel
from .m0_config import (
    LongContextM0ArtifactConfig,
    LongContextM0GPUConfig,
    LongContextM0ModelConfig,
    LongContextM0RuntimeConfig,
)
from .m1_capacity_config import M1CapacitySLO, _UniqueKeySafeLoader
from .m2_fp8_config import M2M1BoundaryArtifact
from .m2_fp8_integrity import M2_FP8_INTEGRITY_FILE, validate_m2_fp8_artifacts

FORMAL_PREFIX_TOKENS = (2_048, 4_096)
FORMAL_REUSE_PERCENTS = (0, 50, 100)
FORMAL_PROFILE_IDS = frozenset({"apc-off", "apc-on"})
FORMAL_REPEATS = 3
FORMAL_REQUESTS_PER_REUSE = 24
FORMAL_BOUNDARY_POOL_SIZES = (48, 72)
NEGATIVE_M2_IDS = (
    "longctx-v5-m2-fp8-smoke-002",
    "longctx-v5-m2-fp8-smoke-003",
    "longctx-v5-m2-fp8-smoke-004",
)

_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_LOWERCASE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read artifact JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"artifact JSON must be an object: {path}")
    return value


def _mapping(value: object, field: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"artifact field {field} must be a mapping")
    return value


class M3M2NegativeArtifact(StrictFrozenModel):
    """One sealed FP8 incompatibility artifact retained as a negative prerequisite."""

    experiment_id: str
    root: Path

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, value: str) -> str:
        if _PORTABLE_ID.fullmatch(value) is None:
            raise ValueError("M2 experiment_id must be one portable path component")
        return value

    @field_validator("root", mode="before")
    @classmethod
    def parse_root(cls, value: object) -> object:
        return Path(value) if isinstance(value, str) else value

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: Path) -> Path:
        if not value.is_absolute() or value.is_symlink():
            raise ValueError("M2 negative root must be an absolute real directory")
        resolved = value.resolve(strict=False)
        if not resolved.is_dir():
            raise ValueError("M2 negative root must exist")
        return resolved

    @model_validator(mode="after")
    def validate_negative_evidence(self) -> "M3M2NegativeArtifact":
        if self.root.name != self.experiment_id:
            raise ValueError("M2 negative root name must equal experiment_id")
        validate_m2_fp8_artifacts(self.root)
        summary = _read_json(self.root / "summary.json")
        acceptance = _mapping(summary.get("acceptance"), "M2 acceptance")
        if (
            summary.get("schema_version") != "longctx-m2-fp8.v1"
            or summary.get("project_line") != "longctx-v5"
            or summary.get("milestone") != "M2"
            or summary.get("evidence_role") != "smoke"
            or summary.get("m3_started") is not False
            or acceptance.get("passed") is not False
        ):
            raise ValueError("M3 requires sealed, rejected M2 FP8 compatibility evidence")
        return self

    def identity(self) -> dict[str, object]:
        summary = _read_json(self.root / "summary.json")
        return {
            "experiment_id": self.experiment_id,
            "root": str(self.root),
            "source_commit": summary.get("source_commit"),
            "integrity_file": str(self.root / M2_FP8_INTEGRITY_FILE),
        }


class M3SmokeArtifact(StrictFrozenModel):
    """Sealed same-path smoke required before a formal M3 matrix."""

    experiment_id: str
    root: Path

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, value: str) -> str:
        if _PORTABLE_ID.fullmatch(value) is None:
            raise ValueError("M3 smoke experiment_id must be one portable path component")
        return value

    @field_validator("root", mode="before")
    @classmethod
    def parse_root(cls, value: object) -> object:
        return Path(value) if isinstance(value, str) else value

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: Path) -> Path:
        if not value.is_absolute() or value.is_symlink():
            raise ValueError("M3 smoke root must be an absolute real directory")
        resolved = value.resolve(strict=False)
        if not resolved.is_dir():
            raise ValueError("M3 smoke root must exist before formal execution")
        return resolved

    @model_validator(mode="after")
    def validate_smoke(self) -> "M3SmokeArtifact":
        # Local import avoids a config/integrity import cycle.
        from .m3_apc_integrity import validate_m3_apc_artifacts

        if self.root.name != self.experiment_id:
            raise ValueError("M3 smoke root name must equal experiment_id")
        validate_m3_apc_artifacts(self.root)
        summary = _read_json(self.root / "summary.json")
        acceptance = _mapping(summary.get("acceptance"), "M3 smoke acceptance")
        if (
            summary.get("schema_version") != "longctx-m3-apc.v1"
            or summary.get("project_line") != "longctx-v5"
            or summary.get("milestone") != "M3"
            or summary.get("evidence_role") != "smoke"
            or acceptance.get("passed") is not True
            or summary.get("m4_started") is not False
        ):
            raise ValueError("formal M3 requires an accepted same-path smoke artifact")
        return self


class M3APCProfile(StrictFrozenModel):
    """One explicit upstream APC state with request-level cache instrumentation."""

    profile_id: str
    enable_prefix_caching: bool

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        if _LOWERCASE_ID.fullmatch(value) is None:
            raise ValueError("profile_id must use lowercase letters, digits, and hyphens")
        return value

    @model_validator(mode="after")
    def validate_semantics(self) -> "M3APCProfile":
        expected = {"apc-off": False, "apc-on": True}
        if expected.get(self.profile_id) is not self.enable_prefix_caching:
            raise ValueError(f"M3 APC profile semantics are not preregistered: {self.profile_id}")
        return self

    def vllm_args(self) -> dict[str, object]:
        arguments: dict[str, object] = {"enable-prompt-tokens-details": True}
        arguments[
            "enable-prefix-caching" if self.enable_prefix_caching else "no-enable-prefix-caching"
        ] = True
        return arguments


class M3APCContext(StrictFrozenModel):
    """The exact M1 8K last-pre-saturation service point reused by M3."""

    context_id: Literal["context-8k"]
    total_kv_tokens: Literal[8192]
    output_tokens: Literal[128]
    load_id: Literal["mid"]
    offered_requests_per_second: float = Field(gt=0)
    slo: M1CapacitySLO

    @field_validator("offered_requests_per_second")
    @classmethod
    def validate_rate(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("offered_requests_per_second must be finite")
        return value

    @property
    def input_tokens(self) -> int:
        return self.total_kv_tokens - self.output_tokens


class M3APCProtocol(StrictFrozenModel):
    """Frozen minimal paired matrix and one mechanistic pool-boundary bracket."""

    repeats: int = Field(ge=1, le=FORMAL_REPEATS)
    requests_per_reuse: int = Field(ge=2)
    reuse_percents: tuple[Literal[0, 50, 100], ...]
    prefix_tokens: tuple[int, ...]
    measurement_seed: int = Field(ge=0)
    warmup_seed: int = Field(ge=0)
    client_max_concurrency: int = Field(ge=1)
    request_timeout_seconds: float = Field(gt=0)
    burstiness: float
    ignore_eos: Literal[True]
    boundary_pool_sizes: tuple[int, ...]
    boundary_prefix_tokens: Literal[4096]
    boundary_tail_tokens: Literal[64]
    boundary_output_tokens: Literal[8]
    boundary_request_interval_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_protocol(self) -> "M3APCProtocol":
        if not math.isfinite(self.burstiness) or self.burstiness != 1.0:
            raise ValueError("M3 burstiness must equal 1.0")
        if not math.isfinite(self.request_timeout_seconds):
            raise ValueError("M3 request_timeout_seconds must be finite")
        if not math.isfinite(self.boundary_request_interval_seconds):
            raise ValueError("M3 boundary request interval must be finite")
        if self.measurement_seed == self.warmup_seed:
            raise ValueError("M3 measurement and warmup seeds must be disjoint")
        if self.requests_per_reuse % 2:
            raise ValueError("M3 requests_per_reuse must be even for exact 50% reuse")
        if self.reuse_percents != FORMAL_REUSE_PERCENTS:
            raise ValueError("M3 reuse levels must be exactly 0%, 50%, and 100%")
        if any(value <= 0 or value % 16 for value in self.prefix_tokens):
            raise ValueError("M3 prefix lengths must be positive multiples of the 16-token block")
        if tuple(sorted(set(self.prefix_tokens))) != self.prefix_tokens:
            raise ValueError("M3 prefix lengths must be unique and increasing")
        if tuple(sorted(set(self.boundary_pool_sizes))) != self.boundary_pool_sizes:
            raise ValueError("M3 boundary pool sizes must be unique and increasing")
        if self.client_max_concurrency < self.measured_request_count:
            raise ValueError("strict M3 replay requires concurrency at least the trace size")
        return self

    @property
    def measured_request_count(self) -> int:
        return self.requests_per_reuse * len(self.reuse_percents)


class LongContextM3APCConfig(StrictFrozenModel):
    """One v5-only APC smoke or formal E3 protocol."""

    project_line: Literal["longctx-v5"]
    milestone: Literal["M3"]
    experiment_kind: Literal["automatic-prefix-caching"]
    evidence_role: Literal["smoke", "formal"]
    model: LongContextM0ModelConfig
    runtime: LongContextM0RuntimeConfig
    artifacts: LongContextM0ArtifactConfig
    gpu: LongContextM0GPUConfig
    m1_boundaries: M2M1BoundaryArtifact
    m2_negative_artifacts: tuple[M3M2NegativeArtifact, ...]
    smoke_artifact: Optional[M3SmokeArtifact] = None
    profiles: tuple[M3APCProfile, ...]
    context: M3APCContext
    protocol: M3APCProtocol

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="after")
    def validate_identity_and_matrix(self) -> "LongContextM3APCConfig":
        expected_m1 = (self.artifacts.root / self.m1_boundaries.experiment_id).resolve(strict=False)
        if self.m1_boundaries.root != expected_m1:
            raise ValueError("M1 boundaries must live below the fixed v5 artifact root")
        negative_ids = tuple(item.experiment_id for item in self.m2_negative_artifacts)
        if negative_ids != NEGATIVE_M2_IDS:
            raise ValueError("M3 must bind the ordered M2 smoke-002 through smoke-004 evidence")
        for item in self.m2_negative_artifacts:
            expected = (self.artifacts.root / item.experiment_id).resolve(strict=False)
            if item.root != expected:
                raise ValueError("M2 negative evidence must live below the fixed artifact root")
        profile_ids = [profile.profile_id for profile in self.profiles]
        if set(profile_ids) != FORMAL_PROFILE_IDS or len(profile_ids) != 2:
            raise ValueError("M3 profiles must be exactly apc-off and apc-on")
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("M3 profile IDs must be unique")
        self._validate_m1_binding()
        if self.evidence_role == "formal":
            if self.smoke_artifact is None:
                raise ValueError("formal M3 requires a sealed accepted smoke artifact")
            expected_smoke = (self.artifacts.root / self.smoke_artifact.experiment_id).resolve(
                strict=False
            )
            if self.smoke_artifact.root != expected_smoke:
                raise ValueError("M3 smoke must live below the fixed v5 artifact root")
            if self.protocol.prefix_tokens != FORMAL_PREFIX_TOKENS:
                raise ValueError("formal M3 requires exactly 2K and 4K prefixes")
            if self.protocol.repeats != FORMAL_REPEATS:
                raise ValueError("formal M3 requires exactly three paired repeats")
            if self.protocol.requests_per_reuse != FORMAL_REQUESTS_PER_REUSE:
                raise ValueError("formal M3 keeps the minimal 24 requests per reuse cohort")
            if self.protocol.boundary_pool_sizes != FORMAL_BOUNDARY_POOL_SIZES:
                raise ValueError("formal M3 requires the preregistered 48/72 prefix-pool bracket")
            if self.core_run_count != 18 or self.total_gpu_run_count != 20:
                raise ValueError("formal M3 must remain the minimal 18+2 GPU-run protocol")
        else:
            if self.smoke_artifact is not None:
                raise ValueError("M3 smoke must not bind another smoke artifact")
            if len(self.protocol.prefix_tokens) != 1:
                raise ValueError("M3 smoke uses one prefix length on the same execution path")
            if len(self.protocol.boundary_pool_sizes) != 1:
                raise ValueError("M3 smoke uses one small same-path boundary probe")
        return self

    def _validate_m1_binding(self) -> None:
        summary = self.m1_boundaries.summary()
        analysis = _mapping(summary.get("boundary_analysis"), "boundary_analysis")
        raw_contexts = analysis.get("contexts")
        if not isinstance(raw_contexts, list):
            raise ValueError("M1 v2 boundary contexts are unavailable")
        selected: Optional[tuple[int, str, float]] = None
        for raw in raw_contexts:
            row = _mapping(raw, "boundary context")
            if row.get("context_id") != self.context.context_id:
                continue
            joint = _mapping(row.get("joint_saturation_boundary"), "joint boundary")
            point = _mapping(joint.get("last_pre_saturation"), "last_pre_saturation")
            tokens = row.get("context_tokens")
            load_id = point.get("load_id")
            rate = point.get("target_offered_requests_per_second")
            if (
                isinstance(tokens, bool)
                or not isinstance(tokens, int)
                or not isinstance(load_id, str)
                or isinstance(rate, bool)
                or not isinstance(rate, (int, float))
            ):
                raise ValueError("M1 v2 8K last-pre-saturation boundary is malformed")
            selected = (tokens, load_id, float(rate))
        actual = (
            self.context.total_kv_tokens,
            self.context.load_id,
            self.context.offered_requests_per_second,
        )
        if selected != actual:
            raise ValueError("M3 must reuse the exact M1 v2 8K last-pre-saturation boundary")

        source = _read_json(self.m1_boundaries.source_v1_root() / "experiment.json")
        contexts = source.get("contexts")
        if not isinstance(contexts, list):
            raise ValueError("M1 v1 source contexts are unavailable")
        source_slo: Optional[Mapping[object, object]] = None
        for raw in contexts:
            row = _mapping(raw, "M1 v1 context")
            if row.get("context_id") == self.context.context_id:
                source_slo = _mapping(row.get("slo"), "M1 v1 context.slo")
        if source_slo is None or self.context.slo.model_dump(mode="json") != dict(source_slo):
            raise ValueError("M3 changed the frozen M1 8K SLO")

    @property
    def core_run_count(self) -> int:
        # APC-off has no warm state; APC-on runs target-prefix-cold and warm.
        cells_per_prefix = 3
        return len(self.protocol.prefix_tokens) * cells_per_prefix * self.protocol.repeats

    @property
    def total_gpu_run_count(self) -> int:
        return self.core_run_count + len(self.protocol.boundary_pool_sizes)

    def to_tuning_config(
        self,
        profile: M3APCProfile,
        *,
        warmup_requests: int,
        boundary: bool = False,
    ) -> TuningConfig:
        if profile not in self.profiles:
            raise ValueError("M3 trial must use a preregistered APC profile")
        if boundary and not profile.enable_prefix_caching:
            raise ValueError("M3 prefix-pool boundary is only defined with APC enabled")
        sample_size = (
            max(self.protocol.boundary_pool_sizes)
            if boundary
            else self.protocol.measured_request_count
        )
        output_tokens = (
            self.protocol.boundary_output_tokens if boundary else self.context.output_tokens
        )
        input_tokens = (
            self.protocol.boundary_prefix_tokens + self.protocol.boundary_tail_tokens
            if boundary
            else self.context.input_tokens
        )
        identity = self.model.identity()
        return TuningConfig(
            model=str(self.model.local_path),
            model_revision=identity.revision,
            tokenizer=str(self.model.local_path),
            gpu=GPUConfig(device_ids=list(self.gpu.device_ids), count=1),
            slo=SLOConfig(
                ttft_ms=self.context.slo.ttft_ms,
                tpot_ms=self.context.slo.tpot_ms,
                e2e_ms=self.context.slo.e2e_ms,
            ),
            constraints=Constraints(
                max_error_rate=self.context.slo.max_error_rate_ppm / 1_000_000,
                max_peak_vram_mb=None,
                max_memory_utilization=1.0,
                require_no_oom=True,
                require_server_alive=True,
            ),
            workload=WorkloadConfig(
                name=f"longctx-v5-m3-{profile.profile_id}",
                dataset_name=identity.repository_id,
                sample_size=sample_size,
                prompt_length_distribution="uniform",
                warmup_requests=warmup_requests,
                max_concurrency=max(self.protocol.client_max_concurrency, sample_size),
                concurrent_requests=max(self.protocol.client_max_concurrency, sample_size),
                request_rate=self.context.offered_requests_per_second,
                capacity_request_rates=[],
                capacity_repeats=1,
                burstiness=self.protocol.burstiness,
                max_tokens=output_tokens,
                fixed_input_tokens=input_tokens,
                fixed_output_tokens=output_tokens,
                ignore_eos=self.protocol.ignore_eos,
                seed=self.protocol.measurement_seed,
                request_timeout_seconds=self.protocol.request_timeout_seconds,
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
                timeout_minutes=30,
                prune_enabled=False,
                n_startup_trials=0,
                seed=self.protocol.measurement_seed,
                methods=["default"],
                repeat_count=1,
                top_candidates=1,
                holdout_enabled=False,
                resume=False,
            ),
            baseline=BaselineConfig(
                enabled=False,
                num_requests=sample_size,
                max_tokens=output_tokens,
            ),
            adaptive_prefill=AdaptivePrefillConfig(
                enabled=False,
                decision_log_enabled=False,
            ),
            vllm_args=profile.vllm_args(),
        )


def load_longctx_m3_apc_config(config_path: str | Path) -> LongContextM3APCConfig:
    """Load one duplicate-key-free long-context v5 M3 YAML file."""
    path = Path(config_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"longctx-v5 M3 APC config not found: {path}")
    if path.suffix.casefold() not in {".yaml", ".yml"}:
        raise ValueError("longctx-v5 M3 APC config must use YAML")
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"unable to read longctx-v5 M3 APC config {path}: {error}") from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError("longctx-v5 M3 APC YAML root must be a string-keyed mapping")
    normalized = dict(payload)
    for field in ("m2_negative_artifacts", "profiles"):
        if isinstance(normalized.get(field), list):
            normalized[field] = tuple(normalized[field])
    protocol = normalized.get("protocol")
    if isinstance(protocol, dict):
        protocol = dict(protocol)
        for field in ("reuse_percents", "prefix_tokens", "boundary_pool_sizes"):
            if isinstance(protocol.get(field), list):
                protocol[field] = tuple(protocol[field])
        normalized["protocol"] = protocol
    try:
        return LongContextM3APCConfig.model_validate(normalized)
    except ValidationError as error:
        raise ValueError(f"invalid longctx-v5 M3 APC configuration: {error}") from error


__all__ = [
    "LongContextM3APCConfig",
    "M3APCContext",
    "M3APCProfile",
    "M3APCProtocol",
    "load_longctx_m3_apc_config",
]
