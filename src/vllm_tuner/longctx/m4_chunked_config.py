"""Strict configuration for long-context v5 M4 Chunked Prefill experiments."""

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
from .m3_apc_config import M3M2NegativeArtifact
from .m3_apc_integrity import M3_APC_INTEGRITY_FILE, validate_m3_apc_artifacts

FORMAL_PROFILE_IDS = (
    "production-default",
    "native-chunk-1024",
    "native-chunk-512",
)
FORMAL_LONG_PREFILL_TOKENS = (4_096, 8_192)
FORMAL_REPEATS = 3
FORMAL_DECODE_REQUESTS = 48
FORMAL_INJECTION_OFFSETS = (17.25, 35.25, 53.25)
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


class M4M3Artifact(StrictFrozenModel):
    """The accepted sealed 20/20 M3 formal artifact required by M4."""

    experiment_id: Literal["longctx-v5-m3-apc-formal-001"]
    root: Path

    @field_validator("root", mode="before")
    @classmethod
    def parse_root(cls, value: object) -> object:
        return Path(value) if isinstance(value, str) else value

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: Path) -> Path:
        if not value.is_absolute() or value.is_symlink():
            raise ValueError("M3 formal root must be an absolute real directory")
        resolved = value.resolve(strict=False)
        if not resolved.is_dir():
            raise ValueError("M3 formal root must exist before M4")
        return resolved

    @model_validator(mode="after")
    def validate_formal(self) -> "M4M3Artifact":
        if self.root.name != self.experiment_id:
            raise ValueError("M3 formal root name must equal experiment_id")
        seal = validate_m3_apc_artifacts(self.root)
        attestation = _mapping(seal.get("attestation"), "M3 attestation")
        summary = _read_json(self.root / "summary.json")
        execution = _mapping(summary.get("execution"), "M3 execution")
        acceptance = _mapping(summary.get("acceptance"), "M3 acceptance")
        if (
            summary.get("schema_version") != "longctx-m3-apc.v1"
            or summary.get("project_line") != "longctx-v5"
            or summary.get("milestone") != "M3"
            or summary.get("evidence_role") != "formal"
            or execution.get("planned_jobs") != 20
            or execution.get("completed_jobs") != 20
            or execution.get("failed_jobs") != 0
            or acceptance.get("passed") is not True
            or summary.get("m4_started") is not False
            or attestation.get("accepted") is not True
            or attestation.get("m4_started") is not False
        ):
            raise ValueError("M4 requires the accepted sealed 20/20 M3 formal artifact")
        return self

    def identity(self) -> dict[str, object]:
        summary = _read_json(self.root / "summary.json")
        return {
            "experiment_id": self.experiment_id,
            "root": str(self.root),
            "source_commit": summary.get("source_commit"),
            "integrity_file": str(self.root / M3_APC_INTEGRITY_FILE),
        }


class M4SmokeArtifact(StrictFrozenModel):
    """Accepted same-path M4 smoke required before formal execution."""

    experiment_id: str
    root: Path

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, value: str) -> str:
        if _PORTABLE_ID.fullmatch(value) is None:
            raise ValueError("M4 smoke experiment_id must be portable")
        return value

    @field_validator("root", mode="before")
    @classmethod
    def parse_root(cls, value: object) -> object:
        return Path(value) if isinstance(value, str) else value

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: Path) -> Path:
        if not value.is_absolute() or value.is_symlink():
            raise ValueError("M4 smoke root must be an absolute real directory")
        resolved = value.resolve(strict=False)
        if not resolved.is_dir():
            raise ValueError("M4 smoke root must exist before formal execution")
        return resolved

    @model_validator(mode="after")
    def validate_smoke(self) -> "M4SmokeArtifact":
        from .m4_chunked_integrity import validate_m4_chunked_artifacts

        if self.root.name != self.experiment_id:
            raise ValueError("M4 smoke root name must equal experiment_id")
        validate_m4_chunked_artifacts(self.root)
        summary = _read_json(self.root / "summary.json")
        acceptance = _mapping(summary.get("acceptance"), "M4 smoke acceptance")
        if (
            summary.get("schema_version") != "longctx-m4-chunked-prefill.v1"
            or summary.get("project_line") != "longctx-v5"
            or summary.get("milestone") != "M4"
            or summary.get("evidence_role") != "smoke"
            or acceptance.get("passed") is not True
            or summary.get("m5_started") is not False
        ):
            raise ValueError("formal M4 requires an accepted same-path smoke artifact")
        return self


class M4ChunkedProfile(StrictFrozenModel):
    """One preregistered native vLLM Chunked Prefill profile."""

    profile_id: str
    max_num_batched_tokens: Optional[int] = Field(default=None, ge=1)
    max_num_partial_prefills: Optional[int] = Field(default=None, ge=1)
    max_long_partial_prefills: Optional[int] = Field(default=None, ge=1)
    long_prefill_token_threshold: Optional[int] = Field(default=None, ge=1)

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        if _LOWERCASE_ID.fullmatch(value) is None:
            raise ValueError("profile_id must use lowercase letters, digits, and hyphens")
        return value

    @model_validator(mode="after")
    def validate_semantics(self) -> "M4ChunkedProfile":
        expected: dict[str, tuple[Optional[int], Optional[int], Optional[int], Optional[int]]] = {
            "production-default": (None, None, None, None),
            "native-chunk-1024": (1_024, 2, 1, 2_048),
            "native-chunk-512": (512, 2, 1, 2_048),
        }
        actual = (
            self.max_num_batched_tokens,
            self.max_num_partial_prefills,
            self.max_long_partial_prefills,
            self.long_prefill_token_threshold,
        )
        if expected.get(self.profile_id) != actual:
            raise ValueError(f"M4 profile is not preregistered: {self.profile_id}")
        return self

    @property
    def production_default(self) -> bool:
        return self.profile_id == "production-default"

    @property
    def resolved_max_num_batched_tokens(self) -> int:
        return self.max_num_batched_tokens or 2_048

    def vllm_args(self) -> dict[str, object]:
        if self.production_default:
            return {}
        assert self.max_num_batched_tokens is not None
        assert self.max_num_partial_prefills is not None
        assert self.max_long_partial_prefills is not None
        assert self.long_prefill_token_threshold is not None
        return {
            "enable-chunked-prefill": True,
            "long-prefill-token-threshold": self.long_prefill_token_threshold,
            "max-long-partial-prefills": self.max_long_partial_prefills,
            "max-num-batched-tokens": self.max_num_batched_tokens,
            "max-num-partial-prefills": self.max_num_partial_prefills,
        }


class M4Protocol(StrictFrozenModel):
    """Frozen decode-stream and long-prefill injection protocol."""

    repeats: int = Field(ge=1, le=FORMAL_REPEATS)
    long_prefill_tokens: tuple[int, ...]
    decode_requests: int = Field(ge=8)
    decode_input_tokens: Literal[256]
    decode_output_tokens: Literal[256]
    decode_interval_seconds: float = Field(gt=0)
    injection_offsets_seconds: tuple[float, ...]
    long_output_tokens: Literal[32]
    measurement_seed: int = Field(ge=0)
    warmup_seed: int = Field(ge=0)
    warmup_requests: Literal[2]
    client_max_concurrency: int = Field(ge=1)
    request_timeout_seconds: float = Field(gt=0)
    burstiness: float
    ignore_eos: Literal[True]

    @model_validator(mode="after")
    def validate_protocol(self) -> "M4Protocol":
        if self.measurement_seed == self.warmup_seed:
            raise ValueError("M4 measurement and warmup seeds must be disjoint")
        if not math.isfinite(self.burstiness) or self.burstiness != 1.0:
            raise ValueError("M4 burstiness must equal 1.0")
        if not math.isfinite(self.decode_interval_seconds):
            raise ValueError("M4 decode interval must be finite")
        if not math.isfinite(self.request_timeout_seconds):
            raise ValueError("M4 request timeout must be finite")
        if tuple(sorted(set(self.long_prefill_tokens))) != self.long_prefill_tokens:
            raise ValueError("M4 long-prefill lengths must be unique and increasing")
        if any(value < 4_096 or value > 8_192 or value % 16 for value in self.long_prefill_tokens):
            raise ValueError("M4 long-prefill lengths must be block-aligned 4K through 8K")
        offsets = self.injection_offsets_seconds
        if not offsets or tuple(sorted(set(offsets))) != offsets:
            raise ValueError("M4 injection offsets must be unique and increasing")
        if any(not math.isfinite(value) or value <= 0 for value in offsets):
            raise ValueError("M4 injection offsets must be finite and positive")
        decode_span = (self.decode_requests - 1) * self.decode_interval_seconds
        if offsets[0] <= self.decode_interval_seconds * 2 or offsets[-1] >= decode_span:
            raise ValueError("M4 injections must occur inside an established decode stream")
        if self.client_max_concurrency < self.measured_request_count:
            raise ValueError("strict M4 replay requires concurrency at least trace size")
        return self

    @property
    def measured_request_count(self) -> int:
        return self.decode_requests + len(self.injection_offsets_seconds)


class LongContextM4ChunkedConfig(StrictFrozenModel):
    """One v5-only M4 native Chunked Prefill smoke or formal protocol."""

    project_line: Literal["longctx-v5"]
    milestone: Literal["M4"]
    experiment_kind: Literal["chunked-prefill-interference"]
    evidence_role: Literal["smoke", "formal"]
    model: LongContextM0ModelConfig
    runtime: LongContextM0RuntimeConfig
    artifacts: LongContextM0ArtifactConfig
    gpu: LongContextM0GPUConfig
    m1_boundaries: M2M1BoundaryArtifact
    m2_negative_artifacts: tuple[M3M2NegativeArtifact, ...]
    m3_artifact: M4M3Artifact
    smoke_artifact: Optional[M4SmokeArtifact] = None
    profiles: tuple[M4ChunkedProfile, ...]
    slo: M1CapacitySLO
    protocol: M4Protocol

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="after")
    def validate_identity_and_matrix(self) -> "LongContextM4ChunkedConfig":
        expected_m1 = (self.artifacts.root / self.m1_boundaries.experiment_id).resolve(strict=False)
        if self.m1_boundaries.root != expected_m1:
            raise ValueError("M1 boundaries must live below the fixed v5 artifact root")
        negative_ids = tuple(item.experiment_id for item in self.m2_negative_artifacts)
        if negative_ids != NEGATIVE_M2_IDS:
            raise ValueError("M4 must bind the ordered M2 smoke-002 through smoke-004 evidence")
        for item in self.m2_negative_artifacts:
            expected = (self.artifacts.root / item.experiment_id).resolve(strict=False)
            if item.root != expected:
                raise ValueError("M2 negative evidence must live below the fixed artifact root")
        expected_m3 = (self.artifacts.root / self.m3_artifact.experiment_id).resolve(strict=False)
        if self.m3_artifact.root != expected_m3:
            raise ValueError("M3 formal evidence must live below the fixed artifact root")
        profile_ids = tuple(profile.profile_id for profile in self.profiles)
        if profile_ids != FORMAL_PROFILE_IDS:
            raise ValueError("M4 profiles must keep the preregistered order and exact minimal set")
        self._validate_frozen_slo()
        if self.evidence_role == "formal":
            if self.smoke_artifact is None:
                raise ValueError("formal M4 requires a sealed accepted smoke artifact")
            expected_smoke = (self.artifacts.root / self.smoke_artifact.experiment_id).resolve(
                strict=False
            )
            if self.smoke_artifact.root != expected_smoke:
                raise ValueError("M4 smoke must live below the fixed artifact root")
            if self.protocol.long_prefill_tokens != FORMAL_LONG_PREFILL_TOKENS:
                raise ValueError("formal M4 requires exactly 4K and 8K long prefills")
            if self.protocol.repeats != FORMAL_REPEATS:
                raise ValueError("formal M4 requires exactly three paired repeats")
            if self.protocol.decode_requests != FORMAL_DECODE_REQUESTS:
                raise ValueError("formal M4 keeps the preregistered 48-request decode stream")
            if self.protocol.injection_offsets_seconds != FORMAL_INJECTION_OFFSETS:
                raise ValueError("formal M4 injection offsets changed")
            if self.total_gpu_run_count != 18:
                raise ValueError("formal M4 must remain the minimal 18-run matrix")
        else:
            if self.smoke_artifact is not None:
                raise ValueError("M4 smoke must not bind another smoke artifact")
            if self.protocol.repeats != 1 or self.protocol.long_prefill_tokens != (4_096,):
                raise ValueError("M4 smoke uses one 4K same-path repeat")
            if len(self.protocol.injection_offsets_seconds) != 1:
                raise ValueError("M4 smoke uses one in-stream long-prefill injection")
            if self.total_gpu_run_count != 3:
                raise ValueError("M4 smoke must remain the minimal three-profile check")
        return self

    def _validate_frozen_slo(self) -> None:
        source = _read_json(self.m1_boundaries.source_v1_root() / "experiment.json")
        contexts = source.get("contexts")
        if not isinstance(contexts, list):
            raise ValueError("M1 source contexts are unavailable")
        source_slo: Optional[Mapping[object, object]] = None
        for raw in contexts:
            row = _mapping(raw, "M1 source context")
            if row.get("context_id") == "context-8k":
                source_slo = _mapping(row.get("slo"), "M1 context-8k SLO")
        if source_slo is None or self.slo.model_dump(mode="json") != dict(source_slo):
            raise ValueError("M4 changed the frozen M1 context-8k SLO")

    @property
    def total_gpu_run_count(self) -> int:
        return len(self.profiles) * len(self.protocol.long_prefill_tokens) * self.protocol.repeats

    def to_tuning_config(self, profile: M4ChunkedProfile) -> TuningConfig:
        if profile not in self.profiles:
            raise ValueError("M4 trial must use a preregistered profile")
        identity = self.model.identity()
        count = self.protocol.measured_request_count
        trace_span = (self.protocol.decode_requests - 1) * self.protocol.decode_interval_seconds
        request_rate = (count - 1) / trace_span
        return TuningConfig(
            model=str(self.model.local_path),
            model_revision=identity.revision,
            tokenizer=str(self.model.local_path),
            gpu=GPUConfig(device_ids=list(self.gpu.device_ids), count=1),
            slo=SLOConfig(
                ttft_ms=self.slo.ttft_ms,
                tpot_ms=self.slo.tpot_ms,
                e2e_ms=self.slo.e2e_ms,
            ),
            constraints=Constraints(
                max_error_rate=self.slo.max_error_rate_ppm / 1_000_000,
                max_peak_vram_mb=None,
                max_memory_utilization=1.0,
                require_no_oom=True,
                require_server_alive=True,
            ),
            workload=WorkloadConfig(
                name=f"longctx-v5-m4-{profile.profile_id}",
                dataset_name=identity.repository_id,
                sample_size=count,
                prompt_length_distribution="weighted",
                warmup_requests=self.protocol.warmup_requests,
                max_concurrency=self.protocol.client_max_concurrency,
                concurrent_requests=self.protocol.client_max_concurrency,
                request_rate=request_rate,
                capacity_request_rates=[],
                capacity_repeats=1,
                burstiness=self.protocol.burstiness,
                max_tokens=self.protocol.decode_output_tokens,
                fixed_input_tokens=None,
                fixed_output_tokens=None,
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
                num_requests=count,
                max_tokens=self.protocol.decode_output_tokens,
            ),
            adaptive_prefill=AdaptivePrefillConfig(enabled=False, decision_log_enabled=False),
            vllm_args=profile.vllm_args(),
        )


def load_longctx_m4_chunked_config(config_path: str | Path) -> LongContextM4ChunkedConfig:
    """Load one duplicate-key-free long-context v5 M4 YAML file."""
    path = Path(config_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"longctx-v5 M4 config not found: {path}")
    if path.suffix.casefold() not in {".yaml", ".yml"}:
        raise ValueError("longctx-v5 M4 config must use YAML")
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"unable to read longctx-v5 M4 config {path}: {error}") from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError("longctx-v5 M4 YAML root must be a string-keyed mapping")
    normalized = dict(payload)
    for field in ("m2_negative_artifacts", "profiles"):
        if isinstance(normalized.get(field), list):
            normalized[field] = tuple(normalized[field])
    protocol = normalized.get("protocol")
    if isinstance(protocol, dict):
        protocol = dict(protocol)
        for field in ("long_prefill_tokens", "injection_offsets_seconds"):
            if isinstance(protocol.get(field), list):
                protocol[field] = tuple(protocol[field])
        normalized["protocol"] = protocol
    try:
        return LongContextM4ChunkedConfig.model_validate(normalized)
    except ValidationError as error:
        raise ValueError(f"invalid longctx-v5 M4 configuration: {error}") from error


__all__ = [
    "LongContextM4ChunkedConfig",
    "M4ChunkedProfile",
    "M4Protocol",
    "load_longctx_m4_chunked_config",
]
