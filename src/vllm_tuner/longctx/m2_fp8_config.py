"""Strict configuration for long-context v5 M2 FP8 KV experiments."""

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
from .m1_capacity_integrity import validate_m1_capacity_artifacts
from .m2_fp8_integrity import validate_m2_fp8_artifacts

FORMAL_CONTEXT_TOKENS = frozenset({8_192, 16_384, 32_768})
FORMAL_PROFILE_IDS = frozenset({"bf16-auto", "fp8-dynamic"})
SMOKE_PROFILE_IDS = frozenset({"bf16-auto", "fp8-dynamic", "fp8-unit-fallback"})
FORMAL_REPEATS = 3
FORMAL_MEASUREMENT_SECONDS = 180
FORMAL_MINIMUM_REQUESTS = 100

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


def _absolute_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"artifact field {field} must be an absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"artifact field {field} must be an absolute path")
    return path.resolve(strict=False)


class M2M1BoundaryArtifact(StrictFrozenModel):
    """Accepted M1 v2 boundaries and their immutable 27-run v1 source."""

    experiment_id: str
    root: Path

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, value: str) -> str:
        if _PORTABLE_ID.fullmatch(value) is None:
            raise ValueError("M1 boundary experiment_id must be one portable path component")
        return value

    @field_validator("root", mode="before")
    @classmethod
    def parse_root(cls, value: object) -> object:
        return Path(value) if isinstance(value, str) else value

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: Path) -> Path:
        if not value.is_absolute() or value.is_symlink():
            raise ValueError("M1 boundary root must be an absolute real directory")
        resolved = value.resolve(strict=False)
        if not resolved.is_dir():
            raise ValueError("M1 boundary root must exist")
        return resolved

    @model_validator(mode="after")
    def validate_accepted_boundaries(self) -> "M2M1BoundaryArtifact":
        if self.root.name != self.experiment_id:
            raise ValueError("M1 boundary root name must equal experiment_id")
        validate_m1_capacity_artifacts(self.root)
        summary = _read_json(self.root / "summary.json")
        acceptance = _mapping(summary.get("acceptance"), "acceptance")
        analysis = _mapping(summary.get("boundary_analysis"), "boundary_analysis")
        if (
            summary.get("schema_version") != "longctx-m1-capacity-boundaries.v2"
            or summary.get("project_line") != "longctx-v5"
            or summary.get("milestone") != "M1"
            or acceptance.get("passed") is not True
            or analysis.get("accepted") is not True
            or analysis.get("numeric_thresholds_modified") is not False
            or summary.get("gpu_runs_executed") != 0
        ):
            raise ValueError("M2 requires accepted, zero-GPU, threshold-preserving M1 v2 evidence")
        source = _mapping(summary.get("source"), "source")
        source_root = _absolute_path(source.get("root"), "source.root")
        validate_m1_capacity_artifacts(source_root)
        if source_root.name != source.get("experiment_id"):
            raise ValueError("M1 v2 source root identity mismatch")
        return self

    def summary(self) -> dict[str, object]:
        return _read_json(self.root / "summary.json")

    def source_v1_root(self) -> Path:
        source = _mapping(self.summary().get("source"), "source")
        return _absolute_path(source.get("root"), "source.root")


class M2SmokeArtifact(StrictFrozenModel):
    """Sealed, accepted same-path smoke required before a formal M2 matrix."""

    experiment_id: str
    root: Path

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, value: str) -> str:
        if _PORTABLE_ID.fullmatch(value) is None:
            raise ValueError("M2 smoke experiment_id must be one portable path component")
        return value

    @field_validator("root", mode="before")
    @classmethod
    def parse_root(cls, value: object) -> object:
        return Path(value) if isinstance(value, str) else value

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: Path) -> Path:
        if not value.is_absolute() or value.is_symlink():
            raise ValueError("M2 smoke root must be an absolute real directory")
        resolved = value.resolve(strict=False)
        if not resolved.is_dir():
            raise ValueError("M2 smoke root must exist before formal execution")
        return resolved

    @model_validator(mode="after")
    def validate_smoke(self) -> "M2SmokeArtifact":
        if self.root.name != self.experiment_id:
            raise ValueError("M2 smoke root name must equal experiment_id")
        validate_m2_fp8_artifacts(self.root)
        summary = _read_json(self.root / "summary.json")
        acceptance = _mapping(summary.get("acceptance"), "acceptance")
        if (
            summary.get("schema_version") != "longctx-m2-fp8.v1"
            or summary.get("project_line") != "longctx-v5"
            or summary.get("milestone") != "M2"
            or summary.get("evidence_role") != "smoke"
            or acceptance.get("passed") is not True
        ):
            raise ValueError("formal M2 requires an accepted same-path smoke artifact")
        return self

    def summary(self) -> dict[str, object]:
        return _read_json(self.root / "summary.json")


class M2FP8Profile(StrictFrozenModel):
    """One explicit KV dtype, scale source, and expected automatic backend resolution."""

    profile_id: str
    kv_cache_dtype: Literal["auto", "fp8"]
    calculate_kv_scales: bool
    scale_source: Literal["model-dtype", "dynamic-first-forward", "unit-fallback"]
    expected_attention_backend: Literal["FLASH_ATTN", "FLASHINFER"]
    backend_resolution: Literal["production-default", "automatic-fp8-fallback"]

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        if _LOWERCASE_ID.fullmatch(value) is None:
            raise ValueError("profile_id must use lowercase letters, digits, and hyphens")
        return value

    @model_validator(mode="after")
    def validate_semantics(self) -> "M2FP8Profile":
        expected = {
            "bf16-auto": (
                "auto",
                False,
                "model-dtype",
                "FLASH_ATTN",
                "production-default",
            ),
            "fp8-dynamic": (
                "fp8",
                True,
                "dynamic-first-forward",
                "FLASHINFER",
                "automatic-fp8-fallback",
            ),
            "fp8-unit-fallback": (
                "fp8",
                False,
                "unit-fallback",
                "FLASHINFER",
                "automatic-fp8-fallback",
            ),
        }
        actual = (
            self.kv_cache_dtype,
            self.calculate_kv_scales,
            self.scale_source,
            self.expected_attention_backend,
            self.backend_resolution,
        )
        if expected.get(self.profile_id) != actual:
            raise ValueError(f"M2 FP8 profile semantics are not preregistered: {self.profile_id}")
        return self

    def vllm_args(self) -> dict[str, object]:
        """Return only the intended E2 overrides; BF16 remains production default."""
        if self.profile_id == "bf16-auto":
            return {}
        arguments: dict[str, object] = {"kv-cache-dtype": "fp8"}
        if self.calculate_kv_scales:
            arguments["calculate-kv-scales"] = True
        return arguments


class M2FP8Context(StrictFrozenModel):
    """One M1-derived pre-saturation context/load reused without retuning."""

    context_id: str
    total_kv_tokens: int = Field(gt=128, le=32_768)
    output_tokens: Literal[128]
    load_id: Literal["mid"]
    offered_requests_per_second: float = Field(gt=0)
    slo: M1CapacitySLO

    @field_validator("context_id")
    @classmethod
    def validate_context_id(cls, value: str) -> str:
        if _LOWERCASE_ID.fullmatch(value) is None:
            raise ValueError("context_id must use lowercase letters, digits, and hyphens")
        return value

    @field_validator("offered_requests_per_second")
    @classmethod
    def validate_rate(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("offered_requests_per_second must be finite")
        return value

    @property
    def input_tokens(self) -> int:
        return self.total_kv_tokens - self.output_tokens


class M2FP8Protocol(StrictFrozenModel):
    """Paired trace, quality probe, timeout, and formal repeat controls."""

    repeats: int = Field(ge=1, le=FORMAL_REPEATS)
    measurement_seconds: int = Field(ge=1)
    minimum_measured_requests: int = Field(ge=2)
    quality_probe_count: int = Field(ge=1, le=4)
    quality_output_tokens: int = Field(ge=8, le=64)
    quality_seed: int = Field(ge=0)
    measurement_seed: int = Field(ge=0)
    quality_prompt_index_offset: int = Field(ge=1_000_000)
    client_max_concurrency: int = Field(ge=1)
    request_timeout_seconds: float = Field(gt=0)
    burstiness: float
    ignore_eos: Literal[True]

    @model_validator(mode="after")
    def validate_protocol(self) -> "M2FP8Protocol":
        if not math.isfinite(self.burstiness) or self.burstiness != 1.0:
            raise ValueError("M2 burstiness must equal 1.0")
        if self.quality_seed == self.measurement_seed:
            raise ValueError("quality and measurement seeds must be disjoint")
        if not math.isfinite(self.request_timeout_seconds):
            raise ValueError("request_timeout_seconds must be finite")
        return self

    def measured_request_count(self, context: M2FP8Context) -> int:
        duration_count = (
            math.ceil(context.offered_requests_per_second * self.measurement_seconds) + 1
        )
        return max(self.minimum_measured_requests, duration_count)


class LongContextM2FP8Config(StrictFrozenModel):
    """One v5-only FP8 compatibility smoke or paired formal E2 matrix."""

    project_line: Literal["longctx-v5"]
    milestone: Literal["M2"]
    experiment_kind: Literal["fp8-kv-cache"]
    evidence_role: Literal["smoke", "formal"]
    model: LongContextM0ModelConfig
    runtime: LongContextM0RuntimeConfig
    artifacts: LongContextM0ArtifactConfig
    gpu: LongContextM0GPUConfig
    m1_boundaries: M2M1BoundaryArtifact
    smoke_artifact: Optional[M2SmokeArtifact] = None
    profiles: tuple[M2FP8Profile, ...]
    contexts: tuple[M2FP8Context, ...]
    protocol: M2FP8Protocol

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="after")
    def validate_identity_and_matrix(self) -> "LongContextM2FP8Config":
        expected_m1 = (self.artifacts.root / self.m1_boundaries.experiment_id).resolve(strict=False)
        if self.m1_boundaries.root != expected_m1:
            raise ValueError("M1 boundaries must live below the fixed v5 artifact root")
        if self.smoke_artifact is not None:
            expected_smoke = (self.artifacts.root / self.smoke_artifact.experiment_id).resolve(
                strict=False
            )
            if self.smoke_artifact.root != expected_smoke:
                raise ValueError("M2 smoke must live below the fixed v5 artifact root")

        profile_ids = [profile.profile_id for profile in self.profiles]
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("M2 profile IDs must be unique")
        required_profiles = (
            FORMAL_PROFILE_IDS if self.evidence_role == "formal" else SMOKE_PROFILE_IDS
        )
        if set(profile_ids) != required_profiles:
            raise ValueError(
                f"{self.evidence_role} M2 profiles must equal {sorted(required_profiles)}"
            )
        if not self.contexts:
            raise ValueError("M2 contexts must not be empty")
        totals = [context.total_kv_tokens for context in self.contexts]
        if totals != sorted(set(totals)):
            raise ValueError("M2 contexts must be unique and ordered by increasing KV tokens")
        self._validate_m1_bindings()

        largest_count = max(
            self.protocol.measured_request_count(context) for context in self.contexts
        )
        if self.protocol.client_max_concurrency < largest_count:
            raise ValueError(
                "strict paired replay requires client_max_concurrency at least every request count"
            )
        if self.evidence_role == "formal":
            if self.smoke_artifact is None:
                raise ValueError("formal M2 requires a sealed accepted smoke_artifact")
            if set(totals) != FORMAL_CONTEXT_TOKENS or len(self.contexts) != 3:
                raise ValueError("formal M2 requires exactly the M1 8K, 16K, and 32K contexts")
            if self.protocol.repeats != FORMAL_REPEATS:
                raise ValueError("formal M2 requires exactly three paired repeats")
            if self.protocol.measurement_seconds < FORMAL_MEASUREMENT_SECONDS:
                raise ValueError("formal M2 requires at least 180 seconds per trace")
            if self.protocol.minimum_measured_requests < FORMAL_MINIMUM_REQUESTS:
                raise ValueError("formal M2 requires at least 100 requests per trial")
        elif self.smoke_artifact is not None:
            raise ValueError("smoke M2 must not bind another smoke artifact")
        return self

    def _validate_m1_bindings(self) -> None:
        summary = self.m1_boundaries.summary()
        analysis = _mapping(summary.get("boundary_analysis"), "boundary_analysis")
        raw_boundaries = analysis.get("contexts")
        if not isinstance(raw_boundaries, list):
            raise ValueError("M1 v2 boundary contexts are unavailable")
        boundaries: dict[str, tuple[int, str, float]] = {}
        for raw in raw_boundaries:
            row = _mapping(raw, "boundary context")
            joint = _mapping(row.get("joint_saturation_boundary"), "joint boundary")
            point = _mapping(joint.get("last_pre_saturation"), "last_pre_saturation")
            context_id = row.get("context_id")
            tokens = row.get("context_tokens")
            load_id = point.get("load_id")
            rate = point.get("target_offered_requests_per_second")
            if (
                not isinstance(context_id, str)
                or isinstance(tokens, bool)
                or not isinstance(tokens, int)
                or not isinstance(load_id, str)
                or isinstance(rate, bool)
                or not isinstance(rate, (int, float))
            ):
                raise ValueError("M1 v2 last-pre-saturation boundary is malformed")
            boundaries[context_id] = (tokens, load_id, float(rate))

        source_experiment = _read_json(self.m1_boundaries.source_v1_root() / "experiment.json")
        raw_contexts = source_experiment.get("contexts")
        if not isinstance(raw_contexts, list):
            raise ValueError("M1 v1 source contexts are unavailable")
        source_slos: dict[str, Mapping[object, object]] = {}
        for raw in raw_contexts:
            row = _mapping(raw, "M1 v1 context")
            context_id = row.get("context_id")
            if isinstance(context_id, str):
                source_slos[context_id] = _mapping(row.get("slo"), "M1 v1 context.slo")

        for context in self.contexts:
            expected = boundaries.get(context.context_id)
            actual = (
                context.total_kv_tokens,
                context.load_id,
                context.offered_requests_per_second,
            )
            if expected != actual:
                raise ValueError(
                    f"M2 context {context.context_id} must reuse the exact M1 v2 "
                    "last-pre-saturation boundary"
                )
            slo = source_slos.get(context.context_id)
            if slo is None or context.slo.model_dump(mode="json") != dict(slo):
                raise ValueError(f"M2 context {context.context_id} changed a frozen M1 SLO")

    @property
    def formal_acceptance_eligible(self) -> bool:
        return self.evidence_role == "formal"

    def to_tuning_config(
        self,
        profile: M2FP8Profile,
        context: M2FP8Context,
    ) -> TuningConfig:
        if profile not in self.profiles or context not in self.contexts:
            raise ValueError("M2 trial must use preregistered profile and context objects")
        sample_size = self.protocol.measured_request_count(context)
        measurement_span = (sample_size - 1) / context.offered_requests_per_second
        timeout_minutes = max(
            20,
            math.ceil(15 + (measurement_span + self.protocol.request_timeout_seconds) / 60),
        )
        identity = self.model.identity()
        return TuningConfig(
            model=str(self.model.local_path),
            model_revision=identity.revision,
            tokenizer=str(self.model.local_path),
            gpu=GPUConfig(device_ids=list(self.gpu.device_ids), count=1),
            slo=SLOConfig(
                ttft_ms=context.slo.ttft_ms,
                tpot_ms=context.slo.tpot_ms,
                e2e_ms=context.slo.e2e_ms,
            ),
            constraints=Constraints(
                max_error_rate=context.slo.max_error_rate_ppm / 1_000_000,
                max_peak_vram_mb=None,
                max_memory_utilization=1.0,
                require_no_oom=True,
                require_server_alive=True,
            ),
            workload=WorkloadConfig(
                name=f"longctx-v5-m2-{profile.profile_id}-{context.context_id}",
                dataset_name=identity.repository_id,
                sample_size=sample_size,
                prompt_length_distribution="uniform",
                warmup_requests=self.protocol.quality_probe_count,
                max_concurrency=self.protocol.client_max_concurrency,
                concurrent_requests=self.protocol.client_max_concurrency,
                request_rate=context.offered_requests_per_second,
                capacity_request_rates=[],
                capacity_repeats=1,
                burstiness=self.protocol.burstiness,
                max_tokens=context.output_tokens,
                fixed_input_tokens=context.input_tokens,
                fixed_output_tokens=context.output_tokens,
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
                timeout_minutes=timeout_minutes,
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
                max_tokens=context.output_tokens,
            ),
            adaptive_prefill=AdaptivePrefillConfig(
                enabled=False,
                decision_log_enabled=False,
            ),
            vllm_args=profile.vllm_args(),
        )


def load_longctx_m2_fp8_config(config_path: str | Path) -> LongContextM2FP8Config:
    """Load one duplicate-key-free long-context v5 M2 YAML file."""
    path = Path(config_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"longctx-v5 M2 FP8 config not found: {path}")
    if path.suffix.casefold() not in {".yaml", ".yml"}:
        raise ValueError("longctx-v5 M2 FP8 config must use YAML")
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"unable to read longctx-v5 M2 FP8 config {path}: {error}") from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError("longctx-v5 M2 FP8 YAML root must be a string-keyed mapping")
    normalized = dict(payload)
    if isinstance(normalized.get("profiles"), list):
        normalized["profiles"] = tuple(normalized["profiles"])
    if isinstance(normalized.get("contexts"), list):
        normalized["contexts"] = tuple(normalized["contexts"])
    try:
        return LongContextM2FP8Config.model_validate(normalized)
    except ValidationError as error:
        raise ValueError(f"invalid longctx-v5 M2 FP8 configuration: {error}") from error


__all__ = [
    "LongContextM2FP8Config",
    "M2FP8Context",
    "M2FP8Profile",
    "load_longctx_m2_fp8_config",
]
