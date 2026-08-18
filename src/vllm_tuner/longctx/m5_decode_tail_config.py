"""Strict configuration for long-context v5 M5 Decode-tail validation."""

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
from .m4_chunked_integrity import M4_CHUNKED_INTEGRITY_FILE, validate_m4_chunked_artifacts

FORMAL_PROFILE_IDS = ("production-default", "decode-tail-1024")
FORMAL_REPEATS = 3
FORMAL_DECODE_REQUESTS = 48
FORMAL_TARGET = {
    "cohort_id": "target",
    "prompt_seed": 2026081851,
    "arrival_seed": 2026081853,
    "warmup_seed": 2026081852,
    "long_prefill_tokens": (4_096, 8_192, 4_096),
    "injection_offsets_seconds": (17.25, 35.25, 53.25),
}
FORMAL_HELD_OUT = {
    "cohort_id": "held-out",
    "prompt_seed": 2026081951,
    "arrival_seed": 2026081953,
    "warmup_seed": 2026081952,
    "long_prefill_tokens": (8_192, 4_096, 8_192),
    "injection_offsets_seconds": (13.5, 33.0, 56.25),
}

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


class M5M4Artifact(StrictFrozenModel):
    """The accepted sealed 18/18 M4 formal artifact required by M5."""

    experiment_id: Literal["longctx-v5-m4-chunked-formal-001"]
    root: Path

    @field_validator("root", mode="before")
    @classmethod
    def parse_root(cls, value: object) -> object:
        return Path(value) if isinstance(value, str) else value

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: Path) -> Path:
        if not value.is_absolute() or value.is_symlink():
            raise ValueError("M4 formal root must be an absolute real directory")
        resolved = value.resolve(strict=False)
        if not resolved.is_dir():
            raise ValueError("M4 formal root must exist before M5")
        return resolved

    @model_validator(mode="after")
    def validate_formal(self) -> "M5M4Artifact":
        if self.root.name != self.experiment_id:
            raise ValueError("M4 formal root name must equal experiment_id")
        seal = validate_m4_chunked_artifacts(self.root)
        attestation = _mapping(seal.get("attestation"), "M4 attestation")
        summary = _read_json(self.root / "summary.json")
        execution = _mapping(summary.get("execution"), "M4 execution")
        acceptance = _mapping(summary.get("acceptance"), "M4 acceptance")
        analysis = _mapping(summary.get("analysis"), "M4 analysis")
        selection = _mapping(analysis.get("selection"), "M4 selection")
        if (
            summary.get("schema_version") != "longctx-m4-chunked-prefill.v1"
            or summary.get("project_line") != "longctx-v5"
            or summary.get("milestone") != "M4"
            or summary.get("evidence_role") != "formal"
            or execution.get("planned_jobs") != 18
            or execution.get("completed_jobs") != 18
            or execution.get("failed_jobs") != 0
            or acceptance.get("passed") is not True
            or selection.get("profile_id") != "production-default"
            or summary.get("m5_started") is not False
            or attestation.get("accepted") is not True
            or attestation.get("m5_started") is not False
        ):
            raise ValueError("M5 requires sealed M4 18/18 with production-default selection")
        return self

    def identity(self) -> dict[str, object]:
        summary = _read_json(self.root / "summary.json")
        return {
            "experiment_id": self.experiment_id,
            "root": str(self.root),
            "source_commit": summary.get("source_commit"),
            "integrity_file": str(self.root / M4_CHUNKED_INTEGRITY_FILE),
        }


class M5SmokeArtifact(StrictFrozenModel):
    """Accepted same-path M5 smoke required before formal execution."""

    experiment_id: str
    root: Path

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, value: str) -> str:
        if _PORTABLE_ID.fullmatch(value) is None:
            raise ValueError("M5 smoke experiment_id must be portable")
        return value

    @field_validator("root", mode="before")
    @classmethod
    def parse_root(cls, value: object) -> object:
        return Path(value) if isinstance(value, str) else value

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: Path) -> Path:
        if not value.is_absolute() or value.is_symlink():
            raise ValueError("M5 smoke root must be an absolute real directory")
        resolved = value.resolve(strict=False)
        if not resolved.is_dir():
            raise ValueError("M5 smoke root must exist before formal execution")
        return resolved

    @model_validator(mode="after")
    def validate_smoke(self) -> "M5SmokeArtifact":
        from .m5_decode_tail_integrity import validate_m5_decode_tail_artifacts

        if self.root.name != self.experiment_id:
            raise ValueError("M5 smoke root name must equal experiment_id")
        validate_m5_decode_tail_artifacts(self.root)
        summary = _read_json(self.root / "summary.json")
        acceptance = _mapping(summary.get("acceptance"), "M5 smoke acceptance")
        if (
            summary.get("schema_version") != "longctx-m5-decode-tail.v1"
            or summary.get("project_line") != "longctx-v5"
            or summary.get("milestone") != "M5"
            or summary.get("evidence_role") != "smoke"
            or acceptance.get("passed") is not True
            or summary.get("m6_started") is not False
        ):
            raise ValueError("formal M5 requires an accepted same-path smoke artifact")
        return self


class M5DecodeTailProfile(StrictFrozenModel):
    """One of the only two preregistered M5 native vLLM profiles."""

    profile_id: str
    long_prefill_token_threshold: Optional[int] = Field(default=None, ge=1)

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        if _LOWERCASE_ID.fullmatch(value) is None:
            raise ValueError("profile_id must use lowercase letters, digits, and hyphens")
        return value

    @model_validator(mode="after")
    def validate_semantics(self) -> "M5DecodeTailProfile":
        expected = {"production-default": None, "decode-tail-1024": 1_024}
        if expected.get(self.profile_id) != self.long_prefill_token_threshold:
            raise ValueError(f"M5 profile is not preregistered: {self.profile_id}")
        return self

    @property
    def production_default(self) -> bool:
        return self.profile_id == "production-default"

    @property
    def resolved_max_num_batched_tokens(self) -> int:
        return 2_048

    @property
    def max_num_partial_prefills(self) -> None:
        return None

    @property
    def max_long_partial_prefills(self) -> None:
        return None

    def vllm_args(self) -> dict[str, object]:
        if self.production_default:
            return {}
        return {
            "enable-chunked-prefill": True,
            "long-prefill-token-threshold": 1_024,
        }


class M5Cohort(StrictFrozenModel):
    """One frozen target or held-out mixed-prefill trace identity."""

    cohort_id: Literal["target", "held-out"]
    prompt_seed: int = Field(ge=0)
    arrival_seed: int = Field(ge=0)
    warmup_seed: int = Field(ge=0)
    long_prefill_tokens: tuple[Literal[4096, 8192], ...]
    injection_offsets_seconds: tuple[float, ...]

    @model_validator(mode="after")
    def validate_cohort(self) -> "M5Cohort":
        if len({self.prompt_seed, self.arrival_seed, self.warmup_seed}) != 3:
            raise ValueError("M5 prompt, arrival, and warmup seeds must be disjoint")
        if len(self.long_prefill_tokens) != len(self.injection_offsets_seconds):
            raise ValueError("M5 mixed-prefill lengths and injection offsets must align")
        if set(self.long_prefill_tokens) != {4_096, 8_192}:
            raise ValueError("M5 each cohort must mix both 4K and 8K long prefills")
        offsets = self.injection_offsets_seconds
        if not offsets or tuple(sorted(set(offsets))) != offsets:
            raise ValueError("M5 injection offsets must be unique and increasing")
        if any(not math.isfinite(value) or value <= 0 for value in offsets):
            raise ValueError("M5 injection offsets must be finite and positive")
        return self


class M5Protocol(StrictFrozenModel):
    """Frozen stable-decode target and held-out validation protocol."""

    repeats: int = Field(ge=1, le=FORMAL_REPEATS)
    decode_requests: int = Field(ge=12)
    decode_input_tokens: Literal[256]
    decode_output_tokens: Literal[256]
    decode_interval_seconds: float = Field(gt=0)
    decode_arrival_jitter_fraction: float = Field(ge=0, lt=0.25)
    long_output_tokens: Literal[32]
    warmup_requests: Literal[3]
    client_max_concurrency: int = Field(ge=1)
    request_timeout_seconds: float = Field(gt=0)
    ignore_eos: Literal[True]
    cohorts: tuple[M5Cohort, ...]

    @model_validator(mode="after")
    def validate_protocol(self) -> "M5Protocol":
        if not math.isfinite(self.decode_interval_seconds):
            raise ValueError("M5 decode interval must be finite")
        if not math.isfinite(self.decode_arrival_jitter_fraction):
            raise ValueError("M5 decode arrival jitter must be finite")
        if not math.isfinite(self.request_timeout_seconds):
            raise ValueError("M5 request timeout must be finite")
        ids = tuple(cohort.cohort_id for cohort in self.cohorts)
        if len(set(ids)) != len(ids):
            raise ValueError("M5 cohort identities must be unique")
        seeds = [
            seed
            for cohort in self.cohorts
            for seed in (cohort.prompt_seed, cohort.arrival_seed, cohort.warmup_seed)
        ]
        if len(set(seeds)) != len(seeds):
            raise ValueError("M5 target and held-out seed sets must be disjoint")
        decode_span = (self.decode_requests - 1) * self.decode_interval_seconds
        if any(
            cohort.injection_offsets_seconds[0] <= self.decode_interval_seconds * 2
            or cohort.injection_offsets_seconds[-1] >= decode_span
            for cohort in self.cohorts
        ):
            raise ValueError("M5 injections must occur inside an established decode stream")
        if self.client_max_concurrency < self.measured_request_count:
            raise ValueError("strict M5 replay requires concurrency at least trace size")
        return self

    @property
    def measured_request_count(self) -> int:
        return self.decode_requests + max(
            len(cohort.long_prefill_tokens) for cohort in self.cohorts
        )


class LongContextM5DecodeTailConfig(StrictFrozenModel):
    """One v5-only M5 Decode-tail smoke or formal protocol."""

    project_line: Literal["longctx-v5"]
    milestone: Literal["M5"]
    experiment_kind: Literal["decode-tail-non-inferiority"]
    evidence_role: Literal["smoke", "formal"]
    model: LongContextM0ModelConfig
    runtime: LongContextM0RuntimeConfig
    artifacts: LongContextM0ArtifactConfig
    gpu: LongContextM0GPUConfig
    m4_artifact: M5M4Artifact
    smoke_artifact: Optional[M5SmokeArtifact] = None
    profiles: tuple[M5DecodeTailProfile, ...]
    slo: M1CapacitySLO
    protocol: M5Protocol

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="after")
    def validate_identity_and_matrix(self) -> "LongContextM5DecodeTailConfig":
        expected_m4 = (self.artifacts.root / self.m4_artifact.experiment_id).resolve(strict=False)
        if self.m4_artifact.root != expected_m4:
            raise ValueError("M4 formal evidence must live below the fixed v5 artifact root")
        if tuple(profile.profile_id for profile in self.profiles) != FORMAL_PROFILE_IDS:
            raise ValueError("M5 profiles must keep the exact preregistered two-profile order")
        m4_experiment = _read_json(self.m4_artifact.root / "experiment.json")
        if self.slo.model_dump(mode="json") != m4_experiment.get("slo"):
            raise ValueError("M5 changed the frozen M4/M1 SLO")
        if self.evidence_role == "formal":
            if self.smoke_artifact is None:
                raise ValueError("formal M5 requires a sealed accepted smoke artifact")
            expected_smoke = (self.artifacts.root / self.smoke_artifact.experiment_id).resolve(
                strict=False
            )
            if self.smoke_artifact.root != expected_smoke:
                raise ValueError("M5 smoke must live below the fixed artifact root")
            if self.protocol.repeats != FORMAL_REPEATS:
                raise ValueError("formal M5 requires exactly three paired repeats")
            if self.protocol.decode_requests != FORMAL_DECODE_REQUESTS:
                raise ValueError("formal M5 keeps the 48-request stable decode stream")
            expected_cohorts = (FORMAL_TARGET, FORMAL_HELD_OUT)
            observed_cohorts = tuple(
                cohort.model_dump(mode="python") for cohort in self.protocol.cohorts
            )
            if observed_cohorts != expected_cohorts:
                raise ValueError("formal M5 target or held-out trace identity changed")
            if self.total_gpu_run_count != 12:
                raise ValueError("formal M5 must remain the frozen 12-run matrix")
        else:
            if self.smoke_artifact is not None:
                raise ValueError("M5 smoke must not bind another smoke artifact")
            if self.protocol.repeats != 1 or len(self.protocol.cohorts) != 1:
                raise ValueError("M5 smoke uses one target cohort and one paired repeat")
            if self.protocol.cohorts[0].cohort_id != "target":
                raise ValueError("M5 smoke must exercise the target path")
            if self.total_gpu_run_count != 2:
                raise ValueError("M5 smoke must remain the minimal two-profile check")
        return self

    @property
    def total_gpu_run_count(self) -> int:
        return len(self.profiles) * len(self.protocol.cohorts) * self.protocol.repeats

    def to_tuning_config(self, profile: M5DecodeTailProfile, cohort: M5Cohort) -> TuningConfig:
        if profile not in self.profiles or cohort not in self.protocol.cohorts:
            raise ValueError("M5 trial must use a preregistered profile and cohort")
        identity = self.model.identity()
        count = self.protocol.decode_requests + len(cohort.long_prefill_tokens)
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
                name=f"longctx-v5-m5-{cohort.cohort_id}-{profile.profile_id}",
                dataset_name=identity.repository_id,
                sample_size=count,
                prompt_length_distribution="weighted",
                warmup_requests=self.protocol.warmup_requests,
                max_concurrency=self.protocol.client_max_concurrency,
                concurrent_requests=self.protocol.client_max_concurrency,
                request_rate=request_rate,
                capacity_request_rates=[],
                capacity_repeats=1,
                burstiness=1.0,
                max_tokens=self.protocol.decode_output_tokens,
                fixed_input_tokens=None,
                fixed_output_tokens=None,
                ignore_eos=self.protocol.ignore_eos,
                seed=cohort.prompt_seed,
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
                seed=cohort.prompt_seed,
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


def load_longctx_m5_decode_tail_config(
    config_path: str | Path,
) -> LongContextM5DecodeTailConfig:
    """Load one duplicate-key-free long-context v5 M5 YAML file."""
    path = Path(config_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"longctx-v5 M5 config not found: {path}")
    if path.suffix.casefold() not in {".yaml", ".yml"}:
        raise ValueError("longctx-v5 M5 config must use YAML")
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"unable to read longctx-v5 M5 config {path}: {error}") from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError("longctx-v5 M5 YAML root must be a string-keyed mapping")
    normalized = dict(payload)
    if isinstance(normalized.get("profiles"), list):
        normalized["profiles"] = tuple(normalized["profiles"])
    protocol = normalized.get("protocol")
    if isinstance(protocol, dict):
        protocol = dict(protocol)
        cohorts = protocol.get("cohorts")
        if isinstance(cohorts, list):
            normalized_cohorts = []
            for raw in cohorts:
                if not isinstance(raw, dict):
                    normalized_cohorts.append(raw)
                    continue
                cohort = dict(raw)
                for field in ("long_prefill_tokens", "injection_offsets_seconds"):
                    if isinstance(cohort.get(field), list):
                        cohort[field] = tuple(cohort[field])
                normalized_cohorts.append(cohort)
            protocol["cohorts"] = tuple(normalized_cohorts)
        normalized["protocol"] = protocol
    try:
        return LongContextM5DecodeTailConfig.model_validate(normalized)
    except ValidationError as error:
        raise ValueError(f"invalid longctx-v5 M5 configuration: {error}") from error


__all__ = [
    "LongContextM5DecodeTailConfig",
    "M5Cohort",
    "M5DecodeTailProfile",
    "M5Protocol",
    "load_longctx_m5_decode_tail_config",
]
