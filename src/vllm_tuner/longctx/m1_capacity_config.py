"""Strict long-context v5 M1 capacity-sweep configuration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

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

M1_INITIALIZATION_SCHEMA = "longctx-m1-init.v2"
M1_INITIALIZATION_INTEGRITY_FILE = "m1-integrity.json"
FORMAL_CONTEXT_TOTAL_KV_TOKENS = frozenset({8_192, 16_384, 32_768})
FORMAL_REPEATS = 3
FORMAL_MINIMUM_MEASUREMENT_SECONDS = 180
FORMAL_MINIMUM_REQUESTS = 100
WARMUP_PROMPT_INDEX_FLOOR = 1_000_000

_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_LOWERCASE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")


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


def _json_mapping(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read initialization artifact JSON {path}: {error}") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"initialization artifact JSON must be an object: {path}")
    return value


def _mapping(value: object, field: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"initialization artifact field {field} must be a mapping")
    return value


def _json_absolute_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"initialization artifact field {field} must be an absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"initialization artifact field {field} must be an absolute path")
    return path.resolve(strict=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"initialization artifact must not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative != M1_INITIALIZATION_INTEGRITY_FILE:
            files[relative] = path
    return files


def _validate_initialization_seal(root: Path, experiment_id: str) -> None:
    integrity_path = root / M1_INITIALIZATION_INTEGRITY_FILE
    if not integrity_path.is_file() or integrity_path.is_symlink():
        raise ValueError("initialization_artifact.root must contain an M1 integrity seal")
    integrity = _json_mapping(integrity_path)
    if integrity.get("schema") != M1_INITIALIZATION_SCHEMA:
        raise ValueError("initialization artifact integrity schema mismatch")
    if integrity.get("identity") != experiment_id:
        raise ValueError("initialization artifact integrity identity mismatch")
    expected = _mapping(integrity.get("files"), "integrity.files")
    if any(not isinstance(name, str) for name in expected):
        raise ValueError("initialization artifact integrity file names must be strings")
    actual = _artifact_files(root)
    if set(expected) != set(actual):
        raise ValueError("initialization artifact integrity file set mismatch")
    for name, path in actual.items():
        record = _mapping(expected[name], f"integrity.files.{name}")
        if set(record) != {"size_bytes", "sha256"}:
            raise ValueError(f"invalid initialization artifact integrity record: {name}")
        size_bytes = record.get("size_bytes")
        digest = record.get("sha256")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes != path.stat().st_size
            or not isinstance(digest, str)
            or digest != _sha256_file(path)
        ):
            raise ValueError(f"initialization artifact checksum mismatch: {name}")

    required = {"experiment.json", "manifest.json", "summary.json"}
    missing = sorted(required - set(actual))
    if missing:
        raise ValueError("initialization artifact is missing required files: " + ", ".join(missing))
    manifest = _json_mapping(root / "manifest.json")
    summary = _json_mapping(root / "summary.json")
    if manifest.get("schema_version") != M1_INITIALIZATION_SCHEMA:
        raise ValueError("initialization artifact manifest schema mismatch")
    if summary.get("schema_version") != M1_INITIALIZATION_SCHEMA:
        raise ValueError("initialization artifact summary schema mismatch")
    if (
        manifest.get("experiment_id") != experiment_id
        or summary.get("experiment_id") != experiment_id
    ):
        raise ValueError("initialization artifact experiment identity mismatch")
    if manifest.get("source_commit") != summary.get("source_commit"):
        raise ValueError("initialization artifact source identity mismatch")
    for field in (
        "primary_error_passed",
        "extrapolation_error_passed",
        "initialization_validation_passed",
    ):
        if summary.get(field) is not True:
            raise ValueError(f"initialization artifact is not accepted: {field} is not true")
    manifest_model = _mapping(manifest.get("model"), "manifest.model")
    manifest_runtime = _mapping(manifest.get("runtime"), "manifest.runtime")
    if manifest_model.get("matches_lock") is not True:
        raise ValueError("initialization artifact model identity did not match its lock")
    if manifest_runtime.get("matches_lock") is not True:
        raise ValueError("initialization artifact runtime identity did not match its lock")


class M1InitializationArtifactBinding(StrictFrozenModel):
    """Exact, sealed, successful initialization evidence required by E1."""

    experiment_id: str
    root: Path

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, value: str) -> str:
        if _PORTABLE_ID.fullmatch(value) is None:
            raise ValueError("initialization experiment_id must be one portable path component")
        return value

    @field_validator("root", mode="before")
    @classmethod
    def parse_root(cls, value: object) -> object:
        if isinstance(value, str):
            return Path(value)
        return value

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("initialization_artifact.root must be an absolute path")
        if value.is_symlink():
            raise ValueError("initialization_artifact.root must be an existing real directory")
        resolved = value.resolve(strict=False)
        if not resolved.is_dir():
            raise ValueError("initialization_artifact.root must be an existing real directory")
        return resolved

    @model_validator(mode="after")
    def validate_sealed_success(self) -> "M1InitializationArtifactBinding":
        if self.root.name != self.experiment_id:
            raise ValueError("initialization artifact root name must equal experiment_id")
        _validate_initialization_seal(self.root, self.experiment_id)
        return self


class M1CapacityServerProfile(StrictFrozenModel):
    """Expected vLLM 0.16 production defaults, recorded but never passed as overrides."""

    name: Literal["production-default"]
    expected_gpu_memory_utilization_ppm: Literal[900_000]
    expected_max_model_len: Literal[32_768]
    expected_max_num_seqs: Literal[256]
    expected_max_num_batched_tokens: Literal[2_048]
    inherit_upstream_defaults: Literal[True]


class M1CapacitySLO(StrictFrozenModel):
    """Per-request SLO thresholds used to compute capacity goodput."""

    ttft_ms: float = Field(gt=0)
    tpot_ms: float = Field(gt=0)
    e2e_ms: float = Field(gt=0)
    max_error_rate_ppm: int = Field(ge=0, le=1_000_000)

    @field_validator("ttft_ms", "tpot_ms", "e2e_ms")
    @classmethod
    def validate_finite_latency(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("SLO latency thresholds must be finite")
        return value

    @model_validator(mode="after")
    def validate_e2e_threshold(self) -> "M1CapacitySLO":
        if self.e2e_ms < self.ttft_ms:
            raise ValueError("SLO e2e_ms must be at least ttft_ms")
        return self


class M1CapacityLoad(StrictFrozenModel):
    """One preregistered open-loop offered load."""

    load_id: str
    offered_requests_per_second: float = Field(gt=0)

    @field_validator("load_id")
    @classmethod
    def validate_load_id(cls, value: str) -> str:
        if _LOWERCASE_ID.fullmatch(value) is None:
            raise ValueError("load_id must use lowercase letters, digits, and hyphens")
        return value

    @field_validator("offered_requests_per_second")
    @classmethod
    def validate_finite_load(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("offered load must be finite")
        return value


class M1CapacityContext(StrictFrozenModel):
    """One fixed total-KV context, its SLO, and three offered loads."""

    context_id: str
    total_kv_tokens: int = Field(gt=128, le=32_768)
    output_tokens: Literal[128]
    loads: tuple[M1CapacityLoad, ...]
    slo: M1CapacitySLO

    @field_validator("context_id")
    @classmethod
    def validate_context_id(cls, value: str) -> str:
        if _LOWERCASE_ID.fullmatch(value) is None:
            raise ValueError("context_id must use lowercase letters, digits, and hyphens")
        return value

    @model_validator(mode="after")
    def validate_loads(self) -> "M1CapacityContext":
        if not self.loads:
            raise ValueError("capacity context loads must not be empty")
        ids = [load.load_id for load in self.loads]
        if len(ids) != len(set(ids)):
            raise ValueError("capacity load IDs must be unique within each context")
        rates = [load.offered_requests_per_second for load in self.loads]
        if rates != sorted(rates) or len(rates) != len(set(rates)):
            raise ValueError("capacity offered loads must be unique and strictly increasing")
        return self

    @property
    def input_tokens(self) -> int:
        """Prompt tokens that, with the fixed output, produce the requested total KV."""
        return self.total_kv_tokens - self.output_tokens


class M1CapacityMeasurementProtocol(StrictFrozenModel):
    """Long-trace duration, repeat, timeout, and disjoint prompt controls."""

    repeats: int = Field(ge=1, le=FORMAL_REPEATS)
    measurement_seconds: int = Field(ge=1)
    minimum_measured_requests: int = Field(ge=1)
    warmup_requests: int = Field(ge=1)
    warmup_seed: int = Field(ge=0)
    measurement_seed: int = Field(ge=0)
    warmup_prompt_index_offset: int = Field(ge=WARMUP_PROMPT_INDEX_FLOOR)
    client_max_concurrency: int = Field(ge=1)
    request_timeout_seconds: float = Field(gt=0)
    burstiness: float
    ignore_eos: Literal[True]

    @field_validator("request_timeout_seconds")
    @classmethod
    def validate_finite_timeout(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("request timeout must be finite")
        return value

    @field_validator("burstiness")
    @classmethod
    def require_poisson_open_loop(cls, value: float) -> float:
        if not math.isfinite(value) or value != 1.0:
            raise ValueError("formal capacity burstiness must equal 1.0")
        return value

    @model_validator(mode="after")
    def validate_disjoint_seeds(self) -> "M1CapacityMeasurementProtocol":
        if self.warmup_seed == self.measurement_seed:
            raise ValueError("warmup and measurement seeds must be disjoint across repeats")
        return self

    def warmup_seed_for_repeat(self, repeat_index: int) -> int:
        """Return the one fixed warmup seed replayed for every paired repeat."""
        self._validate_repeat_index(repeat_index)
        return self.warmup_seed

    def measurement_seed_for_repeat(self, repeat_index: int) -> int:
        """Return the one fixed measured seed replayed for every paired repeat."""
        self._validate_repeat_index(repeat_index)
        return self.measurement_seed

    def warmup_prompt_start_for_repeat(self, repeat_index: int) -> int:
        """Return the fixed warmup prompt offset replayed for every paired repeat."""
        self._validate_repeat_index(repeat_index)
        return self.warmup_prompt_index_offset

    def measured_request_count(self, load: M1CapacityLoad) -> int:
        """Cover both the minimum duration and the preregistered request floor."""
        duration_count = math.ceil(load.offered_requests_per_second * self.measurement_seconds) + 1
        return max(self.minimum_measured_requests, duration_count)

    def _validate_repeat_index(self, repeat_index: int) -> None:
        if isinstance(repeat_index, bool) or not 0 <= repeat_index < self.repeats:
            raise ValueError(f"repeat_index must be between 0 and {self.repeats - 1}")


class M1CapacityKneePolicy(StrictFrozenModel):
    """Preregistered joint overload signals and knee selection rule."""

    repeat_aggregation: Literal["median"]
    minimum_valid_repeats: int = Field(ge=1, le=FORMAL_REPEATS)
    throughput_plateau_max_gain_ppm: int = Field(gt=0, lt=1_000_000)
    queue_growth_min_requests_per_second: float = Field(gt=0)
    minimum_peak_waiting_requests: int = Field(ge=1)
    minimum_slo_attainment_ppm: int = Field(gt=0, le=1_000_000)
    minimum_achieved_to_offered_ppm: int = Field(gt=0, le=1_000_000)
    minimum_completion_ppm: int = Field(gt=0, le=1_000_000)
    max_p99_dispatch_delay_ms: float = Field(gt=0)
    maximum_preemptions_for_stable: int = Field(ge=0)
    maximum_timeouts_for_stable: int = Field(ge=0)
    require_zero_oom_events: Literal[True]
    minimum_joint_signal_repeats: int = Field(ge=1, le=FORMAL_REPEATS)
    required_prefix_cache_hits_delta: Literal[0]
    overload_rule: Literal["throughput-plateau-and-positive-queue-growth-and-slo-failure"]
    selection_rule: Literal["highest-stable-load-before-first-joint-overload"]
    no_overload_result: Literal["right-censored-above-highest-load"]
    below_lowest_result: Literal["left-censored-below-lowest-load"]

    @field_validator("queue_growth_min_requests_per_second", "max_p99_dispatch_delay_ms")
    @classmethod
    def validate_finite_knee_float(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("knee floating thresholds must be finite")
        return value


class LongContextM1CapacityConfig(StrictFrozenModel):
    """One v5-only E1 matrix; only formal matrices can satisfy M1 acceptance."""

    project_line: Literal["longctx-v5"]
    milestone: Literal["M1"]
    experiment_kind: Literal["capacity-sweep"]
    evidence_role: Literal["smoke", "pilot", "formal"]
    model: LongContextM0ModelConfig
    runtime: LongContextM0RuntimeConfig
    artifacts: LongContextM0ArtifactConfig
    gpu: LongContextM0GPUConfig
    initialization_artifact: M1InitializationArtifactBinding
    server_profile: M1CapacityServerProfile
    contexts: tuple[M1CapacityContext, ...]
    protocol: M1CapacityMeasurementProtocol
    knee_policy: M1CapacityKneePolicy

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="after")
    def validate_identity_and_matrix(self) -> "LongContextM1CapacityConfig":
        expected_initialization_root = (
            self.artifacts.root / self.initialization_artifact.experiment_id
        ).resolve(strict=False)
        if self.initialization_artifact.root != expected_initialization_root:
            raise ValueError("initialization artifact must live below the fixed v5 artifact root")
        self._validate_initialization_identity()

        if not self.contexts:
            raise ValueError("capacity contexts must not be empty")
        context_ids = [context.context_id for context in self.contexts]
        if len(context_ids) != len(set(context_ids)):
            raise ValueError("capacity context IDs must be unique")
        totals = [context.total_kv_tokens for context in self.contexts]
        if len(totals) != len(set(totals)):
            raise ValueError("capacity context total KV lengths must be unique")
        if totals != sorted(totals):
            raise ValueError("capacity contexts must be ordered by increasing total KV tokens")
        if self.knee_policy.minimum_valid_repeats > self.protocol.repeats:
            raise ValueError("knee minimum_valid_repeats cannot exceed configured repeats")
        if self.knee_policy.minimum_joint_signal_repeats != self.knee_policy.minimum_valid_repeats:
            raise ValueError("joint knee signals must use every valid repeat")
        largest_point = max(
            self.protocol.measured_request_count(load)
            for context in self.contexts
            for load in context.loads
        )
        if self.protocol.client_max_concurrency < largest_point:
            raise ValueError(
                "strict open-loop capacity measurement requires client_max_concurrency "
                "at least every measured request count"
            )

        if self.evidence_role == "formal":
            if len(self.contexts) != 3 or set(totals) != FORMAL_CONTEXT_TOTAL_KV_TOKENS:
                raise ValueError("formal M1 requires exactly the 8K, 16K, and 32K contexts")
            if any(len(context.loads) != 3 for context in self.contexts):
                raise ValueError("formal M1 requires exactly three offered loads per context")
            if self.protocol.repeats != FORMAL_REPEATS:
                raise ValueError("formal M1 requires exactly three repeats per capacity point")
            if self.protocol.measurement_seconds < FORMAL_MINIMUM_MEASUREMENT_SECONDS:
                raise ValueError("formal M1 measurement_seconds must be at least 180")
            if self.protocol.minimum_measured_requests < FORMAL_MINIMUM_REQUESTS:
                raise ValueError("formal M1 minimum_measured_requests must be at least 100")
            if self.knee_policy.minimum_valid_repeats != FORMAL_REPEATS:
                raise ValueError("formal M1 knee decisions require all three repeats")
        return self

    def _validate_initialization_identity(self) -> None:
        experiment = _json_mapping(self.initialization_artifact.root / "experiment.json")
        if (
            experiment.get("project_line") != "longctx-v5"
            or experiment.get("milestone") != "M1"
            or experiment.get("experiment_kind") != "planner-initialization-validation"
        ):
            raise ValueError("bound artifact is not longctx-v5 M1 initialization evidence")
        model = _mapping(experiment.get("model"), "experiment.model")
        runtime = _mapping(experiment.get("runtime"), "experiment.runtime")
        artifacts = _mapping(experiment.get("artifacts"), "experiment.artifacts")
        gpu = _mapping(experiment.get("gpu"), "experiment.gpu")
        if (
            _json_absolute_path(model.get("local_path"), "model.local_path")
            != self.model.local_path
        ):
            raise ValueError("capacity model path differs from initialization evidence")
        if _json_absolute_path(model.get("lock_path"), "model.lock_path") != self.model.lock_path:
            raise ValueError("capacity model lock differs from initialization evidence")
        if (
            _json_absolute_path(runtime.get("lock_path"), "runtime.lock_path")
            != self.runtime.lock_path
        ):
            raise ValueError("capacity runtime lock differs from initialization evidence")
        if _json_absolute_path(artifacts.get("root"), "artifacts.root") != self.artifacts.root:
            raise ValueError("capacity artifact root differs from initialization evidence")
        if gpu.get("count") != self.gpu.count or gpu.get("device_ids") != list(self.gpu.device_ids):
            raise ValueError("capacity GPU selection differs from initialization evidence")

        manifest = _json_mapping(self.initialization_artifact.root / "manifest.json")
        manifest_model = _mapping(manifest.get("model"), "manifest.model")
        manifest_runtime = _mapping(manifest.get("runtime"), "manifest.runtime")
        identity = self.model.identity()
        expected_model_facts = {
            "lock_path": self.model.lock_path,
            "model_dir": self.model.local_path,
            "expected_repository_id": identity.repository_id,
            "expected_revision": identity.revision,
            "expected_parameter_count": identity.parameter_count,
        }
        for field, expected in expected_model_facts.items():
            actual = manifest_model.get(field)
            if isinstance(expected, Path):
                try:
                    matches = _json_absolute_path(actual, f"manifest.model.{field}") == expected
                except ValueError:
                    matches = False
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(f"capacity model identity differs from initialization {field}")
        try:
            runtime_lock_matches = (
                _json_absolute_path(manifest_runtime.get("lock_path"), "manifest.runtime.lock_path")
                == self.runtime.lock_path
            )
        except ValueError:
            runtime_lock_matches = False
        if not runtime_lock_matches:
            raise ValueError("capacity runtime identity differs from initialization lock_path")

    @property
    def formal_acceptance_eligible(self) -> bool:
        """Return true only for the full formal 3x3x3 long-trace matrix."""
        return self.evidence_role == "formal"

    def require_formal_acceptance(self) -> None:
        """Fail closed when smoke or pilot evidence is presented for M1 acceptance."""
        if not self.formal_acceptance_eligible:
            raise ValueError(
                "smoke and pilot capacity evidence cannot satisfy formal M1 acceptance"
            )

    def to_tuning_config(
        self,
        context: M1CapacityContext,
        load: M1CapacityLoad,
        repeat_index: int,
    ) -> TuningConfig:
        """Adapt one E1 point while preserving actual upstream production defaults."""
        matching_context = next(
            (
                candidate
                for candidate in self.contexts
                if candidate.context_id == context.context_id
            ),
            None,
        )
        if matching_context != context:
            raise ValueError("context is not a preregistered member of this capacity matrix")
        matching_load = next(
            (candidate for candidate in context.loads if candidate.load_id == load.load_id),
            None,
        )
        if matching_load != load:
            raise ValueError("load is not preregistered for the selected capacity context")
        measurement_seed = self.protocol.measurement_seed_for_repeat(repeat_index)
        sample_size = self.protocol.measured_request_count(load)
        identity = self.model.identity()
        measurement_span = (sample_size - 1) / load.offered_requests_per_second
        timeout_minutes = max(
            20,
            math.ceil(15 + (measurement_span + self.protocol.request_timeout_seconds) / 60),
        )
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
                name=f"longctx-v5-m1-capacity-{context.context_id}-{load.load_id}",
                dataset_name=identity.repository_id,
                sample_size=sample_size,
                prompt_length_distribution="uniform",
                warmup_requests=self.protocol.warmup_requests,
                max_concurrency=self.protocol.client_max_concurrency,
                concurrent_requests=self.protocol.client_max_concurrency,
                request_rate=load.offered_requests_per_second,
                capacity_request_rates=[],
                capacity_repeats=1,
                burstiness=self.protocol.burstiness,
                max_tokens=context.output_tokens,
                fixed_input_tokens=context.input_tokens,
                fixed_output_tokens=context.output_tokens,
                ignore_eos=True,
                seed=measurement_seed,
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
                seed=measurement_seed,
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
            # E1 is the real upstream production default. The expected profile above is
            # evidence to verify after startup, not authorization to pass init-only flags.
            vllm_args={},
        )


def load_longctx_m1_capacity_config(config_path: str | Path) -> LongContextM1CapacityConfig:
    """Load one duplicate-key-free long-context v5 M1 capacity YAML file."""
    path = Path(config_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"longctx-v5 M1 capacity config not found: {path}")
    if path.suffix.casefold() not in {".yaml", ".yml"}:
        raise ValueError("longctx-v5 M1 capacity config must use YAML")
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"unable to read longctx-v5 M1 capacity config {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("longctx-v5 M1 capacity YAML root must be a mapping")
    if any(not isinstance(key, str) for key in payload):
        raise ValueError("longctx-v5 M1 capacity YAML root keys must be strings")

    normalized = dict(payload)
    raw_contexts = normalized.get("contexts")
    if isinstance(raw_contexts, list):
        contexts: list[object] = []
        for raw_context in raw_contexts:
            if isinstance(raw_context, dict):
                context = dict(raw_context)
                if isinstance(context.get("loads"), list):
                    context["loads"] = tuple(context["loads"])
                contexts.append(context)
            else:
                contexts.append(raw_context)
        normalized["contexts"] = tuple(contexts)
    try:
        return LongContextM1CapacityConfig.model_validate(normalized)
    except ValidationError as error:
        raise ValueError(f"invalid longctx-v5 M1 capacity configuration: {error}") from error
