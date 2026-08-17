"""Strict M1 initialization-probe configuration for KV Planner validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from vllm_tuner.config.models import (
    AdaptivePrefillConfig,
    GPUConfig,
    TelemetryConfig,
    TuningConfig,
)

from .kv_capacity_planner import (
    ContextDistributionSpec,
    SafetyPolicy,
    StrictFrozenModel,
)
from .m0_config import (
    LongContextM0ArtifactConfig,
    LongContextM0GPUConfig,
    LongContextM0ModelConfig,
    LongContextM0RuntimeConfig,
)

_PROBE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
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


class M1InitializationProbe(StrictFrozenModel):
    probe_id: str
    role: Literal["calibration", "validation", "extrapolation"]
    gpu_memory_utilization_ppm: int = Field(gt=0, le=1_000_000)
    max_model_len: int = Field(gt=0)
    max_num_seqs: int = Field(gt=0)
    repeats: int = Field(default=1, ge=1, le=3)

    @field_validator("probe_id")
    @classmethod
    def validate_probe_id(cls, value: str) -> str:
        if _PROBE_ID.fullmatch(value) is None:
            raise ValueError("probe_id must use lowercase letters, digits, and hyphens")
        return value


class LongContextM1Config(StrictFrozenModel):
    project_line: Literal["longctx-v5"]
    milestone: Literal["M1"]
    experiment_kind: Literal["planner-initialization-validation"]
    model: LongContextM0ModelConfig
    runtime: LongContextM0RuntimeConfig
    artifacts: LongContextM0ArtifactConfig
    gpu: LongContextM0GPUConfig
    probes: tuple[M1InitializationProbe, ...]
    deployment_distribution: ContextDistributionSpec
    safety: SafetyPolicy

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="after")
    def validate_matrix(self) -> "LongContextM1Config":
        if not self.probes:
            raise ValueError("M1 probes must not be empty")
        ids = [probe.probe_id for probe in self.probes]
        if len(set(ids)) != len(ids):
            raise ValueError("M1 probe IDs must be unique")
        calibration = [probe for probe in self.probes if probe.role == "calibration"]
        validation = [probe for probe in self.probes if probe.role == "validation"]
        extrapolation = [probe for probe in self.probes if probe.role == "extrapolation"]
        calibration_runs = sum(probe.repeats for probe in calibration)
        if calibration_runs < 6 or any(probe.repeats < 2 for probe in calibration):
            raise ValueError("M1 requires at least three calibration points with two repeats each")
        calibration_utils = {probe.gpu_memory_utilization_ppm for probe in calibration}
        if len(calibration_utils) < 3:
            raise ValueError("M1 calibration requires at least three utilization points")
        calibration_lengths = {probe.max_model_len for probe in calibration}
        if len(calibration_lengths) != 1:
            raise ValueError("M1 calibration runs must share one max_model_len profile")
        if not validation:
            raise ValueError("M1 requires an in-profile held-out validation point")
        if any(probe.max_model_len not in calibration_lengths for probe in validation):
            raise ValueError("primary validation must use the calibration max_model_len profile")
        if any(probe.gpu_memory_utilization_ppm in calibration_utils for probe in validation):
            raise ValueError("primary validation utilization must be held out from calibration")
        context_lengths = {probe.max_model_len for probe in (*validation, *extrapolation)}
        if len(context_lengths) < 3:
            raise ValueError("M1 requires three validation/extrapolation context points")
        max_num_seqs_values = {probe.max_num_seqs for probe in self.probes}
        if len(max_num_seqs_values) != 1:
            raise ValueError("all M1 probes must share one max_num_seqs profile class")

        model_config_path = self.model.local_path / "config.json"
        try:
            model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"unable to read model config {model_config_path}: {error}") from error
        max_positions = model_config.get("max_position_embeddings")
        if isinstance(max_positions, bool) or not isinstance(max_positions, int):
            raise ValueError("model config max_position_embeddings must be an integer")
        if any(probe.max_model_len > max_positions for probe in self.probes):
            raise ValueError("M1 probe max_model_len exceeds model capacity")
        return self

    def to_tuning_config(self, probe: M1InitializationProbe) -> TuningConfig:
        identity = self.model.identity()
        return TuningConfig(
            model=str(self.model.local_path),
            model_revision=identity.revision,
            tokenizer=str(self.model.local_path),
            gpu=GPUConfig(device_ids=list(self.gpu.device_ids), count=1),
            telemetry=TelemetryConfig(enabled=False, collect_nvml=False),
            adaptive_prefill=AdaptivePrefillConfig(
                enabled=False,
                decision_log_enabled=False,
            ),
            vllm_args={
                "gpu-memory-utilization": probe.gpu_memory_utilization_ppm / 1_000_000,
                "max-model-len": probe.max_model_len,
                "max-num-seqs": probe.max_num_seqs,
            },
        )


def load_longctx_m1_config(config_path: str | Path) -> LongContextM1Config:
    path = Path(config_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"longctx-v5 M1 config not found: {path}")
    if path.suffix.casefold() not in {".yaml", ".yml"}:
        raise ValueError("longctx-v5 M1 config must use YAML")
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"unable to read longctx-v5 M1 config {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("longctx-v5 M1 YAML root must be a mapping")
    normalized = dict(payload)
    if isinstance(normalized.get("probes"), list):
        normalized["probes"] = tuple(normalized["probes"])
    distribution = normalized.get("deployment_distribution")
    if isinstance(distribution, dict) and isinstance(distribution.get("bins"), list):
        normalized_distribution = dict(distribution)
        normalized_distribution["bins"] = tuple(normalized_distribution["bins"])
        normalized["deployment_distribution"] = normalized_distribution
    try:
        return LongContextM1Config.model_validate(normalized)
    except ValidationError as error:
        raise ValueError(f"invalid longctx-v5 M1 configuration: {error}") from error
