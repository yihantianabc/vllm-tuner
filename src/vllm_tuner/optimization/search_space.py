"""Validated, vLLM-effective parameter search space."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from typing import Any, Optional

from vllm_tuner.config.models import TuningConfig

logger = logging.getLogger(__name__)


TUNABLE_PARAMETERS = frozenset({"gpu_memory_utilization", "max_num_seqs", "max_num_batched_tokens"})
FIXED_PARAMETERS = {
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
}


def canonical_parameter_name(name: str) -> str:
    """Normalize YAML/CLI spelling to a Python parameter name."""
    return name.strip().lstrip("-").replace("-", "_")


class VLLMSearchSpace:
    """Single-GPU search space containing only parameters consumed by vLLM."""

    DEFAULT_RANGES: dict[str, tuple[float, float, type[float]]] = {
        "gpu_memory_utilization": (0.60, 0.95, float),
    }
    DEFAULT_CATEGORICAL: dict[str, list[int]] = {
        "max_num_seqs": [8, 16, 32, 64, 128],
        "max_num_batched_tokens": [1024, 2048, 4096, 8192],
    }

    def __init__(self, config: TuningConfig, num_gpus: int = 1):
        if num_gpus != 1:
            raise ValueError("SLOTune core search supports exactly one GPU")
        self.config = config
        self.num_gpus = num_gpus
        self.ranges = {
            "gpu_memory_utilization": config.search_space.gpu_memory_utilization,
        }
        self.categorical = {
            "max_num_seqs": list(config.search_space.max_num_seqs),
            "max_num_batched_tokens": list(config.search_space.max_num_batched_tokens),
        }
        self.fixed = dict(FIXED_PARAMETERS)
        self._validate_space()
        self._validate_vllm_arg_conflicts()

    def _validate_space(self) -> None:
        """Validate bounds and obvious cross-parameter relationships."""
        low, high = self.ranges["gpu_memory_utilization"]
        if not 0 < low <= high < 1:
            raise ValueError("invalid gpu_memory_utilization range")
        for name, values in self.categorical.items():
            if not values or any(not isinstance(value, int) or value <= 0 for value in values):
                raise ValueError(f"{name} choices must be positive integers")
        if min(self.categorical["max_num_batched_tokens"]) < min(self.categorical["max_num_seqs"]):
            raise ValueError("max_num_batched_tokens must be at least max_num_seqs")

    def _validate_vllm_arg_conflicts(self) -> None:
        configured = {canonical_parameter_name(name) for name in self.config.vllm_args}
        conflicts = configured.intersection(TUNABLE_PARAMETERS | FIXED_PARAMETERS.keys())
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(
                f"vllm_args duplicates trial/fixed parameters: {names}; define each parameter once"
            )

    def should_suggest(self, param_name: str) -> bool:
        """Return whether a parameter belongs to the tunable space."""
        return canonical_parameter_name(param_name) in TUNABLE_PARAMETERS

    def get_parameter_names(self) -> list[str]:
        """Return all tunable parameter names in stable order."""
        return [
            "gpu_memory_utilization",
            "max_num_seqs",
            "max_num_batched_tokens",
        ]

    def apply_params(self, trial: Any, params: dict[str, Any]) -> dict[str, Any]:
        """Ask an Optuna-compatible trial for one valid server configuration."""
        unexpected = set(params).intersection(TUNABLE_PARAMETERS | FIXED_PARAMETERS.keys())
        if unexpected:
            raise ValueError(f"trial parameters were already supplied: {sorted(unexpected)}")
        low, high = self.ranges["gpu_memory_utilization"]
        suggested: dict[str, Any] = {
            "gpu_memory_utilization": trial.suggest_float("gpu_memory_utilization", low, high),
            "max_num_seqs": trial.suggest_categorical(
                "max_num_seqs", self.categorical["max_num_seqs"]
            ),
            "max_num_batched_tokens": trial.suggest_categorical(
                "max_num_batched_tokens", self.categorical["max_num_batched_tokens"]
            ),
            **self.fixed,
        }
        if not self.validate_params(suggested):
            raise ValueError(f"sampler produced an invalid configuration: {suggested}")
        return suggested

    def sample_random(self, rng: random.Random) -> dict[str, Any]:
        """Draw a deterministic random baseline configuration."""
        low, high = self.ranges["gpu_memory_utilization"]
        params = {
            "gpu_memory_utilization": rng.uniform(low, high),
            "max_num_seqs": rng.choice(self.categorical["max_num_seqs"]),
            "max_num_batched_tokens": rng.choice(self.categorical["max_num_batched_tokens"]),
            **self.fixed,
        }
        return params

    def get_default_params(self) -> dict[str, Any]:
        """Return only fixed parameters so vLLM owns all other defaults."""
        return dict(self.fixed)

    def get_fixed_params(self) -> dict[str, Any]:
        """Return parameters that cannot vary in the core experiment."""
        return dict(self.fixed)

    def get_bounds(self, param: str) -> Optional[tuple[Any, Any]]:
        """Return continuous bounds when available."""
        return self.ranges.get(canonical_parameter_name(param))

    def get_categories(self, param: str) -> Optional[list[Any]]:
        """Return categorical choices when available."""
        values = self.categorical.get(canonical_parameter_name(param))
        return list(values) if values is not None else None

    def validate_params(self, params: dict[str, Any]) -> bool:
        """Validate values, fixed parallelism, and unknown parameter names."""
        allowed = TUNABLE_PARAMETERS | FIXED_PARAMETERS.keys() | {"_trial_id"}
        if set(params).difference(allowed):
            logger.warning("Unknown vLLM trial parameters: %s", sorted(set(params) - allowed))
            return False

        if "gpu_memory_utilization" in params:
            value = params["gpu_memory_utilization"]
            low, high = self.ranges["gpu_memory_utilization"]
            if not isinstance(value, (int, float)) or not low <= float(value) <= high:
                return False
        for name, values in self.categorical.items():
            if name in params and params[name] not in values:
                return False
        for name, fixed_value in self.fixed.items():
            if params.get(name, fixed_value) != fixed_value:
                return False
        if (
            "max_num_batched_tokens" in params
            and "max_num_seqs" in params
            and params["max_num_batched_tokens"] < params["max_num_seqs"]
        ):
            return False
        return True

    def manifest(self) -> dict[str, Any]:
        """Return a stable serializable search-space description."""
        return {
            "ranges": self.ranges,
            "categorical": self.categorical,
            "fixed": self.fixed,
        }

    def checksum(self) -> str:
        """Hash the effective search space for safe resume validation."""
        payload = json.dumps(self.manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_search_space(config: TuningConfig, num_gpus: int = 1) -> VLLMSearchSpace:
    """Create the validated effective search space."""
    return VLLMSearchSpace(config, num_gpus)


def get_default_max_num_seqs_range() -> tuple[int, int]:
    """Return the min/max default sequence choices for compatibility."""
    values = VLLMSearchSpace.DEFAULT_CATEGORICAL["max_num_seqs"]
    return min(values), max(values)
