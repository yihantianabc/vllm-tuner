"""Define search space for vLLM parameters."""

import logging
from typing import Dict, Any, List, Tuple, Callable, Optional

from vllm_tuner.config.models import TuningConfig

logger = logging.getLogger(__name__)


class VLLMSearchSpace:
    """Define and manage search space for vLLM tuning parameters."""

    DEFAULT_RANGES = {
        "batch_size": (1, 256, int),
        "max_num_batched_tokens": (2048, 32768, int),
        "max_num_seqs": (32, 1024, int),
        "gpu_memory_utilization": (0.6, 0.99, float),
    }

    DEFAULT_CATEGORICAL = {
        "tensor_parallel_size": [1, 2, 4, 8],
        "pipeline_parallel_size": [1, 2, 4],
    }

    def __init__(self, config: TuningConfig, num_gpus: int = 1):
        self.config = config
        self.num_gpus = num_gpus
        self.ranges = self._get_ranges()
        self.categorical = self._get_categorical()

    def _get_ranges(self) -> Dict[str, Tuple[Any, Any, Callable]]:
        """Get parameter ranges, respecting config overrides."""
        ranges = {}

        for param, (default_min, default_max, dtype) in self.DEFAULT_RANGES.items():
            override = getattr(self.config.search_space, param)

            if override is not None:
                min_val, max_val = override
            else:
                min_val, max_val = default_min, default_max

            if dtype is int:
                ranges[param] = (min_val, max_val)
            else:
                ranges[param] = (min_val, max_val)

        return ranges

    def _get_categorical(self) -> Dict[str, List[Any]]:
        """Get categorical parameters, respecting config overrides."""
        categorical = {}

        for param, default_values in self.DEFAULT_CATEGORICAL.items():
            override = getattr(self.config.search_space, param)

            if override is not None:
                values = override
            else:
                values = default_values

            if param == "tensor_parallel_size":
                values = [v for v in values if v <= self.num_gpus]

            categorical[param] = values

        return categorical

    def should_suggest(self, param_name: str) -> bool:
        """Check if a parameter should be suggested based on GPU count."""
        if param_name == "tensor_parallel_size":
            return self.num_gpus > 1

        if param_name == "pipeline_parallel_size":
            return self.num_gpus > 1

        return True

    def get_parameter_names(self) -> List[str]:
        """Get list of all parameters in search space."""
        names = []

        for param in self.ranges:
            if self.should_suggest(param):
                names.append(param)

        for param in self.categorical:
            if self.should_suggest(param):
                names.append(param)

        return names

    def apply_params(self, trial, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply trial suggestions to create parameter dict."""
        trial_params = {}

        for param, (min_val, max_val) in self.ranges.items():
            if param in self.DEFAULT_RANGES:
                dtype = self.DEFAULT_RANGES[param][2]

                if not self.should_suggest(param):
                    trial_params[param] = min_val
                    continue

                if dtype is int:
                    trial_params[param] = trial.suggest_int(param, min_val, max_val)
                else:
                    trial_params[param] = trial.suggest_float(param, min_val, max_val)

        for param, values in self.categorical.items():
            if not self.should_suggest(param):
                trial_params[param] = values[0] if values else 1
                continue

            trial_params[param] = trial.suggest_categorical(param, values)

        return trial_params

    def get_fixed_params(self) -> Dict[str, Any]:
        """Get fixed parameters (values not in search space)."""
        fixed = {}

        for param in self.ranges:
            if not self.should_suggest(param):
                min_val, _ = self.ranges[param]
                dtype = self.DEFAULT_RANGES[param][2]
                fixed[param] = min_val

        for param, values in self.categorical.items():
            if not self.should_suggest(param):
                fixed[param] = values[0] if values else 1

        return fixed

    def get_bounds(self, param: str) -> Optional[Tuple[Any, Any]]:
        """Get bounds for a parameter."""
        if param in self.ranges:
            return self.ranges[param]
        return None

    def get_categories(self, param: str) -> Optional[List[Any]]:
        """Get categories for a parameter."""
        if param in self.categorical:
            return self.categorical[param]
        return None

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameter values."""
        for param, value in params.items():
            if param in self.ranges:
                min_val, max_val = self.ranges[param]
                dtype = self.DEFAULT_RANGES[param][2]

                if dtype is int:
                    if not isinstance(value, int) or value < min_val or value > max_val:
                        logger.warning(
                            f"Invalid {param}: {value}, range: [{min_val}, {max_val}]"
                        )
                        return False
                else:
                    if (
                        not isinstance(value, (float, int))
                        or value < min_val
                        or value > max_val
                    ):
                        logger.warning(
                            f"Invalid {param}: {value}, range: [{min_val}, {max_val}]"
                        )
                        return False

            if param in self.categorical:
                if value not in self.categorical[param]:
                    logger.warning(
                        f"Invalid {param}: {value}, options: {self.categorical[param]}"
                    )
                    return False

        return True


def get_search_space(
    config: TuningConfig,
    num_gpus: int = 1,
) -> VLLMSearchSpace:
    """Create search space from configuration."""
    return VLLMSearchSpace(config, num_gpus)


def get_default_batch_size_range() -> Tuple[int, int]:
    """Get default batch size range."""
    return VLLMSearchSpace.DEFAULT_RANGES["batch_size"][:2]


def get_default_max_num_seqs_range() -> Tuple[int, int]:
    """Get default max_num_seqs range."""
    return VLLMSearchSpace.DEFAULT_RANGES["max_num_seqs"][:2]
