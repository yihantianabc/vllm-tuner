"""Configuration models using Pydantic for validation."""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic_settings import BaseSettings


class GPUConfig(BaseModel):
    """GPU configuration."""

    device_ids: List[int] = Field(default_factory=list, description="GPU device IDs to use")
    count: int = Field(default=1, ge=1, description="Number of GPUs")

    @model_validator(mode="after")
    def validate_count(self) -> "GPUConfig":
        """Ensure count matches device_ids length if device_ids is provided."""
        if self.device_ids and self.count != len(self.device_ids):
            # Auto-update count to match device_ids
            self.count = len(self.device_ids)
        return self


class WeightedObjectives(BaseModel):
    """Multi-objective optimization weights (must sum to 100)."""

    throughput: int = Field(
        default=60, ge=0, le=100, description="Weight for throughput maximization"
    )
    latency: int = Field(default=30, ge=0, le=100, description="Weight for latency minimization")
    memory: int = Field(default=10, ge=0, le=100, description="Weight for memory efficiency")

    @field_validator("throughput", "latency", "memory")
    @classmethod
    def validate_weights(cls, v: int) -> int:
        if v < 0 or v > 100:
            raise ValueError("Weights must be between 0 and 100")
        return v

    @model_validator(mode="after")
    def check_sum(self) -> "WeightedObjectives":
        total = self.throughput + self.latency + self.memory
        if total != 100:
            raise ValueError(f"Weights must sum to 100, got {total}")
        return self


class Constraints(BaseModel):
    """Tuning constraints."""

    max_latency_ms: Optional[int] = Field(
        default=None, ge=0, description="Maximum acceptable latency in milliseconds"
    )
    max_memory_utilization: Optional[float] = Field(
        default=None, ge=0, le=1, description="Maximum memory utilization (0-1)"
    )
    throughput_min: Optional[float] = Field(
        default=None, ge=0, description="Minimum throughput requirement (requests/sec)"
    )


class SearchSpaceOverride(BaseModel):
    """Custom search space overrides."""

    batch_size: Optional[tuple[int, int]] = None
    max_num_batched_tokens: Optional[tuple[int, int]] = None
    max_num_seqs: Optional[tuple[int, int]] = None
    gpu_memory_utilization: Optional[tuple[float, float]] = None
    tensor_parallel_size: Optional[List[int]] = None
    pipeline_parallel_size: Optional[List[int]] = None


class WorkloadConfig(BaseModel):
    """Benchmark workload configuration."""

    name: str = Field(default="alpaca", description="Workload name")
    dataset_name: str = Field(default="tatsu-lab/alpaca", description="HuggingFace dataset name")
    sample_size: int = Field(default=100, ge=1, description="Number of prompts to sample")
    prompt_length_distribution: str = Field(
        default="auto",
        pattern="^(auto|uniform|weighted)$",
        description="Prompt length sampling strategy",
    )
    warmup_requests: int = Field(default=5, ge=0, description="Number of warmup requests")
    concurrent_requests: int = Field(
        default=10, ge=1, description="Number of concurrent client requests"
    )
    max_tokens: int = Field(default=256, ge=1, description="Max output tokens per request")


class StudySettings(BaseModel):
    """Optuna study settings."""

    min_trials: int = Field(default=20, ge=1, description="Minimum number of trials to run")
    timeout_minutes: int = Field(
        default=60, ge=1, description="Maximum time to run study in minutes"
    )
    storage_backend: str = Field(
        default="sqlite:///studies/optuna.db", description="Optuna storage backend URL"
    )
    prune_enabled: bool = Field(default=True, description="Enable early pruning with MedianPruner")
    n_startup_trials: int = Field(
        default=5, ge=0, description="Number of trials before pruning starts"
    )


class BaselineConfig(BaseModel):
    """Baseline generation configuration."""

    enabled: bool = Field(
        default=True, ge=True, description="Enable baseline generation before optimization"
    )
    num_requests: int = Field(
        default=1000,
        ge=1,
        description="Number of baseline requests (overrides workload.sample_size)",
    )
    max_tokens: int = Field(default=256, ge=1, description="Max output tokens per baseline request")


class TuningConfig(BaseModel):
    """Master configuration for vLLM tuner."""

    model: str = Field(
        default="gpt2", description="Model name or path (e.g., 'gpt2', 'meta-llama/Llama-2-7b')"
    )
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    objectives: WeightedObjectives = Field(default_factory=WeightedObjectives)
    constraints: Constraints = Field(default_factory=Constraints)
    search_space: SearchSpaceOverride = Field(default_factory=SearchSpaceOverride)
    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)
    study: StudySettings = Field(default_factory=StudySettings)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)

    vllm_args: dict = Field(
        default_factory=dict, description="Additional arguments to pass to vLLM server"
    )
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    objectives: WeightedObjectives = Field(default_factory=WeightedObjectives)
    constraints: Constraints = Field(default_factory=Constraints)
    search_space: SearchSpaceOverride = Field(default_factory=SearchSpaceOverride)
    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)
    study: StudySettings = Field(default_factory=StudySettings)

    vllm_args: dict = Field(
        default_factory=dict, description="Additional arguments to pass to vLLM server"
    )

    @field_validator("gpu")
    @classmethod
    def validate_gpu_config(cls, v: GPUConfig) -> GPUConfig:
        if not v.device_ids:
            if v.count == 1:
                v.device_ids = [0]
            else:
                v.device_ids = list(range(v.count))
        else:
            v.count = len(v.device_ids)
        return v

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TunerSettings(BaseSettings):
    """Runtime settings from environment variables."""

    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")
    study_output_dir: str = Field(default="studies", description="Directory for study outputs")
    html_output_dir: str = Field(default="reports", description="Directory for HTML reports")

    class Config:
        env_prefix = "VLLM_TUNER_"
        env_file = ".env"
