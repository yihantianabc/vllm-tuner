"""Typed experiment and trial records shared across SLOTune components."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


def utc_now_iso() -> str:
    """Return an unambiguous UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class TrialStatus(str, Enum):
    """Lifecycle and terminal states for one server configuration."""

    CREATED = "CREATED"
    STARTING = "STARTING"
    READY = "READY"
    WARMING_UP = "WARMING_UP"
    MEASURING = "MEASURING"
    COLLECTING = "COLLECTING"
    STOPPING = "STOPPING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    INFEASIBLE = "INFEASIBLE"
    PRUNED = "PRUNED"

    @property
    def terminal(self) -> bool:
        """Return whether no later lifecycle transition is allowed."""
        return self in {
            TrialStatus.COMPLETE,
            TrialStatus.FAILED,
            TrialStatus.INFEASIBLE,
            TrialStatus.PRUNED,
        }


class EnvironmentFingerprint(BaseModel):
    """Software and hardware identity needed to interpret an experiment."""

    python_version: str
    platform: str
    packages: dict[str, Optional[str]] = Field(default_factory=dict)
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None
    gpu: list[dict[str, Any]] = Field(default_factory=list)
    cpu: Optional[str] = None
    memory_bytes: Optional[int] = None


class ModelWeightFingerprint(BaseModel):
    """Content identity for one local model weight file."""

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str

    model_config = ConfigDict(extra="forbid")


class ExperimentSpec(BaseModel):
    """Immutable inputs that define comparability and resume compatibility."""

    experiment_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    model: str
    model_revision: Optional[str] = None
    tokenizer: Optional[str] = None
    model_config_sha256: Optional[str] = None
    model_metadata_sha256: Optional[str] = None
    tokenizer_sha256: Optional[str] = None
    model_weight_files: list[ModelWeightFingerprint] = Field(default_factory=list)
    model_weights_sha256: Optional[str] = None
    trace_sha256: str
    holdout_trace_sha256: Optional[str] = None
    workload: dict[str, Any]
    slo: dict[str, Any]
    constraints: dict[str, Any] = Field(default_factory=dict)
    gpu_config: dict[str, Any] = Field(default_factory=dict)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    study: dict[str, Any] = Field(default_factory=dict)
    vllm_args: dict[str, Any] = Field(default_factory=dict)
    search_space: dict[str, Any]
    search_space_sha256: str
    experiment_config_sha256: Optional[str] = None
    seed: int
    environment: EnvironmentFingerprint
    source_commit: Optional[str] = None
    source_tree_sha256: Optional[str] = None
    dirty_worktree: bool = False
    artifact_schema_version: str = "4"
    report_artifacts: dict[str, Any] = Field(default_factory=dict)
    artifact_warnings: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class TrialResult(BaseModel):
    """Structured outcome with non-overlapping client, engine, and GPU namespaces."""

    trial_id: str
    method: str
    status: TrialStatus
    params: dict[str, Any]
    started_at: str = Field(default_factory=utc_now_iso)
    finished_at: Optional[str] = None
    measurement_seconds: Optional[float] = Field(default=None, ge=0)
    client: dict[str, Any] = Field(default_factory=dict)
    engine: dict[str, Any] = Field(default_factory=dict)
    gpu: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    failure_reason: Optional[dict[str, Any]] = None
    last_server_status: Optional[dict[str, Any]] = None
    cleanup_status: Optional[dict[str, Any]] = None
    artifacts: dict[str, str] = Field(default_factory=dict)

    @property
    def selectable(self) -> bool:
        """Return whether this trial may participate in best-candidate selection."""
        return (
            self.status == TrialStatus.COMPLETE
            and bool(self.constraints.get("feasible", False))
            and self.cleanup_status is not None
            and self.cleanup_status.get("clean") is True
        )
