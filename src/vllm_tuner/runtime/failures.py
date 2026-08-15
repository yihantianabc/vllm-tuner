"""Specific failure taxonomy for server startup and benchmark trials."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class FailureType(str, Enum):
    """Actionable terminal failure categories."""

    OOM = "OOM"
    PORT_IN_USE = "PORT_IN_USE"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    REQUEST_ERROR = "REQUEST_ERROR"
    STARTUP_TIMEOUT = "STARTUP_TIMEOUT"
    PROCESS_EXITED = "PROCESS_EXITED"
    TELEMETRY_ERROR = "TELEMETRY_ERROR"
    CANCELLED = "CANCELLED"
    CLEANUP_ERROR = "CLEANUP_ERROR"
    ARTIFACT_ERROR = "ARTIFACT_ERROR"
    UNKNOWN = "UNKNOWN"


class FailureReason(BaseModel):
    """Serializable failure evidence, separate from optimizer objective values."""

    type: FailureType
    message: str
    phase: Optional[str] = None
    exit_code: Optional[int] = None
    retryable: bool = False
    evidence: list[str] = Field(default_factory=list)


class UnsafeCleanupError(RuntimeError):
    """Fatal signal that cleanup could not be positively verified."""

    def __init__(self, message: str, *, result: Optional[Any] = None) -> None:
        super().__init__(message)
        self.result = result


OOM_PATTERNS = (
    re.compile(r"cuda out of memory", re.IGNORECASE),
    re.compile(r"torch\.OutOfMemoryError", re.IGNORECASE),
    re.compile(r"cublas_status_alloc_failed", re.IGNORECASE),
    re.compile(r"failed to allocate.*(?:cuda|gpu)", re.IGNORECASE),
)
PORT_PATTERNS = (
    re.compile(r"address already in use", re.IGNORECASE),
    re.compile(r"errno\s*98", re.IGNORECASE),
)
ARGUMENT_PATTERNS = (
    re.compile(r"unrecognized arguments?", re.IGNORECASE),
    re.compile(r"invalid choice", re.IGNORECASE),
    re.compile(r"argument .* expected", re.IGNORECASE),
)
MODEL_PATTERNS = (
    re.compile(r"failed to load (?:the )?model", re.IGNORECASE),
    re.compile(r"model .* (?:not found|does not exist)", re.IGNORECASE),
    re.compile(r"repository .* not found", re.IGNORECASE),
    re.compile(r"incorrect path_or_model_id", re.IGNORECASE),
)


def _matches(patterns: tuple[re.Pattern[str], ...], text: str) -> list[str]:
    return [pattern.pattern for pattern in patterns if pattern.search(text)]


def classify_failure(
    error: BaseException | str,
    *,
    log_text: str = "",
    phase: Optional[str] = None,
    exit_code: Optional[int] = None,
) -> FailureReason:
    """Classify explicit evidence without treating every RuntimeError as an OOM."""
    message = str(error)
    combined = f"{message}\n{log_text}"
    evidence: list[str] = []

    evidence = _matches(OOM_PATTERNS, combined)
    if evidence:
        kind = FailureType.OOM
    else:
        evidence = _matches(PORT_PATTERNS, combined)
        if evidence:
            kind = FailureType.PORT_IN_USE
        else:
            evidence = _matches(ARGUMENT_PATTERNS, combined)
            if evidence:
                kind = FailureType.INVALID_ARGUMENT
            else:
                evidence = _matches(MODEL_PATTERNS, combined)
                if evidence:
                    kind = FailureType.MODEL_LOAD_ERROR
                elif "timeout" in message.lower() and phase in {"STARTING", "READY"}:
                    kind = FailureType.STARTUP_TIMEOUT
                elif phase in {"WARMING_UP", "MEASURING"}:
                    kind = FailureType.REQUEST_ERROR
                elif phase == "STOPPING":
                    kind = FailureType.CLEANUP_ERROR
                elif exit_code is not None:
                    kind = FailureType.PROCESS_EXITED
                else:
                    kind = FailureType.UNKNOWN

    retryable = kind in {
        FailureType.PORT_IN_USE,
        FailureType.REQUEST_ERROR,
        FailureType.STARTUP_TIMEOUT,
        FailureType.TELEMETRY_ERROR,
    }
    return FailureReason(
        type=kind,
        message=message,
        phase=phase,
        exit_code=exit_code,
        retryable=retryable,
        evidence=evidence,
    )
