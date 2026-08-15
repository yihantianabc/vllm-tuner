"""Reliable vLLM process and trial lifecycle management."""

from .failures import FailureReason, FailureType, UnsafeCleanupError, classify_failure
from .controller import TrialController
from .server import (
    CleanupStatus,
    ManagedVLLMServer,
    ServerStatus,
    find_free_port,
    port_is_available,
)
from .state_machine import InvalidTransitionError, TrialStateMachine

__all__ = [
    "CleanupStatus",
    "FailureReason",
    "FailureType",
    "InvalidTransitionError",
    "ManagedVLLMServer",
    "ServerStatus",
    "TrialController",
    "TrialStateMachine",
    "UnsafeCleanupError",
    "classify_failure",
    "find_free_port",
    "port_is_available",
]
