"""Deterministic multi-phase traces for adaptive Prefill evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from dataclasses import replace as dataclass_replace
from typing import Any, Sequence

from .generator import generate_trace
from .trace import TraceEntry, WorkloadTrace


@dataclass(frozen=True)
class NonStationaryPhaseSpec:
    """One contiguous workload phase and its arrival process."""

    name: str
    source_profile: str
    count: int
    request_rate: float
    burstiness: float
    fixed_input_tokens: int | None = None
    fixed_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("phase name must not be empty")
        if self.count <= 0:
            raise ValueError("phase count must be positive")
        if self.request_rate <= 0:
            raise ValueError("phase request_rate must be positive")
        if self.burstiness <= 0:
            raise ValueError("phase burstiness must be positive")


DEFAULT_PILOT_PHASES: dict[str, NonStationaryPhaseSpec] = {
    "decode_heavy": NonStationaryPhaseSpec(
        name="decode_heavy",
        source_profile="chat",
        count=12,
        request_rate=12.0,
        burstiness=1.0,
        fixed_input_tokens=256,
        fixed_output_tokens=256,
    ),
    "long_prefill_burst": NonStationaryPhaseSpec(
        name="long_prefill_burst",
        source_profile="rag",
        count=8,
        request_rate=32.0,
        burstiness=0.4,
        fixed_input_tokens=4096,
        fixed_output_tokens=32,
    ),
    "mixed": NonStationaryPhaseSpec(
        name="mixed",
        source_profile="mixed",
        count=12,
        request_rate=12.0,
        burstiness=1.2,
    ),
}

CALIBRATION_PHASE_ORDER = ("decode_heavy", "long_prefill_burst", "mixed")
HELDOUT_PHASE_ORDER = ("long_prefill_burst", "mixed", "decode_heavy")


def _phase_seed(seed: int, name: str) -> int:
    """Derive an order-independent deterministic seed for one named phase."""
    return seed + sum((index + 1) * ord(character) for index, character in enumerate(name))


def generate_nonstationary_trace(
    phases: Sequence[NonStationaryPhaseSpec],
    *,
    seed: int,
    tokenizer: Any,
    phase_gap_seconds: float = 0.05,
) -> WorkloadTrace:
    """Join independently seeded phases without exposing labels to the controller."""
    if not phases:
        raise ValueError("at least one non-stationary phase is required")
    if phase_gap_seconds < 0:
        raise ValueError("phase_gap_seconds must be non-negative")
    names = [phase.name for phase in phases]
    if len(set(names)) != len(names):
        raise ValueError("phase names must be unique")

    entries: list[TraceEntry] = []
    phase_start = 0.0
    for phase_index, phase in enumerate(phases):
        generated = generate_trace(
            phase.source_profile,
            count=phase.count,
            request_rate=phase.request_rate,
            burstiness=phase.burstiness,
            seed=_phase_seed(seed, phase.name),
            tokenizer=tokenizer,
            fixed_input_tokens=phase.fixed_input_tokens,
            fixed_output_tokens=phase.fixed_output_tokens,
        )
        if phase_index:
            phase_start += phase_gap_seconds
        for request_index, entry in enumerate(generated.entries):
            entries.append(
                entry.model_copy(
                    update={
                        "request_id": f"{phase.name}-{request_index:06d}",
                        "scheduled_offset_seconds": round(
                            phase_start + entry.scheduled_offset_seconds, 9
                        ),
                        "profile": phase.name,
                        "shared_prefix_id": (
                            f"{phase.name}-shared" if entry.shared_prefix_id is not None else None
                        ),
                    }
                )
            )
        phase_start = entries[-1].scheduled_offset_seconds

    return WorkloadTrace(
        seed=seed,
        profile="nonstationary",
        request_rate=None,
        burstiness=1.0,
        entries=entries,
    )


def phase_boundaries(trace: WorkloadTrace) -> list[dict[str, float | int | str]]:
    """Summarize contiguous labeled phases for offline analysis only."""
    boundaries: list[dict[str, float | int | str]] = []
    for entry in trace.entries:
        if not boundaries or boundaries[-1]["phase"] != entry.profile:
            boundaries.append(
                {
                    "phase": entry.profile,
                    "start_seconds": entry.scheduled_offset_seconds,
                    "end_seconds": entry.scheduled_offset_seconds,
                    "requests": 1,
                }
            )
        else:
            boundaries[-1]["end_seconds"] = entry.scheduled_offset_seconds
            boundaries[-1]["requests"] = int(boundaries[-1]["requests"]) + 1
    return boundaries


def phase_manifest(phases: Sequence[NonStationaryPhaseSpec]) -> list[dict[str, Any]]:
    """Return a JSON-ready copy of the frozen phase configuration."""
    return [asdict(phase) for phase in phases]


def multiply_phase_counts(
    phases: Sequence[NonStationaryPhaseSpec], multiplier: int
) -> list[NonStationaryPhaseSpec]:
    """Expand measured requests without changing phase proportions or arrival processes."""
    if multiplier <= 0:
        raise ValueError("phase count multiplier must be positive")
    return [dataclass_replace(phase, count=phase.count * multiplier) for phase in phases]


def empirical_request_rate(trace: WorkloadTrace) -> float:
    """Return the finite trace's realized rate over its scheduled arrival span."""
    if len(trace.entries) < 2:
        raise ValueError("empirical request rate requires at least two requests")
    span = trace.entries[-1].scheduled_offset_seconds - trace.entries[0].scheduled_offset_seconds
    if span <= 0:
        raise ValueError("empirical request rate requires a positive arrival span")
    return (len(trace.entries) - 1) / span


def scale_trace_to_empirical_rate(
    trace: WorkloadTrace, target_requests_per_second: float
) -> WorkloadTrace:
    """Scale only arrival offsets while preserving exact requests and phase order."""
    if not math.isfinite(target_requests_per_second) or target_requests_per_second <= 0:
        raise ValueError("target request rate must be finite and positive")
    current_rate = empirical_request_rate(trace)
    scale = current_rate / target_requests_per_second
    origin = trace.entries[0].scheduled_offset_seconds
    entries = [
        entry.model_copy(
            update={
                "scheduled_offset_seconds": round(
                    origin + (entry.scheduled_offset_seconds - origin) * scale,
                    9,
                )
            }
        )
        for entry in trace.entries
    ]
    return trace.model_copy(
        update={
            "request_rate": target_requests_per_second,
            "entries": entries,
        }
    )
