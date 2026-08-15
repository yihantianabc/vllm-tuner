"""Named workload profiles used for capacity and holdout experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadProfile:
    """Token-length ranges and prefix-sharing behavior."""

    name: str
    input_token_range: tuple[int, int]
    output_token_range: tuple[int, int]
    shared_prefix_ratio: float = 0.0
    description: str = ""


PROFILES: dict[str, WorkloadProfile] = {
    "chat": WorkloadProfile(
        "chat", (192, 320), (96, 160), 0.0, "Short input with decode-oriented output"
    ),
    "rag": WorkloadProfile(
        "rag",
        (1792, 2304),
        (96, 160),
        0.5,
        "Long prefill with a shared retrieval prefix",
    ),
    "mixed": WorkloadProfile(
        "mixed",
        (256, 4096),
        (64, 256),
        0.25,
        "Bimodal prefill lengths for HOL analysis",
    ),
    "codegen": WorkloadProfile("codegen", (384, 640), (384, 640), 0.0, "Long decode workload"),
}


def get_profile(name: str) -> WorkloadProfile:
    """Resolve a profile by name with an actionable error."""
    try:
        return PROFILES[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unknown workload profile {name!r}; choose {sorted(PROFILES)}") from error
