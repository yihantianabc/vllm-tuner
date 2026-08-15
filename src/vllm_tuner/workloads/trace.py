"""Immutable JSONL workload traces with stable checksums."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TraceEntry(BaseModel):
    """One scheduled request independent of the server configuration."""

    request_id: str
    scheduled_offset_seconds: float = Field(ge=0)
    prompt: str
    input_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=1)
    profile: str
    shared_prefix_id: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class WorkloadTrace(BaseModel):
    """Ordered trace that can be reused across every candidate configuration."""

    seed: int
    profile: str
    request_rate: Optional[float] = Field(default=None, gt=0)
    burstiness: float = Field(default=1.0, gt=0)
    entries: list[TraceEntry]

    @model_validator(mode="after")
    def validate_entries(self) -> "WorkloadTrace":
        if not self.entries:
            raise ValueError("trace must contain at least one request")
        request_ids = [entry.request_id for entry in self.entries]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("trace request IDs must be unique")
        offsets = [entry.scheduled_offset_seconds for entry in self.entries]
        if offsets != sorted(offsets):
            raise ValueError("trace entries must be ordered by scheduled time")
        return self

    def iter_jsonl(self) -> Iterator[str]:
        """Yield stable JSON rows."""
        for entry in self.entries:
            yield json.dumps(
                entry.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )

    def write(self, path: str | Path) -> Path:
        """Persist the exact trace used by every trial."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(self.iter_jsonl()) + "\n", encoding="utf-8")
        return destination

    def checksum(self) -> str:
        """Hash canonical rows rather than model serialization metadata."""
        payload = "\n".join(self.iter_jsonl()) + "\n"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def read(
        cls,
        path: str | Path,
        *,
        seed: int,
        profile: str,
        request_rate: Optional[float],
        burstiness: float,
    ) -> "WorkloadTrace":
        """Load validated rows from JSONL."""
        entries: list[TraceEntry] = []
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    entries.append(TraceEntry.model_validate_json(line))
                except Exception as error:
                    raise ValueError(f"Invalid trace row {line_number}: {error}") from error
        return cls(
            seed=seed,
            profile=profile,
            request_rate=request_rate,
            burstiness=burstiness,
            entries=entries,
        )
