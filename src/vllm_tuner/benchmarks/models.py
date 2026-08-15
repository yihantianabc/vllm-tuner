"""Typed models shared by benchmark clients and result parsers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class RequestStatus(str, Enum):
    """Terminal state of a benchmark request."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class RequestSpec:
    """A single OpenAI-compatible generation request.

    ``scheduled_at`` is an optional absolute ``perf_counter_ns`` timestamp. It is
    normally populated by the load generator rather than by callers.
    """

    request_id: str
    prompt: str
    model: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    ignore_eos: bool = False
    input_tokens: Optional[int] = None
    scheduled_at: Optional[int] = None
    endpoint: str = "/v1/completions"
    extra_body: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.input_tokens is not None and self.input_tokens < 0:
            raise ValueError("input_tokens must be non-negative")
        if not self.endpoint.startswith("/"):
            raise ValueError("endpoint must start with '/'")

    def to_payload(self, default_model: Optional[str] = None) -> dict[str, Any]:
        """Build an OpenAI-compatible streaming request body."""

        model = self.model or default_model
        if not model:
            raise ValueError("A model must be set on RequestSpec or the client")

        payload: dict[str, Any] = {
            "model": model,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.ignore_eos:
            payload["ignore_eos"] = True
        payload.update(self.extra_body)

        # Streaming is a measurement invariant and must not be disabled by an
        # accidental extra_body override.
        payload["stream"] = True
        stream_options = payload.get("stream_options")
        if not isinstance(stream_options, dict):
            stream_options = {}
        payload["stream_options"] = {**stream_options, "include_usage": True}
        return payload


@dataclass
class RequestResult:
    """Raw result and monotonic timestamps for one request.

    Timestamps are integer nanoseconds from ``time.perf_counter_ns``. Optional
    timestamps remain ``None`` when a backend does not expose the corresponding
    per-request measurement; the parser never invents an E2E timestamp.
    """

    request_id: str
    scheduled_at: Optional[int] = None
    sent_at: Optional[int] = None
    first_token_at: Optional[int] = None
    finished_at: Optional[int] = None
    input_tokens: int = 0
    output_tokens: int = 0
    token_timestamps: list[int] = field(default_factory=list)
    event_timestamps: list[int] = field(default_factory=list)
    token_timestamps_valid: bool = True
    token_timestamp_source: str = "provided"
    status: RequestStatus = RequestStatus.FAILED
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    output_text: str = ""
    http_status: Optional[int] = None
    token_count_source: Optional[str] = None
    warmup: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = RequestStatus(self.status)
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if (
            self.first_token_at is None
            and self.token_timestamps
            and not self.metadata.get("first_token_at_explicit", False)
        ):
            self.first_token_at = self.token_timestamps[0]

    @property
    def success(self) -> bool:
        """Whether the request completed without a client or server error."""

        return self.status == RequestStatus.SUCCESS

    @property
    def ttft_ns(self) -> Optional[int]:
        """Time from sending the request to the first non-empty output event."""

        if self.sent_at is None or self.first_token_at is None:
            return None
        return max(0, self.first_token_at - self.sent_at)

    @property
    def e2e_ns(self) -> Optional[int]:
        """Time from sending the request until the response stream finishes."""

        if self.sent_at is None or self.finished_at is None:
            return None
        return max(0, self.finished_at - self.sent_at)

    @property
    def tpot_ns(self) -> Optional[float]:
        """Mean decode time per output token, excluding the first token."""

        if self.first_token_at is None or self.finished_at is None:
            return None
        if self.output_tokens <= 1:
            return None
        return max(0, self.finished_at - self.first_token_at) / (self.output_tokens - 1)

    @property
    def itl_ns(self) -> list[int]:
        """Arrival gaps between adjacent output tokens when token evidence is valid."""

        if not self.token_timestamps_valid:
            return []
        return [
            max(0, current - previous)
            for previous, current in zip(self.token_timestamps, self.token_timestamps[1:])
        ]

    @property
    def inter_event_latency_ns(self) -> list[int]:
        """Arrival gaps between adjacent non-empty SSE output events."""

        return [
            max(0, current - previous)
            for previous, current in zip(self.event_timestamps, self.event_timestamps[1:])
        ]

    def to_dict(self, include_metrics: bool = True) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        data = asdict(self)
        data["status"] = self.status.value
        if include_metrics:
            data.update(
                {
                    "ttft_ms": (None if self.ttft_ns is None else self.ttft_ns / 1_000_000),
                    "tpot_ms": (None if self.tpot_ns is None else self.tpot_ns / 1_000_000),
                    "itl_ms": [value / 1_000_000 for value in self.itl_ns],
                    "inter_event_latency_ms": [
                        value / 1_000_000 for value in self.inter_event_latency_ns
                    ],
                    "e2e_ms": None if self.e2e_ns is None else self.e2e_ns / 1_000_000,
                }
            )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RequestResult":
        """Load a request result while ignoring stored derived metrics."""

        fields = {
            "request_id",
            "scheduled_at",
            "sent_at",
            "first_token_at",
            "finished_at",
            "input_tokens",
            "output_tokens",
            "token_timestamps",
            "event_timestamps",
            "token_timestamps_valid",
            "token_timestamp_source",
            "status",
            "error_type",
            "error_message",
            "output_text",
            "http_status",
            "token_count_source",
            "warmup",
            "metadata",
        }
        values = {key: value for key, value in data.items() if key in fields}
        return cls(**values)


@dataclass(frozen=True)
class SLOThresholds:
    """Optional per-request latency objectives, expressed in milliseconds."""

    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    e2e_ms: Optional[float] = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass
class BenchmarkResult:
    """A benchmark run containing raw requests and aggregate measurements."""

    backend: str
    started_at: Optional[int] = None
    finished_at: Optional[int] = None
    request_results: list[RequestResult] = field(default_factory=list)
    warmup_results: list[RequestResult] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    raw_result: dict[str, Any] = field(default_factory=dict)
    output_path: Optional[Path] = None
    command: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def metrics(self) -> dict[str, Any]:
        """Compatibility alias for aggregate metrics."""

        return self.aggregate

    @property
    def requests(self) -> list[RequestResult]:
        """Compatibility alias for measured request results."""

        return self.request_results

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation including raw requests."""

        return {
            "backend": self.backend,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "request_results": [result.to_dict() for result in self.request_results],
            "warmup_results": [result.to_dict() for result in self.warmup_results],
            "aggregate": self.aggregate,
            "raw_result": self.raw_result,
            "output_path": str(self.output_path) if self.output_path else None,
            "command": self.command,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "warnings": self.warnings,
        }
