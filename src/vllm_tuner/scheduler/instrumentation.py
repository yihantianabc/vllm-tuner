"""vLLM-independent signal collection and decision-log records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol


class RequestView(Protocol):
    """Request attributes used by the scheduler instrumentation."""

    @property
    def request_id(self) -> str: ...

    @property
    def arrival_time(self) -> float: ...

    @property
    def num_prompt_tokens(self) -> int: ...

    @property
    def num_computed_tokens(self) -> int: ...

    @property
    def num_output_tokens(self) -> int: ...


@dataclass(frozen=True)
class SchedulerSignals:
    """The three controller signals plus queue-size diagnostics."""

    timestamp: float
    decode_backlog: int
    oldest_prefill_wait_ms: float
    kv_cache_usage: float
    running_requests: int
    waiting_requests: int
    prefill_request_ids: frozenset[str]


@dataclass(frozen=True)
class SchedulerStepRecord:
    """One auditable scheduler decision and its observed outcome."""

    timestamp: float
    step_index: int
    controller_state: str
    decode_backlog: int
    oldest_prefill_wait_ms: float
    kv_cache_usage: float
    prefill_cap: int
    scheduled_decode_tokens: int
    scheduled_prefill_tokens: int
    total_scheduled_tokens: int
    running_requests: int
    waiting_requests: int
    preemption_delta: int
    scheduler_cpu_time_us: float
    reason_code: str


def is_prefill_request(request: RequestView) -> bool:
    """Return whether a request has not produced its first output token."""
    return request.num_output_tokens == 0


def collect_scheduler_signals(
    requests: Mapping[str, RequestView],
    *,
    now: float,
    kv_cache_usage: float,
    running_requests: int,
    waiting_requests: int,
) -> SchedulerSignals:
    """Collect controller inputs without mutating vLLM request or queue state."""
    decode_backlog = 0
    oldest_prefill_wait_ms = 0.0
    prefill_request_ids: set[str] = set()
    for request_id, request in requests.items():
        if is_prefill_request(request):
            prefill_request_ids.add(request_id)
            wait_ms = max(0.0, (now - request.arrival_time) * 1000.0)
            oldest_prefill_wait_ms = max(oldest_prefill_wait_ms, wait_ms)
        else:
            decode_backlog += 1
    return SchedulerSignals(
        timestamp=now,
        decode_backlog=decode_backlog,
        oldest_prefill_wait_ms=oldest_prefill_wait_ms,
        kv_cache_usage=min(1.0, max(0.0, kv_cache_usage)),
        running_requests=running_requests,
        waiting_requests=waiting_requests,
        prefill_request_ids=frozenset(prefill_request_ids),
    )


def split_scheduled_tokens(
    num_scheduled_tokens: Mapping[str, int],
    prefill_request_ids: frozenset[str],
) -> tuple[int, int]:
    """Split a stock SchedulerOutput into decode and prefill token totals."""
    scheduled_prefill_tokens = sum(
        tokens
        for request_id, tokens in num_scheduled_tokens.items()
        if request_id in prefill_request_ids
    )
    total = sum(num_scheduled_tokens.values())
    return total - scheduled_prefill_tokens, scheduled_prefill_tokens


def has_unfinished_prefill(
    requests: Mapping[str, RequestView], request_ids: frozenset[str]
) -> bool:
    """Return whether pre-step Prefill requests still have prompt work."""
    return any(
        request_id in requests
        and requests[request_id].num_output_tokens == 0
        and requests[request_id].num_computed_tokens < requests[request_id].num_prompt_tokens
        for request_id in request_ids
    )


class SchedulerDecisionWriter:
    """Line-buffered JSONL writer owned by one EngineCore scheduler instance."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)

    def write(self, record: SchedulerStepRecord) -> None:
        """Append one complete JSON object as an independently readable line."""
        self._handle.write(json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n")

    def close(self) -> None:
        """Flush and close the decision log."""
        if not self._handle.closed:
            self._handle.close()
