"""OpenAI-compatible SSE benchmark client with nanosecond timestamps."""

from __future__ import annotations

import asyncio
import codecs
import json
import logging
import math
import random
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Union

import httpx

from .metrics import aggregate_request_results
from .models import (
    BenchmarkResult,
    RequestResult,
    RequestSpec,
    RequestStatus,
    SLOThresholds,
)

logger = logging.getLogger(__name__)

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class SSEEvent:
    """One decoded Server-Sent Event."""

    data: str
    event: Optional[str] = None
    event_id: Optional[str] = None
    retry: Optional[int] = None


class SSEDecoder:
    """Incrementally decode SSE records across arbitrary HTTP byte chunks."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""
        self._data_lines: list[str] = []
        self._event: Optional[str] = None
        self._event_id: Optional[str] = None
        self._retry: Optional[int] = None

    def feed(self, chunk: Union[bytes, str]) -> list[SSEEvent]:
        """Consume bytes or text and return every complete SSE event."""

        text = self._decoder.decode(chunk) if isinstance(chunk, bytes) else chunk
        self._buffer += text
        events: list[SSEEvent] = []

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.endswith("\r"):
                line = line[:-1]
            event = self._consume_line(line)
            if event is not None:
                events.append(event)
        return events

    def close(self) -> list[SSEEvent]:
        """Flush the UTF-8 decoder and a final event without a blank line."""

        self._buffer += self._decoder.decode(b"", final=True)
        events: list[SSEEvent] = []
        if self._buffer:
            line = self._buffer[:-1] if self._buffer.endswith("\r") else self._buffer
            self._buffer = ""
            event = self._consume_line(line)
            if event is not None:
                events.append(event)
        event = self._dispatch()
        if event is not None:
            events.append(event)
        return events

    def _consume_line(self, line: str) -> Optional[SSEEvent]:
        if not line:
            return self._dispatch()
        if line.startswith(":"):
            return None

        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]

        if field == "data":
            self._data_lines.append(value)
        elif field == "event":
            self._event = value
        elif field == "id" and "\x00" not in value:
            self._event_id = value
        elif field == "retry":
            try:
                self._retry = int(value)
            except ValueError:
                pass
        return None

    def _dispatch(self) -> Optional[SSEEvent]:
        if not self._data_lines:
            self._event = None
            self._retry = None
            return None
        event = SSEEvent(
            data="\n".join(self._data_lines),
            event=self._event,
            event_id=self._event_id,
            retry=self._retry,
        )
        self._data_lines = []
        self._event = None
        self._retry = None
        return event


def token_counter_from_tokenizer(tokenizer: object) -> TokenCounter:
    """Adapt a Hugging Face-style tokenizer to a simple token counter."""

    def count(text: str) -> int:
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            try:
                tokens = encode(text, add_special_tokens=False)
            except TypeError:
                tokens = encode(text)
            return len(tokens)

        if callable(tokenizer):
            try:
                encoded = tokenizer(text, add_special_tokens=False)
            except TypeError:
                encoded = tokenizer(text)
            if isinstance(encoded, dict):
                return len(encoded["input_ids"])
            return len(getattr(encoded, "input_ids"))
        raise TypeError("tokenizer must be callable or provide encode()")

    return count


def _choice_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""

    content: object
    if "text" in choice:
        content = choice.get("text")
    elif isinstance(choice.get("delta"), dict):
        content = choice["delta"].get("content")
    elif isinstance(choice.get("message"), dict):
        content = choice["message"].get("content")
    else:
        content = ""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                pieces.append(part["text"])
        return "".join(pieces)
    return ""


def _choice_token_ids(payload: dict[str, Any]) -> tuple[Optional[list[int]], Optional[str]]:
    """Return vLLM delta token IDs, distinguishing an absent field from an empty delta."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None, None
    choice = choices[0]
    raw_ids = choice.get("token_ids")
    delta = choice.get("delta")
    if raw_ids is None and isinstance(delta, dict):
        raw_ids = delta.get("token_ids")
    if raw_ids is None:
        return None, None
    if not isinstance(raw_ids, list) or any(type(value) is not int for value in raw_ids):
        return None, "choice.token_ids must be a list of integers"
    return list(raw_ids), None


class SSEBenchmarkClient:
    """Measure streamed requests against an OpenAI-compatible vLLM endpoint."""

    def __init__(
        self,
        base_url: str,
        model: Optional[str] = None,
        *,
        timeout: Union[float, httpx.Timeout] = 300.0,
        token_counter: Optional[TokenCounter] = None,
        tokenizer: Optional[object] = None,
        clock_ns: Optional[Callable[[], int]] = None,
        require_done: bool = True,
        strict_token_count: bool = True,
        require_token_ids: bool = False,
    ) -> None:
        if token_counter is not None and tokenizer is not None:
            raise ValueError("Pass token_counter or tokenizer, not both")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.token_counter = token_counter or (
            token_counter_from_tokenizer(tokenizer) if tokenizer is not None else None
        )
        self._clock_ns = clock_ns or time.perf_counter_ns
        self.require_done = require_done
        self.strict_token_count = strict_token_count
        self.require_token_ids = require_token_ids

    async def send_request(
        self,
        request: RequestSpec,
        client: Optional[httpx.AsyncClient] = None,
        *,
        scheduled_at: Optional[int] = None,
        warmup: bool = False,
    ) -> RequestResult:
        """Send one request and preserve every raw client-side measurement."""

        payload = request.to_payload(self.model)
        if self.require_token_ids:
            # Verified against pinned vLLM 0.16 CompletionRequest/ChatCompletionRequest.
            # This client option is opt-in so generic OpenAI-compatible backends
            # are not sent a vLLM extension they may reject.
            payload["return_token_ids"] = True
        if request.endpoint.endswith("/chat/completions"):
            payload.pop("prompt", None)
            payload["messages"] = [{"role": "user", "content": request.prompt}]

        owns_client = client is None
        http_client = client or httpx.AsyncClient(timeout=self.timeout, trust_env=False)
        scheduled = scheduled_at if scheduled_at is not None else request.scheduled_at
        sent_at = self._clock_ns()
        output_parts: list[str] = []
        token_timestamps: list[int] = []
        event_timestamps: list[int] = []
        timestamp_state = {
            "saw_token_ids": False,
            "saw_text_without_token_ids": False,
        }
        usage: dict[str, Any] = {}
        saw_done = False
        completion_ns: Optional[int] = None

        try:
            url = f"{self.base_url}{request.endpoint}"
            async with http_client.stream(
                "POST", url, json=payload, timeout=self.timeout
            ) as response:
                if response.is_error:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    return self._failed_result(
                        request,
                        scheduled,
                        sent_at,
                        "http_error",
                        f"HTTP {response.status_code}: {body[:500]}",
                        warmup=warmup,
                        http_status=response.status_code,
                    )

                decoder = SSEDecoder()
                async for chunk in response.aiter_bytes():
                    for event in decoder.feed(chunk):
                        done, error = self._consume_event(
                            event,
                            output_parts,
                            token_timestamps,
                            event_timestamps,
                            timestamp_state,
                            usage,
                        )
                        if error is not None:
                            return self._failed_result(
                                request,
                                scheduled,
                                sent_at,
                                error[0],
                                error[1],
                                warmup=warmup,
                                output_text="".join(output_parts),
                                token_timestamps=token_timestamps,
                                event_timestamps=event_timestamps,
                            )
                        if done:
                            saw_done = True
                            completion_ns = self._clock_ns()
                            break
                    if saw_done:
                        break

                if not saw_done:
                    for event in decoder.close():
                        done, error = self._consume_event(
                            event,
                            output_parts,
                            token_timestamps,
                            event_timestamps,
                            timestamp_state,
                            usage,
                        )
                        if error is not None:
                            return self._failed_result(
                                request,
                                scheduled,
                                sent_at,
                                error[0],
                                error[1],
                                warmup=warmup,
                                output_text="".join(output_parts),
                                token_timestamps=token_timestamps,
                                event_timestamps=event_timestamps,
                            )
                        if done and not saw_done:
                            saw_done = True
                            completion_ns = self._clock_ns()

                # EOF is the completion boundary for backends where DONE is
                # optional. Record it before leaving the response context so
                # transport cleanup/aclose latency cannot pollute E2E or TPOT.
                if completion_ns is None:
                    completion_ns = self._clock_ns()

            finished_at = completion_ns
            if self.require_done and not saw_done:
                return self._failed_result(
                    request,
                    scheduled,
                    sent_at,
                    "protocol_error",
                    "SSE stream ended before data: [DONE]",
                    warmup=warmup,
                    finished_at=finished_at,
                    output_text="".join(output_parts),
                    token_timestamps=token_timestamps,
                    event_timestamps=event_timestamps,
                )

            output_text = "".join(output_parts)
            try:
                input_tokens, output_tokens, count_source = self._resolve_token_counts(
                    request, output_text, usage
                )
            except (KeyError, TypeError, ValueError) as error:
                return self._failed_result(
                    request,
                    scheduled,
                    sent_at,
                    "token_count_error",
                    str(error),
                    warmup=warmup,
                    finished_at=finished_at,
                    output_text=output_text,
                    token_timestamps=token_timestamps,
                    event_timestamps=event_timestamps,
                )

            token_timestamps_valid = (
                bool(timestamp_state["saw_token_ids"])
                and not timestamp_state["saw_text_without_token_ids"]
                and len(token_timestamps) == output_tokens
            )
            token_timestamp_source = (
                "vllm_delta_token_ids"
                if token_timestamps_valid
                else (
                    "vllm_delta_token_ids_count_mismatch"
                    if timestamp_state["saw_token_ids"]
                    else "unavailable"
                )
            )

            return RequestResult(
                request_id=request.request_id,
                scheduled_at=scheduled,
                sent_at=sent_at,
                first_token_at=event_timestamps[0] if event_timestamps else None,
                finished_at=finished_at,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                token_timestamps=token_timestamps,
                event_timestamps=event_timestamps,
                token_timestamps_valid=token_timestamps_valid,
                token_timestamp_source=token_timestamp_source,
                status=RequestStatus.SUCCESS,
                output_text=output_text,
                http_status=200,
                token_count_source=count_source,
                warmup=warmup,
                metadata={
                    "saw_done": saw_done,
                    "usage": usage,
                    "first_token_at_explicit": True,
                    "token_timestamp_count": len(token_timestamps),
                    "event_timestamp_count": len(event_timestamps),
                    "token_timestamp_count_matches_output_tokens": token_timestamps_valid,
                },
            )
        except (httpx.TimeoutException, asyncio.TimeoutError) as error:
            logger.warning("Benchmark request %s timed out: %s", request.request_id, error)
            return self._failed_result(
                request,
                scheduled,
                sent_at,
                "timeout",
                str(error) or "request timed out",
                status=RequestStatus.TIMEOUT,
                warmup=warmup,
                output_text="".join(output_parts),
                token_timestamps=token_timestamps,
                event_timestamps=event_timestamps,
            )
        except httpx.HTTPError as error:
            logger.warning("Benchmark request %s failed: %s", request.request_id, error)
            return self._failed_result(
                request,
                scheduled,
                sent_at,
                "transport_error",
                str(error),
                warmup=warmup,
                output_text="".join(output_parts),
                token_timestamps=token_timestamps,
                event_timestamps=event_timestamps,
            )
        except Exception as error:
            logger.exception("Unexpected benchmark request failure: %s", request.request_id)
            return self._failed_result(
                request,
                scheduled,
                sent_at,
                "client_error",
                f"{type(error).__name__}: {error}",
                warmup=warmup,
                output_text="".join(output_parts),
                token_timestamps=token_timestamps,
                event_timestamps=event_timestamps,
            )
        finally:
            if owns_client:
                await http_client.aclose()

    def _consume_event(
        self,
        event: SSEEvent,
        output_parts: list[str],
        token_timestamps: list[int],
        event_timestamps: list[int],
        timestamp_state: dict[str, bool],
        usage: dict[str, Any],
    ) -> tuple[bool, Optional[tuple[str, str]]]:
        data = event.data.strip()
        if data == "[DONE]":
            return True, None
        if not data:
            return False, None
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as error:
            return False, ("invalid_json", f"Invalid SSE JSON: {error}")
        if not isinstance(payload, dict):
            return False, ("invalid_event", "SSE data must decode to a JSON object")

        server_error = payload.get("error")
        if server_error:
            if isinstance(server_error, dict):
                message = str(server_error.get("message", server_error))
            else:
                message = str(server_error)
            return False, ("server_error", message)

        event_usage = payload.get("usage")
        if isinstance(event_usage, dict):
            usage.update(event_usage)

        text = _choice_text(payload)
        token_ids, token_error = _choice_token_ids(payload)
        if token_error is not None:
            return False, ("invalid_event", token_error)

        arrival_ns: Optional[int] = None
        if text or token_ids:
            arrival_ns = self._clock_ns()
        if text:
            output_parts.append(text)
            if arrival_ns is not None:
                event_timestamps.append(arrival_ns)
            if not token_ids:
                timestamp_state["saw_text_without_token_ids"] = True
        if token_ids:
            timestamp_state["saw_token_ids"] = True
            assert arrival_ns is not None
            token_timestamps.extend([arrival_ns] * len(token_ids))
        return False, None

    def _resolve_token_counts(
        self, request: RequestSpec, output_text: str, usage: dict[str, Any]
    ) -> tuple[int, int, str]:
        prompt_usage = usage.get("prompt_tokens")
        output_usage = usage.get("completion_tokens")

        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None
        input_source: Optional[str] = None
        output_source: Optional[str] = None

        if isinstance(prompt_usage, int) and prompt_usage >= 0:
            input_tokens = prompt_usage
            input_source = "usage"
        elif request.input_tokens is not None:
            input_tokens = request.input_tokens
            input_source = "request"
        elif self.token_counter is not None:
            input_tokens = self.token_counter(request.prompt)
            input_source = "tokenizer"

        if isinstance(output_usage, int) and output_usage >= 0:
            output_tokens = output_usage
            output_source = "usage"
        elif self.token_counter is not None:
            output_tokens = self.token_counter(output_text)
            output_source = "tokenizer"

        if input_tokens is None or output_tokens is None:
            if self.strict_token_count:
                raise ValueError(
                    "Token counts unavailable: enable stream usage, provide input_tokens, "
                    "or configure a tokenizer"
                )
            input_tokens = input_tokens or 0
            output_tokens = output_tokens or 0
            return input_tokens, output_tokens, "unavailable"
        if input_tokens == 0 and request.prompt:
            if self.token_counter is None:
                if self.strict_token_count:
                    raise ValueError("Usage reported zero input tokens for a non-empty prompt")
            else:
                input_tokens = self.token_counter(request.prompt)
                input_source = "tokenizer"
        if output_tokens == 0 and output_text:
            if self.token_counter is None:
                if self.strict_token_count:
                    raise ValueError("Usage reported zero output tokens for non-empty output")
            else:
                output_tokens = self.token_counter(output_text)
                output_source = "tokenizer"
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Tokenizer returned a negative token count")

        source = (
            input_source if input_source == output_source else f"{input_source}+{output_source}"
        )
        return input_tokens, output_tokens, source or "unavailable"

    def _failed_result(
        self,
        request: RequestSpec,
        scheduled_at: Optional[int],
        sent_at: int,
        error_type: str,
        error_message: str,
        *,
        status: RequestStatus = RequestStatus.FAILED,
        warmup: bool,
        finished_at: Optional[int] = None,
        output_text: str = "",
        token_timestamps: Optional[list[int]] = None,
        event_timestamps: Optional[list[int]] = None,
        http_status: Optional[int] = None,
    ) -> RequestResult:
        timestamps = list(token_timestamps or [])
        events = list(event_timestamps or [])
        return RequestResult(
            request_id=request.request_id,
            scheduled_at=scheduled_at,
            sent_at=sent_at,
            first_token_at=events[0] if events else None,
            finished_at=finished_at if finished_at is not None else self._clock_ns(),
            input_tokens=request.input_tokens or 0,
            output_tokens=0,
            token_timestamps=timestamps,
            event_timestamps=events,
            token_timestamps_valid=False,
            token_timestamp_source="failed_request",
            status=status,
            error_type=error_type,
            error_message=error_message,
            output_text=output_text,
            http_status=http_status,
            warmup=warmup,
            metadata={"first_token_at_explicit": True},
        )

    async def run(
        self,
        requests: list[RequestSpec],
        *,
        warmup_requests: int = 0,
        request_rate: float = float("inf"),
        burstiness: float = 1.0,
        max_concurrency: Optional[int] = None,
        seed: int = 0,
        slo: Optional[SLOThresholds] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> BenchmarkResult:
        """Run warmup and measured requests with explicit gather failures."""

        if warmup_requests < 0:
            raise ValueError("warmup_requests must be non-negative")
        if request_rate <= 0:
            raise ValueError("request_rate must be positive")
        if not math.isfinite(burstiness) or burstiness <= 0:
            raise ValueError("burstiness must be positive and finite")
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")

        owns_client = client is None
        http_client = client or httpx.AsyncClient(timeout=self.timeout, trust_env=False)
        warmup_results: list[RequestResult] = []

        try:
            if requests:
                for index in range(warmup_requests):
                    original = requests[index % len(requests)]
                    warmup_spec = replace(
                        original, request_id=f"warmup-{index}-{original.request_id}"
                    )
                    warmup_results.append(
                        await self.send_request(warmup_spec, http_client, warmup=True)
                    )

            started_at = self._clock_ns()
            offsets = self._arrival_offsets(len(requests), request_rate, burstiness, seed)
            semaphore = asyncio.Semaphore(max_concurrency or max(1, len(requests)))

            async def execute(spec: RequestSpec, offset_s: float) -> RequestResult:
                scheduled_at = started_at + int(offset_s * 1_000_000_000)
                delay_s = (scheduled_at - self._clock_ns()) / 1_000_000_000
                if delay_s > 0:
                    await asyncio.sleep(delay_s)
                async with semaphore:
                    return await self.send_request(
                        spec, http_client, scheduled_at=scheduled_at, warmup=False
                    )

            outcomes = await asyncio.gather(
                *(execute(spec, offset) for spec, offset in zip(requests, offsets)),
                return_exceptions=True,
            )
            finished_at = self._clock_ns()

            measured_results: list[RequestResult] = []
            for spec, offset, outcome in zip(requests, offsets, outcomes):
                if isinstance(outcome, BaseException):
                    measured_results.append(
                        RequestResult(
                            request_id=spec.request_id,
                            scheduled_at=started_at + int(offset * 1_000_000_000),
                            sent_at=None,
                            finished_at=finished_at,
                            input_tokens=spec.input_tokens or 0,
                            status=RequestStatus.FAILED,
                            error_type="task_exception",
                            error_message=f"{type(outcome).__name__}: {outcome}",
                        )
                    )
                else:
                    measured_results.append(outcome)

            aggregate = aggregate_request_results(
                measured_results,
                started_at=started_at,
                finished_at=finished_at,
                slo=slo,
            )
            return BenchmarkResult(
                backend="sse",
                started_at=started_at,
                finished_at=finished_at,
                request_results=measured_results,
                warmup_results=warmup_results,
                aggregate=aggregate,
            )
        finally:
            if owns_client:
                await http_client.aclose()

    @staticmethod
    def _arrival_offsets(
        count: int, request_rate: float, burstiness: float, seed: int
    ) -> list[float]:
        if count <= 0:
            return []
        if request_rate == float("inf"):
            return [0.0] * count

        generator = random.Random(seed)
        offsets = [0.0]
        # Public SLOTune ``burstiness`` is inter-arrival CV. For a Gamma
        # distribution CV = 1 / sqrt(shape), while scale = mean / shape.
        mean = 1.0 / request_rate
        shape = 1.0 / (burstiness * burstiness)
        scale = mean / shape
        for _ in range(1, count):
            offsets.append(offsets[-1] + generator.gammavariate(shape, scale))
        return offsets

    run_benchmark = run


# Concise compatibility aliases.
SSEClient = SSEBenchmarkClient
SSEParser = SSEDecoder
