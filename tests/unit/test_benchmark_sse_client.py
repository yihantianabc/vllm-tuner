"""Tests for incremental SSE parsing and streamed request measurement."""

import json
import statistics
import time
from pathlib import Path
from typing import AsyncIterator

import httpx
import pytest

from vllm_tuner.benchmarks.models import RequestResult, RequestSpec, RequestStatus
from vllm_tuner.benchmarks.sse_client import SSEBenchmarkClient, SSEDecoder

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "sse"


class ChunkedStream(httpx.AsyncByteStream):
    """Yield fixture chunks exactly as separate HTTP transport chunks."""

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk.encode("utf-8")

    async def aclose(self) -> None:
        return None


class TimeoutStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        raise httpx.ReadTimeout("fixture timeout")
        yield b""  # pragma: no cover - makes this an async generator

    async def aclose(self) -> None:
        return None


class CloseAwareStream(ChunkedStream):
    """Expose whether transport cleanup happened before the metric boundary."""

    def __init__(self, chunks: list[str]) -> None:
        super().__init__(chunks)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class CompletionBeforeCloseClock:
    def __init__(self, stream: CloseAwareStream) -> None:
        self.stream = stream
        self.values = iter((1_000, 1_100, 1_200))

    def __call__(self) -> int:
        value = next(self.values)
        if value == 1_200:
            assert self.stream.closed is False
        return value


class DeterministicClock:
    def __init__(self, *timestamps: int) -> None:
        self.timestamps = iter(timestamps)

    def __call__(self) -> int:
        return next(self.timestamps)


def _fixture(name: str) -> list[str]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_decoder_handles_split_and_multiple_events() -> None:
    decoder = SSEDecoder()
    events = []

    for chunk in _fixture("split_event.json"):
        events.extend(decoder.feed(chunk))
    events.extend(decoder.close())

    assert json.loads(events[0].data)["choices"][0]["text"] == "hello"
    assert len(events) == 4
    assert events[-1].data == "[DONE]"


def test_decoder_supports_crlf_comments_and_multiline_data() -> None:
    decoder = SSEDecoder()
    events = []
    for chunk in _fixture("comments_and_multiline.json"):
        events.extend(decoder.feed(chunk))

    assert json.loads(events[0].data)["choices"][0]["text"] == "ok"
    assert events[1].data == "[DONE]"


def test_decoder_flushes_final_event_without_blank_line() -> None:
    decoder = SSEDecoder()

    assert decoder.feed(b'data: {"value": 1}') == []
    assert json.loads(decoder.close()[0].data) == {"value": 1}


@pytest.mark.asyncio
async def test_client_streams_true_and_measures_split_event() -> None:
    captured_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return httpx.Response(200, stream=ChunkedStream(_fixture("split_event.json")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = SSEBenchmarkClient(
            "http://test",
            "model",
            clock_ns=DeterministicClock(1_000, 1_100, 1_150, 1_200),
        )
        result = await client.send_request(
            RequestSpec(
                request_id="split",
                prompt="prompt",
                max_tokens=2,
                extra_body={
                    "stream": False,
                    "stream_options": {"include_usage": False},
                },
            ),
            http_client,
        )

    assert captured_payload["stream"] is True
    assert captured_payload["stream_options"] == {"include_usage": True}
    assert result.status == RequestStatus.SUCCESS
    assert result.output_text == "hello world"
    assert result.input_tokens == 3
    assert result.output_tokens == 2
    assert result.token_count_source == "usage"
    assert result.sent_at == 1_000
    assert result.first_token_at == 1_100
    assert result.finished_at == 1_200
    assert result.token_timestamps == [1_100, 1_150]


@pytest.mark.asyncio
async def test_empty_text_does_not_start_ttft_and_one_chunk_has_multiple_events() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ChunkedStream(_fixture("multiple_events.json")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = SSEBenchmarkClient(
            "http://test",
            "model",
            clock_ns=DeterministicClock(10, 25, 40, 55),
        )
        result = await client.send_request(
            RequestSpec(request_id="multi", prompt="prompt"), http_client
        )

    assert result.output_text == "first second"
    assert result.first_token_at == 25
    assert result.token_timestamps == [25, 40]
    assert result.finished_at == 55


@pytest.mark.asyncio
async def test_done_timestamp_excludes_response_context_cleanup() -> None:
    stream = CloseAwareStream(
        [
            'data: {"choices":[{"text":"answer"}],'
            '"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\ndata: [DONE]\n\n'
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        result = await SSEBenchmarkClient(
            "http://test",
            "model",
            clock_ns=CompletionBeforeCloseClock(stream),
        ).send_request(RequestSpec(request_id="cleanup", prompt="prompt"), http_client)

    assert stream.closed is True
    assert result.sent_at == 1_000
    assert result.first_token_at == 1_100
    assert result.finished_at == 1_200
    assert result.e2e_ns == 200


@pytest.mark.asyncio
async def test_tokenizer_counts_tokens_when_usage_is_absent() -> None:
    chunks = ['data: {"choices":[{"text":"two words"}]}\n\ndata: [DONE]\n\n']

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ChunkedStream(chunks))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = SSEBenchmarkClient(
            "http://test",
            "model",
            token_counter=lambda text: len(text.split()),
        )
        result = await client.send_request(
            RequestSpec(request_id="tokens", prompt="three input words"), http_client
        )

    assert result.status == RequestStatus.SUCCESS
    assert result.input_tokens == 3
    assert result.output_tokens == 2
    assert result.token_count_source == "tokenizer"


@pytest.mark.asyncio
async def test_http_error_is_an_explicit_request_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = SSEBenchmarkClient("http://test", "model")
        result = await client.send_request(
            RequestSpec(request_id="http", prompt="prompt"), http_client
        )

    assert result.status == RequestStatus.FAILED
    assert result.error_type == "http_error"
    assert result.http_status == 503
    assert "unavailable" in (result.error_message or "")


@pytest.mark.asyncio
async def test_timeout_is_an_explicit_request_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=TimeoutStream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = SSEBenchmarkClient("http://test", "model")
        result = await client.send_request(
            RequestSpec(request_id="timeout", prompt="prompt"), http_client
        )

    assert result.status == RequestStatus.TIMEOUT
    assert result.error_type == "timeout"


@pytest.mark.asyncio
async def test_missing_done_is_a_protocol_failure() -> None:
    chunks = [
        'data: {"choices":[{"text":"answer"}],'
        '"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n'
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ChunkedStream(chunks))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = SSEBenchmarkClient("http://test", "model")
        result = await client.send_request(
            RequestSpec(request_id="truncated", prompt="prompt"), http_client
        )

    assert result.status == RequestStatus.FAILED
    assert result.error_type == "protocol_error"


@pytest.mark.asyncio
async def test_run_records_gather_exceptions_and_excludes_warmup(monkeypatch) -> None:
    benchmark_client = SSEBenchmarkClient("http://test", "model")

    async def fake_send(
        request: RequestSpec,
        client: httpx.AsyncClient,
        *,
        scheduled_at=None,
        warmup=False,
    ) -> RequestResult:
        if request.request_id == "bad":
            raise RuntimeError("escaped task error")
        now = time.perf_counter_ns()
        return RequestResult(
            request_id=request.request_id,
            scheduled_at=scheduled_at,
            sent_at=now,
            first_token_at=now,
            finished_at=now,
            input_tokens=1,
            output_tokens=1,
            token_timestamps=[now],
            status=RequestStatus.SUCCESS,
            warmup=warmup,
        )

    monkeypatch.setattr(benchmark_client, "send_request", fake_send)
    result = await benchmark_client.run(
        [
            RequestSpec(request_id="good", prompt="prompt"),
            RequestSpec(request_id="bad", prompt="prompt"),
        ],
        warmup_requests=1,
        max_concurrency=2,
    )

    assert len(result.warmup_results) == 1
    assert result.aggregate["num_requests"] == 2
    assert result.aggregate["completed"] == 1
    assert result.aggregate["failed"] == 1
    assert result.request_results[1].error_type == "task_exception"
    assert "escaped task error" in (result.request_results[1].error_message or "")


@pytest.mark.asyncio
async def test_required_vllm_token_ids_measure_multi_token_chunks() -> None:
    captured_payload = {}
    chunks = [
        'data: {"choices":[{"text":"ab","token_ids":[11,12]}]}\n\n'
        'data: {"choices":[{"text":"c","token_ids":[13]}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":3}}\n\n'
        "data: [DONE]\n\n"
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return httpx.Response(200, stream=ChunkedStream(chunks))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        result = await SSEBenchmarkClient(
            "http://test",
            "model",
            require_token_ids=True,
            clock_ns=DeterministicClock(1_000, 1_100, 1_150, 1_200),
        ).send_request(
            RequestSpec(
                request_id="token-ids",
                prompt="prompt",
                extra_body={"return_token_ids": False},
            ),
            http_client,
        )

    assert captured_payload["return_token_ids"] is True
    assert result.status == RequestStatus.SUCCESS
    assert result.token_timestamps == [1_100, 1_100, 1_150]
    assert result.event_timestamps == [1_100, 1_150]
    assert result.token_timestamps_valid is True
    assert result.token_timestamp_source == "vllm_delta_token_ids"
    assert result.itl_ns == [0, 50]
    assert result.inter_event_latency_ns == [50]


@pytest.mark.asyncio
async def test_token_count_mismatch_keeps_raw_arrivals_but_disables_itl() -> None:
    chunks = [
        'data: {"choices":[{"text":"ab","token_ids":[11,12]}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":3}}\n\n'
        "data: [DONE]\n\n"
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ChunkedStream(chunks))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        result = await SSEBenchmarkClient(
            "http://test",
            "model",
            require_token_ids=True,
            clock_ns=DeterministicClock(10, 20, 30),
        ).send_request(RequestSpec(request_id="mismatch", prompt="prompt"), http_client)

    assert result.status == RequestStatus.SUCCESS
    assert result.output_tokens == 3
    assert result.token_timestamps == [20, 20]
    assert result.event_timestamps == [20]
    assert result.token_timestamps_valid is False
    assert result.token_timestamp_source == "vllm_delta_token_ids_count_mismatch"
    assert result.itl_ns == []
    assert result.to_dict()["itl_ms"] == []


@pytest.mark.asyncio
async def test_backend_without_token_ids_reports_only_inter_event_latency() -> None:
    chunks = [
        'data: {"choices":[{"text":"a"}]}\n\n'
        'data: {"choices":[{"text":" b"}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":3}}\n\n'
        "data: [DONE]\n\n"
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ChunkedStream(chunks))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        result = await SSEBenchmarkClient(
            "http://test",
            "model",
            require_token_ids=True,
            clock_ns=DeterministicClock(10, 20, 30, 40),
        ).send_request(RequestSpec(request_id="events", prompt="prompt"), http_client)

    assert result.status == RequestStatus.SUCCESS
    assert result.first_token_at == 20
    assert result.token_timestamps == []
    assert result.event_timestamps == [20, 30]
    assert result.token_timestamps_valid is False
    assert result.token_timestamp_source == "unavailable"
    assert result.itl_ns == []
    assert result.inter_event_latency_ns == [10]


@pytest.mark.asyncio
async def test_empty_text_token_does_not_start_ttft() -> None:
    chunks = [
        'data: {"choices":[{"text":"","token_ids":[11]}]}\n\n'
        'data: {"choices":[{"text":"answer","token_ids":[12]}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ChunkedStream(chunks))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        result = await SSEBenchmarkClient(
            "http://test",
            "model",
            require_token_ids=True,
            clock_ns=DeterministicClock(10, 20, 30, 40),
        ).send_request(RequestSpec(request_id="empty-token", prompt="prompt"), http_client)

    assert result.token_timestamps == [20, 30]
    assert result.event_timestamps == [30]
    assert result.first_token_at == 30
    assert result.ttft_ns == 20
    assert result.itl_ns == [10]


def test_arrival_offsets_are_seeded_and_burstiness_is_interarrival_cv() -> None:
    count = 50_000
    request_rate = 4.0
    burstiness = 1.5

    first = SSEBenchmarkClient._arrival_offsets(count, request_rate, burstiness, seed=2026)
    second = SSEBenchmarkClient._arrival_offsets(count, request_rate, burstiness, seed=2026)
    intervals = [right - left for left, right in zip(first, first[1:])]
    mean = statistics.fmean(intervals)
    cv = statistics.pstdev(intervals, mu=mean) / mean

    assert first == second
    assert mean == pytest.approx(1.0 / request_rate, rel=0.03)
    assert cv == pytest.approx(burstiness, rel=0.03)


@pytest.mark.asyncio
@pytest.mark.parametrize("burstiness", [0.0, -1.0, float("inf"), float("-inf"), float("nan")])
async def test_run_rejects_non_positive_or_non_finite_burstiness(
    burstiness: float,
) -> None:
    client = SSEBenchmarkClient("http://test", "model")

    with pytest.raises(ValueError, match="burstiness must be positive and finite"):
        await client.run([], burstiness=burstiness)
