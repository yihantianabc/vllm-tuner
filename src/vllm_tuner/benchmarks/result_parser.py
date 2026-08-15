"""Parse vLLM benchmark JSON without discarding detailed request data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from .metrics import aggregate_request_results
from .models import BenchmarkResult, RequestResult, RequestStatus, SLOThresholds

JSONSource = Union[str, Path, Mapping[str, Any]]
_DETAIL_FIELDS = {
    "input_lens",
    "prompt_lens",
    "output_lens",
    "ttfts",
    "tpots",
    "itls",
    "e2els",
    "latencies",
    "start_times",
    "end_times",
    "generated_texts",
    "errors",
    "request_ids",
    "request_results",
    "requests",
}


class BenchmarkResultError(ValueError):
    """Raised when saved benchmark output is missing required measurements."""


def load_benchmark_json(path: Union[str, Path]) -> dict[str, Any]:
    """Load a JSON result, accepting append-mode JSONL and returning its last run."""

    result_path = Path(path)
    text = result_path.read_text(encoding="utf-8").strip()
    if not text:
        raise BenchmarkResultError(f"Benchmark result is empty: {result_path}")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        values = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise BenchmarkResultError(
                    f"Invalid JSONL at {result_path}:{line_number}: {error}"
                ) from error
        if not values:
            raise BenchmarkResultError(f"Benchmark result is empty: {result_path}")
        value = values[-1]
    if not isinstance(value, dict):
        raise BenchmarkResultError("Top-level benchmark result must be a JSON object")
    return value


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _integer(value: object, default: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return default


def _sequence(raw: Mapping[str, Any], *keys: str) -> Sequence[Any]:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, list):
            return value
    return []


def _at(values: Sequence[Any], index: int, default: Any = None) -> Any:
    return values[index] if index < len(values) else default


def _seconds_to_ns(value: object) -> Optional[int]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return int(round(float(value) * 1_000_000_000))


def _error_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _parse_native_request_results(raw: Mapping[str, Any]) -> list[RequestResult]:
    values = raw.get("request_results", raw.get("requests"))
    if not isinstance(values, list) or not all(isinstance(value, dict) for value in values):
        return []
    return [RequestResult.from_dict(value) for value in values]


def _parse_vllm_detailed_results(raw: Mapping[str, Any]) -> list[RequestResult]:
    input_lens = _sequence(raw, "input_lens", "prompt_lens")
    output_lens = _sequence(raw, "output_lens")
    ttfts = _sequence(raw, "ttfts")
    tpots = _sequence(raw, "tpots")
    itls = _sequence(raw, "itls")
    start_times = _sequence(raw, "start_times")
    end_times = _sequence(raw, "end_times")
    e2els = _sequence(raw, "e2els", "latencies")
    errors = _sequence(raw, "errors")
    generated_texts = _sequence(raw, "generated_texts")
    request_ids = _sequence(raw, "request_ids")

    lengths = [
        len(input_lens),
        len(output_lens),
        len(ttfts),
        len(tpots),
        len(itls),
        len(start_times),
        len(end_times),
        len(e2els),
        len(errors),
        len(generated_texts),
        len(request_ids),
    ]
    count = max(lengths, default=0)
    if count == 0:
        return []

    results = []
    for index in range(count):
        error_message = _error_text(_at(errors, index, ""))
        status = RequestStatus.FAILED if error_message else RequestStatus.SUCCESS
        output_tokens = max(0, _integer(_at(output_lens, index, 0)))
        sent_at = _seconds_to_ns(_at(start_times, index))
        ttft_ns = _seconds_to_ns(_at(ttfts, index))
        first_token_at = (
            sent_at + ttft_ns
            if sent_at is not None
            and ttft_ns is not None
            and status == RequestStatus.SUCCESS
            and output_tokens > 0
            else None
        )

        token_timestamps: list[int] = []
        if first_token_at is not None:
            token_timestamps.append(first_token_at)
            recent = first_token_at
            raw_itls = _at(itls, index, [])
            if isinstance(raw_itls, list):
                for raw_itl in raw_itls:
                    itl_ns = _seconds_to_ns(raw_itl)
                    if itl_ns is not None:
                        recent += max(0, itl_ns)
                        token_timestamps.append(recent)

        finished_at = _seconds_to_ns(_at(end_times, index))
        finished_at_source = "end_times" if finished_at is not None else None
        e2e_ns = _seconds_to_ns(_at(e2els, index))
        if finished_at is None and sent_at is not None and e2e_ns is not None:
            finished_at = sent_at + e2e_ns
            finished_at_source = "e2els"
        if finished_at is None and first_token_at is not None:
            tpot_ns = _seconds_to_ns(_at(tpots, index))
            if tpot_ns is not None and output_tokens > 0:
                # vLLM 0.16's detailed JSON saves per-request TPOT but omits
                # end_times/e2els. TPOT is derived from that request's measured
                # latency, so this algebraically restores its last-token time.
                finished_at = first_token_at + max(0, output_tokens - 1) * max(0, tpot_ns)
                finished_at_source = "official_tpot"
            elif token_timestamps:
                # Older variants expose TTFT + ITLs only. The final observed
                # token timestamp is still real per-request timing evidence.
                finished_at = token_timestamps[-1]
                finished_at_source = "token_timestamps"

        results.append(
            RequestResult(
                request_id=str(_at(request_ids, index, f"request-{index}")),
                scheduled_at=sent_at,
                sent_at=sent_at,
                first_token_at=first_token_at,
                finished_at=finished_at,
                input_tokens=max(0, _integer(_at(input_lens, index, 0))),
                output_tokens=output_tokens,
                token_timestamps=token_timestamps,
                token_timestamps_valid=(
                    status == RequestStatus.SUCCESS and len(token_timestamps) == output_tokens
                ),
                token_timestamp_source=(
                    "official_vllm_itls"
                    if status == RequestStatus.SUCCESS and len(token_timestamps) == output_tokens
                    else "official_vllm_itls_count_mismatch"
                ),
                status=status,
                error_type="vllm_bench_error" if error_message else None,
                error_message=error_message or None,
                output_text=str(_at(generated_texts, index, "") or ""),
                token_count_source="vllm_bench",
                metadata={
                    "source": "vllm_bench",
                    "per_request_e2e_available": finished_at is not None,
                    "finished_at_source": finished_at_source,
                    "official_tpot_ms": (
                        _number(_at(tpots, index)) * 1000 if _at(tpots, index) is not None else None
                    ),
                },
            )
        )
    return results


def _normalise_aggregate(raw: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = {key: value for key, value in raw.items() if key not in _DETAIL_FIELDS}
    completed = _integer(raw.get("completed", raw.get("successful_requests", 0)))
    num_prompts = _integer(raw.get("num_prompts", raw.get("num_requests", 0)))
    failed = _integer(raw.get("failed", raw.get("failed_requests", -1)), -1)
    if failed < 0:
        failed = max(0, num_prompts - completed)

    aggregate.update(
        {
            "num_requests": num_prompts or completed + failed,
            "completed": completed,
            "failed": failed,
            "total_input_tokens": _integer(
                raw.get("total_input_tokens", raw.get("total_input", 0))
            ),
            "total_output_tokens": _integer(
                raw.get("total_output_tokens", raw.get("total_output", 0))
            ),
            "duration": _number(raw.get("duration")),
        }
    )

    # vLLM calls this metric E2EL. SLOTune uses E2E but preserves the original
    # names in raw_result for exact provenance.
    for key, value in list(raw.items()):
        if "e2el" in key:
            aggregate[key.replace("e2el", "e2e")] = value
    return aggregate


def validate_benchmark_result(result: BenchmarkResult) -> None:
    """Reject internally inconsistent official results before tuning uses them."""

    completed = _integer(result.aggregate.get("completed"))
    failed = _integer(result.aggregate.get("failed"))
    total = _integer(result.aggregate.get("num_requests"))
    if total and completed + failed != total:
        raise BenchmarkResultError(
            f"completed ({completed}) + failed ({failed}) does not equal requests ({total})"
        )
    if completed > 0:
        input_tokens = _integer(result.aggregate.get("total_input_tokens"))
        output_tokens = _integer(result.aggregate.get("total_output_tokens"))
        if input_tokens <= 0:
            raise BenchmarkResultError("Successful benchmark reported no input tokens")
        if output_tokens <= 0:
            raise BenchmarkResultError("Successful benchmark reported no output tokens")


def parse_vllm_benchmark_result(
    source: JSONSource,
    *,
    slo: Optional[SLOThresholds] = None,
    validate: bool = True,
) -> BenchmarkResult:
    """Parse official ``vllm bench serve`` JSON and preserve its raw payload."""

    output_path: Optional[Path]
    if isinstance(source, Mapping):
        raw = dict(source)
        output_path = None
    else:
        output_path = Path(source)
        raw = load_benchmark_json(output_path)

    request_results = _parse_native_request_results(raw)
    if not request_results:
        request_results = _parse_vllm_detailed_results(raw)

    aggregate = _normalise_aggregate(raw)
    if request_results:
        measured_starts = [
            result.sent_at for result in request_results if result.sent_at is not None
        ]
        recomputed_start = min(measured_starts) if measured_starts else None
        raw_duration = _number(raw.get("duration"))
        recomputed_finish = (
            recomputed_start + int(raw_duration * 1_000_000_000)
            if recomputed_start is not None and raw_duration > 0
            else None
        )
        recomputed = aggregate_request_results(
            request_results,
            started_at=recomputed_start,
            finished_at=recomputed_finish,
            slo=slo,
            include_request_results=False,
        )
        aggregate["recomputed_from_detailed"] = recomputed
        aggregate["request_results"] = [result.to_dict() for result in request_results]
        if slo is not None and aggregate.get("request_goodput") is None:
            aggregate["request_goodput"] = recomputed["request_goodput"]
            aggregate["good_completed"] = recomputed["good_completed"]

    warnings: list[str] = []
    if not request_results:
        warnings.append(
            "Official JSON has no per-request arrays; run vllm bench serve with --save-detailed"
        )
    elif any(result.finished_at is None for result in request_results if result.success):
        warnings.append(
            "This vLLM JSON version omits per-request E2E timestamps; aggregate E2E remains "
            "authoritative and no per-request value was fabricated"
        )
    if any(
        result.success and result.token_timestamp_source == "official_vllm_itls_count_mismatch"
        for result in request_results
    ):
        warnings.append(
            "Official per-request ITL arrays contain fewer intervals than required by the "
            "reported output token counts; native arrays are preserved, but token-level "
            "per-request ITL is unavailable and no interval was fabricated"
        )

    result = BenchmarkResult(
        backend=str(raw.get("backend", raw.get("endpoint_type", "vllm"))),
        request_results=request_results,
        aggregate=aggregate,
        raw_result=raw,
        output_path=output_path,
        warnings=warnings,
    )
    if validate:
        validate_benchmark_result(result)
    return result


class VLLMResultParser:
    """Object-oriented facade for dependency injection in experiment code."""

    def __init__(self, *, validate: bool = True) -> None:
        self.validate = validate

    def parse(self, source: JSONSource, *, slo: Optional[SLOThresholds] = None) -> BenchmarkResult:
        """Parse one raw official result."""

        return parse_vllm_benchmark_result(source, slo=slo, validate=self.validate)


parse_vllm_result = parse_vllm_benchmark_result
parse_benchmark_result = parse_vllm_benchmark_result
