"""Tests for official vLLM benchmark result parsing."""

import json

import pytest

from vllm_tuner.benchmarks.models import RequestStatus
from vllm_tuner.benchmarks.result_parser import (
    BenchmarkResultError,
    load_benchmark_json,
    parse_vllm_benchmark_result,
)


def _official_result() -> dict:
    return {
        "backend": "vllm",
        "model_id": "model",
        "num_prompts": 2,
        "duration": 2.0,
        "completed": 1,
        "failed": 1,
        "total_input_tokens": 3,
        "total_output_tokens": 2,
        "request_throughput": 0.5,
        "mean_ttft_ms": 100.0,
        "p99_ttft_ms": 100.0,
        "mean_e2el_ms": 150.0,
        "p99_e2el_ms": 150.0,
        "input_lens": [3, 4],
        "output_lens": [2, 0],
        "ttfts": [0.1, 0.0],
        "tpots": [0.05, 0.0],
        "itls": [[0.05], []],
        "start_times": [10.0, 11.0],
        "generated_texts": ["two tokens", ""],
        "errors": ["", "connection failed"],
    }


def test_parse_official_detailed_result_preserves_raw_requests() -> None:
    raw = _official_result()

    result = parse_vllm_benchmark_result(raw)

    assert result.raw_result == raw
    assert result.aggregate["completed"] == 1
    assert result.aggregate["failed"] == 1
    assert result.aggregate["mean_e2e_ms"] == 150.0
    assert len(result.request_results) == 2
    first, failed = result.request_results
    assert first.sent_at == 10_000_000_000
    assert first.first_token_at == 10_100_000_000
    assert first.token_timestamps == [10_100_000_000, 10_150_000_000]
    assert first.finished_at == 10_150_000_000
    assert first.e2e_ns == 150_000_000
    assert first.tpot_ns == 50_000_000
    assert first.metadata["finished_at_source"] == "official_tpot"
    assert first.input_tokens == 3
    assert first.output_tokens == 2
    assert failed.status == RequestStatus.FAILED
    assert failed.error_type == "vllm_bench_error"
    assert result.warnings == []


def test_parser_uses_real_per_request_e2e_when_backend_provides_it() -> None:
    raw = _official_result()
    raw["e2els"] = [0.16, 0.0]

    result = parse_vllm_benchmark_result(raw)

    assert result.request_results[0].finished_at == 10_160_000_000
    assert result.request_results[0].e2e_ns == 160_000_000
    assert result.request_results[0].metadata["finished_at_source"] == "e2els"


def test_parser_falls_back_to_observed_itl_timestamp_when_tpot_is_absent() -> None:
    raw = _official_result()
    raw.pop("tpots")

    result = parse_vllm_benchmark_result(raw)

    first = result.request_results[0]
    assert first.finished_at == 10_150_000_000
    assert first.metadata["finished_at_source"] == "token_timestamps"


def test_parser_does_not_fabricate_missing_official_itl_interval() -> None:
    raw = _official_result()
    raw["total_output_tokens"] = 3
    raw["output_lens"] = [3, 0]
    # Pinned vLLM versions can emit fewer intervals than output_tokens - 1.
    raw["itls"] = [[0.05], []]

    result = parse_vllm_benchmark_result(raw)

    first = result.request_results[0]
    assert first.token_timestamps == [10_100_000_000, 10_150_000_000]
    assert first.token_timestamps_valid is False
    assert first.token_timestamp_source == "official_vllm_itls_count_mismatch"
    assert first.itl_ns == []
    assert result.raw_result["itls"][0] == [0.05]
    assert any("fewer intervals" in warning for warning in result.warnings)


def test_parser_rejects_successful_zero_token_result() -> None:
    raw = _official_result()
    raw["total_output_tokens"] = 0

    with pytest.raises(BenchmarkResultError, match="no output tokens"):
        parse_vllm_benchmark_result(raw)


def test_parser_can_skip_validation_for_failure_forensics() -> None:
    raw = _official_result()
    raw["total_input_tokens"] = 0
    raw["total_output_tokens"] = 0

    result = parse_vllm_benchmark_result(raw, validate=False)

    assert result.aggregate["total_input_tokens"] == 0
    assert result.aggregate["total_output_tokens"] == 0


def test_load_jsonl_returns_last_append_mode_run(tmp_path) -> None:
    path = tmp_path / "bench.jsonl"
    path.write_text(
        json.dumps({"completed": 1}) + "\n" + json.dumps({"completed": 2}) + "\n",
        encoding="utf-8",
    )

    assert load_benchmark_json(path) == {"completed": 2}


def test_parser_requires_json_object(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(BenchmarkResultError, match="JSON object"):
        load_benchmark_json(path)
