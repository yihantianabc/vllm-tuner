"""Tests for benchmark latency definitions and reduction."""

import pytest

from vllm_tuner.benchmarks.metrics import (
    aggregate_request_results,
    calculate_e2e_ms,
    calculate_inter_event_latency_ms,
    calculate_itl_ms,
    calculate_tpot_ms,
    calculate_ttft_ms,
    numpy_percentile,
    request_meets_slo,
)
from vllm_tuner.benchmarks.models import RequestResult, RequestStatus, SLOThresholds


def _successful_result(request_id: str = "request-1", *, warmup: bool = False) -> RequestResult:
    return RequestResult(
        request_id=request_id,
        scheduled_at=900_000_000,
        sent_at=1_000_000_000,
        first_token_at=1_100_000_000,
        finished_at=1_220_000_000,
        input_tokens=10,
        output_tokens=4,
        token_timestamps=[1_100_000_000, 1_140_000_000, 1_200_000_000],
        status=RequestStatus.SUCCESS,
        warmup=warmup,
    )


def test_request_metrics_match_hand_calculated_values() -> None:
    """TTFT, TPOT, ITL and E2E use their independent measurement points."""

    result = _successful_result()

    assert calculate_ttft_ms(result) == 100.0
    assert calculate_tpot_ms(result) == 40.0
    assert calculate_itl_ms(result) == [40.0, 60.0]
    assert calculate_e2e_ms(result) == 220.0


def test_invalid_token_timestamps_do_not_pollute_itl_distribution() -> None:
    result = RequestResult(
        request_id="event-only",
        sent_at=1_000_000,
        first_token_at=1_100_000,
        finished_at=1_400_000,
        input_tokens=1,
        output_tokens=3,
        token_timestamps=[1_100_000, 1_200_000],
        event_timestamps=[1_100_000, 1_300_000],
        token_timestamps_valid=False,
        token_timestamp_source="vllm_delta_token_ids_count_mismatch",
        status=RequestStatus.SUCCESS,
    )

    assert calculate_itl_ms(result) == []
    assert calculate_inter_event_latency_ms(result) == [0.2]
    aggregate = aggregate_request_results([result])
    assert aggregate["itl_count"] == 0
    assert aggregate["inter_event_latency_count"] == 1
    assert aggregate["mean_inter_event_latency_ms"] == 0.2


def test_tpot_is_undefined_for_a_single_output_token() -> None:
    result = RequestResult(
        request_id="one-token",
        sent_at=10,
        first_token_at=20,
        finished_at=30,
        output_tokens=1,
        token_timestamps=[20],
        status=RequestStatus.SUCCESS,
    )

    assert calculate_tpot_ms(result) is None
    assert request_meets_slo(result, SLOThresholds(tpot_ms=0))


def test_numpy_percentile_uses_interpolation() -> None:
    assert numpy_percentile([1.0, 2.0, 3.0, 4.0], 75) == 3.25
    assert numpy_percentile([], 99) is None
    with pytest.raises(ValueError, match="between 0 and 100"):
        numpy_percentile([1.0], 101)


def test_aggregate_excludes_warmup_and_keeps_failures_and_raw_results() -> None:
    success = _successful_result()
    second = RequestResult(
        request_id="request-2",
        sent_at=1_500_000_000,
        first_token_at=1_550_000_000,
        finished_at=1_700_000_000,
        input_tokens=5,
        output_tokens=2,
        token_timestamps=[1_550_000_000, 1_650_000_000],
        status=RequestStatus.SUCCESS,
    )
    failed = RequestResult(
        request_id="request-3",
        sent_at=1_600_000_000,
        finished_at=1_800_000_000,
        status=RequestStatus.TIMEOUT,
        error_type="timeout",
    )
    warmup = _successful_result("warmup", warmup=True)

    aggregate = aggregate_request_results(
        [success, second, failed, warmup],
        started_at=1_000_000_000,
        finished_at=3_000_000_000,
        percentiles=(50, 99),
        slo=SLOThresholds(ttft_ms=75, tpot_ms=200, e2e_ms=250),
    )

    assert aggregate["num_requests"] == 3
    assert aggregate["completed"] == 2
    assert aggregate["failed"] == 1
    assert aggregate["error_types"] == {"timeout": 1}
    assert aggregate["total_input_tokens"] == 15
    assert aggregate["total_output_tokens"] == 6
    assert aggregate["request_throughput"] == 1.0
    assert aggregate["good_completed"] == 1
    assert aggregate["request_goodput"] == 0.5
    assert aggregate["mean_ttft_ms"] == 75.0
    assert aggregate["p50_ttft_ms"] == 75.0
    assert len(aggregate["request_results"]) == 3


def test_missing_measurements_remain_none_instead_of_zero() -> None:
    result = RequestResult(
        request_id="missing",
        status=RequestStatus.SUCCESS,
        input_tokens=1,
        output_tokens=1,
    )

    aggregate = aggregate_request_results([result])

    assert aggregate["mean_ttft_ms"] is None
    assert aggregate["mean_tpot_ms"] is None
    assert aggregate["mean_e2e_ms"] is None
    assert aggregate["duration"] is None
