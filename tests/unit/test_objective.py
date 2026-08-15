"""Hand-calculated SLO goodput and constraint tests."""

from vllm_tuner.config.models import Constraints, SLOConfig
from vllm_tuner.tuning.objective import compute_slo_goodput, evaluate_request_slo


def _request(request_id: str, ttft: float, tpot: float, e2e: float, success: bool = True):
    return {
        "request_id": request_id,
        "status": "COMPLETE" if success else "FAILED",
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "e2e_ms": e2e,
        "input_tokens": 10,
        "output_tokens": 4,
    }


def test_per_request_slo_uses_independent_latency_definitions() -> None:
    slo = SLOConfig(ttft_ms=100, tpot_ms=20, e2e_ms=200)
    result = evaluate_request_slo(_request("a", 90, 21, 150), slo)
    assert result.good is False
    assert result.violations == ("tpot_slo",)


def test_goodput_math_and_token_totals() -> None:
    requests = [
        _request("a", 50, 10, 100),
        _request("b", 60, 12, 120),
    ]
    result = compute_slo_goodput(
        requests,
        measurement_seconds=2.0,
        offered_requests=4,
        slo=SLOConfig(ttft_ms=100, tpot_ms=20, e2e_ms=200),
        constraints=Constraints(max_error_rate=0),
        gpu={"peak_memory_mb": 1000, "peak_memory_utilization": 0.5},
    )
    assert result.goodput_requests_per_sec == 1.0
    assert result.offered_requests_per_sec == 2.0
    assert result.achieved_requests_per_sec == 1.0
    assert result.total_input_tokens == 20
    assert result.total_output_tokens == 8
    assert result.constraints.feasible is True


def test_token_totals_exclude_failed_requests_within_error_budget() -> None:
    result = compute_slo_goodput(
        [
            _request("ok", 50, 10, 100),
            _request("failed", 0, 0, 0, success=False),
        ],
        measurement_seconds=2.0,
        offered_requests=2,
        slo=SLOConfig(ttft_ms=100, tpot_ms=20, e2e_ms=200),
        constraints=Constraints(max_error_rate=0.5),
        gpu={"peak_memory_mb": 1000, "peak_memory_utilization": 0.5},
    )

    assert result.completed_requests == 1
    assert result.failed_requests == 1
    assert result.total_input_tokens == 10
    assert result.total_output_tokens == 4
    assert result.constraints.feasible is True


def test_configured_open_loop_rate_is_not_diluted_by_backlog_duration() -> None:
    requests = [_request(f"request-{index}", 50, 10, 100) for index in range(10)]
    result = compute_slo_goodput(
        requests,
        measurement_seconds=5.0,
        offered_requests=10,
        offered_requests_per_second=8.0,
        slo=SLOConfig(ttft_ms=100, tpot_ms=20, e2e_ms=200),
        constraints=Constraints(max_error_rate=0),
        gpu={"peak_memory_mb": 1000, "peak_memory_utilization": 0.5},
    )

    assert result.offered_requests_per_sec == 8.0
    assert result.achieved_requests_per_sec == 2.0


def test_error_and_oom_are_hard_constraints() -> None:
    result = compute_slo_goodput(
        [_request("a", 50, 10, 100, success=False)],
        measurement_seconds=1,
        slo=SLOConfig(ttft_ms=100, tpot_ms=20, e2e_ms=200),
        constraints=Constraints(max_error_rate=0, require_no_oom=True),
        engine={"oom_count": 1},
        server_alive=False,
    )
    assert result.constraints.feasible is False
    assert {"error_rate", "oom", "server_exit"}.issubset(result.constraints.violations)


def test_request_oom_is_hard_even_when_error_rate_is_within_limit() -> None:
    requests = [_request(f"ok-{index}", 50, 10, 100) for index in range(99)]
    failed = _request("oom", 0, 0, 0, success=False)
    failed["error_type"] = "server_error"
    failed["error_message"] = "torch.OutOfMemoryError: CUDA out of memory"
    requests.append(failed)

    result = compute_slo_goodput(
        requests,
        measurement_seconds=10,
        slo=SLOConfig(ttft_ms=100, tpot_ms=20, e2e_ms=200),
        constraints=Constraints(
            max_error_rate=0.01,
            require_no_oom=True,
            max_memory_utilization=None,
        ),
        gpu={"peak_memory_mb": 1000},
    )

    assert result.constraints.values["request_oom_count"] == 1
    assert "error_rate" not in result.constraints.violations
    assert "oom" in result.constraints.violations
    assert result.constraints.feasible is False


def test_configured_vram_constraints_require_real_gpu_evidence() -> None:
    result = compute_slo_goodput(
        [_request("ok", 10, 2, 20)],
        measurement_seconds=1.0,
        slo=SLOConfig(ttft_ms=100, tpot_ms=20, e2e_ms=1000),
        constraints=Constraints(
            max_peak_vram_mb=1000,
            max_memory_utilization=0.9,
        ),
        gpu={"available": False},
    )

    assert result.constraints.feasible is False
    assert {
        "missing_peak_vram",
        "missing_memory_utilization",
    }.issubset(result.constraints.violations)


def test_named_latency_and_throughput_constraints_are_enforced() -> None:
    result = compute_slo_goodput(
        [_request("ok", 10, 2, 20)],
        measurement_seconds=2.0,
        slo=SLOConfig(ttft_ms=100, tpot_ms=20, e2e_ms=1000),
        constraints=Constraints(
            max_peak_vram_mb=None,
            max_memory_utilization=None,
            max_latency_ms=15,
            throughput_min=1.0,
        ),
    )

    assert {"max_latency", "throughput_min"}.issubset(result.constraints.violations)
