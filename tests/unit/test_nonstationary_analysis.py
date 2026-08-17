"""Offline phase metrics and Oracle tests."""

from vllm_tuner.analysis.nonstationary import (
    aggregate_policy_trials,
    select_phase_oracle,
    summarize_labeled_requests,
)
from vllm_tuner.workloads.trace import TraceEntry, WorkloadTrace


def trace() -> WorkloadTrace:
    return WorkloadTrace(
        seed=7,
        profile="nonstationary",
        entries=[
            TraceEntry(
                request_id="decode-0",
                scheduled_offset_seconds=0.0,
                prompt="a",
                input_tokens=1,
                output_tokens=2,
                profile="decode",
            ),
            TraceEntry(
                request_id="prefill-0",
                scheduled_offset_seconds=1.0,
                prompt="b",
                input_tokens=2,
                output_tokens=1,
                profile="prefill",
            ),
        ],
    )


def rows(decode_tpot: float, prefill_ttft: float) -> list[dict[str, object]]:
    return [
        {
            "request_id": "decode-0",
            "status": "success",
            "ttft_ms": 10.0,
            "tpot_ms": decode_tpot,
            "itl_ms": [decode_tpot, decode_tpot + 1],
            "e2e_ms": 20.0,
            "output_tokens": 2,
            "scheduled_at": 1_000_000_000,
            "sent_at": 1_010_000_000,
            "finished_at": 1_020_000_000,
        },
        {
            "request_id": "prefill-0",
            "status": "success",
            "ttft_ms": prefill_ttft,
            "tpot_ms": 2.0,
            "itl_ms": [2.0],
            "e2e_ms": prefill_ttft + 2,
            "output_tokens": 1,
            "scheduled_at": 2_000_000_000,
            "sent_at": 2_000_000_000,
            "finished_at": 2_100_000_000,
        },
    ]


def test_phase_summary_joins_labels_and_applies_all_slos() -> None:
    summary = summarize_labeled_requests(
        trace(),
        rows(8.0, 120.0),
        slo={"ttft_ms": 100.0, "tpot_ms": 10.0, "e2e_ms": 200.0},
    )

    assert summary["overall"]["slo_good_requests"] == 1
    assert summary["phases"]["decode"]["p99_tpot_ms"] == 8.0
    assert summary["phases"]["decode"]["p99_itl_ms"] == 8.99
    assert summary["phases"]["decode"]["p99_client_wait_ms"] == 10.0
    assert summary["phases"]["prefill"]["slo_good_fraction"] == 0.0


def test_oracle_can_expose_different_fixed_winners_by_phase() -> None:
    slo = {"ttft_ms": 100.0, "tpot_ms": 10.0, "e2e_ms": 200.0}
    low = aggregate_policy_trials([summarize_labeled_requests(trace(), rows(5.0, 120.0), slo=slo)])
    high = aggregate_policy_trials([summarize_labeled_requests(trace(), rows(15.0, 50.0), slo=slo)])

    oracle = select_phase_oracle(
        {"fixed-low": low, "fixed-high": high},
        eligible_policies=["fixed-low", "fixed-high"],
    )

    assert oracle["phases"]["decode"]["policy"] == "fixed-low"
    assert oracle["phases"]["prefill"]["policy"] == "fixed-high"
    assert oracle["distinct_winners"] == ["fixed-high", "fixed-low"]
