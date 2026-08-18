"""Focused tests for the M5 engineering KV materiality guardrail."""

from __future__ import annotations

from types import SimpleNamespace

from vllm_tuner.longctx.m5_decode_tail_engineering import (
    MAX_KV_USAGE_ABSOLUTE_DELTA,
    analyze_m5_engineering_records,
)


def _records(peak_delta: float, p95_delta: float) -> list[SimpleNamespace]:
    records: list[SimpleNamespace] = []
    for cohort in ("target", "held-out"):
        for repeat in range(3):
            baseline = SimpleNamespace(
                cohort_id=cohort,
                profile_id="production-default",
                repeat_index=repeat,
                kv_usage=SimpleNamespace(p95=0.01, maximum=0.04),
            )
            candidate = SimpleNamespace(
                cohort_id=cohort,
                profile_id="decode-tail-1024",
                repeat_index=repeat,
                kv_usage=SimpleNamespace(
                    p95=0.01 + p95_delta,
                    maximum=0.04 + peak_delta,
                ),
            )
            records.extend((baseline, candidate))
    return records


def _base_analysis(peak_delta: float) -> dict[str, object]:
    paired = []
    for cohort in ("target", "held-out"):
        pairs = [
            {"repeat_index": repeat, "kv_usage_maximum_delta": peak_delta} for repeat in range(3)
        ]
        aggregates = {
            "kv_usage_maximum_delta": {
                "minimum": peak_delta,
                "median": peak_delta,
                "maximum": peak_delta,
            }
        }
        checks = {
            "paired_repeat_count": True,
            "decode_interference_itl_p99_median_improvement_ge_25pct": True,
            "decode_goodput_median_change_ge_minus_0_5pct": True,
            "decode_goodput_each_repeat_ge_minus_1pct": True,
            "long_prefill_ttft_p99_median_degradation_le_15pct": True,
            "decode_tpot_p99_median_degradation_le_2pct": True,
            "waiting_has_no_opposite_median_worsening": True,
            "kv_usage_has_no_opposite_median_worsening": peak_delta <= 0,
            "zero_oom_timeout_preemption": True,
        }
        acceptance = {
            "passed": peak_delta <= 0,
            "checks": checks,
            "failure_reasons": (
                [] if peak_delta <= 0 else ["kv_usage_has_no_opposite_median_worsening"]
            ),
            "paired_metrics": aggregates,
        }
        paired.append(
            {
                "cohort_id": cohort,
                "repeat_count": 3,
                "pairs": pairs,
                "aggregates": aggregates,
                "acceptance": acceptance,
            }
        )
    return {
        "schema_version": "old",
        "paired": paired,
        "cohort_acceptance": {},
        "acceptance": {"passed": False},
        "decision": {"positive_result": False},
        "preregistered_thresholds": {},
    }


def test_negligible_three_block_scale_peak_is_accepted(monkeypatch: object) -> None:
    peak_delta = 3 / 14_614
    assert peak_delta < MAX_KV_USAGE_ABSOLUTE_DELTA
    monkeypatch.setattr(
        "vllm_tuner.longctx.m5_decode_tail_engineering.analyze_m5_records",
        lambda records, formal: _base_analysis(peak_delta),
    )
    analysis = analyze_m5_engineering_records(_records(peak_delta, 0.0))
    assert analysis["acceptance"]["passed"] is True
    assert analysis["decision"]["profile_id"] == "decode-tail-1024"
    assert all(
        value["checks"]["kv_usage_has_no_material_worsening"] is True
        for value in analysis["cohort_acceptance"].values()
    )


def test_material_kv_growth_is_still_rejected(monkeypatch: object) -> None:
    peak_delta = MAX_KV_USAGE_ABSOLUTE_DELTA * 2
    monkeypatch.setattr(
        "vllm_tuner.longctx.m5_decode_tail_engineering.analyze_m5_records",
        lambda records, formal: _base_analysis(peak_delta),
    )
    analysis = analyze_m5_engineering_records(_records(peak_delta, peak_delta))
    assert analysis["acceptance"]["passed"] is False
    assert analysis["decision"]["profile_id"] == "production-default"
