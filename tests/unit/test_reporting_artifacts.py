"""Evidence-first report and plot artifact tests."""

import html
import json

import plotly.graph_objects as go

from vllm_tuner.reporting.plots import (
    capacity_curve,
    figure_data_available,
    summarize_capacity_rows,
    telemetry_timeline,
)
from vllm_tuner.reporting.report import comparison_rows, generate_report


def test_capacity_curve_does_not_turn_missing_results_into_zero() -> None:
    figure = capacity_curve([{"offered_requests_per_sec": 4.0}])

    assert figure_data_available(figure) is False
    assert [trace.name for trace in figure.data] == ["Target offered load"]


def test_capacity_summary_reports_median_range_and_failed_repeats() -> None:
    summaries = summarize_capacity_rows(
        [
            {
                "target_offered_requests_per_sec": 4.0,
                "empirical_scheduled_requests_per_sec": 4.2,
                "achieved_requests_per_sec": 3.0,
                "goodput_requests_per_sec": 2.0,
                "status": "COMPLETE",
                "feasible": True,
            },
            {
                "target_offered_requests_per_sec": 4.0,
                "empirical_scheduled_requests_per_sec": 3.8,
                "achieved_requests_per_sec": 4.0,
                "goodput_requests_per_sec": 3.0,
                "status": "COMPLETE",
                "feasible": True,
            },
            {
                "target_offered_requests_per_sec": 4.0,
                "empirical_scheduled_requests_per_sec": 99.0,
                # Partial values from a failed point are diagnostic only and must
                # never move the measured capacity summary.
                "achieved_requests_per_sec": 100.0,
                "goodput_requests_per_sec": 100.0,
                "status": "FAILED",
                "feasible": False,
            },
        ]
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["offered_requests_per_sec"] == 4.0
    assert summary["target_offered_requests_per_sec"] == 4.0
    assert summary["repeat_count"] == 3
    assert summary["complete_count"] == 2
    assert summary["feasible_count"] == 2
    assert summary["failed_count"] == 1
    assert summary["measured_count"] == 2
    assert summary["median_empirical_scheduled_requests_per_sec"] == 4.0
    assert summary["min_empirical_scheduled_requests_per_sec"] == 3.8
    assert summary["max_empirical_scheduled_requests_per_sec"] == 4.2
    assert summary["median_achieved_requests_per_sec"] == 3.5
    assert summary["min_achieved_requests_per_sec"] == 3.0
    assert summary["max_achieved_requests_per_sec"] == 4.0
    assert summary["median_goodput_requests_per_sec"] == 2.5
    assert summary["min_goodput_requests_per_sec"] == 2.0
    assert summary["max_goodput_requests_per_sec"] == 3.0
    assert summary["median_p99_ttft_ms"] is None


def test_repeat_comparison_keeps_distinct_tpe_candidates_separate() -> None:
    rows = comparison_rows(
        [
            {
                "method": "tpe",
                "repeat_of": candidate,
                "parameters": {"max_num_seqs": 8 * candidate},
                "status": "COMPLETE",
                "feasible": True,
                "goodput_requests_per_sec": float(candidate),
            }
            for candidate in (1, 2)
        ]
    )

    assert len(rows) == 2
    assert {row["method"] for row in rows} == {"tpe-1", "tpe-2"}


def test_telemetry_timeline_reads_namespaced_engine_and_gpu_series() -> None:
    figure = telemetry_timeline(
        [{"first_token_at": 1_100_000_000, "ttft_ms": 10.0}],
        [
            {
                "monotonic_ns": 1_000_000_000,
                "metrics": {
                    "num_requests_waiting": 2,
                    "kv_cache_usage_perc": 0.5,
                },
            }
        ],
        [
            {
                "monotonic_ns": 1_200_000_000,
                "gpu_utilization_percent": 75,
                "memory_used_mb": 512,
            }
        ],
    )

    assert figure_data_available(figure) is True
    assert {trace.name for trace in figure.data} == {
        "client.ttft_ms",
        "engine.waiting",
        "engine.kv_cache_usage",
        "gpu.utilization_percent",
        "gpu.memory_used_mb",
    }


def test_report_records_html_fallback_when_static_renderer_is_missing(
    tmp_path, monkeypatch
) -> None:
    def unavailable_renderer(self, *args, **kwargs):
        raise ValueError("kaleido is not installed")

    monkeypatch.setattr(go.Figure, "write_image", unavailable_renderer)
    paths = generate_report(
        tmp_path,
        manifest={
            "experiment_id": "exp",
            "model": "model",
            "trace_sha256": "search",
            "holdout_trace_sha256": "holdout",
            "search_space_sha256": "space",
        },
        trials=[
            {
                "trial_number": 0,
                "method": "default",
                "status": "COMPLETE",
                "feasible": True,
                "offered_requests_per_sec": 2.0,
                "achieved_requests_per_sec": 1.8,
                "goodput_requests_per_sec": 1.5,
                "p99_ttft_ms": 20.0,
            }
        ],
        capacity_sweep=[
            {
                "offered_requests_per_sec": 2.0,
                "target_offered_requests_per_sec": 2.0,
                "empirical_scheduled_requests_per_sec": 2.25,
                "achieved_requests_per_sec": 1.8,
                "goodput_requests_per_sec": 1.5,
                "status": "COMPLETE",
                "feasible": True,
            }
        ],
    )

    plot_manifest = json.loads(paths["plot_manifest"].read_text())
    for record in plot_manifest["plots"].values():
        assert (tmp_path / record["html"]).exists()
        assert record["static_image_available"] is False
        assert "kaleido is not installed" in record["fallback_reason"]
    assert plot_manifest["plots"]["capacity_curve"]["data_available"] is True
    assert plot_manifest["plots"]["telemetry_timeline"]["data_available"] is False
    assert "interactive HTML fallback" in paths["markdown"].read_text()
    assert "Holdout trace SHA-256: `holdout`" in paths["markdown"].read_text()
    assert "Target offered req/s | Empirical scheduled req/s" in paths["markdown"].read_text()
    assert "| 2.000 | 2.250 (2.250–2.250) |" in paths["markdown"].read_text()


def test_report_renders_validated_default_negative_conditions_and_preemptions(
    tmp_path, monkeypatch
) -> None:
    def unavailable_renderer(self, *args, **kwargs):
        raise ValueError("no renderer")

    monkeypatch.setattr(go.Figure, "write_image", unavailable_renderer)
    best = {
        "validated": True,
        "candidate": "default-0",
        "method": "default",
        "metric_provenance": "median_of_complete_feasible_repeats",
        "parameters": {"max_num_seqs": 8},
        "repeat_metrics": {
            "goodput_requests_per_sec": {
                "median": 3.0,
                "min": 2.9,
                "max": 3.1,
                "count": 3,
            }
        },
        "holdout_metrics": {
            "goodput_requests_per_sec": {
                "median": 2.8,
                "min": 2.7,
                "max": 2.9,
                "count": 3,
            }
        },
    }
    condition = {
        "trace_name": "held_out",
        "metric": "p99_ttft",
        "adaptive_value": 0.12,
        "fixed_value": 0.10,
        "fixed_budget": 1024,
        "relative_gain": -0.2,
        "explanation": "Adaptive overhead regressed held-out TTFT.",
    }
    paths = generate_report(
        tmp_path,
        manifest={"experiment_id": "exp", "model": "model"},
        trials=[],
        best=best,
        scheduler_results=[
            {
                "trace": "held_out",
                "policy": "adaptive",
                "budget": None,
                "goodput": 3.0,
                "p99_ttft": 0.12,
                "p99_tpot": 0.01,
                "fairness_index": 0.99,
                "starvation_count": 1,
                "preemption_count": 3,
            }
        ],
        scheduler_negative_conditions=[condition],
    )

    markdown = paths["markdown"].read_text()
    assert "## Validated best" in markdown
    assert "validated default configuration remained best" in markdown
    assert "no tuning improvement is claimed" in markdown
    assert "| Starvation | Preemptions |" in markdown
    assert "| 1 | 3 |" in markdown
    assert condition["explanation"] in markdown
    html_text = paths["html"].read_text()
    assert "scheduler_negative_conditions" in html_text
    assert "Adaptive overhead regressed held-out TTFT." in html.unescape(html_text)
