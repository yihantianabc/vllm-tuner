"""High-level orchestration test without starting a real GPU server."""

import json
from pathlib import Path

import pandas as pd
import pytest

from vllm_tuner.config.models import (
    StudySettings,
    TelemetryConfig,
    TuningConfig,
    WorkloadConfig,
)
from vllm_tuner.experiment.artifacts import SUMMARY_COMPACT_FILE
from vllm_tuner.experiment.models import (
    EnvironmentFingerprint,
    ExperimentSpec,
    TrialResult,
    TrialStatus,
)
from vllm_tuner.experiment.manifest import sha256_file
from vllm_tuner.experiment.runner import SLOTuneExperimentRunner
from vllm_tuner.profiling.nvml_session import NVMLSample, summarize_nvml_samples
from vllm_tuner.tuning.optimizer import SearchMethod, SearchRun, SearchTrial


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens)


def _clean_cleanup_status():
    return {
        "clean": True,
        "process_group_empty": True,
        "port_available": True,
        "gpu_clean": True,
    }


class FakeTrialController:
    calls = []

    def __init__(self, config, trace, artifacts, **kwargs):
        self.trace = trace
        self.artifacts = artifacts

    def _write_measured_artifacts(self, trial_id, params, num_requests=1):
        base = Path("trials") / trial_id
        requests = [
            {
                "request_id": f"{trial_id}-request-{index}",
                "status": "success",
                "input_tokens": 1,
                "output_tokens": 1,
            }
            for index in range(num_requests)
        ]
        aggregate = {
            "num_requests": num_requests,
            "completed": num_requests,
            "failed": 0,
            "total_input_tokens": num_requests,
            "total_output_tokens": num_requests,
            "duration": 1.0,
        }
        self.artifacts.write_json(
            base / "server-command.json",
            {"argv": ["python", "-m", "fake_vllm"], "environment": {}},
        )
        self.artifacts.write_json(base / "params.json", params)
        self.artifacts.write_json(
            base / "status.json",
            {"status": "COMPLETE", "history": [{"current": "COMPLETE"}]},
        )
        self.artifacts.write_jsonl(
            base / "request-results.jsonl",
            requests,
        )
        self.artifacts.write_json(
            base / "benchmark-raw.json",
            {"backend": "fake", "request_results": requests, "aggregate": aggregate},
        )
        self.artifacts.write_jsonl(base / "prometheus.jsonl", [])
        self.artifacts.write_jsonl(base / "nvml.jsonl", [])
        self.artifacts.write_text(base / "server.log", "fake vLLM server completed\n")
        self.artifacts.write_json(base / "cleanup.json", _clean_cleanup_status())
        return aggregate

    async def run_trial(self, params, trial_id, method):
        type(self).calls.append(trial_id)
        aggregate = self._write_measured_artifacts(trial_id, params)
        return TrialResult(
            trial_id=trial_id,
            method=method,
            status=TrialStatus.COMPLETE,
            params=params,
            measurement_seconds=1.0,
            client={
                **aggregate,
                "goodput_requests_per_sec": 2.0,
                "offered_requests_per_sec": 2.0,
                "achieved_requests_per_sec": 2.0,
                "p99_ttft_ms": 10.0,
            },
            constraints={"feasible": True, "violations": []},
            cleanup_status=_clean_cleanup_status(),
        )


class CapacityTrialController(FakeTrialController):
    async def run_trial(self, params, trial_id, method):
        if trial_id == "capacity-rate-1-repeat-0":
            raise RuntimeError("injected capacity failure")
        rate = float(self.trace.request_rate or 2.0)
        scheduled_span = (
            self.trace.entries[-1].scheduled_offset_seconds
            - self.trace.entries[0].scheduled_offset_seconds
        )
        empirical_scheduled_rate = (
            (len(self.trace.entries) - 1) / scheduled_span if scheduled_span > 0 else None
        )
        aggregate = self._write_measured_artifacts(trial_id, params, num_requests=3)
        return TrialResult(
            trial_id=trial_id,
            method=method,
            status=TrialStatus.COMPLETE,
            params=params,
            measurement_seconds=1.0,
            client={
                **aggregate,
                "goodput_requests_per_sec": 0.75 * rate,
                "offered_requests_per_sec": rate,
                "target_offered_requests_per_sec": rate,
                "empirical_scheduled_requests_per_sec": empirical_scheduled_rate,
                "achieved_requests_per_sec": 0.9 * rate,
                "request_throughput": 0.9 * rate,
                "output_throughput": 10.0 * rate,
                "total_token_throughput": 12.0 * rate,
                "p50_ttft_ms": 5.0 + rate,
                "p95_ttft_ms": 8.0 + rate,
                "p99_ttft_ms": 10.0 + rate,
                "p50_tpot_ms": 1.0,
                "p95_tpot_ms": 1.5,
                "p99_tpot_ms": 2.0,
                "p50_e2e_ms": 20.0,
                "p95_e2e_ms": 25.0,
                "p99_e2e_ms": 30.0,
                "completed": 3,
                "failed": 0,
                "error_rate": 0.0,
                "error_types": {},
            },
            engine={
                "num_requests_waiting": {"peak": rate},
                "kv_cache_usage_perc": {"peak": 0.5},
                "num_preemptions_total": {"delta": 1.0},
            },
            gpu={
                "peak_memory_mb": 512.0,
                "p95_memory_mb": 500.0,
                "mean_gpu_utilization_percent": 75.0,
                "energy_joules": 100.0,
                "energy_per_output_token_joules": 1.25,
            },
            constraints={"feasible": True, "violations": []},
            cleanup_status=_clean_cleanup_status(),
        )


def _fake_scheduler():
    metrics = {
        "goodput": 1.0,
        "p99_ttft": 0.1,
        "p99_tpot": 0.01,
        "fairness_index": 1.0,
        "starvation_count": 0,
        "preemption_count": 0,
    }
    section = {
        "trace_name": "calibration",
        "adaptive": {
            "policy_name": "adaptive",
            "seed": 1,
            "metrics": metrics,
            "requests": [{"request_id": "scheduler-request"}],
            "steps": [{"step": 1}],
            "decisions": [{"budget": 512}],
        },
        "fixed_baselines": {},
        "best_fixed_budget": None,
        "goodput_gain_vs_best": 0.0,
        "negative_gain_conditions": [],
    }
    return {
        "calibration": section,
        "held_out": {**section, "trace_name": "held_out"},
        "has_negative_result": False,
        "negative_gain_conditions": [],
    }


@pytest.mark.asyncio
async def test_experiment_runner_writes_search_repeat_and_report(tmp_path, monkeypatch) -> None:
    config = TuningConfig(
        model="fake-model",
        workload=WorkloadConfig(
            name="chat",
            dataset_name="unused",
            sample_size=2,
            warmup_requests=0,
            max_tokens=2,
        ),
        telemetry=TelemetryConfig(enabled=False),
        study=StudySettings(
            trial_budget=1,
            methods=["default"],
            repeat_count=1,
            top_candidates=1,
            holdout_enabled=False,
        ),
    )
    runner = SLOTuneExperimentRunner(
        config,
        "experiment",
        results_root=tmp_path,
        repository=".",
        tokenizer=FakeTokenizer(),
    )
    monkeypatch.setattr("vllm_tuner.experiment.runner.TrialController", FakeTrialController)
    monkeypatch.setattr(runner, "_save_environment", lambda: None)
    monkeypatch.setattr(runner, "_scheduler_ablation", lambda trace, holdout: _fake_scheduler())

    summary = await runner.run()

    assert summary["best"]["goodput_requests_per_sec"] == 2.0
    assert (runner.artifacts.root / "aggregate/trials.parquet").exists()
    assert (runner.artifacts.root / "aggregate/repeated-results.parquet").exists()
    assert (runner.artifacts.root / "aggregate/holdout-results.parquet").exists()
    assert (runner.artifacts.root / "aggregate/capacity-sweep.parquet").exists()
    assert (runner.artifacts.root / "report/report.html").exists()
    assert (runner.artifacts.root / "report/report.md").exists()
    assert (runner.artifacts.root / "report/capacity-curve.html").exists()
    assert (runner.artifacts.root / "report/search-trajectory.html").exists()
    assert (runner.artifacts.root / "report/telemetry-timeline.html").exists()
    assert (runner.artifacts.root / "report/plot-manifest.json").exists()
    assert (runner.artifacts.root / "trace.jsonl").exists()
    assert (runner.artifacts.root / "holdout-trace.jsonl").exists()
    manifest = json.loads((runner.artifacts.root / "manifest.json").read_text())
    assert manifest["holdout_trace_sha256"]
    assert manifest["report_artifacts"]["plots"]["capacity_curve"]["data_available"] is False
    assert manifest["report_artifacts"]["plots"]["telemetry_timeline"]["data_available"] is False
    trial_directories = list((runner.artifacts.root / "trials").iterdir())
    assert trial_directories
    for trial_dir in trial_directories:
        for name in (
            "request-results.jsonl",
            "prometheus.jsonl",
            "nvml.jsonl",
            "server.log",
            "summary.json",
            "artifact-status.json",
        ):
            assert (trial_dir / name).exists()
    assert (runner.artifacts.root / "summary.json").exists()
    assert (runner.artifacts.root / SUMMARY_COMPACT_FILE).exists()
    assert (runner.artifacts.root / "experiment-integrity.json").exists()
    assert (runner.artifacts.root / "lineage.json").exists()
    assert (runner.artifacts.root / "experiment-audit.json").exists()
    runner.artifacts.validate_experiment_integrity()
    raw_scheduler = runner.artifacts.root / "aggregate/scheduler-ablation.json"
    assert summary["scheduler_ablation"]["raw_sha256"] == sha256_file(raw_scheduler)
    assert "requests" not in summary["scheduler_ablation"]["calibration"]["adaptive"]
    scheduler_raw = json.loads(raw_scheduler.read_text())
    assert scheduler_raw["calibration"]["adaptive"]["requests"]
    original_summary = json.loads((runner.artifacts.root / "summary.json").read_text())
    assert original_summary["scheduler_ablation"] == scheduler_raw
    assert "experiment_attestation" not in original_summary
    assert summary == json.loads((runner.artifacts.root / SUMMARY_COMPACT_FILE).read_text())
    repeat_summary = json.loads(
        (runner.artifacts.trials_dir / "repeat-default-0-0" / "summary.json").read_text()
    )
    assert repeat_summary["method"] == "default"
    assert repeat_summary["phase"] == "repeat"
    assert repeat_summary["source_method"] == "default"
    assert repeat_summary["source_trial_id"] == "default-0000"
    report_markdown = (runner.artifacts.report_dir / "report.md").read_text(encoding="utf-8")
    assert "validated default configuration remained best" in report_markdown


@pytest.mark.asyncio
async def test_capacity_sweep_repeats_defaults_and_continues_after_failure(
    tmp_path, monkeypatch
) -> None:
    config = TuningConfig(
        model="fake-model",
        workload=WorkloadConfig(
            name="chat",
            dataset_name="unused",
            sample_size=3,
            warmup_requests=0,
            max_tokens=2,
            capacity_request_rates=[1.0, 4.0],
            capacity_repeats=2,
        ),
        telemetry=TelemetryConfig(enabled=False),
        study=StudySettings(
            trial_budget=1,
            methods=["default"],
            repeat_count=1,
            top_candidates=1,
            holdout_enabled=False,
        ),
    )
    runner = SLOTuneExperimentRunner(
        config,
        "capacity-experiment",
        results_root=tmp_path,
        repository=".",
        tokenizer=FakeTokenizer(),
    )
    monkeypatch.setattr("vllm_tuner.experiment.runner.TrialController", CapacityTrialController)
    monkeypatch.setattr(runner, "_save_environment", lambda: None)
    monkeypatch.setattr(runner, "_scheduler_ablation", lambda trace, holdout: _fake_scheduler())

    summary = await runner.run()

    points = summary["capacity_sweep"]["points"]
    assert len(points) == 4
    assert {point["offered_requests_per_sec"] for point in points} == {1.0, 4.0}
    assert all(
        point["target_offered_requests_per_sec"] == point["offered_requests_per_sec"]
        for point in points
    )
    assert all(
        point["measured_offered_requests_per_sec"] == point["target_offered_requests_per_sec"]
        for point in points
    )
    assert all(point["empirical_scheduled_requests_per_sec"] is not None for point in points)
    assert sum(point["status"] == "FAILED" for point in points) == 1
    assert sum(point["status"] == "COMPLETE" for point in points) == 3
    assert all(
        point["parameters"] == {"tensor_parallel_size": 1, "pipeline_parallel_size": 1}
        for point in points
    )

    for rate_slug in ("1", "4"):
        for repeat in range(2):
            trial_dir = runner.artifacts.trials_dir / f"capacity-rate-{rate_slug}-repeat-{repeat}"
            assert (trial_dir / "capacity-trace.jsonl").exists()
            assert (trial_dir / "capacity-trace.sha256").exists()
            assert (trial_dir / "capacity-point.json").exists()
            assert (trial_dir / "request-results.jsonl").exists()
            assert (trial_dir / "summary.json").exists()

    raw = pd.read_parquet(runner.artifacts.root / "aggregate/capacity-sweep.parquet")
    grouped = pd.read_parquet(runner.artifacts.root / "aggregate/capacity-sweep-summary.parquet")
    assert len(raw) == 4
    assert {
        "target_offered_requests_per_sec",
        "empirical_scheduled_requests_per_sec",
        "output_throughput",
        "p50_ttft_ms",
        "peak_waiting_requests",
        "peak_kv_cache_usage",
        "preemptions",
        "peak_memory_mb",
        "p95_memory_mb",
        "mean_gpu_utilization_percent",
        "energy_joules",
        "energy_per_output_token_joules",
    }.issubset(raw.columns)
    assert len(grouped) == 2
    rate_one = grouped.loc[grouped["offered_requests_per_sec"] == 1.0].iloc[0]
    assert rate_one["repeat_count"] == 2
    assert rate_one["failed_count"] == 1
    assert rate_one["median_output_throughput"] == 10.0
    assert rate_one["median_peak_waiting_requests"] == 1.0
    assert rate_one["median_p95_memory_mb"] == 500.0
    assert rate_one["median_energy_per_output_token_joules"] == 1.25
    complete_rate_one = next(
        point
        for point in points
        if point["target_offered_requests_per_sec"] == 1.0 and point["status"] == "COMPLETE"
    )
    capacity_trace = (
        runner.artifacts.trials_dir / complete_rate_one["trial_id"] / "capacity-trace.jsonl"
    )
    trace_rows = [json.loads(line) for line in capacity_trace.read_text().splitlines()]
    expected_empirical = (len(trace_rows) - 1) / (
        trace_rows[-1]["scheduled_offset_seconds"] - trace_rows[0]["scheduled_offset_seconds"]
    )
    assert complete_rate_one["empirical_scheduled_requests_per_sec"] == pytest.approx(
        expected_empirical
    )
    assert complete_rate_one["empirical_scheduled_requests_per_sec"] != pytest.approx(1.0)
    assert rate_one["median_empirical_scheduled_requests_per_sec"] == pytest.approx(
        expected_empirical
    )

    plot_manifest = json.loads((runner.artifacts.report_dir / "plot-manifest.json").read_text())
    capacity_plot = plot_manifest["plots"]["capacity_curve"]
    assert capacity_plot["data_available"] is True
    assert capacity_plot["data_source"] == "aggregate/capacity-sweep.parquet"


def _fast_report(output_dir, **kwargs):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    markdown = destination / "report.md"
    html = destination / "report.html"
    plot_manifest = destination / "plot-manifest.json"
    markdown.write_text("# fake report\n")
    html.write_text("<html></html>\n")
    plot_manifest.write_text(json.dumps({"schema_version": 1, "plots": {}}) + "\n")
    return {"markdown": markdown, "html": html, "plot_manifest": plot_manifest}


def _search_trial(
    *,
    number=0,
    value=10.0,
    status=TrialStatus.COMPLETE,
    repeat_of=None,
    holdout=False,
):
    feasible = status == TrialStatus.COMPLETE
    result = TrialResult(
        trial_id=f"trial-{number}-{repeat_of}-{holdout}",
        method="default",
        status=status,
        params={"max_num_seqs": 8},
        client={"goodput_requests_per_sec": value} if value is not None else {},
        constraints={"feasible": feasible, "violations": [] if feasible else ["failed"]},
        cleanup_status=_clean_cleanup_status() if feasible else None,
    )
    return SearchTrial(
        number=number,
        method=SearchMethod.DEFAULT,
        params={"max_num_seqs": 8},
        status=status,
        objective=value if feasible else None,
        result=result,
        repeat_of=repeat_of,
        holdout=holdout,
    )


def test_top_candidates_preserve_same_parameters_from_each_method() -> None:
    params = {"max_num_seqs": 8}
    runs = {
        method: SearchRun(
            method=method,
            requested_budget=1,
            trials=[
                SearchTrial(
                    number=0,
                    method=method,
                    params=dict(params),
                    status=TrialStatus.COMPLETE,
                    objective=float(index + 1),
                )
            ],
        )
        for index, method in enumerate(
            (SearchMethod.DEFAULT, SearchMethod.RANDOM, SearchMethod.TPE)
        )
    }

    candidates = SLOTuneExperimentRunner._top_candidates(runs, count=3)

    assert [candidate.method for candidate in candidates] == [
        SearchMethod.DEFAULT,
        SearchMethod.RANDOM,
        SearchMethod.TPE,
    ]
    assert all(candidate.params == params for candidate in candidates)


def test_final_best_requires_every_repeat_holdout_and_no_large_degradation(tmp_path) -> None:
    config = TuningConfig(
        model="fake-model",
        telemetry=TelemetryConfig(enabled=False),
        study=StudySettings(
            trial_budget=1,
            methods=["default"],
            repeat_count=3,
            top_candidates=1,
            holdout_enabled=True,
            holdout_min_goodput_ratio=0.8,
        ),
    )
    runner = SLOTuneExperimentRunner(
        config,
        "validation",
        results_root=tmp_path,
        repository=".",
        tokenizer=FakeTokenizer(),
    )
    candidate = _search_trial(value=12.0)
    repeats = [
        _search_trial(number=index, value=value, repeat_of=0)
        for index, value in enumerate((10.0, 11.0, 12.0))
    ]
    incomplete_holdout = [
        _search_trial(number=0, value=9.0, repeat_of=0, holdout=True),
        _search_trial(
            number=1,
            value=None,
            status=TrialStatus.FAILED,
            repeat_of=0,
            holdout=True,
        ),
        _search_trial(number=2, value=9.0, repeat_of=0, holdout=True),
    ]

    rows, selected, best = runner._validate_candidates([candidate], repeats, incomplete_holdout)
    assert selected is None
    assert best is None
    assert rows[0]["rejection_reasons"] == ["not_all_holdouts_complete_and_feasible"]

    degraded_holdout = [
        _search_trial(number=index, value=5.0, repeat_of=0, holdout=True) for index in range(3)
    ]
    rows, selected, best = runner._validate_candidates([candidate], repeats, degraded_holdout)
    assert selected is None
    assert best is None
    assert rows[0]["rejection_reasons"] == ["holdout_goodput_degraded"]

    valid_holdout = [
        _search_trial(number=index, value=9.0, repeat_of=0, holdout=True) for index in range(3)
    ]
    assert candidate.result is not None
    candidate.result.client.update(
        {
            "achieved_requests_per_sec": 999.0,
            "p99_ttft_ms": 999.0,
        }
    )
    for trial, ttft in zip(repeats, (10.0, 20.0, 30.0)):
        assert trial.result is not None
        trial.result.client.update(
            {
                "achieved_requests_per_sec": ttft / 2,
                "p99_ttft_ms": ttft,
            }
        )
    for trial, ttft in zip(valid_holdout, (40.0, 50.0, 60.0)):
        assert trial.result is not None
        trial.result.client.update(
            {
                "achieved_requests_per_sec": ttft / 2,
                "p99_ttft_ms": ttft,
            }
        )

    rows, selected, best = runner._validate_candidates([candidate], repeats, valid_holdout)
    assert selected is candidate
    assert best is not None
    assert best["validated"] is True
    assert best["goodput_requests_per_sec"] == 11.0
    assert best["metric_provenance"] == "median_of_complete_feasible_repeats"
    assert best["p99_ttft_ms"] == 20.0
    assert best["achieved_requests_per_sec"] == 10.0
    assert best["repeat_metrics"]["p99_ttft_ms"] == {
        "count": 3,
        "median": 20.0,
        "min": 10.0,
        "max": 30.0,
    }
    assert best["holdout_metrics"]["p99_ttft_ms"]["median"] == 50.0
    assert best["search_observation"]["goodput_requests_per_sec"] == 12.0
    assert best["search_observation"]["p99_ttft_ms"] == 999.0
    assert "trial_id" not in best
    assert "trial_number" not in best
    assert "search_goodput_requests_per_sec" not in best
    assert rows[0]["holdout_to_repeat_goodput_ratio"] == pytest.approx(9.0 / 11.0)


def test_selectable_telemetry_requires_continuous_core_and_energy_evidence(tmp_path) -> None:
    config = TuningConfig(
        model="fake-model",
        telemetry=TelemetryConfig(
            enabled=True,
            collect_nvml=True,
            collect_energy=True,
        ),
        study=StudySettings(
            trial_budget=1,
            methods=["default"],
            repeat_count=1,
            top_candidates=1,
            holdout_enabled=False,
        ),
    )
    runner = SLOTuneExperimentRunner(
        config,
        "telemetry-gate",
        results_root=tmp_path,
        repository=".",
        tokenizer=FakeTokenizer(),
    )
    gpu_summary = summarize_nvml_samples(
        [
            NVMLSample(
                0,
                0,
                "a",
                memory_used_mb=900,
                gpu_utilization_percent=40,
                power_w=100,
            ),
            NVMLSample(
                0,
                1_000_000_000,
                "b",
                memory_used_mb=1000,
                gpu_utilization_percent=50,
                power_w=150,
            ),
            NVMLSample(
                0,
                2_000_000_000,
                "c",
                memory_used_mb=950,
                gpu_utilization_percent=60,
                power_w=200,
            ),
        ],
        output_tokens=600,
    )
    result = TrialResult(
        trial_id="trial",
        method="default",
        status=TrialStatus.COMPLETE,
        params={"max_num_seqs": 8},
        engine={
            "available": True,
            "sample_count": 5,
            "successful_sample_count": 5,
            "num_requests_running": {"available": True},
            "num_requests_waiting": {"available": True},
            "kv_cache_usage_perc": {"available": True},
            "num_preemptions_total": {"available": True, "delta": 0},
            "prompt_tokens_total": {"available": True, "delta": 100},
            "generation_tokens_total": {"available": True, "delta": 50},
        },
        gpu=gpu_summary,
        constraints={"feasible": True},
    )

    assert runner._telemetry_evidence_errors(result) == []

    result.engine["successful_sample_count"] = 1
    result.engine["generation_tokens_total"] = {"available": True, "delta": None}
    result.gpu["gpu_utilization_percent"] = {"available": False}
    result.gpu["energy_per_output_token_joules"] = None
    errors = runner._telemetry_evidence_errors(result)

    assert "engine.successful_sample_count<2" in errors
    assert "engine.sample_coverage<0.8" in errors
    assert "engine.generation_tokens_total.delta" in errors
    assert "gpu.gpu_utilization_percent" in errors
    assert "gpu.energy_per_output_token_joules" in errors


def test_formal_runner_rejects_dirty_source_before_creating_artifacts(
    tmp_path, monkeypatch
) -> None:
    trace = tmp_path / "trace.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    trace.write_text('{"request_id":"one"}\n', encoding="utf-8")
    holdout.write_text('{"request_id":"held-out"}\n', encoding="utf-8")
    manifest = ExperimentSpec(
        experiment_id="formal-dirty",
        model="fake-model",
        trace_sha256="trace",
        holdout_trace_sha256="holdout",
        workload={"name": "chat"},
        slo={"ttft_ms": 100},
        search_space={"max_num_seqs": [1]},
        search_space_sha256="space",
        seed=1,
        environment=EnvironmentFingerprint(python_version="3", platform="test"),
        source_commit="commit",
        source_tree_sha256="dirty-tree",
        dirty_worktree=True,
    )
    monkeypatch.setattr("vllm_tuner.experiment.runner.build_manifest", lambda **kwargs: manifest)
    runner = SLOTuneExperimentRunner(
        TuningConfig(model="fake-model"),
        "formal-dirty",
        results_root=tmp_path / "results",
        repository=tmp_path,
        tokenizer=FakeTokenizer(),
        require_clean_source=True,
    )

    with pytest.raises(ValueError, match="clean Git worktree"):
        runner._initialize_artifacts(trace, holdout)

    assert not runner.artifacts.root.exists()


@pytest.mark.asyncio
async def test_identical_search_and_holdout_trace_is_rejected(tmp_path) -> None:
    trace_path = tmp_path / "same.jsonl"
    rows = [
        {
            "request_id": "request-0",
            "scheduled_offset_seconds": 0.0,
            "prompt": "hello",
            "input_tokens": 1,
            "output_tokens": 1,
            "profile": "chat",
        },
        {
            "request_id": "request-1",
            "scheduled_offset_seconds": 0.5,
            "prompt": "world",
            "input_tokens": 1,
            "output_tokens": 1,
            "profile": "chat",
        },
    ]
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    runner = SLOTuneExperimentRunner(
        TuningConfig(
            model="fake-model",
            workload=WorkloadConfig(name="chat", request_rate=2.0, sample_size=2),
            telemetry=TelemetryConfig(enabled=False),
            study=StudySettings(trial_budget=1, methods=["default"], repeat_count=1),
        ),
        "identical-traces",
        results_root=tmp_path,
        repository=".",
        trace_path=trace_path,
        holdout_trace_path=trace_path,
        tokenizer=FakeTokenizer(),
    )

    with pytest.raises(ValueError, match="identical"):
        await runner.run()
    assert not runner.artifacts.root.exists()


@pytest.mark.asyncio
async def test_identical_trace_is_allowed_when_holdout_is_disabled(tmp_path, monkeypatch) -> None:
    trace_path = tmp_path / "same.jsonl"
    rows = [
        {
            "request_id": "request-0",
            "scheduled_offset_seconds": 0.0,
            "prompt": "hello",
            "input_tokens": 1,
            "output_tokens": 1,
            "profile": "chat",
        }
    ]
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    runner = SLOTuneExperimentRunner(
        TuningConfig(
            model="fake-model",
            workload=WorkloadConfig(name="chat", sample_size=1),
            telemetry=TelemetryConfig(enabled=False),
            study=StudySettings(
                trial_budget=1,
                methods=["default"],
                repeat_count=1,
                holdout_enabled=False,
            ),
        ),
        "identical-traces-without-holdout",
        results_root=tmp_path,
        repository=".",
        trace_path=trace_path,
        holdout_trace_path=trace_path,
        tokenizer=FakeTokenizer(),
    )

    def reached_artifact_initialization(*args, **kwargs):
        raise RuntimeError("artifact initialization reached")

    monkeypatch.setattr(runner, "_initialize_artifacts", reached_artifact_initialization)
    with pytest.raises(RuntimeError, match="artifact initialization reached"):
        await runner.run()


@pytest.mark.asyncio
async def test_resume_replays_complete_trials_without_rerunning(tmp_path, monkeypatch) -> None:
    config = TuningConfig(
        model="fake-model",
        workload=WorkloadConfig(
            name="chat",
            dataset_name="unused",
            sample_size=2,
            warmup_requests=0,
            max_tokens=2,
        ),
        telemetry=TelemetryConfig(enabled=False),
        study=StudySettings(
            trial_budget=1,
            methods=["default"],
            repeat_count=1,
            top_candidates=1,
            holdout_enabled=False,
        ),
    )
    monkeypatch.setattr("vllm_tuner.experiment.runner.TrialController", FakeTrialController)
    monkeypatch.setattr("vllm_tuner.experiment.runner.generate_report", _fast_report)
    FakeTrialController.calls = []
    first = SLOTuneExperimentRunner(
        config,
        "resume-experiment",
        results_root=tmp_path,
        repository=".",
        tokenizer=FakeTokenizer(),
    )
    monkeypatch.setattr(first, "_save_environment", lambda: None)
    monkeypatch.setattr(first, "_scheduler_ablation", lambda trace, holdout: _fake_scheduler())
    first_summary = await first.run()
    first_calls = list(FakeTrialController.calls)
    assert first_calls == ["default-0000", "repeat-default-0-0"]

    config.study.resume = True
    resumed = SLOTuneExperimentRunner(
        config,
        "resume-experiment",
        results_root=tmp_path,
        repository=".",
        tokenizer=FakeTokenizer(),
    )
    monkeypatch.setattr(resumed, "_save_environment", lambda: None)
    monkeypatch.setattr(resumed, "_scheduler_ablation", lambda trace, holdout: _fake_scheduler())
    resumed_summary = await resumed.run()

    assert FakeTrialController.calls == first_calls
    assert resumed_summary["best"] == first_summary["best"]


def test_selectable_trial_without_required_raw_evidence_is_failed(tmp_path) -> None:
    config = TuningConfig(
        model="fake-model",
        telemetry=TelemetryConfig(enabled=False),
        study=StudySettings(trial_budget=1, methods=["default"]),
    )
    runner = SLOTuneExperimentRunner(
        config,
        "artifact-gate",
        results_root=tmp_path,
        repository=".",
        tokenizer=FakeTokenizer(),
    )
    runner.artifacts.initialize()
    result = TrialResult(
        trial_id="default-0000",
        method="default",
        status=TrialStatus.COMPLETE,
        params={"max_num_seqs": 8},
        client={"goodput_requests_per_sec": 2.0},
        constraints={"feasible": True, "violations": []},
        cleanup_status=_clean_cleanup_status(),
    )

    runner._finalize_trial_artifacts(result)

    assert result.status == TrialStatus.FAILED
    assert result.failure_reason["type"] == "ARTIFACT_UNAVAILABLE"
    saved = json.loads((runner.artifacts.trials_dir / "default-0000" / "summary.json").read_text())
    assert saved["status"] == "FAILED"


def test_complete_trial_without_verified_cleanup_is_cleanup_error(tmp_path) -> None:
    runner = SLOTuneExperimentRunner(
        TuningConfig(
            model="fake-model",
            telemetry=TelemetryConfig(enabled=False),
            study=StudySettings(trial_budget=1, methods=["default"]),
        ),
        "cleanup-gate",
        results_root=tmp_path,
        repository=".",
        tokenizer=FakeTokenizer(),
    )
    runner.artifacts.initialize()
    result = TrialResult(
        trial_id="default-0000",
        method="default",
        status=TrialStatus.COMPLETE,
        params={"max_num_seqs": 8},
        constraints={"feasible": True, "violations": []},
        cleanup_status=None,
    )

    runner._finalize_trial_artifacts(result)

    assert result.status == TrialStatus.FAILED
    assert result.failure_reason["type"] == "CLEANUP_ERROR"
    assert "cleanup_error" in result.constraints["violations"]
    runner.artifacts.validate_trial_integrity(result.trial_id)


def test_fresh_semantic_mismatch_is_failed_and_lifecycle_is_synchronized(tmp_path) -> None:
    runner = SLOTuneExperimentRunner(
        TuningConfig(
            model="fake-model",
            telemetry=TelemetryConfig(enabled=False),
            study=StudySettings(trial_budget=1, methods=["default"]),
        ),
        "semantic-gate",
        results_root=tmp_path,
        repository=".",
        tokenizer=FakeTokenizer(),
    )
    runner.artifacts.initialize()
    trial_id = "default-0000"
    base = Path("trials") / trial_id
    request = {
        "request_id": "request-0",
        "status": "success",
        "input_tokens": 1,
        "output_tokens": 1,
    }
    raw_aggregate = {
        "num_requests": 1,
        "completed": 1,
        "failed": 0,
        "total_input_tokens": 1,
        "total_output_tokens": 1,
    }
    result = TrialResult(
        trial_id=trial_id,
        method="default",
        status=TrialStatus.COMPLETE,
        params={"max_num_seqs": 8},
        client={**raw_aggregate, "total_output_tokens": 999},
        constraints={"feasible": True, "violations": []},
        cleanup_status=_clean_cleanup_status(),
    )
    runner.artifacts.write_json(base / "server-command.json", {"argv": ["vllm", "serve"]})
    runner.artifacts.write_json(base / "params.json", result.params)
    runner.artifacts.write_json(
        base / "status.json",
        {
            "status": "COMPLETE",
            "terminal": True,
            "history": [{"previous": "STOPPING", "current": "COMPLETE"}],
        },
    )
    runner.artifacts.write_jsonl(base / "request-results.jsonl", [request])
    runner.artifacts.write_json(
        base / "benchmark-raw.json",
        {"backend": "fake", "request_results": [request], "aggregate": raw_aggregate},
    )
    runner.artifacts.write_jsonl(base / "prometheus.jsonl", [])
    runner.artifacts.write_jsonl(base / "nvml.jsonl", [])
    runner.artifacts.write_text(base / "server.log", "server completed\n")
    runner.artifacts.write_json(base / "cleanup.json", result.cleanup_status)

    runner._finalize_trial_artifacts(result)

    assert result.status == TrialStatus.FAILED
    assert result.failure_reason["type"] == "ARTIFACT_INCONSISTENT"
    assert "artifact_inconsistent" in result.constraints["violations"]
    lifecycle = json.loads((runner.artifacts.trials_dir / trial_id / "status.json").read_text())
    assert lifecycle["status"] == "FAILED"
    assert lifecycle["terminal"] is True
    assert lifecycle["history"][0]["current"] == "COMPLETE"
    assert lifecycle["history"][-1]["current"] == "FAILED"
    assert lifecycle["history"][-1]["source"] == "artifact_finalizer"
    loaded = runner.artifacts.load_trial_result(trial_id)
    assert loaded is not None
    runner.artifacts.validate_cached_trial(loaded, require_telemetry=False)
