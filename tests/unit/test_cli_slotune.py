"""CLI coverage for immutable SLOTune experiment artifacts."""

import json
from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from vllm_tuner.cli.main import app
from vllm_tuner.reporting.export import export_best_config

runner = CliRunner()


def _write_experiment(results_root: Path, experiment_id: str = "exp") -> Path:
    experiment = results_root / experiment_id
    aggregate = experiment / "aggregate"
    report = experiment / "report"
    aggregate.mkdir(parents=True)
    report.mkdir()
    parameters = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }
    best = {
        "method": "default",
        "status": "COMPLETE",
        "feasible": True,
        "validated": True,
        "candidate": "default-0",
        "metric_provenance": "median_of_complete_feasible_repeats",
        "search_observation": {
            "trial_id": "default-0000",
            "trial_number": 0,
            "method": "default",
            "parameters": parameters,
            "goodput_requests_per_sec": 9.9,
        },
        "repeat_required": 3,
        "repeat_complete_feasible": 3,
        "holdout_required": True,
        "holdout_complete_feasible": 3,
        "repeat_metrics": {
            "goodput_requests_per_sec": {
                "count": 3,
                "median": 3.5,
                "min": 3.4,
                "max": 3.6,
            },
            "achieved_requests_per_sec": {
                "count": 3,
                "median": 3.8,
                "min": 3.7,
                "max": 3.9,
            },
            "p99_ttft_ms": {
                "count": 3,
                "median": 24.0,
                "min": 23.0,
                "max": 25.0,
            },
        },
        "holdout_metrics": {
            "goodput_requests_per_sec": {
                "count": 3,
                "median": 3.3,
                "min": 3.2,
                "max": 3.4,
            }
        },
        "parameters": parameters,
        "goodput_requests_per_sec": 3.5,
        "offered_requests_per_sec": 4.0,
        "achieved_requests_per_sec": 3.8,
        "p99_ttft_ms": 24.0,
        "p99_tpot_ms": 3.0,
        "p99_e2e_ms": 80.0,
    }
    repetitions = [
        {
            "trial_id": f"repeat-default-0-{repeat}",
            "trial_number": repeat,
            "method": "default",
            "parameters": parameters,
            "repeat_of": 0,
            "holdout": False,
            "status": "COMPLETE",
            "feasible": True,
            "goodput_requests_per_sec": goodput,
        }
        for repeat, goodput in enumerate((3.4, 3.5, 3.6))
    ]
    holdouts = [
        {
            "trial_id": f"holdout-default-0-{repeat}",
            "trial_number": repeat,
            "method": "default",
            "parameters": parameters,
            "repeat_of": 0,
            "holdout": True,
            "status": "COMPLETE",
            "feasible": True,
            "goodput_requests_per_sec": goodput,
        }
        for repeat, goodput in enumerate((3.2, 3.3, 3.4))
    ]
    manifest = {
        "experiment_id": experiment_id,
        "created_at": "2026-08-15T00:00:00+00:00",
        "model": "fake/model",
        "model_revision": "revision-1",
        "tokenizer": "fake/tokenizer",
        "vllm_args": {"max-model-len": 4096},
        "source_commit": "abc123",
        "trace_sha256": "search-trace",
        "holdout_trace_sha256": "holdout-trace",
        "artifact_warnings": [],
    }
    summary = {
        "experiment_id": experiment_id,
        "manifest": manifest,
        "search": {"default": [best]},
        "best": best,
        "candidate_validation": [
            {
                "candidate": "default-0",
                "method": "default",
                "parameters": parameters,
                "validated": True,
            }
        ],
        "repetitions": repetitions,
        "holdout": holdouts,
        "capacity_sweep": {"points": [], "by_rate": []},
        "report": {
            "html": str(report / "report.html"),
            "markdown": str(report / "report.md"),
        },
    }
    (experiment / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (experiment / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (report / "report.html").write_text("<html>SLOTune static report</html>", encoding="utf-8")
    (report / "report.md").write_text("# Existing detailed report\n", encoding="utf-8")
    pd.DataFrame([best]).to_parquet(aggregate / "trials.parquet", index=False)
    pd.DataFrame(holdouts).to_parquet(aggregate / "holdout-results.parquet", index=False)
    return experiment


def test_report_reuses_or_copies_existing_static_html(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)
    reused = runner.invoke(
        app,
        ["report", "-n", "exp", "--results-root", str(tmp_path), "-f", "html"],
    )
    assert reused.exit_code == 0, reused.output
    assert f"HTML report reused: {experiment / 'report/report.html'}" in reused.output

    copied_path = tmp_path / "copied.html"
    copied = runner.invoke(
        app,
        [
            "report",
            "-n",
            "exp",
            "--results-root",
            str(tmp_path),
            "-f",
            "html",
            "-o",
            str(copied_path),
        ],
    )
    assert copied.exit_code == 0, copied.output
    assert copied_path.read_text() == "<html>SLOTune static report</html>"


def test_report_exports_current_summary_and_aggregate_parquet(tmp_path: Path) -> None:
    _write_experiment(tmp_path)
    json_path = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "report",
            "-n",
            "exp",
            "--results-root",
            str(tmp_path),
            "-f",
            "json",
            "-o",
            str(json_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(json_path.read_text())
    assert payload["schema"] == "slotune-experiment-report-v1"
    assert payload["summary"]["best"]["parameters"]["tensor_parallel_size"] == 1
    assert payload["aggregates"]["trials"]["available"] is True
    assert payload["aggregates"]["trials"]["row_count"] == 1
    assert payload["aggregates"]["capacity_sweep"]["available"] is False
    assert payload["aggregates"]["capacity_sweep"]["records"] == []


def test_report_markdown_uses_slotune_metrics_and_validation(tmp_path: Path) -> None:
    _write_experiment(tmp_path)
    markdown_path = tmp_path / "summary.md"
    result = runner.invoke(
        app,
        [
            "report",
            "-n",
            "exp",
            "--results-root",
            str(tmp_path),
            "-f",
            "markdown",
            "-o",
            str(markdown_path),
        ],
    )
    assert result.exit_code == 0, result.output
    markdown = markdown_path.read_text()
    assert "# SLOTune experiment: exp" in markdown
    assert "SLO goodput, repeat median (range): `3.500 (3.400–3.600; n=3)`" in markdown
    assert "Search SLO goodput (selection observation only): `9.900`" in markdown
    assert "Holdout validation: `passed`" in markdown
    assert "tensor_parallel_size" in markdown
    assert "Average Latency" not in markdown


def test_export_reads_best_parameters_after_exact_holdout_validation(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)
    result = runner.invoke(
        app,
        ["export", "-n", "exp", "--results-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    exported = yaml.safe_load((experiment / "best.yaml").read_text())
    assert exported["source_experiment"] == "exp"
    assert exported["model"] == "fake/model"
    assert exported["model_revision"] == "revision-1"
    assert exported["tokenizer"] == "fake/tokenizer"
    assert exported["vllm_params"] == {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }
    assert exported["base_vllm_args"] == {"max-model-len": 4096}
    assert exported["vllm_args"] == {
        "max-model-len": 4096,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }
    assert exported["performance_metrics"]["goodput_requests_per_sec"] == 3.5
    assert exported["validation"]["validated"] is True
    assert exported["validation"]["matching_repeat_trial_ids"] == [
        "repeat-default-0-0",
        "repeat-default-0-1",
        "repeat-default-0-2",
    ]
    assert exported["validation"]["matching_holdout_trial_ids"] == [
        "holdout-default-0-0",
        "holdout-default-0-1",
        "holdout-default-0-2",
    ]


def test_export_rejects_unvalidated_or_empty_best(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)
    summary_path = experiment / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["best"]["validated"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    unvalidated = runner.invoke(
        app,
        ["export", "-n", "exp", "--results-root", str(tmp_path)],
    )
    assert unvalidated.exit_code == 1
    assert "summary.best.validated is not true" in unvalidated.output

    summary["best"] = None
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    empty = runner.invoke(
        app,
        ["export", "-n", "exp", "--results-root", str(tmp_path)],
    )
    assert empty.exit_code == 1
    assert "summary.best is empty" in empty.output


def test_export_rejects_same_parameters_from_a_different_method(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)
    summary_path = experiment / "summary.json"
    summary = json.loads(summary_path.read_text())
    for row in summary["repetitions"]:
        row["method"] = "random"
    for row in summary["holdout"]:
        row["method"] = "random"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = runner.invoke(
        app,
        ["export", "-n", "exp", "--results-root", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "every exact repeat and holdout row" in result.output
    assert not (experiment / "best.yaml").exists()


def test_export_rejects_conflicting_manifest_and_validated_parameters(
    tmp_path: Path,
) -> None:
    experiment = _write_experiment(tmp_path)
    summary_path = experiment / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["manifest"]["vllm_args"]["tensor-parallel-size"] = 2
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = runner.invoke(
        app,
        ["export", "-n", "exp", "--results-root", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "conflict with manifest.vllm_args" in result.output
    assert not (experiment / "best.yaml").exists()


def test_export_helper_retains_explicit_legacy_compatibility(tmp_path: Path) -> None:
    output = tmp_path / "legacy.yaml"

    export_best_config(
        {
            "parameters": {"max_num_seqs": 8},
            "metrics": {"throughput_requests_per_sec": 2.0},
            "trial_number": 4,
            "state": "COMPLETE",
        },
        output,
    )

    exported = yaml.safe_load(output.read_text())
    assert exported["vllm_params"] == {"max_num_seqs": 8}
    assert exported["performance_metrics"]["throughput_requests_per_sec"] == 2.0
    assert exported["candidate_info"]["status"] == "COMPLETE"
    assert "model" not in exported


def test_list_studies_shows_manifest_best_status_and_legacy_label(tmp_path: Path) -> None:
    _write_experiment(tmp_path)
    legacy = tmp_path / "old" / "configs"
    legacy.mkdir(parents=True)
    (legacy / "summary.json").write_text(json.dumps({"best_trial": {}}))

    result = runner.invoke(app, ["list-studies", "--results-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Found 2 studies" in result.output
    assert "Status: COMPLETE (summary.json present)" in result.output
    assert "Manifest: model=fake/model" in result.output
    assert (
        "Best: trial=default-0, status=COMPLETE, goodput=3.500 req/s, validated=yes"
        in result.output
    )
    assert "old [legacy layout]" in result.output
    assert "not interpreted as SLOTune" in result.output


def test_report_rejects_legacy_layout_instead_of_mixing_schemas(tmp_path: Path) -> None:
    legacy = tmp_path / "old" / "configs"
    legacy.mkdir(parents=True)
    (legacy / "summary.json").write_text(json.dumps({"best_trial": {}}))

    result = runner.invoke(
        app,
        ["report", "-n", "old", "--results-root", str(tmp_path), "-f", "json"],
    )

    assert result.exit_code == 1
    assert "Legacy study layout detected" in result.output


def test_tune_help_exposes_manifest_validated_resume() -> None:
    result = runner.invoke(app, ["tune", "--help"])

    assert result.exit_code == 0, result.output
    assert "--resume" in result.output
    assert "immutable" in result.output
    assert "manifest" in result.output
    assert "--allow-dirty-source" in result.output
    assert "formal runs" in result.output
    assert "require clean Git" in result.output
