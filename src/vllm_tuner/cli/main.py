"""CLI interface for vLLM tuner using Typer."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import typer

from vllm_tuner.config.validation import (
    load_yaml_config,
    validate_study_name,
    create_study_dirs,
    TunerSettings,
)
from vllm_tuner.reporting.dashboard import ProgressDashboard
from vllm_tuner.reporting.export import export_study_summary
from vllm_tuner.reporting.html import generate_html_report
from vllm_tuner.tuner.study_manager import StudyManager

logging.getLogger("httpx").setLevel(logging.WARNING)

app = typer.Typer(
    name="vllm-tuner",
    help="Auto-tuner for vLLM to maximize throughput and minimize latency",
    add_completion=False,
)

settings = TunerSettings()
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@app.command()
def tune(
    config: str = typer.Option(
        "config/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file",
    ),
    study_name: str = typer.Option(
        None,
        "--study-name",
        "-n",
        help="Name of the tuning study (required)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Override model name",
    ),
    gpu_count: Optional[int] = typer.Option(
        None,
        "--gpu-count",
        "-g",
        help="Override number of GPUs",
    ),
    with_progress: bool = typer.Option(
        False,
        "--with-progress",
        help="Enable progress bar (disabled by default)",
    ),
    generate_baseline: bool = typer.Option(
        True,
        "--baseline/--no-baseline",
        help="Generate baseline metrics before optimization",
    ),
):
    """
    Run a tuning study to optimize vLLM parameters.
    """
    if study_name is None:
        study_name = "default_tune"

    study_name = validate_study_name(study_name)
    logger.info(f"Starting tune study: {study_name}")

    try:
        config_obj = load_yaml_config(config)
        if model:
            config_obj.model = model
        if gpu_count:
            config_obj.gpu.count = gpu_count

        dirs = create_study_dirs(study_name, settings)
        logger.info(f"Study directories created: {dirs}")

        # Generate baseline BEFORE optimization
        if generate_baseline and config_obj.baseline.enabled:
            typer.echo("\n" + "=" * 80)
            typer.echo("GENERATING BASELINE METRICS")
            typer.echo("=" * 80 + "\n")

            try:
                baseline_output_dir = dirs["study"] / "baseline"
                from vllm_tuner.baseline.runner import generate_baseline_from_config

                async def run_baseline():
                    await generate_baseline_from_config(config_obj, baseline_output_dir)

                asyncio.run(run_baseline())

                typer.echo(f"\n✓ Baseline saved to: {baseline_output_dir}")

                # Print quick summary
                baseline_file = baseline_output_dir / "baseline_summary.txt"
                if baseline_file.exists():
                    with open(baseline_file) as f:
                        typer.echo(f.read())

                typer.echo("\n" + "=" * 80)
                typer.echo("STARTING OPTIMIZATION")
                typer.echo("=" * 80 + "\n")

            except Exception as e:
                logger.error(f"Baseline generation failed: {e}")
                if generate_baseline:  # If explicitly requested, abort
                    raise typer.Exit(1)
                else:  # If enabled by default, just warn and continue
                    logger.warning("Continuing without baseline...")
                    typer.echo(
                        "\n⚠ Baseline generation failed, continuing without baseline",
                        err=True,
                    )

        study_manager = StudyManager(config_obj, study_name, dirs["study"])

        if with_progress:
            dashboard = ProgressDashboard(study_name)
            dashboard.start(total_trials=config_obj.study.min_trials)

            async def run_with_progress():
                result = await study_manager.run_study()
                dashboard.stop()
                return result

            asyncio.run(run_with_progress())
        else:
            asyncio.run(study_manager.run_study())

        summary = study_manager.get_study_summary()

        typer.echo(f"\n✓ Study '{study_name}' completed successfully!")
        typer.echo(
            f"  Best throughput: {summary.get('best_trial', {}).get('metrics', {}).get('throughput_requests_per_sec', 0):.2f} req/s"
        )
        typer.echo(
            f"  Best latency: {summary.get('best_trial', {}).get('metrics', {}).get('avg_latency_ms', 0):.2f} ms"
        )

        exported = export_study_summary(
            summary,
            study_manager.optimizer.get_all_trials(),
            dirs["configs"],
        )

        typer.echo("\nConfiguration saved to:")
        for key, path in exported.items():
            typer.echo(f"  {path}")

        baseline_data = None
        baseline_file = dirs["study"] / "baseline" / "baseline_metrics.json"
        if baseline_file.exists():
            try:
                with open(baseline_file) as f:
                    baseline_full = json.load(f)
                    baseline_data = baseline_full.get("metrics", {})
                    logger.info("Loaded baseline metrics for reporting")
            except Exception as e:
                logger.warning(f"Failed to load baseline metrics: {e}")

        try:
            html_path = dirs["reports"] / "report.html"
            generate_html_report(
                study_name,
                dirs["reports"],
                summary,
                study_manager.optimizer.get_all_trials(),
                baseline_data=baseline_data,
            )
            typer.echo(f"\nHTML report: {html_path}")
        except Exception as e:
            logger.warning(f"Failed to generate HTML report: {e}")

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Tune command failed: {e}", exc_info=True)
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def report(
    study_name: str = typer.Option(
        ...,
        "--study-name",
        "-n",
        help="Name of the study to report",
    ),
    format: str = typer.Option(
        "html",
        "--format",
        "-f",
        help="Report format (html, json, markdown)",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: reports/<study_name>/report.<format>)",
    ),
):
    """
    Generate a report from completed study.
    """
    study_name = validate_study_name(study_name)

    try:
        dirs = create_study_dirs(study_name, settings)

        summary_path = dirs["configs"] / "summary.json"
        trials_path = dirs["configs"] / "trials.json"

        if not summary_path.exists() or not trials_path.exists():
            typer.echo(f"Error: Study data not found for '{study_name}'", err=True)
            raise typer.Exit(1)

        import json

        with open(summary_path, "r") as f:
            summary = json.load(f)

        with open(trials_path, "r") as f:
            trials_data = json.load(f)

        baseline_data = None
        baseline_file = dirs["study"] / "baseline" / "baseline_metrics.json"
        if baseline_file.exists():
            try:
                with open(baseline_file) as f:
                    baseline_full = json.load(f)
                    baseline_data = baseline_full.get("metrics", {})
                    logger.info("Loaded baseline metrics for reporting")
            except Exception as e:
                logger.warning(f"Failed to load baseline metrics: {e}")

        if output is None:
            output = str(dirs["reports"] / f"report.{format}")

        if format == "html":
            generate_html_report(
                study_name,
                dirs["reports"],
                summary,
                trials_data,
                baseline_data=baseline_data,
            )
            typer.echo(f"HTML report generated: {output}")
        elif format == "json":
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump({"summary": summary, "trials": trials_data}, f, indent=2)
            typer.echo(f"JSON report generated: {output}")
        elif format == "markdown":
            markdown = _generate_markdown_summary(summary, trials_data)
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                f.write(markdown)
            typer.echo(f"Markdown report generated: {output}")
        else:
            typer.echo(f"Error: Unsupported format '{format}'", err=True)
            raise typer.Exit(1)

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Report command failed: {e}", exc_info=True)
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


def _generate_markdown_summary(
    summary: dict,
    trials_data: list[dict],
) -> str:
    """Generate markdown summary."""
    markdown = f"""# vLLM Tuner Report: {summary.get("study_name", "Unknown")}

## Study Summary

- **Total Trials**: {summary.get("num_trials", 0)}
- **Study Storage**: {summary.get("study_storage", "N/A")}

## Best Configuration

### Parameters
"""

    best = summary.get("best_trial", {})
    params = best.get("parameters", {})

    for key, value in params.items():
        markdown += f"- **{key}**: {value}\n"

    metrics = best.get("metrics", {})

    markdown += f"""
### Performance Metrics
- **Throughput**: {metrics.get("throughput_requests_per_sec", 0):.2f} requests/sec
- **Average Latency**: {metrics.get("avg_latency_ms", 0):.2f} ms
- **P50 Latency**: {metrics.get("p50_latency_ms", 0):.2f} ms
- **P95 Latency**: {metrics.get("p95_latency_ms", 0):.2f} ms
- **P99 Latency**: {metrics.get("p99_latency_ms", 0):.2f} ms
- **Token Throughput**: {metrics.get("throughput_tokens_per_sec", 0):.2f} tokens/sec

## Trial Results

"""

    markdown += "| Trial # | Throughput (req/s) | Latency (ms) | State |\n"
    markdown += "|---------|-------------------|-------------|-------|\n"

    for trial in trials_data[:20]:
        t_num = trial.get("trial_number", 0)
        t_throughput = trial.get("metrics", {}).get("throughput_requests_per_sec", 0)
        t_latency = trial.get("metrics", {}).get("avg_latency_ms", 0)
        t_state = trial.get("state", "Unknown")

        markdown += f"| {t_num} | {t_throughput:.2f} | {t_latency:.2f} | {t_state} |\n"

    if len(trials_data) > 20:
        markdown += f"\n*... and {len(trials_data) - 20} more trials*\n"

    return markdown


@app.command()
def export(
    study_name: str = typer.Option(
        ...,
        "--study-name",
        "-n",
        help="Name of the study to export",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: configs/<study_name>/best.yaml)",
    ),
    format: str = typer.Option(
        "yaml",
        "--format",
        "-f",
        help="Export format (yaml, json)",
    ),
):
    """
    Export the best configuration from a study.
    """
    study_name = validate_study_name(study_name)
    logger.info(f"Exporting best config from study: {study_name}")

    try:
        dirs = create_study_dirs(study_name, settings)

        summary_path = dirs["configs"] / "summary.json"

        if not summary_path.exists():
            typer.echo(f"Error: Study data not found for '{study_name}'", err=True)
            raise typer.Exit(1)

        import json

        with open(summary_path, "r") as f:
            summary = json.load(f)

        best = summary.get("best_trial", {})

        if not best:
            typer.echo(
                f"Error: No successful trials found in study '{study_name}'", err=True
            )
            raise typer.Exit(1)

        if output is None:
            output = str(dirs["configs"] / f"best.{format}")

        from vllm_tuner.reporting.export import export_best_config

        export_best_config(best, Path(output), format)

        typer.echo(f"Best configuration exported to: {output}")

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Export command failed: {e}", exc_info=True)
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def list_studies():
    """
    List all available studies.
    """
    studies_dir = Path(settings.study_output_dir)

    if not studies_dir.exists():
        typer.echo("No studies found")
        return

    studies = list(studies_dir.iterdir())

    if not studies:
        typer.echo("No studies found")
        return

    typer.echo(f"Found {len(studies)} studies:")
    for study in studies:
        typer.echo(f"  - {study.name}")
        summary_path = study / "configs" / "summary.json"
        if summary_path.exists():
            import json

            with open(summary_path, "r") as f:
                summary = json.load(f)
            best = summary.get("best_trial", {})
            metrics = best.get("metrics", {})
            typer.echo(
                f"    Throughput: {metrics.get('throughput_requests_per_sec', 0):.2f} req/s"
            )


def main():
    app()


if __name__ == "__main__":
    main()
