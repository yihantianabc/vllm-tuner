"""CLI registration coverage for long-context v5 M1 capacity commands."""

from click.utils import strip_ansi
from typer.main import get_command
from typer.testing import CliRunner

from vllm_tuner.cli.main import app

runner = CliRunner()


def test_m1_capacity_run_command_registers_isolated_matrix_arguments() -> None:
    result = runner.invoke(app, ["longctx-m1-capacity", "--help"])

    assert result.exit_code == 0, result.output
    help_output = strip_ansi(result.output)
    assert "--config" in help_output
    assert "--experiment-id" in help_output
    assert "--resume" in help_output

    command = get_command(app).commands["longctx-m1-capacity"]
    options = {parameter.name: parameter for parameter in command.params}
    assert options["config"].default == ("experiments/long_context/v5/m1-capacity-smoke.yaml")
    assert set(options["config"].opts) == {"--config", "-c"}
    assert options["experiment_id"].default == "longctx-v5-m1-capacity-smoke-001"
    assert set(options["experiment_id"].opts) == {"--experiment-id", "-n"}
    assert options["resume"].default is False
    assert "checksum-valid" in options["resume"].help


def test_m1_capacity_status_command_registers_operator_paths() -> None:
    result = runner.invoke(app, ["longctx-m1-capacity-status", "--help"])

    assert result.exit_code == 0, result.output
    help_output = strip_ansi(result.output)
    assert "--artifact-root" in help_output
    assert "--experiment-id" in help_output

    command = get_command(app).commands["longctx-m1-capacity-status"]
    options = {parameter.name: parameter for parameter in command.params}
    assert options["artifact_root"].default == "/root/autodl-tmp/longctx-v5-artifacts"
    assert options["artifact_root"].opts == ["--artifact-root"]
    assert options["experiment_id"].default == "longctx-v5-m1-capacity-smoke-001"
    assert set(options["experiment_id"].opts) == {"--experiment-id", "-n"}
