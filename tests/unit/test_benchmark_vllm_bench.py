"""Tests for the official vLLM bench serve adapter."""

import json
import sys

import pytest

from vllm_tuner.benchmarks.models import BenchmarkResult, SLOThresholds
from vllm_tuner.benchmarks.vllm_bench import (
    BenchmarkAdapter,
    BenchmarkExecutionError,
    VLLMBenchAdapter,
    VLLMBenchConfig,
)


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_build_command_covers_reproducible_traffic_and_lengths(tmp_path) -> None:
    config = VLLMBenchConfig(
        base_url="http://127.0.0.1:8000/",
        model="Qwen/Qwen3-0.6B",
        output_path=tmp_path / "raw.json",
        num_prompts=32,
        request_rate=12.5,
        burstiness=0.5,
        max_concurrency=8,
        fixed_input_len=128,
        fixed_output_len=64,
        ignore_eos=True,
        seed=2026,
        warmup_requests=3,
        slo=SLOThresholds(ttft_ms=100, tpot_ms=20, e2e_ms=1000),
        metadata={"run": "holdout"},
    )

    command = VLLMBenchAdapter().build_command(config)

    assert command[:5] == [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
    ]
    assert _value_after(command, "--base-url") == "http://127.0.0.1:8000"
    assert _value_after(command, "--num-prompts") == "32"
    assert _value_after(command, "--request-rate") == "12.5"
    # SLOTune's public value is inter-arrival CV; vLLM expects Gamma shape.
    assert _value_after(command, "--burstiness") == "4"
    assert _value_after(command, "--max-concurrency") == "8"
    assert _value_after(command, "--input-len") == "128"
    assert _value_after(command, "--output-len") == "64"
    assert _value_after(command, "--random-range-ratio") == "0"
    assert _value_after(command, "--num-warmups") == "3"
    assert _value_after(command, "--seed") == "2026"
    assert "--ignore-eos" in command
    assert "--save-result" in command
    assert "--save-detailed" in command
    goodput_index = command.index("--goodput")
    assert command[goodput_index + 1 : goodput_index + 4] == [
        "ttft:100",
        "tpot:20",
        "e2el:1000",
    ]
    assert "run=holdout" in command


def test_benchmark_adapter_is_an_interface() -> None:
    assert issubclass(VLLMBenchAdapter, BenchmarkAdapter)
    with pytest.raises(TypeError):
        BenchmarkAdapter()


def test_config_rejects_conflicting_length_aliases(tmp_path) -> None:
    with pytest.raises(ValueError, match="disagree"):
        VLLMBenchConfig(
            base_url="http://test",
            model="model",
            output_path=tmp_path / "raw.json",
            input_len=10,
            fixed_input_len=11,
        )


@pytest.mark.parametrize(
    ("interarrival_cv", "expected_vllm_shape"),
    [
        (0.5, 4.0),
        (1.0, 1.0),
        (1.5, 1.0 / 2.25),
    ],
)
def test_build_command_converts_interarrival_cv_to_vllm_gamma_shape(
    tmp_path, interarrival_cv: float, expected_vllm_shape: float
) -> None:
    config = VLLMBenchConfig(
        base_url="http://test",
        model="model",
        output_path=tmp_path / "raw.json",
        burstiness=interarrival_cv,
    )

    command = VLLMBenchAdapter().build_command(config)

    assert float(_value_after(command, "--burstiness")) == pytest.approx(expected_vllm_shape)


@pytest.mark.parametrize("burstiness", [0.0, -1.0, float("inf"), float("-inf"), float("nan")])
def test_config_rejects_non_positive_or_non_finite_burstiness(tmp_path, burstiness: float) -> None:
    with pytest.raises(ValueError, match="burstiness must be positive and finite"):
        VLLMBenchConfig(
            base_url="http://test",
            model="model",
            output_path=tmp_path / "raw.json",
            burstiness=burstiness,
        )


@pytest.mark.asyncio
async def test_adapter_executes_and_parses_raw_json(tmp_path) -> None:
    raw = {
        "backend": "vllm",
        "num_prompts": 1,
        "duration": 1.0,
        "completed": 1,
        "failed": 0,
        "total_input_tokens": 2,
        "total_output_tokens": 1,
        "input_lens": [2],
        "output_lens": [1],
        "ttfts": [0.01],
        "itls": [[]],
        "start_times": [1.0],
        "generated_texts": ["x"],
        "errors": [""],
    }
    raw_json = json.dumps(raw)
    script = (
        "import pathlib,sys;"
        "a=sys.argv;"
        "p=pathlib.Path(a[a.index('--result-dir')+1])/a[a.index('--result-filename')+1];"
        f"p.write_text({raw_json!r},encoding='utf-8')"
    )
    output_path = tmp_path / "raw.json"
    adapter = VLLMBenchAdapter((sys.executable, "-c", script))
    config = VLLMBenchConfig(
        base_url="http://test",
        model="model",
        output_path=output_path,
        num_prompts=1,
    )

    result = await adapter.run(config)

    assert isinstance(result, BenchmarkResult)
    assert result.aggregate["completed"] == 1
    assert result.output_path == output_path
    assert result.raw_result == raw
    assert result.command[:3] == [sys.executable, "-c", script]


@pytest.mark.asyncio
async def test_adapter_reports_nonzero_exit(tmp_path) -> None:
    adapter = VLLMBenchAdapter((sys.executable, "-c", "import sys;sys.exit(7)"))
    config = VLLMBenchConfig(
        base_url="http://test",
        model="model",
        output_path=tmp_path / "raw.json",
        num_prompts=1,
    )

    with pytest.raises(BenchmarkExecutionError, match="exited with 7"):
        await adapter.run(config)


@pytest.mark.asyncio
async def test_adapter_terminates_timed_out_process(tmp_path) -> None:
    adapter = VLLMBenchAdapter((sys.executable, "-c", "import time;time.sleep(30)"))
    config = VLLMBenchConfig(
        base_url="http://test",
        model="model",
        output_path=tmp_path / "raw.json",
        num_prompts=1,
        timeout_s=0.05,
    )

    with pytest.raises(BenchmarkExecutionError, match="timed out"):
        await adapter.run(config)
