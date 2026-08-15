"""Adapter for the official ``vllm bench serve`` measurement backend."""

from __future__ import annotations

import asyncio
import math
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

from .models import BenchmarkResult, SLOThresholds
from .result_parser import VLLMResultParser


class BenchmarkExecutionError(RuntimeError):
    """The benchmark subprocess failed or did not produce its promised artifact."""


@dataclass
class VLLMBenchConfig:
    """Reproducible arguments for ``vllm bench serve``.

    ``burstiness`` follows SLOTune's workload convention: it is the
    coefficient of variation (CV) of inter-arrival times. The official vLLM
    flag with the same name instead expects the Gamma shape parameter, so the
    adapter converts it at the CLI boundary.
    """

    base_url: str
    model: str
    output_path: Union[str, Path]
    num_prompts: int = 100
    backend: str = "vllm"
    endpoint: str = "/v1/completions"
    dataset_name: str = "random"
    dataset_path: Optional[Union[str, Path]] = None
    request_rate: float = float("inf")
    burstiness: float = 1.0
    max_concurrency: Optional[int] = None
    input_len: Optional[int] = None
    output_len: Optional[int] = None
    fixed_input_len: Optional[int] = None
    fixed_output_len: Optional[int] = None
    ignore_eos: bool = False
    seed: int = 0
    warmup_requests: int = 0
    percentile_metrics: tuple[str, ...] = ("ttft", "tpot", "itl", "e2el")
    metric_percentiles: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0)
    slo: Optional[SLOThresholds] = None
    metadata: dict[str, str] = field(default_factory=dict)
    extra_args: tuple[str, ...] = ()
    timeout_s: Optional[float] = None

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)
        if self.dataset_path is not None:
            self.dataset_path = Path(self.dataset_path)
        if self.num_prompts < 1:
            raise ValueError("num_prompts must be positive")
        if self.request_rate <= 0:
            raise ValueError("request_rate must be positive")
        if not math.isfinite(self.burstiness) or self.burstiness <= 0:
            raise ValueError("burstiness must be positive and finite")
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must be non-negative")
        if not self.endpoint.startswith("/"):
            raise ValueError("endpoint must start with '/'")

        self.input_len = self._resolve_length("input_len", self.input_len, self.fixed_input_len)
        self.output_len = self._resolve_length("output_len", self.output_len, self.fixed_output_len)
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not self.percentile_metrics:
            raise ValueError("percentile_metrics must not be empty")
        if not self.metric_percentiles or any(
            percentile < 0 or percentile > 100 for percentile in self.metric_percentiles
        ):
            raise ValueError("metric_percentiles must contain values from 0 to 100")

    @staticmethod
    def _resolve_length(
        name: str, canonical: Optional[int], fixed_alias: Optional[int]
    ) -> Optional[int]:
        if canonical is not None and fixed_alias is not None and canonical != fixed_alias:
            raise ValueError(f"{name} and fixed_{name} disagree")
        value = canonical if canonical is not None else fixed_alias
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")
        return value


class BenchmarkAdapter(ABC):
    """Interface implemented by trustworthy benchmark measurement backends."""

    @abstractmethod
    async def run(
        self, config: VLLMBenchConfig, *, slo: Optional[SLOThresholds] = None
    ) -> BenchmarkResult:
        """Execute one benchmark and return parsed raw and aggregate results."""


class VLLMBenchAdapter(BenchmarkAdapter):
    """Run the official vLLM CLI without a shell and parse its detailed JSON."""

    def __init__(
        self,
        executable: Optional[Union[str, Sequence[str]]] = None,
        *,
        parser: Optional[VLLMResultParser] = None,
    ) -> None:
        self.command_prefix: tuple[str, ...]
        if executable is None:
            # Bind the official CLI to the interpreter that imported SLOTune.
            # This works for absolute-path virtualenv entry points even when
            # that virtualenv's ``bin`` directory is absent from PATH.
            self.command_prefix = (
                sys.executable,
                "-m",
                "vllm.entrypoints.cli.main",
            )
        elif isinstance(executable, str):
            self.command_prefix = (executable,)
        else:
            self.command_prefix = tuple(executable)
        if not self.command_prefix:
            raise ValueError("executable must not be empty")
        self.parser = parser or VLLMResultParser(validate=True)

    def build_command(
        self, config: VLLMBenchConfig, *, slo: Optional[SLOThresholds] = None
    ) -> list[str]:
        """Build a vLLM 0.16-compatible, reproducible command line."""

        output_path = Path(config.output_path)
        # SLOTune exposes inter-arrival CV, whereas vLLM names the Gamma shape
        # parameter ``--burstiness``. Gamma CV = 1 / sqrt(shape), hence this
        # conversion keeps official and SLOTune-generated traffic equivalent.
        vllm_burstiness = 1.0 / (config.burstiness * config.burstiness)
        command = [
            *self.command_prefix,
            "bench",
            "serve",
            "--backend",
            config.backend,
            "--base-url",
            config.base_url.rstrip("/"),
            "--endpoint",
            config.endpoint,
            "--model",
            config.model,
            "--dataset-name",
            config.dataset_name,
            "--num-prompts",
            str(config.num_prompts),
            "--request-rate",
            self._format_float(config.request_rate),
            "--burstiness",
            self._format_float(vllm_burstiness),
            "--seed",
            str(config.seed),
            "--num-warmups",
            str(config.warmup_requests),
            "--percentile-metrics",
            ",".join(config.percentile_metrics),
            "--metric-percentiles",
            ",".join(self._format_float(value) for value in config.metric_percentiles),
            "--save-result",
            "--save-detailed",
            "--disable-tqdm",
            "--result-dir",
            str(output_path.parent),
            "--result-filename",
            output_path.name,
        ]
        if config.dataset_path is not None:
            command.extend(["--dataset-path", str(config.dataset_path)])
        if config.max_concurrency is not None:
            command.extend(["--max-concurrency", str(config.max_concurrency)])
        if config.input_len is not None:
            command.extend(["--input-len", str(config.input_len)])
        if config.output_len is not None:
            command.extend(["--output-len", str(config.output_len)])
        if config.dataset_name == "random" and (
            config.input_len is not None or config.output_len is not None
        ):
            # vLLM 0.16 defaults to zero, but writing it explicitly makes the
            # fixed-length experiment protocol self-contained.
            command.extend(["--random-range-ratio", "0"])
        if config.ignore_eos:
            command.append("--ignore-eos")
        effective_slo = slo or config.slo
        if effective_slo is not None:
            goodput = []
            if effective_slo.ttft_ms is not None:
                goodput.append(f"ttft:{self._format_float(effective_slo.ttft_ms)}")
            if effective_slo.tpot_ms is not None:
                goodput.append(f"tpot:{self._format_float(effective_slo.tpot_ms)}")
            if effective_slo.e2e_ms is not None:
                goodput.append(f"e2el:{self._format_float(effective_slo.e2e_ms)}")
            if goodput:
                command.extend(["--goodput", *goodput])
        for key in sorted(config.metadata):
            command.extend(["--metadata", f"{key}={config.metadata[key]}"])
        command.extend(config.extra_args)
        return command

    async def run(
        self, config: VLLMBenchConfig, *, slo: Optional[SLOThresholds] = None
    ) -> BenchmarkResult:
        """Execute the official CLI, enforce timeout, then parse its raw JSON."""

        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        effective_slo = slo or config.slo
        command = self.build_command(config, slo=effective_slo)

        try:
            environment = os.environ.copy()
            no_proxy = environment.get("NO_PROXY", environment.get("no_proxy", ""))
            loopback = ("127.0.0.1", "localhost")
            entries = [item.strip() for item in no_proxy.split(",") if item.strip()]
            for host in loopback:
                if host not in entries:
                    entries.append(host)
            environment["NO_PROXY"] = ",".join(entries)
            environment["no_proxy"] = environment["NO_PROXY"]
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=environment,
            )
        except OSError as error:
            raise BenchmarkExecutionError(
                f"Unable to start {' '.join(self.command_prefix)}: {error}"
            ) from error

        try:
            if config.timeout_s is None:
                stdout_bytes, stderr_bytes = await process.communicate()
            else:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=config.timeout_s
                )
        except asyncio.TimeoutError as error:
            await self._stop_process(process)
            raise BenchmarkExecutionError(
                f"vllm bench serve timed out after {config.timeout_s} seconds"
            ) from error
        except asyncio.CancelledError:
            await self._stop_process(process)
            raise

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        if process.returncode != 0:
            raise BenchmarkExecutionError(
                f"vllm bench serve exited with {process.returncode}: {stderr[-2000:]}"
            )
        if not output_path.is_file():
            raise BenchmarkExecutionError(
                f"vllm bench serve succeeded but did not create {output_path}"
            )

        result = self.parser.parse(output_path, slo=effective_slo)
        result.command = command
        result.stdout = stdout
        result.stderr = stderr
        result.output_path = output_path
        return result

    @staticmethod
    async def _stop_process(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    @staticmethod
    def _format_float(value: float) -> str:
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return str(int(value)) if float(value).is_integer() else str(value)


OfficialVLLMBenchAdapter = VLLMBenchAdapter
