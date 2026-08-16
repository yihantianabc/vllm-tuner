# SLOTune

SLOTune is a single-GPU experiment system for reproducible, SLO-aware vLLM tuning and
for studying adaptive chunked-prefill/token-budget scheduling. It is an AI-infrastructure
research project—not a chat UI, web control plane, or Kubernetes platform.

> **Evidence status:** two completed Qwen2.5-3B-Instruct formal experiments are preserved on
> the data disk and summarized in the checked-in [result snapshot](docs/results/qwen25-3b-34a25a2.md).
> The Qwen3-0.6B runs remain correctness smoke tests only. Neither formal workload met the
> preregistered 15% goodput or 20% p99-TTFT improvement target; that negative result is retained.

## Fork attribution

Forked from jranaraki/vllm-tuner.

My work focuses on benchmark correctness, SLO-aware optimization, cross-layer observability, reproducibility, and scheduling experiments.

> **SLOTune is based on and forked from
> [`jranaraki/vllm-tuner`](https://github.com/jranaraki/vllm-tuner), originally authored by
> Javad Anaraki and distributed under the MIT License.**

The frozen upstream point used for the SLOTune refactor is recorded in
[`docs/BASELINE_20260815.md`](docs/BASELINE_20260815.md). The upstream project supplied the
initial Typer CLI, Pydantic configuration, Optuna tuning skeleton, vLLM launcher, basic GPU
collection, baseline runner, and HTML-report foundation.

### My Contributions

The table distinguishes implementation and automated evidence from measured artifacts. Commit
links point to the corresponding revisions in the project's public GitHub fork: `34a25a2` is
the clean measurement revision and `ad36ee8` is the clean post-run attestation-tool revision.
Attestation does not retroactively change measurement provenance.

| Area | SLOTune contribution | Code | Tests | Artifact or evidence | Commit |
|---|---|---|---|---|---|
| Benchmark correctness | Streaming SSE framing, typed request records, raw results, and explicit TTFT/TPOT/ITL/E2E/token semantics | [`sse_client.py`](src/vllm_tuner/benchmarks/sse_client.py), [`metrics.py`](src/vllm_tuner/benchmarks/metrics.py), [`vllm_bench.py`](src/vllm_tuner/benchmarks/vllm_bench.py) | [`test_benchmark_sse_client.py`](tests/unit/test_benchmark_sse_client.py), [`test_benchmark_metrics.py`](tests/unit/test_benchmark_metrics.py), [`test_benchmark_vllm_bench.py`](tests/unit/test_benchmark_vllm_bench.py) | [Measurement contract](docs/METHODOLOGY.md#measurement-correctness); live official/SSE comparison recorded in the [result snapshot](docs/results/qwen25-3b-34a25a2.md#official-vs-sse-cross-check) | [`aa9d70a`](https://github.com/yihantianabc/vllm-tuner/commit/aa9d70aafa3634bc517c1c6890e8eab8b0e6ba02) |
| Reproducibility | Frozen seeded traces, checksums, manifests, environment fingerprints, isolated atomic artifacts, held-out traces, and integrity validation | [`trace.py`](src/vllm_tuner/workloads/trace.py), [`manifest.py`](src/vllm_tuner/experiment/manifest.py), [`artifacts.py`](src/vllm_tuner/experiment/artifacts.py) | [`test_workload_trace.py`](tests/unit/test_workload_trace.py), [`test_experiment_artifacts.py`](tests/unit/test_experiment_artifacts.py) | [Artifact contract](docs/METHODOLOGY.md#artifact-acceptance); [formal evidence snapshot](docs/results/qwen25-3b-34a25a2.md#artifact-audit) | [`aa9d70a`](https://github.com/yihantianabc/vllm-tuner/commit/aa9d70aafa3634bc517c1c6890e8eab8b0e6ba02), [`0d605c3`](https://github.com/yihantianabc/vllm-tuner/commit/0d605c350cb37c86602484a194c2abdce3da17e2), [`34a25a2`](https://github.com/yihantianabc/vllm-tuner/commit/34a25a2e10951bfab1c2a86b4c60aff5bef785df), [`ad36ee8`](https://github.com/yihantianabc/vllm-tuner/commit/ad36ee8e0e15a6d0502a35f9e794b056b9522a82) |
| Runtime and observability | Trial state machine, structured failure taxonomy, process-group cleanup, aligned Prometheus and continuous NVML sampling | [`controller.py`](src/vllm_tuner/runtime/controller.py), [`session.py`](src/vllm_tuner/profiling/session.py), [`prometheus.py`](src/vllm_tuner/profiling/prometheus.py), [`nvml_session.py`](src/vllm_tuner/profiling/nvml_session.py) | [`test_trial_controller.py`](tests/unit/test_trial_controller.py), [`test_telemetry_session.py`](tests/unit/test_telemetry_session.py), [`test_prometheus.py`](tests/unit/test_prometheus.py), [`test_nvml_session.py`](tests/unit/test_nvml_session.py) | Both formal roots contain aligned engine/GPU series and cleanup evidence for all 96 trials | [`aa9d70a`](https://github.com/yihantianabc/vllm-tuner/commit/aa9d70aafa3634bc517c1c6890e8eab8b0e6ba02) |
| SLO-aware tuning | Constrained SLO-goodput objective, effective single-GPU search space, equal-budget default/random/TPE, repeats, and holdout validation | [`objective.py`](src/vllm_tuner/tuning/objective.py), [`optimizer.py`](src/vllm_tuner/tuning/optimizer.py), [`runner.py`](src/vllm_tuner/experiment/runner.py) | [`test_objective.py`](tests/unit/test_objective.py), [`test_constrained_optimizer.py`](tests/unit/test_constrained_optimizer.py), [`test_experiment_runner.py`](tests/unit/test_experiment_runner.py) | [Formal result tables](docs/results/qwen25-3b-34a25a2.md#tuning-outcome): repeated and held-out negative results for Chat and RAG | [`aa9d70a`](https://github.com/yihantianabc/vllm-tuner/commit/aa9d70aafa3634bc517c1c6890e8eab8b0e6ba02), [`34a25a2`](https://github.com/yihantianabc/vllm-tuner/commit/34a25a2e10951bfab1c2a86b4c60aff5bef785df) |
| Scheduling experiments | Deterministic fixed/adaptive token-budget simulator with budget conservation, aging, max-wait, admission limits, fairness, starvation, preemption, and downside retention | [`token_budget.py`](src/vllm_tuner/scheduling/token_budget.py), [`admission.py`](src/vllm_tuner/scheduling/admission.py), [`simulator.py`](src/vllm_tuner/scheduling/simulator.py), [`run_scheduler_ablation.py`](scripts/run_scheduler_ablation.py) | [`test_scheduling_token_budget.py`](tests/unit/test_scheduling_token_budget.py), [`test_scheduling_admission.py`](tests/unit/test_scheduling_admission.py), [`test_scheduling_simulator.py`](tests/unit/test_scheduling_simulator.py), [`test_scheduling_script.py`](tests/unit/test_scheduling_script.py) | [Recorded no-gain/regression result](docs/results/qwen25-3b-34a25a2.md#scheduler-simulator-negative-result); explicitly CPU simulator evidence | [`aa9d70a`](https://github.com/yihantianabc/vllm-tuner/commit/aa9d70aafa3634bc517c1c6890e8eab8b0e6ba02) |
| Reporting | Static JSON/Markdown/HTML summaries and plots that preserve infeasible, failed, missing, and negative outcomes | [`report.py`](src/vllm_tuner/reporting/report.py), [`plots.py`](src/vllm_tuner/reporting/plots.py) | [`test_reporting_artifacts.py`](tests/unit/test_reporting_artifacts.py) | [Formal reports and checked-in evidence snapshot](docs/results/qwen25-3b-34a25a2.md) | [`aa9d70a`](https://github.com/yihantianabc/vllm-tuner/commit/aa9d70aafa3634bc517c1c6890e8eab8b0e6ba02), [`b8f2dc1`](https://github.com/yihantianabc/vllm-tuner/commit/b8f2dc10761ac860e1b631d431ed600698e39d83), [`ad36ee8`](https://github.com/yihantianabc/vllm-tuner/commit/ad36ee8e0e15a6d0502a35f9e794b056b9522a82) |

The upstream foundation remains credited above; in particular, this fork does not claim the
original Typer CLI, Pydantic configuration, Optuna skeleton, vLLM launcher, or HTML-report base
as work created from scratch.

## Research questions

SLOTune is designed to answer:

1. Which effective vLLM settings maximize requests that satisfy latency SLOs for one fixed
   model × GPU × workload × software version?
2. Can changes be explained by waiting queue, KV-cache pressure, preemption, GPU utilization,
   and memory timelines rather than a single throughput number?
3. Does an adaptive prefill/token budget improve tail latency or goodput over fixed budgets,
   and under which traces does it tie or regress?

## Real results (artifact-backed only)

| Evidence | Model and workload | Recorded outcome | What it supports |
|---|---|---|---|
| [Frozen bring-up artifact](docs/BASELINE_20260815.md) | Qwen3-0.6B, two local prompts | 2/2 requests completed; no recorded request error or OOM | Historical end-to-end smoke only |
| Current-format local smoke (`/root/autodl-tmp/vllm-tuner-output/slotune-results/smoke-ad36ee8-20260816`) | Qwen3-0.6B, two-request trace | Two COMPLETE/selectable trials; clean measurement/tool revision; recorded lineage; 37 sealed entries including two trial anchors | Current pipeline and fresh attestation wiring only |
| Local 3B preflight (`/root/autodl-tmp/slotune-results/qwen25-3b-preflight-20260815-a`) | Qwen2.5-3B-Instruct, two-request trace | Default evaluation and one repeat completed; no capacity sweep or held-out result | 3B model/pipeline preflight only |
| [Qwen2.5-3B Chat formal run](docs/results/qwen25-3b-34a25a2.md#chat) | RTX 5090; 500 × 128-token synthetic Chat; 48 search + 15 repeat + 15 holdout + 18 capacity trials | 89 COMPLETE, 7 constraint-INFEASIBLE, 0 request failures; TPE-11 repeat goodput 7.948 req/s versus default 7.948; all tested capacity points feasible, so capacity is only a ≥27.883 req/s lower bound | Complete repeated/holdout/capacity evidence; no significant tuning gain |
| [Qwen2.5-3B RAG formal run](docs/results/qwen25-3b-34a25a2.md#rag) | RTX 5090; 500 × 128-token synthetic RAG; same 96-trial protocol | 89 COMPLETE, 7 constraint-INFEASIBLE, 0 request failures; default remains best; observed knee near nominal 16 req/s and two 32 req/s repeats violate TTFT SLO | Complete repeated/holdout/capacity evidence; negative tuning result |

Both formal runs were measured from clean commit
`34a25a2e10951bfab1c2a86b4c60aff5bef785df`; the result snapshot records the environment, trace
hashes, target versus empirical arrival rates, repeat ranges, constraints, and artifact audit.
Clean tool commit `ad36ee8e0e15a6d0502a35f9e794b056b9522a82` subsequently attested both
roots without changing their measurement provenance or original summary/scheduler bytes. The
configs and tests remain protocol evidence; only the linked formal artifacts support these
measurements.

![RAG default capacity curve](docs/results/rag-capacity.png)

The graph is a checked-in copy of the formal artifact's generated capacity plot. The source
artifact and exact table are identified in the [result snapshot](docs/results/qwen25-3b-34a25a2.md#capacity).

## Correct objective and metrics

The optimizer has one objective:

```text
SLO goodput = successful requests satisfying TTFT, TPOT, and E2E SLOs
              --------------------------------------------------------
                              measured seconds
```

Errors, OOM, server exit, peak-VRAM violations, and configured latency violations are hard
constraints. Infeasible or failed trials are never selected as best.

- **TTFT:** request send time to the first non-empty streamed token.
- **ITL:** interval between adjacent output-token arrivals, emitted only when pinned vLLM delta token IDs or the official benchmark provide matching token-level evidence. SSE inter-event latency is stored separately.
- **TPOT:** `(finish - first token) / (output tokens - 1)`; a one-token response uses zero.
- **E2E:** request send time to completion, stored independently rather than reconstructed.
- **Offered load:** scheduled requests per measurement window.
- **Achieved throughput:** successful completions per measurement window.
- **Goodput:** successful completions that also meet every configured SLO per window.

Warmups are marked and excluded. Token totals must come from tokenizer/server usage or a
validated official result. Percentiles use interpolation; task exceptions and missing
telemetry remain explicit instead of becoming zeroes.

See [`docs/user-guide/reports/metrics-explained.md`](docs/user-guide/reports/metrics-explained.md)
for the complete definitions.

## Architecture

```text
fixed ExperimentSpec + trace + SLO + environment
                         │
                         ▼
 Trial Controller: START → READY → WARMUP → MEASURE → COLLECT → STOP
       │                         │                         │
       ▼                         ▼                         ▼
 managed vLLM             official bench/SSE       /metrics + NVML
 process group             per-request results      aligned time series
       └─────────────────────────┬─────────────────────────┘
                                 ▼
              constraint reducer + SLO-goodput objective
                                 ▼
                equal-budget default / random / TPE
                                 ▼
              repeats + held-out trace + static report

 fixed trace ──► deterministic scheduler simulator
                fixed 512…8192 vs adaptive budget
                fairness/starvation/preemption + downside analysis
```

The runtime lifecycle and the simulator are deliberately separate. The adaptive scheduler is
currently a mechanism simulator; unless an artifact explicitly identifies a vLLM runtime
integration, simulator gains must not be presented as measured GPU gains.

## Search method

Only effective single-GPU server parameters are searched:

```yaml
search_space:
  gpu_memory_utilization: [0.60, 0.95]
  max_num_seqs: [8, 16, 32, 64, 128]
  max_num_batched_tokens: [1024, 2048, 4096, 8192]
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
```

`batch_size` is not a valid vLLM server parameter and is rejected. Tensor and pipeline
parallelism are fixed to one for the core single-card experiment. Workload settings such as
request rate, burstiness, and concurrency are experiment inputs—not tuner suggestions.

The `default`, seeded `random`, and constrained `tpe` methods each receive the same number of
measured COMPLETE/INFEASIBLE evaluations. FAIL and PRUNED attempts remain visible but do not
silently count as successful evidence. Top candidates are repeated three times in the formal
configs and rerun against a held-out trace.

## Cross-layer telemetry

During the exact measurement window SLOTune samples:

- vLLM running/waiting requests, KV-cache usage, preemption counters, token counters,
  prefix-cache counters, and available latency/queue metrics from `/metrics`;
- NVML memory, GPU utilization, power, temperature, and clocks;
- client request timestamps, status, error type, and token counts.

Prometheus counters are reduced as window deltas. GPU summaries include peak/mean/p95 where
applicable. Telemetry is collected once more before server shutdown and closes reliably on
cancellation. Collection failure is represented as unavailable data, never fabricated zero.

## Adaptive scheduling simulator

The pure-Python simulator compares configurable fixed budgets (default: 512, 1024, 2048,
4096, 8192) with an adaptive policy driven by decode backlog, prefill backlog, oldest prefill
age, KV pressure, recent p99 TTFT/TPOT, and preemptions. Guardrails include budget bounds,
hysteresis, aging, max-wait swaps, minimum prefill progress, and an admitted-sequence limit.

Every step records its budget and actual decode/prefill tokens. Reports include p50/p99 queue,
TTFT and TPOT, goodput, Jain fairness, starvation, and preemption. Calibration and held-out
comparisons explicitly retain conditions where adaptive behavior has no benefit or regresses.

## Quick start

### Install

```bash
cd /root/autodl-tmp/vllm-tuner
uv venv --seed --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install vllm --torch-backend=auto
```

Large caches and formal artifacts should remain on `/root/autodl-tmp`.

### Short CPU demo

This runs the deterministic calibration/held-out scheduler ablation; it does not need a GPU:

```bash
./scripts/run_demo.sh /root/autodl-tmp/slotune-demo/scheduler
```

The explicit output directory receives `scheduler_ablation.json` and
`scheduler_ablation.md`. The script refuses accidental overwrite.

For the three-to-five-minute walkthrough, pass a completed formal artifact as the optional second
argument. After the CPU simulation finishes, the script checks and prints the pre-generated
report paths, so it never waits for a formal GPU run:

```bash
./scripts/run_demo.sh \
  /root/autodl-tmp/slotune-demo/scheduler-demo \
  /root/autodl-tmp/slotune-results/qwen25-3b-rag-formal-34a25a2
```

### One-command GPU smoke

After installation and local model preparation:

```bash
./scripts/run_data_disk_reproduction.sh slotune_smoke_001
```

This uses [`config/reproduction_smoke.yaml`](config/reproduction_smoke.yaml) and local
Qwen3-0.6B solely to verify model load, streaming requests, telemetry, cleanup, and artifact
generation. Its two-request output is not a performance benchmark.

### Formal 3B configurations

The checked-in profiles target a local Qwen2.5-3B-Instruct on one RTX 5090:

```bash
./scripts/run_reproduction_command.sh tune \
  --config config/formal_3b_chat.yaml \
  --study-name qwen25_3b_chat_001 \
  --results-root /root/autodl-tmp/slotune-results

./scripts/run_reproduction_command.sh tune \
  --config config/formal_3b_rag.yaml \
  --study-name qwen25_3b_rag_001 \
  --results-root /root/autodl-tmp/slotune-results
```

Both configurations use equal budgets for default/random/TPE, `repeat_count: 3`, and held-out
validation. Run names must be unique unless an explicitly compatible resume is intended.
The wrapper exports all data-disk cache and temporary-directory variables into the tuner process
and its vLLM children, and invokes the installed CLI directly so the pinned GPU overlay is not
resynchronized away.

## Current YAML schema

```yaml
model: /root/autodl-tmp/models/Qwen2.5-3B-Instruct
gpu:
  device_ids: [0]
  count: 1
slo:
  ttft_ms: 1000
  tpot_ms: 100
  e2e_ms: null
constraints:
  max_error_rate: 0.01
  max_peak_vram_mb: 31000
  require_no_oom: true
  require_server_alive: true
search_space:
  gpu_memory_utilization: [0.60, 0.95]
  max_num_seqs: [8, 16, 32, 64, 128]
  max_num_batched_tokens: [1024, 2048, 4096, 8192]
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
workload:
  name: chat
  sample_size: 500
  warmup_requests: 30
  max_concurrency: 32
  request_rate: 8
  burstiness: 1.0
  seed: 2026
  capacity_request_rates: [1, 2, 4, 8, 16, 32]
  capacity_repeats: 3
  benchmark_backend: sse
telemetry:
  enabled: true
  interval_ms: 200
  collect_nvml: true
study:
  trial_budget: 16
  methods: [default, random, tpe]
  repeat_count: 3
  top_candidates: 3
  holdout_enabled: true
  resume: false
```

See [`docs/user-guide/configuration.md`](docs/user-guide/configuration.md) and the validated
files under [`config/`](config/).

## Artifacts and failure evidence

Each experiment has an immutable manifest, trace checksum, environment fingerprint, raw
per-request JSONL, Prometheus/NVML series, server command/log, structured status, aggregate
Parquet tables, scheduler ablation, and static reports under the explicit results root.

The trial state machine distinguishes FAILED, INFEASIBLE, and PRUNED. Failure artifacts retain
the phase, exception or classified cause (for example OOM, invalid argument, port conflict,
startup timeout, request failure, or server exit), last server state, and logs. An experiment
with no feasible candidate reports that fact; it does not substitute a failed trial or a
sentinel score.

At light or homogeneous load, an adaptive budget may tie a fixed budget while adding decision
overhead; under sustained KV/decode pressure it may reduce admission and incur preemptions.
These are expected negative/no-benefit conditions to preserve and explain, not results to hide.

## Evidence currently in the repository

- [`docs/BASELINE_20260815.md`](docs/BASELINE_20260815.md): frozen Qwen3-0.6B bring-up
  evidence, clearly scoped as smoke only.
- [`config/formal_3b_chat.yaml`](config/formal_3b_chat.yaml) and
  [`config/formal_3b_rag.yaml`](config/formal_3b_rag.yaml): the exact formal protocols used at
  measurement commit `34a25a2`.
- [`docs/results/qwen25-3b-34a25a2.md`](docs/results/qwen25-3b-34a25a2.md): checked-in evidence
  snapshot for both external data-disk artifact roots, including negative results and plot copies.
- Scheduler unit tests and `scripts/run_scheduler_ablation.py`: deterministic CPU mechanism and
  fairness checks. The recorded 0% goodput gain and TTFT regressions are not GPU runtime claims.

## Limitations and future work

- Core execution is single-node, single-GPU; TP/PP and multi-GPU tuning are intentionally out
  of scope.
- Formal claims remain specific to the recorded model, tokenizer, GPU, trace, vLLM version,
  driver, and environment.
- The adaptive scheduler is not yet wired into vLLM's version-sensitive internal scheduler.
- Formal configurations use the SSE client to replay frozen arrival offsets exactly. Official
  `vllm bench serve` is retained as a live reference/cross-validation backend.
- Energy per output token depends on available NVML power samples.
- Planned work includes 7B/8B validation, runtime scheduler integration pinned to an exact vLLM
  commit, repeated confidence intervals, and broader held-out request rates. M6 prefix-caching
  experiments are explicitly deferred P1 work and do not block the core Definition of Done.

## Documentation and license

- [Documentation index](docs/README.md)
- [Methodology](docs/METHODOLOGY.md)
- [Engineering development log](docs/DEVELOPMENT_LOG.md)
- [Formal experiment protocol](docs/FORMAL_EXPERIMENTS.md)
- [Project-plan audit](docs/PLAN_AUDIT.md)
- [Reproduction guide](REPRODUCTION.md)
- [Project implementation plan](docs/SLOTUNE_PROJECT_PLAN.md)
- [MIT License](LICENSE)

The upstream citation remains available in its project history. When discussing this fork,
attribute the upstream project and distinguish upstream components from the SLOTune changes
listed above.
