# SLOTune project-plan audit

This document audits the repository against
[`SLOTUNE_PROJECT_PLAN.md`](SLOTUNE_PROJECT_PLAN.md). It records implementation evidence and
experimental evidence separately; a checked box in code is not automatically a benchmark claim.
The audit reflects the workspace state on 2026-08-15, before the final project commit.

## Status vocabulary

| Status | Meaning |
|---|---|
| Implemented/tested | The repository contains the implementation and automated contract tests |
| Protocol ready | A validated configuration or command exists, but the required GPU result is absent |
| Smoke/preflight only | A small local artifact exercises the path but cannot support performance claims |
| Formal evidence pending | The plan requires repeated/held-out measurements that are not yet archived and linked |
| Deferred | Explicit P1 or out-of-scope work |

Commit links in the README contribution table say `Pending final commit` until a real final
revision exists. This avoids inventing provenance for a dirty shared workspace.

## Complete plan-section coverage

| Plan sections | Where the requirement is audited or documented | Status |
|---|---|---|
| §0–2 identity, goals, value | [README scope and research questions](../README.md#research-questions), [methodology](METHODOLOGY.md) | Documented |
| §3 P0 correctness gaps | [P0 correctness audit](#p0-correctness-audit) below | Audited requirement by requirement |
| §4–5 target architecture and modules | [README architecture](../README.md#architecture) and milestone code links below | Implemented through the staged module layout |
| §6–12 M0–M6 | [Milestone audit](#milestone-audit) below | M0–M5 software path implemented; formal evidence gaps and M6 deferral explicit |
| §13 formal experiment protocol | [Formal protocol](FORMAL_EXPERIMENTS.md) and [formal-protocol audit](#formal-protocol-audit) | Configured; full formal artifacts pending |
| §14 testing | [Test-plan audit](#test-plan-audit) | Unit contracts present; environment-gated/manual GPU evidence remains explicit |
| §15 artifact layout | [Methodology artifact contract](METHODOLOGY.md#artifact-acceptance) and [artifact-layout audit](#artifact-layout-audit) | Implemented/tested |
| §16 proposed calendar | Not used as an acceptance shortcut | Milestones and evidence, not elapsed days, determine status |
| §17 README packaging | [README](../README.md) and [README audit](#readme-reproduction-and-demo-audit) | Required attribution/contribution/evidence boundaries present; formal result rows pending |
| §18 demo | [CPU demo](../scripts/run_demo.sh), [GPU smoke](../scripts/run_data_disk_reproduction.sh), and [reproduction guide](../REPRODUCTION.md) | Mechanism/wiring demos ready; formal-report segment pending |
| §19 résumé templates | Kept only in the implementation plan | No placeholder percentage is presented as a result |
| §20 risk/degradation paths | [README limitations](../README.md#limitations-and-future-work), formal negative-result rule, and milestone gaps below | Documented; simulator regressions and unavailable telemetry are retained |
| §21 excluded scope | [README scope](../README.md) and limitations | Single-node/single-GPU boundary explicit; web/Kubernetes/multi-GPU claims excluded |
| §22 Definition of Done | [Definition-of-Done audit](#definition-of-done-audit) below | Software criteria largely met; formal experimental criteria not complete |
| §23 implementation sequence | README contribution table and milestone evidence links | Implemented in the planned correctness-first layers |
| §24 official references | Preserved in the source implementation plan | Reference list retained; no external claim is substituted for local evidence |

## Milestone audit

| Milestone | Implementation and test evidence | Experimental evidence | Remaining gap |
|---|---|---|---|
| M0: frozen baseline | [Baseline record](BASELINE_20260815.md), [data-disk setup](../scripts/setup_data_disk_reproduction.sh), [one-command smoke](../scripts/run_data_disk_reproduction.sh), and documentation tests | Historical `reproduction_gpu_20260815_a` records a two-request Qwen3-0.6B bring-up | Final clean commit/branch provenance remains pending; the legacy artifact is not a formal benchmark |
| M1: trustworthy benchmark | [SSE client](../src/vllm_tuner/benchmarks/sse_client.py), [official adapter](../src/vllm_tuner/benchmarks/vllm_bench.py), [result parser](../src/vllm_tuner/benchmarks/result_parser.py), [metric reducer](../src/vllm_tuner/benchmarks/metrics.py), [fixtures](../tests/fixtures), and benchmark unit tests | Current SSE smoke/preflight artifacts exist on the data disk; the official backend is retained for live reference cross-validation | No immutable formal official-vs-SSE comparison artifact is linked, so no numerical agreement claim is made |
| M2: cross-layer telemetry | [Telemetry session](../src/vllm_tuner/profiling/session.py), [Prometheus](../src/vllm_tuner/profiling/prometheus.py), [continuous NVML](../src/vllm_tuner/profiling/nvml_session.py), [time-series reducer](../src/vllm_tuner/profiling/timeseries.py), and telemetry unit tests | Current smoke/preflight artifacts exercise current-format telemetry and explicit availability markers | Formal chat/RAG timelines and any energy-per-token result remain pending |
| M3: reliable trial lifecycle | [State machine](../src/vllm_tuner/runtime/state_machine.py), [controller](../src/vllm_tuner/runtime/controller.py), [server lifecycle](../src/vllm_tuner/runtime/server.py), [failure taxonomy](../src/vllm_tuner/runtime/failures.py), and lifecycle tests | Smoke/preflight artifacts contain structured status, logs, and cleanup evidence | GPU fault-injection evidence across every failure class is not archived as a formal artifact |
| M4: SLO-aware autotuner | [Objective](../src/vllm_tuner/tuning/objective.py), [search space](../src/vllm_tuner/tuning/search_space.py), [constrained optimizer](../src/vllm_tuner/tuning/optimizer.py), [experiment runner](../src/vllm_tuner/experiment/runner.py), and optimizer/runner tests | [Chat and RAG protocols](FORMAL_EXPERIMENTS.md) configure equal default/random/TPE budgets, repeats, held-out validation, and capacity sweeps | Full 3B search/repeat/held-out artifacts are not yet recorded; no tuning improvement is claimed |
| M5: adaptive token budget | [Budget policy](../src/vllm_tuner/scheduling/token_budget.py), [admission controller](../src/vllm_tuner/scheduling/admission.py), [deterministic simulator](../src/vllm_tuner/scheduling/simulator.py), [ablation CLI](../scripts/run_scheduler_ablation.py), and scheduler tests | The CPU demo can emit deterministic calibration/held-out JSON and Markdown including regressions | Runtime integration with a pinned vLLM scheduler and measured GPU ablation are pending |
| M6: prefix caching | Prefix-cache metrics and a shared-prefix RAG profile provide experiment inputs | No cold/warm APC matrix is recorded | Deferred P1: APC on/off, reuse ratios, ordering, bounded reordering, and fairness experiments |

## P0 correctness audit

### Benchmark correctness

| Plan requirement | Evidence | Status |
|---|---|---|
| Real SSE streaming; split/combined event frames; `data`, blank lines, `[DONE]`, empty text, HTTP error, timeout | [SSE implementation](../src/vllm_tuner/benchmarks/sse_client.py), [fixtures](../tests/fixtures/sse), [tests](../tests/unit/test_benchmark_sse_client.py) | Implemented/tested |
| Monotonic per-request send/first-token/token/finish timestamps | [Request models](../src/vllm_tuner/benchmarks/models.py), [SSE client](../src/vllm_tuner/benchmarks/sse_client.py) | Implemented/tested |
| TTFT, TPOT, ITL, E2E, interpolated percentiles, goodput | [Reducer](../src/vllm_tuner/benchmarks/metrics.py), [hand-checked tests](../tests/unit/test_benchmark_metrics.py) | Implemented/tested |
| Token counts are meaningful; warmup excluded; task exceptions retained; raw request output | [Result parser](../src/vllm_tuner/benchmarks/result_parser.py), [controller](../src/vllm_tuner/runtime/controller.py), [artifact store](../src/vllm_tuner/experiment/artifacts.py) | Implemented/tested |
| Official `vllm bench serve` adapter | [Adapter](../src/vllm_tuner/benchmarks/vllm_bench.py), [tests](../tests/unit/test_benchmark_vllm_bench.py) | Implemented/tested; live reference path |
| Same-workload official/SSE numerical cross-validation | [Environment-gated integration test](../tests/integration/test_sse_client.py) | Live cross-check path exists; immutable comparison artifact pending |

Formal configs deliberately select SSE, not the official backend, because formal evidence must
replay frozen `scheduled_offset_seconds`. Official bench remains a live reference with an
independently generated arrival process; reports must label that protocol difference.

### Telemetry correctness

| Plan requirement | Evidence | Status |
|---|---|---|
| vLLM `/metrics` sampling and metric aliases | [Prometheus collector](../src/vllm_tuner/profiling/prometheus.py), [tests](../tests/unit/test_prometheus.py) | Implemented/tested |
| Running/waiting, KV usage, preemptions, token and prefix-cache counters | [Telemetry session](../src/vllm_tuner/profiling/session.py), [fixtures](../tests/fixtures/prometheus) | Implemented/tested |
| Continuous NVML memory/utilization/power/temperature/clocks | [NVML session](../src/vllm_tuner/profiling/nvml_session.py), [tests](../tests/unit/test_nvml_session.py) | Implemented/tested |
| Monotonic alignment, peak/mean/p95 and counter window deltas | [Time-series reducer](../src/vllm_tuner/profiling/timeseries.py), [tests](../tests/unit/test_timeseries.py) | Implemented/tested |
| Cancellation/final sample/missing-data semantics | [Session tests](../tests/unit/test_telemetry_session.py), [artifact finalization](../src/vllm_tuner/experiment/artifacts.py) | Implemented/tested |
| Formal queue/KV/GPU explanation and energy-per-token evidence | [Reporting protocol](FORMAL_EXPERIMENTS.md#reporting-checklist) | Formal evidence pending |

### Optimizer correctness

| Plan requirement | Evidence | Status |
|---|---|---|
| Remove weighted objective and ineffective `batch_size`; fix TP/PP to one | [Configuration models](../src/vllm_tuner/config/models.py), [search-space tests](../tests/unit/test_slotune_config.py), [documentation tests](../tests/unit/test_documentation.py) | Implemented/tested |
| Detect fixed/trial parameter conflicts | [Search-space validation](../src/vllm_tuner/tuning/search_space.py), [tests](../tests/unit/test_slotune_config.py) | Implemented/tested |
| SLO-goodput objective and hard constraints | [Objective](../src/vllm_tuner/tuning/objective.py), [tests](../tests/unit/test_objective.py) | Implemented/tested |
| FAILED/INFEASIBLE/PRUNED remain explicit and cannot become best | [Optimizer](../src/vllm_tuner/tuning/optimizer.py), [tests](../tests/unit/test_constrained_optimizer.py) | Implemented/tested |
| Equal measured budgets for default/random/TPE | [Optimizer](../src/vllm_tuner/tuning/optimizer.py), [formal configs](../config) | Implemented/tested; formal run pending |
| Repeats and held-out candidate validation | [Experiment runner](../src/vllm_tuner/experiment/runner.py), [runner tests](../tests/unit/test_experiment_runner.py) | Implemented/tested; formal run pending |

## M5 acceptance audit

| Acceptance criterion | Evidence | Status |
|---|---|---|
| No step exceeds total token budget | [Simulator tests](../tests/unit/test_scheduling_simulator.py) | Implemented/tested |
| Decode and prefill both progress | [Simulator and admission tests](../tests/unit/test_scheduling_simulator.py) | Implemented/tested |
| Aging, max-wait, and minimum progress prevent infinite starvation | [Admission tests](../tests/unit/test_scheduling_admission.py) | Implemented/tested |
| Same seed/trace reproduces decisions | [Policy and script tests](../tests/unit/test_scheduling_token_budget.py) | Implemented/tested |
| At least two fixed budgets; defaults 512/1024/2048/4096/8192 | [Ablation script](../scripts/run_scheduler_ablation.py), [script tests](../tests/unit/test_scheduling_script.py) | Implemented/tested |
| p50/p99 queue, TTFT, TPOT, goodput, fairness, starvation, preemption | [Simulator](../src/vllm_tuner/scheduling/simulator.py), [tests](../tests/unit/test_scheduling_simulator.py) | Implemented/tested |
| Calibration, held-out, and no-benefit/regression conditions | [Ablation script](../scripts/run_scheduler_ablation.py), [script tests](../tests/unit/test_scheduling_script.py) | Implemented/tested as mechanism evidence |
| Measured vLLM runtime gain | None | Not claimed; runtime integration pending |

## Formal-protocol audit

| Plan requirement | Repository state | Status |
|---|---|---|
| 0.6B is smoke only; formal model is 3B–8B | Qwen3-0.6B smoke plus Qwen2.5-3B-Instruct formal configs | Protocol ready |
| At least chat and RAG workloads | [`formal_3b_chat.yaml`](../config/formal_3b_chat.yaml) and [`formal_3b_rag.yaml`](../config/formal_3b_rag.yaml) | Protocol ready |
| Fixed trace per comparison and separate holdout | Trace generation/checksums and `--trace`/`--holdout-trace` are implemented | Implemented/tested |
| Capacity rates and repeats | Both configs declare 1/2/4/8/16/32 requests/s and `capacity_repeats: 3` | Protocol ready |
| Warmup and at least 500 measured requests | Both formal configs declare 30 warmups and 500 measured requests | Protocol ready |
| Baseline/top candidates repeated three times | `repeat_count: 3`, `top_candidates: 3` | Protocol ready |
| Held-out validation | `holdout_enabled: true` | Protocol ready |
| Formal artifact and honest negative-result analysis | No complete formal chat/RAG artifact is linked | Formal evidence pending |

The plan lists `inf` as a suggested capacity point and prefers 7B/8B when available. The checked-in
formal protocol intentionally uses the explicitly requested finite 1–32 requests/s matrix and a
locally available 3B dense model, which remains inside the plan's accepted 3B–8B range.

## Test-plan audit

- Unit coverage exists for SSE framing, metric math, percentile/goodput, Prometheus deltas, NVML
  aggregates, state transitions, failure classification, search constraints, scheduler
  conservation/fairness, traces, checksums, artifacts, reports, and current documentation.
- A local fake/live SSE integration test exists; the live official-vs-SSE test is explicitly
  environment-gated so ordinary CPU CI does not pretend to run a GPU server.
- GPU smoke and formal performance runs remain explicit manual commands. Performance tests are not
  silently included in ordinary unit CI.

## Artifact-layout audit

[`ArtifactStore`](../src/vllm_tuner/experiment/artifacts.py) implements the planned manifest,
experiment config, search/holdout traces and hashes, environment fingerprint, per-trial raw
records/logs/status, aggregate tables, scheduler ablation, and static report layout. It also adds
`artifact-status.json` availability metadata, capacity-sweep tables/traces, and explicit degraded
markers. Missing evidence is therefore distinguishable from measured zero.

The current data disk contains smoke/preflight directories, including
`/root/autodl-tmp/slotune-results/qwen25-3b-preflight-20260815-a`. That run used a two-request
trace, one default evaluation, one repeat, no capacity sweep, and no held-out result. It is a
3B pipeline preflight—not the formal chat/RAG experiment required by the plan.

## README, reproduction, and demo audit

- [README](../README.md) includes the exact required fork attribution, a code/test/artifact
  contribution table, method and architecture, correct metrics, equal-budget baselines,
  telemetry, simulator scope, failure/negative conditions, one-command smoke, limitations, and an
  artifact-backed real-results register.
- [Reproduction guide](../REPRODUCTION.md) separates legacy, current smoke, 3B preflight/formal,
  and simulator evidence; documents data-disk paths and reproducible commands; and leaves explicit
  pending formal-result rows.
- [CPU demo](../scripts/run_demo.sh) produces deterministic JSON/Markdown to an explicit path.
- [Formal protocol](FORMAL_EXPERIMENTS.md) records the run order, capacity matrix, metrics,
  reporting checklist, and negative-result rule.

The planned demo step that opens a completed formal report cannot yet be advertised because no
formal chat/RAG artifact is linked. The short CPU demo and GPU smoke remain valid demonstrations
of mechanism and wiring only.

## Definition-of-Done audit

| Definition-of-Done item | Status |
|---|---|
| Reliable streaming/official metrics and unit-tested TTFT/TPOT/ITL/E2E/tokens/goodput | Implemented/tested |
| Continuous `/metrics` and NVML sampling | Implemented/tested; formal artifact pending |
| Failed trials excluded from best selection | Implemented/tested |
| Equal default/random/TPE budgets | Implemented/tested; formal artifact pending |
| At least one formal 3B–8B experiment | **Not complete**; 3B preflight is not formal |
| At least two formal workloads | Configured; artifacts pending |
| Three baseline/top-candidate repeats | Configured; artifacts pending |
| Held-out result | Configured; artifact pending |
| Fixed-budget ablation | Implemented/tested in deterministic simulator |
| Adaptive simulator and fairness tests | Implemented/tested |
| Environment, trace, params, raw requests, logs traceable | Implemented/tested; formal artifact pending |
| README upstream attribution and personal-contribution evidence | Complete, final commit links pending |
| Five-minute demo | CPU mechanism demo and GPU smoke commands ready; formal-report segment pending |
| Smoke not presented as performance evidence | Complete |
| Large files remain on `/root/autodl-tmp` | Documented and enforced by reproduction scripts |

The project is therefore implementation-complete for the M1–M5 software path and protocol-ready
for formal experiments, but it does **not** satisfy the plan's experimental Definition of Done
until repeated 3B chat/RAG capacity, holdout, and negative-result artifacts are completed,
validated, and linked.
