# SLOTune project-plan audit

This document audits the repository against
[`SLOTUNE_PROJECT_PLAN.md`](SLOTUNE_PROJECT_PLAN.md). Implementation evidence and experimental
evidence remain separate: passing code does not become a performance claim. This audit reflects
the completed Qwen2.5-3B formal measurements made from clean commit
`34a25a2e10951bfab1c2a86b4c60aff5bef785df` and checked on 2026-08-16.

## Status vocabulary

| Status | Meaning |
|---|---|
| Implemented/tested | The repository contains the implementation and automated contract tests |
| Completed formal evidence | A repeated/held-out GPU artifact has been audited and linked |
| Negative result | The experiment completed but did not meet the preregistered improvement threshold |
| Smoke/preflight only | A small local artifact exercises the path but cannot support performance claims |
| Deferred | Optional P1, runtime-integration, or explicitly out-of-scope work |

The README contribution table links real local revisions `aa9d70a`, `0d605c3`, `b8f2dc1`, and
`34a25a2` through the repository's
[implementation-revisions register](FORMAL_EXPERIMENTS.md#implementation-revisions). It does not
invent upstream GitHub links. The later attestation/documentation revision is not substituted for
the measurement commit.

## Complete plan-section coverage

| Plan sections | Where the requirement is audited or documented | Status |
|---|---|---|
| §0–2 identity, goals, value | [README scope and research questions](../README.md#research-questions), [methodology](METHODOLOGY.md) | Documented |
| §3 P0 correctness gaps | [P0 correctness audit](#p0-correctness-audit) | Audited requirement by requirement |
| §4–5 target architecture and modules | [README architecture](../README.md#architecture) and milestone links below | Implemented through the staged module layout |
| §6–12 M0–M6 | [Milestone audit](#milestone-audit) | Core M0–M5 complete; optional M6 deferred explicitly |
| §13 formal experiment protocol | [Formal protocol](FORMAL_EXPERIMENTS.md), [result snapshot](results/qwen25-3b-34a25a2.md), and [formal audit](#formal-protocol-audit) | Two formal workloads completed and audited |
| §14 testing | [Test-plan audit](#test-plan-audit) | Unit/integration contracts plus manual GPU evidence |
| §15 artifact layout | [Artifact contract](METHODOLOGY.md#artifact-acceptance) and [artifact audit](#artifact-layout-audit) | Implemented, measured, and integrity-checked |
| §16 proposed calendar | Not used as an acceptance shortcut | Milestones and evidence—not elapsed days—determine status |
| §17 README packaging | [README](../README.md) and [packaging audit](#readme-reproduction-and-demo-audit) | Real table, plot, attribution, commits, limits, and negative results present |
| §18 demo | [Demo script](../scripts/run_demo.sh) and [reproduction guide](../REPRODUCTION.md) | CPU ablation plus pre-generated formal report; no live formal GPU wait |
| §19 résumé templates | Kept only in the implementation plan | No placeholder percentage is presented as a result |
| §20 risk/degradation paths | [README limitations](../README.md#limitations-and-future-work) and formal negative-result analysis | Constraint failures and no-benefit/regression cases retained |
| §21 excluded scope | [README scope](../README.md) and limitations | Single-node/single-GPU boundary explicit; excluded claims remain excluded |
| §22 Definition of Done | [Definition-of-Done audit](#definition-of-done-audit) | Core Definition of Done complete with negative performance outcome |
| §23 implementation sequence | README contribution table and revision register | Implemented in correctness-first layers |
| §24 official references | Preserved in the source implementation plan | Reference list retained; local evidence supports local claims |

## Milestone audit

| Milestone | Implementation and test evidence | Experimental evidence | Remaining boundary |
|---|---|---|---|
| M0: frozen baseline | [Baseline record](BASELINE_20260815.md), [data-disk setup](../scripts/setup_data_disk_reproduction.sh), [one-command smoke](../scripts/run_data_disk_reproduction.sh), and documentation tests | Historical 0.6B bring-up remains smoke; final smoke `smoke-34a25a2-20260816` validates the current pipeline; formal manifests record clean commit `34a25a2` | Smoke remains excluded from performance conclusions |
| M1: trustworthy benchmark | [SSE client](../src/vllm_tuner/benchmarks/sse_client.py), [official adapter](../src/vllm_tuner/benchmarks/vllm_bench.py), [result parser](../src/vllm_tuner/benchmarks/result_parser.py), [metric reducer](../src/vllm_tuner/benchmarks/metrics.py), fixtures, and tests | `cross-validation-34a25a2-20260816` matches completed/failed and input/output token totals and validates SSE token timestamps | Sequential backend latencies are a sanity check, not equality evidence |
| M2: cross-layer telemetry | [Telemetry session](../src/vllm_tuner/profiling/session.py), [Prometheus](../src/vllm_tuner/profiling/prometheus.py), [continuous NVML](../src/vllm_tuner/profiling/nvml_session.py), [reducer](../src/vllm_tuner/profiling/timeseries.py), and tests | Both 96-trial formal roots preserve aligned engine/GPU series, availability, power/energy, and generated timelines | Causal conclusions remain limited to aligned observations |
| M3: reliable trial lifecycle | [State machine](../src/vllm_tuner/runtime/state_machine.py), [controller](../src/vllm_tuner/runtime/controller.py), [server lifecycle](../src/vllm_tuner/runtime/server.py), [failure taxonomy](../src/vllm_tuner/runtime/failures.py), and tests | Cleanup validated for all 192 formal trial directories; no residual GPU process, forced SIGKILL, or request-failure misclassification | Exhaustive GPU fault injection for every taxonomy member is not a performance requirement |
| M4: SLO-aware autotuner | [Objective](../src/vllm_tuner/tuning/objective.py), [search space](../src/vllm_tuner/tuning/search_space.py), [optimizer](../src/vllm_tuner/tuning/optimizer.py), [runner](../src/vllm_tuner/experiment/runner.py), and tests | Chat and RAG each contain equal 16-trial method budgets, repeats, holdout, and capacity sweeps | Completed negative result: neither workload met 15% goodput or 20% TTFT threshold |
| M5: adaptive token budget | [Budget policy](../src/vllm_tuner/scheduling/token_budget.py), [admission](../src/vllm_tuner/scheduling/admission.py), [simulator](../src/vllm_tuner/scheduling/simulator.py), [ablation CLI](../scripts/run_scheduler_ablation.py), and tests | Formal reports retain 0% adaptive goodput gain and TTFT regressions on calibration/held-out simulator traces | Runtime integration with vLLM internals is deferred; no GPU scheduler gain is claimed |
| M6: prefix caching | Prefix-cache metrics and a shared-prefix RAG workload provide the prerequisites | No cold/warm APC matrix is claimed | Deferred optional P1: APC on/off, reuse ratios, ordering, and fairness experiments; not a core DoD blocker |

## P0 correctness audit

### Benchmark correctness

| Plan requirement | Evidence | Status |
|---|---|---|
| Real SSE streaming; split/combined frames; `data`, blank lines, `[DONE]`, empty text, HTTP error, timeout | [Implementation](../src/vllm_tuner/benchmarks/sse_client.py), [fixtures](../tests/fixtures/sse), [tests](../tests/unit/test_benchmark_sse_client.py) | Implemented/tested |
| Monotonic send/first-token/token/finish timestamps | [Request models](../src/vllm_tuner/benchmarks/models.py), [SSE client](../src/vllm_tuner/benchmarks/sse_client.py) | Implemented/tested; 48,000 measured records per workload audited |
| TTFT, TPOT, ITL, E2E, interpolated percentiles, goodput | [Reducer](../src/vllm_tuner/benchmarks/metrics.py), [tests](../tests/unit/test_benchmark_metrics.py) | Implemented/tested; formal aggregates checked against raw records |
| Meaningful tokens; warmup excluded; task exceptions/raw output retained | [Parser](../src/vllm_tuner/benchmarks/result_parser.py), [controller](../src/vllm_tuner/runtime/controller.py), [artifact store](../src/vllm_tuner/experiment/artifacts.py) | Implemented/tested and present in formal artifacts |
| Official `vllm bench serve` adapter | [Adapter](../src/vllm_tuner/benchmarks/vllm_bench.py), [tests](../tests/unit/test_benchmark_vllm_bench.py) | Implemented/tested; live reference path |
| Same-prompt official/SSE numerical cross-validation | [Environment-gated integration](../tests/integration/test_sse_client.py), [result snapshot](results/qwen25-3b-34a25a2.md#official-vs-sse-cross-check) | Live artifact completed at `34a25a2` |

Formal configs select SSE because formal evidence must replay frozen `scheduled_offset_seconds`.
The official adapter remains a sequential live reference with an independently generated arrival
process; the cross-check confirms token/count semantics and magnitude, not latency equality.

### Telemetry correctness

| Plan requirement | Evidence | Status |
|---|---|---|
| vLLM `/metrics` sampling and aliases | [Prometheus collector](../src/vllm_tuner/profiling/prometheus.py), [tests](../tests/unit/test_prometheus.py) | Implemented/tested/measured |
| Running/waiting, KV, preemptions, token and prefix-cache counters | [Telemetry session](../src/vllm_tuner/profiling/session.py), [fixtures](../tests/fixtures/prometheus) | Implemented/tested/measured |
| Continuous NVML memory/utilization/power/temperature/clocks | [NVML session](../src/vllm_tuner/profiling/nvml_session.py), [tests](../tests/unit/test_nvml_session.py) | Implemented/tested/measured |
| Monotonic alignment, summaries, counter deltas | [Reducer](../src/vllm_tuner/profiling/timeseries.py), [tests](../tests/unit/test_timeseries.py) | Implemented/tested; formal report inputs audited |
| Cancellation/final sample/missing-data semantics | [Session tests](../tests/unit/test_telemetry_session.py), [artifact finalization](../src/vllm_tuner/experiment/artifacts.py) | Implemented/tested; cleanup verified |
| Formal queue/KV/GPU/energy evidence | [Formal snapshot](results/qwen25-3b-34a25a2.md#artifact-audit) and external reports | Completed formal evidence |

### Optimizer correctness

| Plan requirement | Evidence | Status |
|---|---|---|
| Remove weighted objective/ineffective `batch_size`; TP/PP one | [Configuration](../src/vllm_tuner/config/models.py), [tests](../tests/unit/test_slotune_config.py) | Implemented/tested |
| Detect fixed/trial parameter conflicts | [Search validation](../src/vllm_tuner/tuning/search_space.py), [tests](../tests/unit/test_slotune_config.py) | Implemented/tested |
| SLO-goodput objective and hard constraints | [Objective](../src/vllm_tuner/tuning/objective.py), [tests](../tests/unit/test_objective.py) | Implemented/tested/measured |
| FAILED/INFEASIBLE/PRUNED explicit and excluded from best | [Optimizer](../src/vllm_tuner/tuning/optimizer.py), [tests](../tests/unit/test_constrained_optimizer.py) | Implemented/tested; 14 formal INFEASIBLE outcomes retained |
| Equal measured default/random/TPE budgets | [Optimizer](../src/vllm_tuner/tuning/optimizer.py), [configs](../config), formal roots | 16 × 3 per workload completed |
| Repeats and held-out exact-parameter validation | [Runner](../src/vllm_tuner/experiment/runner.py), [tests](../tests/unit/test_experiment_runner.py), [snapshot](results/qwen25-3b-34a25a2.md#tuning-outcome) | Five candidates × three repeats × two phases per workload completed |

## M5 acceptance audit

| Acceptance criterion | Evidence | Status |
|---|---|---|
| Budget conservation; decode/prefill progress | [Simulator tests](../tests/unit/test_scheduling_simulator.py) | Implemented/tested |
| Aging, max-wait, minimum progress, admission/fairness | [Admission tests](../tests/unit/test_scheduling_admission.py) | Implemented/tested |
| Deterministic same-seed decisions | [Policy/script tests](../tests/unit/test_scheduling_token_budget.py) | Implemented/tested |
| Fixed 512/1024/2048/4096/8192 and adaptive budgets | [Ablation script](../scripts/run_scheduler_ablation.py), [tests](../tests/unit/test_scheduling_script.py) | Completed on calibration and held-out simulator traces |
| Queue/TTFT/TPOT/goodput/fairness/starvation/preemption | [Simulator](../src/vllm_tuner/scheduling/simulator.py), formal reports | Recorded, including negative conditions |
| No-benefit/regression analysis | [Formal snapshot](results/qwen25-3b-34a25a2.md#scheduler-simulator-negative-result) | 0% goodput gain; TTFT regressions retained |
| Measured adaptive vLLM runtime gain | None claimed | Deferred runtime integration, outside the simulator acceptance claim |

## Formal-protocol audit

| Plan requirement | Recorded state | Status |
|---|---|---|
| 0.6B smoke only; formal model 3B–8B | Qwen3-0.6B smoke plus Qwen2.5-3B-Instruct formal artifacts | Completed in accepted model range |
| At least Chat and RAG workloads | Separate 96-trial Chat and RAG roots | Completed formal evidence |
| Fixed comparison trace and separate holdout | Search and holdout JSONL plus SHA-256 in each root | Completed and audited |
| Capacity 1/2/4/8/16/32, three repeats | 18 capacity trials per workload | Completed and audited |
| Warmup and ≥500 measured requests | 30 warmups plus 500 measured requests per trial | Completed and audited |
| Baseline/top candidates repeated three times | Five candidates × three repeat trials per workload | Completed and audited |
| Held-out validation | Five candidates × three separate holdout trials per workload | Completed and audited |
| Honest negative-result analysis | [Result snapshot](results/qwen25-3b-34a25a2.md) | Completed; no threshold-crossing gain claimed |

The plan suggests `inf` as an optional capacity point and prefers 7B/8B when available. The
recorded finite 1–32 matrix follows the explicit required configuration, and the local 3B dense
model is inside the accepted 3B–8B range. Chat remains a capacity lower bound because the highest
tested point was feasible; RAG identifies a knee and constraint boundary. No unmeasured `inf` or
7B/8B result is inferred.

## Test-plan audit

- Unit coverage includes SSE framing, metric math, percentile/goodput, Prometheus deltas, NVML,
  state transitions, failures, search constraints, scheduler conservation/fairness, traces,
  checksums, artifacts, reports, and documentation.
- Fake HTTP/Prometheus integrations and a live official/SSE GPU path are present. Ordinary CPU CI
  keeps the live test environment-gated rather than pretending to start a GPU server.
- GPU performance remains an explicit manual workflow. The completed artifact audit, not ordinary
  unit CI, supplies the formal measurements.

## Artifact-layout audit

[`ArtifactStore`](../src/vllm_tuner/experiment/artifacts.py) implements manifest/config,
search/holdout traces and hashes, environment fingerprint, per-trial requests/telemetry/logs,
structured status, aggregate tables, scheduler ablation, plots, and reports. Per-trial integrity
validated 96/96 directories in each workload. The reviewed additive post-run attestation path is
designed to index those anchors and hash non-trial aggregate/report/environment files without
rewriting sealed formal trials; it must be executed only after its implementation revision is
committed. Each workload has 89 COMPLETE and seven constraint-INFEASIBLE outcomes, with all
48,000 measured requests successful.

The attestation contract emits root `experiment-integrity.json`, `lineage.json`, and
`experiment-audit.json`, plus `aggregate/scheduler-negative-results.json` and
`report/scheduler-negative-results.md`. Additive `summary.compact-v1.json` preserves other root
summary fields while replacing inline scheduler raw rows with compact metrics and a reference to
the unchanged raw artifact by path, size, and SHA-256; the original `summary.json` is preserved
byte-for-byte. `vllm-tuner attest` validates an existing seal idempotently; explicit `--reseal`
validates the old seal before rebuilding and rejects corruption.

The external roots are:

- `/root/autodl-tmp/slotune-results/qwen25-3b-chat-formal-34a25a2`
- `/root/autodl-tmp/slotune-results/qwen25-3b-rag-formal-34a25a2`

The checked-in [snapshot](results/qwen25-3b-34a25a2.md) makes their provenance, key tables,
negative result, and copied plots reviewable while all large files remain on the data disk.

## README, reproduction, and demo audit

- [README](../README.md) preserves the exact upstream attribution, links real revisions and
  artifacts, publishes a real table/graph, and states both negative tuning outcomes.
- [Reproduction guide](../REPRODUCTION.md) separates legacy smoke, current smoke, preflight,
  formal GPU, and CPU simulator evidence; its result register is fully populated.
- [Demo](../scripts/run_demo.sh) runs the deterministic CPU ablation and optionally validates and
  displays an already-generated formal report/capacity plot. It never waits for formal GPU startup.
- [Formal protocol](FORMAL_EXPERIMENTS.md) records protocol, exact result boundary, target versus
  empirical arrival rates, capacity interpretation, and real implementation revisions.

## Definition-of-Done audit

| Definition-of-Done item | Status |
|---|---|
| Reliable streaming/official metrics and tested TTFT/TPOT/ITL/E2E/tokens/goodput | Complete |
| Continuous `/metrics` and NVML sampling | Complete in both formal roots |
| Failed/constraint-infeasible trials excluded from best selection | Complete; all negative outcomes retained |
| Equal default/random/TPE budgets | Complete: 16 measured trials per method/workload |
| At least one formal 3B–8B experiment | Complete: Qwen2.5-3B-Instruct |
| At least two formal workloads | Complete: Chat and RAG |
| Three baseline/top-candidate repeats | Complete |
| Held-out result | Complete for both workloads |
| Fixed token-budget ablation | Complete in the deterministic simulator |
| Adaptive simulator and fairness tests | Complete; 0% gain/regressions retained |
| Environment, trace, params, raw requests, telemetry, logs traceable | Complete and integrity-audited |
| README upstream attribution and contribution evidence | Complete with real local revision links |
| Three-to-five-minute demo | Complete: CPU ablation plus pre-generated formal report path |
| Smoke not presented as performance evidence | Complete |
| Large files remain on `/root/autodl-tmp` | Complete |

The core project therefore satisfies the plan's Definition of Done with an honest negative tuning
result: neither workload crosses the 15% goodput or 20% p99-TTFT target, Chat yields only a tested
capacity lower bound, RAG keeps default as best, and the CPU adaptive simulator records 0% goodput
gain plus TTFT regressions. Optional M6 prefix-caching and a version-pinned adaptive vLLM runtime
integration remain deferred P1 work; neither is misrepresented as completed evidence.
