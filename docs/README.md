# SLOTune documentation

SLOTune documents the evidence chain from a fixed workload trace to a constrained search,
repeats, held-out validation, scheduler ablation, and a static report.

## Start here

- [Project README](../README.md): scope, fork attribution, quick start, and evidence status
- [Methodology](METHODOLOGY.md): metric definitions, lifecycle, optimizer, telemetry, and scheduler
- [Engineering development log](DEVELOPMENT_LOG.md): evidence-backed 2026-08-15—16 timeline,
  commands, debugging decisions, formal execution, audits, and limitations
- [Formal experiments](FORMAL_EXPERIMENTS.md): 3B chat/RAG protocol and reporting rules
- [Qwen2.5-3B formal result](results/qwen25-3b-34a25a2.md): artifact-backed Chat/RAG results,
  capacity plots, negative tuning outcome, and integrity audit
- [Frozen bring-up baseline](BASELINE_20260815.md): 0.6B smoke evidence, not a benchmark
- [Project-plan audit](PLAN_AUDIT.md): completed core M0–M5/Definition-of-Done audit and deferred M6
- [Reproduction guide](../REPRODUCTION.md): data-disk setup, smoke/formal commands, and result register
- [Implementation plan](SLOTUNE_PROJECT_PLAN.md): milestone-level design and acceptance criteria

## User guides

- [Configuration](user-guide/configuration.md)
- [CLI commands](user-guide/cli-commands.md)
- [Metrics explained](user-guide/reports/metrics-explained.md)
- [Experiment comparisons](user-guide/reports/baseline-comparison.md)
- [Custom workload traces](user-guide/examples/custom-workload.md)
- [Validated examples](user-guide/examples/README.md)

## Internals

- [Architecture](architecture/index.md)
- [Search controller](architecture/tuning-engine.md)
- [Developer guide](developer-guide/index.md)
- [Testing](developer-guide/testing.md)
- [Troubleshooting](troubleshooting/common-issues.md)

## Evidence rule

Configuration files and deterministic simulator output are protocols or mechanism evidence.
Only audited artifacts produced by a completed GPU experiment may support performance claims.
Every claim must identify the model, model/tokenizer revision, GPU, vLLM/runtime versions, trace
checksum, SLOs, seed, repeats, and held-out result. Missing telemetry and negative outcomes remain
visible.

The recorded Qwen2.5-3B Chat and RAG artifacts use clean measurement commit `34a25a2`. Neither
workload met the preregistered 15% goodput or 20% p99-TTFT improvement target; the scheduler
result is a CPU simulation with 0% goodput gain and must not be described as a GPU runtime gain.
