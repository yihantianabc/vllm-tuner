# SLOTune documentation

SLOTune documents the evidence chain from a fixed workload trace to a constrained search,
repeats, held-out validation, scheduler ablation, and a static report.

## Start here

- [Project README](../README.md): scope, fork attribution, quick start, and evidence status
- [Methodology](METHODOLOGY.md): metric definitions, lifecycle, optimizer, telemetry, and scheduler
- [Formal experiments](FORMAL_EXPERIMENTS.md): 3B chat/RAG protocol and reporting rules
- [Frozen bring-up baseline](BASELINE_20260815.md): 0.6B smoke evidence, not a benchmark
- [Project-plan audit](PLAN_AUDIT.md): M0–M6 implementation and experimental-evidence gaps
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
Only immutable artifacts produced by a completed GPU experiment may support performance claims.
Every claim must identify the model, model/tokenizer revision, GPU, vLLM/runtime versions, trace
checksum, SLOs, seed, repeats, and held-out result. Missing telemetry and negative outcomes remain
visible.
