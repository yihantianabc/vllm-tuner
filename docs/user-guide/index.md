# SLOTune user guide

SLOTune finds the highest **SLO goodput** among effective vLLM single-GPU settings while retaining
latency, memory, queue, KV-cache, preemption, and failure evidence.

## First run

For a CPU-only deterministic scheduling demo:

```bash
./scripts/run_demo.sh /root/autodl-tmp/slotune-demo/scheduler
```

For the local Qwen3-0.6B GPU correctness smoke:

```bash
./scripts/run_data_disk_reproduction.sh slotune_smoke_001
```

The second command is not a benchmark. Formal templates use the 3B model:

```bash
./scripts/run_reproduction_command.sh tune \
  --config config/formal_3b_chat.yaml \
  --study-name qwen25_3b_chat_001 \
  --results-root /root/autodl-tmp/slotune-results
```

## Core concepts

- A persisted request trace and checksum are reused by default/random/TPE.
- SLO goodput is the only objective; failures and resource/SLO violations are constraints.
- Every search method receives the same measured evaluation budget.
- Warmup data are excluded; request results and telemetry stay raw and namespaced.
- Top candidates are repeated and rerun on a held-out trace.
- Scheduler simulations compare fixed token budgets with adaptive behavior and retain regressions.

## Guides

- [Configuration](configuration.md)
- [CLI](cli-commands.md)
- [Metrics](reports/metrics-explained.md)
- [Comparisons and negative results](reports/baseline-comparison.md)
- [Custom fixed traces](examples/custom-workload.md)
- [Formal protocol](../FORMAL_EXPERIMENTS.md)
- [Troubleshooting](../troubleshooting/common-issues.md)

The core is intentionally one GPU. Multi-GPU, TP, and PP search examples from upstream do not
apply to SLOTune.
