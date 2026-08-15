# Validated configuration examples

Every YAML file in this directory uses the current SLO-goodput schema. Unknown upstream-era
fields such as weighted `objectives` and `batch_size` are intentionally absent and rejected by
the configuration model.

| File | Purpose | Evidence level |
|---|---|---|
| `default.yaml` | compact mixed-profile search | example protocol |
| `simple_tune.yaml` | short 3B development run | development only |
| `latency_optimized.yaml` | tighter chat SLOs | example protocol |
| `multi_gpu_tune.yaml` | legacy filename containing a valid one-GPU RAG example | example protocol |

Formal, repeat-three, held-out protocols live at:

- [`../../../config/formal_3b_chat.yaml`](../../../config/formal_3b_chat.yaml)
- [`../../../config/formal_3b_rag.yaml`](../../../config/formal_3b_rag.yaml)

Run an example with an explicit result root:

```bash
vllm-tuner tune \
  --config docs/user-guide/examples/simple_tune.yaml \
  --study-name development_chat_001 \
  --results-root /root/autodl-tmp/slotune-results
```

These files do not promise a throughput or latency improvement. Results become evidence only
after raw requests, telemetry, environment manifest, repeats, and held-out artifacts have been
validated.

## Migration from upstream examples

| Removed pattern | Current representation |
|---|---|
| `objectives: {throughput, latency, memory}` | `slo` plus `constraints`; maximize goodput |
| `search_space.batch_size` | remove it; use effective vLLM server parameters |
| list-valued TP/PP | scalar `tensor_parallel_size: 1`, `pipeline_parallel_size: 1` |
| `study.min_trials` | `study.trial_budget` per method |
| multi-GPU device lists | unsupported by the core single-GPU protocol |
| optional legacy baseline | equal-budget `default` method |

See [Configuration](../configuration.md) for field semantics and
[Formal experiments](../../FORMAL_EXPERIMENTS.md) for the evidence protocol.
