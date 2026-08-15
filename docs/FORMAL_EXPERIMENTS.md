# Formal 3B experiment protocol

## Status

The repository contains validated formal **protocols**, not fabricated benchmark results.

| Item | Status | Permitted claim |
|---|---|---|
| Qwen3-0.6B reproduction | completed bring-up smoke | end-to-end wiring only |
| Qwen2.5-3B chat config | ready to run | configuration/schema only |
| Qwen2.5-3B RAG config | ready to run | configuration/schema only |
| adaptive scheduler | deterministic simulator tested | mechanism/fairness behavior only |
| runtime adaptive vLLM scheduler | not integrated | no GPU performance claim |

Do not add a throughput, latency, memory, energy, or percentage-improvement number here unless the
supporting immutable artifact is checked and linked.

## Hardware and software scope

The templates assume:

- one NVIDIA RTX 5090 with approximately 32 GB VRAM;
- local `/root/autodl-tmp/models/Qwen2.5-3B-Instruct`;
- pinned vLLM, PyTorch/CUDA, driver, tokenizer, and repository revision captured by the manifest;
- all model caches, temporary compilation data, and experiment artifacts on
  `/root/autodl-tmp`.

The output applies only to the recorded combination. The smoke model Qwen3-0.6B must not be used
as the headline benchmark model.

## Checked-in workloads

### Chat

[`../config/formal_3b_chat.yaml`](../config/formal_3b_chat.yaml) uses the seeded `chat` profile:
roughly 192–320 input tokens, a fixed 128-token measured output, request rate 8/s, and concurrency
capped at 32. It emphasizes decode concurrency and TPOT.

### RAG

[`../config/formal_3b_rag.yaml`](../config/formal_3b_rag.yaml) uses the seeded `rag` profile:
roughly 1792–2304 input tokens, a fixed 128-token measured output, 50% shared-prefix probability,
request rate 4/s, and burstiness 1.5. It emphasizes prefill, TTFT, KV pressure, and prefix behavior.

Both generate 500-request search traces. The held-out trace uses a different deterministic seed
and is not inspected by the optimizer. For externally supplied traces, pass both `--trace` and
`--holdout-trace`; each JSONL row follows the `WorkloadTrace` schema documented in
[`user-guide/examples/custom-workload.md`](user-guide/examples/custom-workload.md).

Both formal configs select `benchmark_backend: sse` to replay the frozen
`scheduled_offset_seconds` values exactly. The official bench adapter remains a live
reference/cross-validation backend and its independently generated arrival process must be labeled
as such.

## Equal-budget protocol

Both configurations use:

```yaml
study:
  trial_budget: 16
  methods: [default, random, tpe]
  repeat_count: 3
  top_candidates: 3
  holdout_enabled: true
  resume: false
```

`trial_budget` is per method, so default/random/TPE each contribute 16 measured outcomes. This is
not a total budget shared among methods. The default method intentionally measures the same vLLM
default server configuration under the same protocol; random and TPE use fixed seeds. Top
candidates are repeated three times, then repeated three times on held-out traffic.

## Run order

1. Verify the model/tokenizer path and available disk space.
2. Ensure no unrelated GPU workload is running.
3. Pin and record runtime versions.
4. Run a uniquely named smoke experiment; inspect cleanup and raw artifacts.
5. Run chat and RAG formal experiments in randomized order if comparing environmental drift.
6. Validate every trial artifact before optimizer submission.
7. Inspect failures, telemetry availability, request token totals, and trace checksums.
8. Report repeats and held-out results, not only the best search observation.

Commands:

```bash
vllm-tuner tune \
  --config config/formal_3b_chat.yaml \
  --study-name qwen25_3b_chat_001 \
  --results-root /root/autodl-tmp/slotune-results

vllm-tuner tune \
  --config config/formal_3b_rag.yaml \
  --study-name qwen25_3b_rag_001 \
  --results-root /root/autodl-tmp/slotune-results
```

For a fixed externally reviewed trace:

```bash
vllm-tuner tune \
  --config config/formal_3b_chat.yaml \
  --study-name qwen25_3b_chat_fixed_trace_001 \
  --trace /root/autodl-tmp/traces/chat-search.jsonl \
  --holdout-trace /root/autodl-tmp/traces/chat-holdout.jsonl \
  --results-root /root/autodl-tmp/slotune-results
```

## Capacity sweep

Both formal configs declare the complete matrix:

```yaml
workload:
  capacity_request_rates: [1, 2, 4, 8, 16, 32]
  capacity_repeats: 3
```

Each rate is repeated three times. Use the same trace family, SLO, environment, and equal search
budget. Do not compare trials that silently changed prompt/output distributions. The scalar
`request_rate` identifies the primary single-run profile; the capacity fields define the formal
sweep protocol.

At every point retain offered rate, achieved throughput, output-token throughput, SLO goodput,
p50/p95/p99 TTFT/TPOT/E2E, errors/timeouts, waiting queue, peak KV usage, preemptions, peak/mean/p95
VRAM, GPU utilization, and optional energy per output token.

## Scheduler ablation

The CPU-only reproducible demo is:

```bash
./scripts/run_demo.sh /root/autodl-tmp/slotune-demo/scheduler
```

It compares fixed 512/1024/2048/4096/8192 budgets with adaptive on deterministic calibration and
held-out traces. JSON and Markdown preserve negative/no-benefit conditions. These data explain
simulator behavior and must remain separate from measured vLLM results.

## Reporting checklist

- Identify model path/revision, tokenizer hash, GPU, vLLM/PyTorch/CUDA/driver, repository commit,
  trace checksum, seed, backend, SLO, search space, and trial budget.
- Distinguish offered, achieved, and goodput.
- Report raw sample counts, failures, repeats, and held-out results.
- State telemetry gaps and whether energy was unavailable.
- Include adaptive scheduler regressions and preemption/fairness costs.
- Avoid causal claims unsupported by aligned time series.
- Restrict conclusions to the recorded environment.

If no feasible candidate or no adaptive gain is found, that is the result. Preserve the failure
or negative-condition artifact and explain the workload/pressure regime rather than replacing it
with an optimistic estimate.
