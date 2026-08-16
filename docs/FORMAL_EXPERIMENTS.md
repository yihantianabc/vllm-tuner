# Formal 3B experiment protocol and recorded outcome

## Status

The repository contains the validated protocols and a checked-in snapshot of two completed
formal artifacts. Raw evidence remains on `/root/autodl-tmp`; all performance statements below
are restricted to those artifacts and the clean measurement revision.

| Item | Status | Permitted claim |
|---|---|---|
| Qwen3-0.6B reproduction | completed bring-up smoke | end-to-end wiring only |
| Qwen2.5-3B Chat | completed: 96 terminal trials | repeated/held-out/capacity result; no significant tuning gain |
| Qwen2.5-3B RAG | completed: 96 terminal trials | default remains best; capacity knee near nominal 16 req/s |
| adaptive scheduler | deterministic CPU simulator completed | 0% goodput gain and retained TTFT regressions; no GPU runtime claim |
| runtime adaptive vLLM scheduler | not integrated | no GPU performance claim |

The formal evidence snapshot is
[`results/qwen25-3b-34a25a2.md`](results/qwen25-3b-34a25a2.md). It links the external roots,
measurement commit, trace and model hashes, repeats, holdout, constraints, negative results, and
artifact audit. Do not add another throughput, latency, memory, energy, or improvement claim
without an equivalently checked artifact.

After a run, `vllm-tuner attest --study-name <id> --results-root <root>` creates or validates the
root integrity seal, phase/source lineage, experiment audit, compact scheduler reference, and
scheduler negative-result views. Rebuilding requires explicit `--reseal`, which validates the old
seal first and rejects corrupt evidence. The compact reference is the additive
`summary.compact-v1.json` sidecar; it does not replace the original root `summary.json`.

## Hardware and software scope

The templates assume:

- one NVIDIA RTX 5090 with approximately 32 GB VRAM;
- local `/root/autodl-tmp/models/Qwen2.5-3B-Instruct`;
- pinned vLLM, PyTorch/CUDA, driver, tokenizer, and repository revision captured by the manifest;
- all model caches, temporary compilation data, and experiment artifacts on
  `/root/autodl-tmp`.

The output applies only to the recorded combination. The smoke model Qwen3-0.6B must not be used
as the headline benchmark model.

The completed runs used one RTX 5090, vLLM 0.16.0, PyTorch 2.9.1+cu130, driver 595.71.05, and
clean source commit `34a25a2e10951bfab1c2a86b4c60aff5bef785df`. Later attestation or
documentation revisions are not measurement provenance.

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

In the recorded result, every Chat point through nominal 32 req/s remained feasible; its maximum
tested median goodput, 27.883 req/s, is therefore only a capacity lower bound. RAG plateaued near
11.8 achieved req/s, with an observed knee near nominal 16 req/s and two of three nominal-32
repeats INFEASIBLE on the 1,500 ms TTFT constraint. See the
[exact capacity tables and plots](results/qwen25-3b-34a25a2.md#capacity).

## Scheduler ablation

The CPU-only reproducible demo is:

```bash
./scripts/run_demo.sh /root/autodl-tmp/slotune-demo/scheduler
```

It compares fixed 512/1024/2048/4096/8192 budgets with adaptive on deterministic calibration and
held-out traces. JSON and Markdown preserve negative/no-benefit conditions. These data explain
simulator behavior and must remain separate from measured vLLM results.

The formal artifacts retained a negative result: adaptive goodput gain was 0% for both workloads
on calibration and held-out simulator traces. Relative to the best fixed budget, p99 TTFT was
worse by 25.36%/15.58% for Chat and 167.22%/332.79% for RAG (calibration/held-out). This does not
measure an adaptive vLLM runtime.

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

## Recorded formal result

Both workloads executed 48 equal-budget search trials, 15 candidate repeats, 15 held-out repeats,
and 18 default capacity trials. Each has 89 COMPLETE and seven constraint-INFEASIBLE outcomes,
zero FAILED/PRUNED outcomes, and 48,000/48,000 successful measured requests. The seven
INFEASIBLE trials in each workload are latency or memory constraint results—not request
failures.

The finite generated traces also make the target/empirical distinction material. Chat's YAML
target is 8.0 req/s while its search/holdout traces realize 8.029529/8.456300 req/s; RAG's target
is 4.0 req/s while its traces realize 4.254534/4.331800 req/s. Goodput slightly above a YAML
target remains below the corresponding empirical arrival rate.

Neither workload reaches the preregistered success threshold of 15% higher goodput or 20% lower
p99 TTFT at equal goodput. Chat's validated TPE candidate differs from repeated default by only
+0.0068% goodput and −3.57% p99 TTFT, and its held-out TTFT is 0.52% worse. RAG retains default
as best; TPE-5's repeat/holdout goodput deltas are −0.0013%/+0.0141%, with TTFT improvements of
7.54%/8.49%. See the [complete snapshot](results/qwen25-3b-34a25a2.md#tuning-outcome).

M6 prefix-caching/APC is explicitly deferred P1 work. The plan marks prefix caching optional; its
deferral does not invalidate the completed core M0–M5 path or the two-workload formal evidence.

## Implementation revisions

These are real commits in the local SLOTune branch; the links deliberately target this repository
rather than implying that unpublished commits exist in the upstream GitHub repository.

| Revision | Scope |
|---|---|
| `aa9d70a` | Trustworthy benchmark, telemetry, lifecycle, constrained tuning, simulator, artifacts, reports, and tests |
| `0d605c3` | Pinned reproducible data-disk GPU environment |
| `b8f2dc1` | Methodology and formal protocol publication |
| `34a25a2` | Disabled-holdout smoke compatibility; clean revision used for all final measurements |
