# SLOTune methodology

## Experimental unit

A result belongs to one immutable combination of:

```text
model + model revision + tokenizer + GPU + software environment
+ fixed request trace + SLO + search space + seed
```

Changing any member creates a different experiment. Search and candidate comparison reuse the
same persisted trace; a separately seeded or explicitly supplied trace is reserved for held-out
validation.

## Measurement correctness

SLOTune stores request-level timestamps and reduces them only after the measurement window:

| Metric | Definition |
|---|---|
| queue | scheduled/send delay before the request first receives service, as exposed by the relevant backend |
| TTFT | request sent to first non-empty streamed token |
| ITL | interval between adjacent output-token arrivals; available only when delta token IDs or official benchmark ITLs match the output-token count |
| TPOT | `(finished_at - first_token_at) / (output_tokens - 1)`; zero for one output token |
| E2E | request sent to completion |
| offered load | scheduled requests divided by the measurement duration |
| achieved throughput | successful completions divided by the duration |
| SLO goodput | successful requests meeting every configured TTFT/TPOT/E2E threshold divided by the duration |

The SSE parser buffers across HTTP chunk boundaries and can emit multiple events from one chunk. Non-empty SSE event arrivals are preserved separately as inter-event latency and never substituted for token ITL.
`[DONE]`, empty text, HTTP errors, timeouts, and asynchronous task exceptions have explicit
paths. Warmup results are marked and excluded. Input/output tokens are counted by a tokenizer,
server usage, or validated official result; zero totals are not accepted as credible evidence.
Percentiles use interpolation rather than a nearest-array-index shortcut.

The formal checked-in configs use the custom SSE client so every persisted arrival offset is
replayed exactly. The official `vllm bench serve` adapter is retained as a live reference backend
for cross-validation. A report must state which backend produced it and must not merge the two
arrival protocols as if they were identical.

## Trial lifecycle and failures

```text
CREATED → STARTING → READY → WARMING_UP → MEASURING
        → COLLECTING → STOPPING → COMPLETE

Any valid phase may terminate as FAILED, INFEASIBLE, or PRUNED.
```

The controller checks both process state and health, records the command/environment/log, manages
the process group, takes a final telemetry sample before stopping the server, and confirms
cleanup. Structured failures distinguish OOM, port conflict, invalid argument, model-load or
startup failure, request error/timeout, telemetry error, and unexpected server exit. A generic
runtime exception is not automatically called OOM.

- **FAILED:** no valid measured outcome; excluded from the objective and best selection.
- **INFEASIBLE:** measurement completed but violated one or more hard constraints; retained for
  constrained-learning evidence and excluded from best selection.
- **PRUNED:** deliberately stopped by the search protocol; never converted into a fake score.

## Objective and constraints

The sole optimization objective is request SLO goodput. There is no 60/30/10 weighted objective.
The core constraints can require:

- error rate at or below its configured threshold;
- no OOM and a live server at the end of measurement;
- peak VRAM and memory utilization below configured safety limits;
- p99 TTFT, TPOT, and E2E within configured SLOs when those thresholds are enabled.

Throughput, latency, and memory remain report dimensions, not arbitrary scalar weights.

## Equal-budget search

The effective server search space is:

- `gpu_memory_utilization`;
- `max_num_seqs`;
- `max_num_batched_tokens`.

`tensor_parallel_size` and `pipeline_parallel_size` are fixed to one. `batch_size` is rejected
because it is not an effective vLLM serving parameter. Fixed `vllm_args` may not duplicate trial
parameters.

The vLLM default, seeded random, and constrained TPE methods receive the same configured count of
measured COMPLETE/INFEASIBLE evaluations. Attempt failures are retained separately. Candidate
selection is direction-aware and excludes non-selectable outcomes. Formal configurations repeat
top candidates three times and then repeat them on the held-out trace.

## Cross-layer telemetry

Telemetry begins immediately before measured traffic and stops before vLLM shutdown. Frames share
a monotonic timestamp and preserve three namespaces:

- `client`: per-request latency/status/tokens and aggregate offered/achieved/goodput;
- `engine`: running/waiting sequences, KV usage, preemption and token counter deltas,
  prefix-cache counters, and available queue/latency histograms from `/metrics`;
- `gpu`: NVML memory, utilization, power, temperature, and clocks.

Prometheus counters are window deltas, not process-lifetime totals. NVML summaries use real time
series for peak, mean, and p95. Optional energy per output token is reported only when power and
output-token data are available. A missing source is marked unavailable with an error; it is not
zero-filled.

## Adaptive token-budget simulator

The deterministic simulator models prompt prefill separately from one-token-per-step decode. Each
step has an auditable total budget and never schedules more tokens than that total. Fixed baselines
default to 512, 1024, 2048, 4096, and 8192.

The adaptive policy observes decode/prefill backlog, oldest prefill age, KV pressure, recent p99
TTFT/TPOT, preemptions, and an available-budget cap. It applies lower/upper bounds, hysteresis,
minimum prefill progress, aging, max-wait admission swaps, and a pressure-aware admitted-sequence
limit. Stable ordering and policy reset make identical trace/seed runs byte-reproducible.

Simulator output includes raw per-request and per-step data plus p50/p99 queue, TTFT and TPOT,
goodput, Jain fairness, starvation, and preemption. The ablation compares calibration and held-out
traces and emits machine-readable negative/no-benefit conditions. This validates mechanisms; it
does not by itself prove a vLLM GPU speedup.

## Artifact acceptance

A formal trial is valid only when required artifacts are complete. The experiment root includes:

```text
manifest.json
experiment.yaml
trace.jsonl + trace.sha256
holdout-trace.sha256
environment/{git-state,python-packages,nvidia-smi,collect-env}.txt
trials/<trial-id>/{server-command,params,status,request-results,
                   benchmark-raw,prometheus,nvml,server.log,summary}.*
aggregate/{trials,repeated-results,holdout-results}.parquet
aggregate/scheduler-ablation.json
report/{report.html,report.md,report.json,...}
summary.json
```

Formal reporting uses repeated results and held-out evidence. It reports medians and ranges or
confidence intervals when available, retains failures, and limits conclusions to the recorded
environment.
