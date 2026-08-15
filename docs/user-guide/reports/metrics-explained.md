# Metrics explained

## Request timeline

```text
scheduled ── sent ── first non-empty token ── token ... token ── finished
               │              │                                  │
               └──── TTFT ────┘                                  │
               └──────────────────── E2E ─────────────────────────┘
                              adjacent arrivals = ITL
```

- **TTFT:** `first_token_at - sent_at`.
- **ITL:** adjacent differences in a validated token timestamp array. SLOTune requests pinned-vLLM delta token IDs for formal SSE trials and requires their count to match authoritative output tokens; otherwise ITL is unavailable and only separately named SSE inter-event latency is reported.
- **TPOT:** `(finished_at - first_token_at) / (output_tokens - 1)`. A successful one-token
  response has TPOT zero because it has no post-first-token interval.
- **E2E:** `finished_at - sent_at`; it is stored directly, never approximated as TTFT plus an
  average.
- **Queue time:** scheduler/backend queue delay as represented in that artifact. Client scheduling
  delay and engine queue time must not be silently merged.

Times are measured with a monotonic clock. Warmups are excluded. Percentiles are interpolated
from raw successful-request samples and the report states sample counts.

## Load and throughput

- **Offered requests/s:** scheduled requests divided by measured seconds.
- **Achieved requests/s:** successful completions divided by measured seconds.
- **Output tokens/s:** validated output tokens divided by measured seconds.
- **SLO goodput:** successful requests meeting every enabled TTFT/TPOT/E2E threshold divided by
  measured seconds.

Offered load can exceed achieved throughput; achieved throughput can exceed goodput. Reporting
only throughput hides overload and SLO misses.

## Constraint metrics

A candidate is selectable only when every configured hard constraint passes: error rate, no OOM,
server alive, peak VRAM/memory utilization, and relevant p99 latency SLOs. FAILED, PRUNED, and
INFEASIBLE states remain separate from numeric objectives.

## Engine telemetry

The engine namespace can contain running/waiting sequences, KV-cache usage, prompt/generation
token counter deltas, preemption deltas, prefix-cache query/hit deltas, and vLLM queue/latency
metrics. Counters are process-window deltas; gauges and histograms retain their proper semantics.

## GPU telemetry

NVML time series include memory used/total, GPU utilization, power, temperature, and clocks.
Summaries use peak, mean, and p95 as appropriate. Energy per output token is present only when
both power integration and validated output-token counts are available.

## Scheduler simulator metrics

- p50/p99 queue, TTFT, and TPOT;
- request goodput and throughput;
- Jain fairness over normalized service outcomes;
- starvation count/rate and maximum service gap;
- preemption count;
- scheduled prefill/decode tokens and per-step budget utilization.

Simulator units and mechanism assumptions are not GPU measurements. Its report labels calibration
and held-out data and preserves equality/regression against fixed budgets.

## Evidence discipline

Do not insert illustrative performance values into project results. A numeric claim requires a
linked immutable artifact with model/GPU/software fingerprints, trace checksum, SLO, repetitions,
and held-out validation.
