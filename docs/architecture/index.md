# SLOTune architecture

```text
ExperimentSpec
  model / hardware / workload / SLO / search space / seed
                              │
                              ▼
                       Experiment Runner
        fixed search trace + separately generated held-out trace
                              │
                              ▼
                       Trial Controller
 START → READY → WARMUP → MEASURE → COLLECT → STOP → terminal status
    │                    │                       │
    ▼                    ▼                       ▼
ManagedVLLMServer   benchmark adapter      TelemetrySession
process group       official or SSE        Prometheus + NVML
    └────────────────────┬───────────────────────┘
                         ▼
                 objective + constraints
                         ▼
        equal-budget default / random / constrained TPE
                         ▼
             repeats + held-out + artifact/report

fixed traces ──► deterministic token-budget simulator ──► ablation
```

## Packages

- `experiment`: immutable manifest, artifact store, high-level orchestration;
- `workloads`: deterministic profiles, arrival generation, JSONL trace/checksum;
- `benchmarks`: request/result models, official adapter, SSE parser/client, metric reducer;
- `runtime`: managed server, state machine, failure classification, trial controller;
- `profiling`: Prometheus parsing, NVML session, aligned time series, telemetry lifecycle;
- `tuning`: SLO-goodput objective, effective search space, equal-budget controller;
- `scheduling`: fixed/adaptive budget policies, fair admission, deterministic simulator;
- `reporting`: plots and static HTML/Markdown/JSON reports;
- `cli`: user-facing experiment entry point.

Legacy upstream packages remain for compatibility, but the components above form the SLOTune
evidence path.

## Ownership boundaries

- Server parameters belong to the tuner.
- Arrival rate, burstiness, concurrency, and token distribution belong to the workload.
- Model/revision, trace, seed, backend, SLO, telemetry interval, and environment are experiment
  constants.
- Runtime scheduler behavior and deterministic simulator behavior are separate evidence types.

This separation prevents the tuner from changing the workload it is being scored against or from
presenting a simulator result as a vLLM measurement.

See [Methodology](../METHODOLOGY.md) for correctness and artifact acceptance.
