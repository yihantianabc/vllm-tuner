# API reference map

- `vllm_tuner.config.models`: `TuningConfig`, `SLOConfig`, `Constraints`,
  `SearchSpaceOverride`, `WorkloadConfig`, `TelemetryConfig`, and `StudySettings`.
- `vllm_tuner.experiment`: manifest, immutable artifact store, models, and experiment runner.
- `vllm_tuner.workloads`: deterministic profiles, generator, and `WorkloadTrace`.
- `vllm_tuner.benchmarks`: request/result models, metric reducer, SSE client, official adapter.
- `vllm_tuner.runtime`: managed vLLM process, failure classification, state machine, controller.
- `vllm_tuner.profiling`: Prometheus, NVML, time-series, and `TelemetrySession`.
- `vllm_tuner.tuning`: SLO-goodput objective, search space, equal-budget controller.
- `vllm_tuner.scheduling`: fixed/adaptive policies, admission, deterministic simulator, ablation.
- `vllm_tuner.reporting`: static plots and reports.

`WeightedObjectives`, legacy optimizer/baseline classes, and upstream reporting paths may remain for
compatibility but are not the SLOTune core protocol. New code should use SLO goodput and the modules
above.

See [Architecture](../architecture/index.md) and [Methodology](../METHODOLOGY.md).
