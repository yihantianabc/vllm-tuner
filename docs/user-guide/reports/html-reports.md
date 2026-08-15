# Static reports

The complete experiment runner produces static HTML, Markdown, JSON, and plot artifacts under the
explicit experiment result root. Reports are views over retained raw evidence, not the sole source
of truth.

Expected sections include:

- manifest and environment identity;
- trial status counts and structured failures;
- default/random/TPE goodput and constraint outcomes;
- repeat and held-out tables;
- offered, achieved, and goodput capacity views;
- TTFT/TPOT/E2E distributions;
- queue, KV, preemption, GPU utilization, and VRAM timelines when available;
- scheduler calibration/held-out comparison;
- limitations and negative/no-benefit conditions.

Legacy study reports can be generated with:

```bash
vllm-tuner report --study-name STUDY --format html
```

For a SLOTune run, use the report paths printed by `vllm-tuner tune` and recorded in
`summary.json`. Do not infer missing raw values from a chart, and do not treat simulator plots as
runtime GPU measurements.
