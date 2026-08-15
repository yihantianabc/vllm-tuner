# Comparisons and negative results

## Runtime search baselines

SLOTune compares three methods with the same measured evaluation budget:

1. `default`: vLLM defaults under the same trace and protocol;
2. `random`: seeded suggestions in the effective search space;
3. `tpe`: seeded constrained TPE suggestions.

The old optional one-off baseline is deprecated. A fair comparison keeps model, server version,
trace, seed, warmup, backend, SLO, telemetry, measurement acceptance, and trial budget fixed.

Reports distinguish search observations, three-repeat candidate validation, and held-out results.
The best search point alone is not a formal conclusion.

## Scheduler baselines

The deterministic scheduler ablation uses fixed 512, 1024, 2048, 4096, and 8192 token budgets by
default, with the same admitted-sequence cap and trace, plus the adaptive policy. At least two
fixed budgets are required. Both calibration and held-out comparisons include latency, goodput,
fairness, starvation, and preemption.

```bash
./scripts/run_demo.sh /root/autodl-tmp/slotune-demo/scheduler
```

## Negative/no-benefit conditions

An adaptive policy may legitimately fail to beat a fixed budget:

- light or homogeneous traffic leaves little backlog signal to exploit;
- hysteresis reacts too slowly to short bursts;
- aggressive KV protection lowers admitted concurrency;
- max-wait swaps improve fairness while increasing preemptions or TPOT;
- a large fixed budget already clears prefill without harming decode;
- a policy tuned on one load does not generalize to held-out timing or length distributions.

The JSON and Markdown artifacts record equality/regression for goodput, p99 TTFT, and p99 TPOT,
including the relevant fixed budget. Do not delete those rows or describe a simulator tie as a GPU
speedup.

## Reading a formal comparison

Check, in order:

1. both runs have matching manifests except for the intended policy/parameters;
2. raw request counts and token totals are credible;
3. no failure or missing telemetry was converted to zero;
4. default/random/TPE used equal measured budgets;
5. repeats show stable direction rather than one favorable sample;
6. held-out goodput and tail latency do not materially regress;
7. queue/KV/preemption/GPU timelines support the proposed explanation.

No measured formal comparison is currently claimed in this document.
