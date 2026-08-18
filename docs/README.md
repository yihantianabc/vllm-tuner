# SLOTune documentation

The active project line is **long-context v5**: Qwen2.5-7B-Instruct on one RTX 5090 with an
independently implemented KV Capacity Planner and artifact-backed analysis of upstream vLLM APC
and Chunked Prefill.

## Start here

- [Project README](../README.md): engineering contribution, headline results, ownership boundary,
  architecture, and quick audit commands
- [Long-context v5 result](results/longctx-v5.md): exact M1–M5 tables, ranges, negative results,
  integrity hashes, environment identity, and artifact paths
- [Reproduction guide](../REPRODUCTION.md): read-only seal audit, offline figure generation, and
  fresh-output M1/M3/M4/M5 commands; FP8 retry is intentionally excluded
- [Resume and interview material](CAREER_MATERIALS.md): final Chinese resume bullets, 30-second and
  2-minute explanations, deep-dive questions, and claim boundaries
- [Long-context v5 plan](SLOTUNE_PROJECT_PLAN.md): milestone design and acceptance criteria

## Current v5 implementation

- [KV Capacity Planner](../src/vllm_tuner/longctx/kv_capacity_planner.py)
- [M1 Planner validation](../src/vllm_tuner/longctx/m1_runner.py) and
  [capacity analysis](../src/vllm_tuner/longctx/m1_capacity_analysis.py)
- [M3 APC experiment](../src/vllm_tuner/longctx/m3_apc_runner.py)
- [M4 Chunked Prefill calibration](../src/vllm_tuner/longctx/m4_chunked_runner.py)
- [M5 Decode-tail validation](../src/vllm_tuner/longctx/m5_decode_tail_runner.py) and
  [engineering reanalysis](../src/vllm_tuner/longctx/m5_decode_tail_engineering.py)
- [M6 figure generator](../scripts/generate_longctx_v5_figures.py)

## Evidence rule

Only sealed GPU artifacts support performance claims. Checked-in YAML, source code, unit tests,
and plots are protocol or presentation evidence; they do not replace raw request records,
telemetry, cleanup, identity locks, and integrity seals.

Current claims must preserve all of these boundaries:

- the Capacity Planner and experiment/evidence pipeline are independent contributions;
- APC and Chunked Prefill are upstream vLLM features that were used and analyzed, not reimplemented;
- M2 FP8 is an incompatibility result with no capacity/quality benefit claim;
- M4 keeps its original `production-default` selection;
- M5's original strict-gate negative remains sealed, while a separate zero-GPU engineering
  artifact selects `decode-tail-1024` under material KV guardrails;
- target/held-out paired medians and ranges are reported instead of a best single run.

## Legacy 3B/TPE line

The following documents describe the earlier 3B constrained-search and CPU Scheduler line. They
remain useful historical evidence but are not inputs to long-context v5 results:

- [Legacy Qwen2.5-3B result](results/qwen25-3b-34a25a2.md)
- [Legacy formal protocol](FORMAL_EXPERIMENTS.md)
- [Engineering development log](DEVELOPMENT_LOG.md)
- [Legacy plan audit](PLAN_AUDIT.md)
- [Frozen 0.6B bring-up baseline](BASELINE_20260815.md)
- [General methodology](METHODOLOGY.md)

## General user and developer guides

- [Configuration](user-guide/configuration.md)
- [CLI commands](user-guide/cli-commands.md)
- [Metrics explained](user-guide/reports/metrics-explained.md)
- [Architecture](architecture/index.md)
- [Developer guide](developer-guide/index.md)
- [Testing](developer-guide/testing.md)
- [Troubleshooting](troubleshooting/common-issues.md)
