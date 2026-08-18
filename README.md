# SLOTune: artifact-backed vLLM long-context optimization

SLOTune is a single-GPU inference-engineering project for **Qwen2.5-7B-Instruct on an
RTX 5090 with upstream vLLM 0.16.0**. The current v5 line combines an independently implemented
KV Capacity Planner with controlled experiments around vLLM's native Automatic Prefix Caching
(APC) and Chunked Prefill.

The final workload-specific profile, `decode-tail-1024`, reduced interference-window Decode ITL
p99 by a paired median **42.8% on the target trace and 42.9% on held-out**, while paired median
Goodput changed by only **-0.027%/-0.014%**. The trade-off was **+5.7%/+6.5%** Long Prefill TTFT
p99 and **+0.50%/+0.41%** Decode TPOT p99. All 12 runs completed without OOM, timeout, or
preemption.

![decode-tail-1024 result](docs/results/longctx-v5-decode-tail.png)

The claim is deliberately narrow: this is a reproducible deployment result for the measured
4K/8K Long Prefill interference workload, model, GPU, and runtime lock—not a general claim that a
1024-token threshold is optimal everywhere.

## What is original work

| Area | Ownership and engineering contribution |
|---|---|
| KV Capacity Planner | **Implemented in this repository.** Models GQA KV geometry, dtype and block rounding, vLLM's null block, calibrated non-KV memory, safety reserves, context distributions, cached tokens, and safe concurrency. |
| Experiment/evidence system | **Implemented in this repository.** Frozen traces, balanced paired repeats, target/held-out cohorts, request-level metrics, Prometheus/NVML collection, cleanup, manifests, checksums, and sealed positive/negative artifacts. |
| APC analysis | Uses **upstream vLLM APC**; the contribution is the real RAG-prefix workload, cold/warm and 0/50/100% reuse matrix, exact hit accounting, and prefix-pool boundary analysis. |
| Chunked Prefill analysis | Uses **upstream vLLM Chunked Prefill**; the contribution is the Decode/Prefill interference protocol, candidate calibration, non-inferiority rules, held-out validation, and deployment decision. |
| FP8 KV investigation | Uses vLLM's native FP8 KV path. The pinned stack failed compatibility smoke, so the failure is retained and no capacity or quality benefit is claimed. |

This project does **not** claim to have implemented APC, Chunked Prefill, PagedAttention, a custom
Scheduler, or a CUDA/Triton kernel. APC and Chunked Prefill were already enabled in the measured
vLLM 0.16.0 production default; `decode-tail-1024` changes only the native
`long_prefill_token_threshold` for the target workload.

## Results at a glance

| Milestone | Evidence | Result |
|---|---:|---|
| M1 Capacity Planner | 6 calibration probes + 3 unseen validation profiles; formal 8K/16K/32K capacity sweep | Planner PASS. Maximum absolute observed KV-block/concurrency error was **0.0446%** versus a 10% target; safe BF16 KV capacity was 192,880 usable tokens, corresponding to 23/11/5 complete 8K/16K/32K contexts. |
| M2 FP8 KV | Compatibility smoke | Negative result. `fp8_e5m2` with `TRITON_ATTN` failed vLLM engine initialization at the attention dtype assertion. The formal matrix was not started and must not be retried on this frozen stack. |
| M3 APC | **20/20** runs: 18 core + 2 pool-boundary | PASS. At 4K prefix, warm TTFT paired-median improvement was **11.0% at 50% reuse** and **55.1% at 100% reuse**; Goodput was not lower in all 18 paired core comparisons. |
| M4 Chunked Prefill calibration | **18/18** runs | PASS as calibration, but the preregistered zero-tolerance Goodput rule kept **production default**. This selection remains sealed and unchanged. |
| M5 Decode-tail deployment | **12/12** runs; 6 target/held-out profile pairs | Engineering PASS for `decode-tail-1024`: repeatable ITL p99 gain with Goodput, TTFT, TPOT, waiting, reliability, and material KV guardrails satisfied. |

The complete tables, environment identity, per-repeat ranges, limitations, and artifact lineage
are in the [long-context v5 result snapshot](docs/results/longctx-v5.md).

### Capacity Planner

The structural core is:

```text
KV bytes/token = layers × 2(K+V) × KV heads × head dimension × dtype bytes
```

The deployable estimate then adds page/block rounding, the reserved null block, calibrated
non-KV residency, a fixed operational reserve, a proportional KV reserve, and the requested
context distribution. Calibration used 75/80/85% GPU-memory-utilization probes; validation used
an unseen 90% point and unseen 8K/16K runtime profiles.

![KV Capacity Planner validation](docs/results/longctx-v5-capacity-planner.png)

The small error is evidence for the locked Qwen2.5-7B/RTX 5090/vLLM profile. The calibrated
non-KV residual is environment-specific, so the number must be recalibrated before changing the
model, GPU, runtime, or graph behavior.

### APC: benefit follows reusable prefix tokens

![APC warm TTFT result](docs/results/longctx-v5-apc.png)

The 0% reuse controls stay near zero, while 2K/4K shared prefixes show larger warm TTFT gains as
reuse increases. This supports the mechanism boundary: APC saves repeated **Prefill** work; it is
not presented as a direct acceleration of every Decode token. A 48-prompt 4K-prefix pool retained
all full hits, while the preregistered 72-prompt pool exposed the first full miss at probe 55.

### Why M4 stayed default but M5 selected 1024

M4 asked a strict calibration question: did a candidate improve Decode interference ITL while
never lowering Decode Goodput direction in at least two of three repeats at both 4K and 8K? Both
native candidates failed that zero-tolerance rule at 8K, despite tiny measured Goodput changes,
so the sealed M4 selection stayed `production-default`.

M5 asked a different, frozen deployment question on new target and held-out traces: can
`decode-tail-1024` deliver at least 25% paired-median ITL p99 improvement while staying within
explicit Goodput, TTFT, TPOT, waiting, reliability, and KV materiality limits? It passed. The
original M5 formal artifact had rejected two target repeats because one 200 ms telemetry sample
rose by three of 14,614 usable KV blocks. That negative artifact remains unchanged; a separate,
zero-GPU engineering reanalysis showed the delta was only 2.625 MiB, with identical paired-median
KV p95, lower waiting, and no reliability event.

## Evidence architecture

```text
model/runtime locks + frozen trace + preregistered acceptance rules
                              │
                              ▼
       managed vLLM lifecycle: READY → WARMUP → MEASURE → CLEANUP
              │                    │                    │
              ▼                    ▼                    ▼
       request-level TTFT/    Prometheus KV/queue/   continuous NVML
       TPOT/ITL/Goodput       preemption counters    GPU telemetry
              └────────────────────┬────────────────────┘
                                   ▼
              balanced paired repeats + target/held-out reducer
                                   ▼
                 manifest + raw records + integrity seal + report
```

Failed startup, incompatible backend, OOM, timeout, missing telemetry, and rejected candidates
remain visible. A smoke or a single best repeat is never promoted to a performance claim.

## Reproduce or audit

Set up the locked data-disk environment:

```bash
cd /root/autodl-tmp/vllm-tuner
./scripts/setup_data_disk_reproduction.sh
```

Regenerate the three checked-in figures without starting vLLM or touching sealed artifacts:

```bash
.venv/bin/python scripts/generate_longctx_v5_figures.py \
  --planner-artifact /root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m1-planner-init-002 \
  --apc-artifact /root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m3-apc-formal-001 \
  --m5-artifact /root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m5-decode-tail-engineering-001 \
  --output-dir docs/results
```

Read-only status checks are available for the sealed GPU roots:

```bash
./scripts/run_longctx_m3_apc.sh --status \
  --experiment-id longctx-v5-m3-apc-formal-001
./scripts/run_longctx_m4_chunked.sh --status \
  --experiment-id longctx-v5-m4-chunked-formal-001
./scripts/run_longctx_m5_decode_tail.sh --status \
  --experiment-id longctx-v5-m5-decode-tail-formal-001
```

The [reproduction guide](REPRODUCTION.md) provides fresh-output commands for M1, M3, M4, and M5
and the offline M5 engineering reanalysis. It intentionally provides no FP8 retry command for the
frozen incompatible stack.

## Sealed evidence roots

All performance claims above resolve to immutable data-disk artifacts:

| Evidence | Path |
|---|---|
| Planner validation | `/root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m1-planner-init-002` |
| Capacity sweep and v2 boundaries | `/root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m1-capacity-formal-001` and `-boundaries-v2` |
| FP8 incompatibility | `/root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m2-fp8-smoke-004` |
| APC formal | `/root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m3-apc-formal-001` |
| Chunked Prefill calibration | `/root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m4-chunked-formal-001` |
| M5 original 12-run evidence | `/root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m5-decode-tail-formal-001` |
| M5 engineering decision | `/root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m5-decode-tail-engineering-001` |

## Environment lock

- Model: `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB
- vLLM: 0.16.0, upstream commit `89a77b10846fd96273cce78d86d2556ea582d26e`
- Python 3.12.3, PyTorch 2.9.1+cu130, CUDA 13.0, driver 595.71.05
- Single GPU, TP=PP=1; no custom Scheduler or patched runtime in M1–M5 evidence

Exact model-file hashes and upstream source hashes are in
[`experiments/long_context/v5/`](experiments/long_context/v5/).

## Fork attribution and legacy line

SLOTune is based on and forked from
[`jranaraki/vllm-tuner`](https://github.com/jranaraki/vllm-tuner), originally authored by Javad
Anaraki and distributed under the MIT License. The upstream project supplied the initial Typer
CLI, Pydantic configuration, Optuna skeleton, vLLM launcher, basic GPU collection, baseline
runner, and HTML-report foundation.

The earlier 3B search/TPE and CPU Scheduler work remains preserved as **legacy evidence**, not as
the source of the v5 long-context claims. See the
[legacy Qwen2.5-3B result](docs/results/qwen25-3b-34a25a2.md) and
[development log](docs/DEVELOPMENT_LOG.md). Current project scope and acceptance rules are in the
[v5 project plan](docs/SLOTUNE_PROJECT_PLAN.md).

## Documentation

- [Long-context v5 result snapshot](docs/results/longctx-v5.md)
- [Reproduction guide](REPRODUCTION.md)
- [Resume and interview material (Chinese)](docs/CAREER_MATERIALS.md)
- [Documentation index](docs/README.md)
- [MIT License](LICENSE)
