# Reproducing long-context v5

This guide separates three actions that must not be confused:

1. **read-only audit** of the existing sealed artifacts;
2. **offline regeneration** of figures or derived analysis from those artifacts;
3. **fresh GPU reproduction** into new artifact directories.

Never point a fresh run or derived output at a reference root. M2 FP8 is incompatible on the
frozen stack and has no retry command in this guide.

## 1. Frozen environment

| Item | Required identity |
|---|---|
| Repository | clean commit; record `git rev-parse HEAD` and `git status --short` |
| Model | `/root/autodl-tmp/models/Qwen2.5-7B-Instruct` |
| Model revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Runtime | vLLM 0.16.0 upstream commit `89a77b1`; no Scheduler patch |
| GPU | NVIDIA GeForce RTX 5090, one GPU, TP=PP=1 |
| Artifact parent | `/root/autodl-tmp/longctx-v5-artifacts` |

The exact model and runtime hashes live in:

```text
experiments/long_context/v5/qwen25-7b-instruct.model.lock.yaml
experiments/long_context/v5/upstream-runtime.lock.yaml
```

Set up the data-disk environment once:

```bash
cd /root/autodl-tmp/vllm-tuner
./scripts/setup_data_disk_reproduction.sh
source scripts/data_disk_reproduction_env.sh
source .venv/bin/activate
```

Record identity before any evidence-producing command:

```bash
git rev-parse HEAD
git status --short
python --version
python -m pip freeze > /root/autodl-tmp/longctx-v5-reproduction-pip-freeze.txt
nvidia-smi
```

## 2. Read-only audit of the reference evidence

The status commands below do not load a model or start vLLM:

```bash
./scripts/run_longctx_m2_fp8.sh --status \
  --experiment-id longctx-v5-m2-fp8-smoke-004
./scripts/run_longctx_m3_apc.sh --status \
  --experiment-id longctx-v5-m3-apc-formal-001
./scripts/run_longctx_m4_chunked.sh --status \
  --experiment-id longctx-v5-m4-chunked-formal-001
./scripts/run_longctx_m5_decode_tail.sh --status \
  --experiment-id longctx-v5-m5-decode-tail-formal-001
```

Validate the exact sealed file sets and hashes for the formal/derived roots:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path

from vllm_tuner.longctx.m1_capacity_integrity import validate_m1_capacity_artifacts
from vllm_tuner.longctx.m2_fp8_integrity import validate_m2_fp8_artifacts
from vllm_tuner.longctx.m3_apc_integrity import validate_m3_apc_artifacts
from vllm_tuner.longctx.m4_chunked_integrity import validate_m4_chunked_artifacts
from vllm_tuner.longctx.m5_decode_tail_integrity import validate_m5_decode_tail_artifacts

base = Path("/root/autodl-tmp/longctx-v5-artifacts")
validators = {
    "longctx-v5-m1-capacity-formal-001": validate_m1_capacity_artifacts,
    "longctx-v5-m1-capacity-formal-001-boundaries-v2": validate_m1_capacity_artifacts,
    "longctx-v5-m2-fp8-smoke-004": validate_m2_fp8_artifacts,
    "longctx-v5-m3-apc-formal-001": validate_m3_apc_artifacts,
    "longctx-v5-m4-chunked-formal-001": validate_m4_chunked_artifacts,
    "longctx-v5-m5-decode-tail-formal-001": validate_m5_decode_tail_artifacts,
    "longctx-v5-m5-decode-tail-engineering-001": validate_m5_decode_tail_artifacts,
}
for experiment_id, validate in validators.items():
    seal = validate(base / experiment_id)
    print(experiment_id, seal.get("schema"), "PASS")
PY
```

The M1 Planner initialization root uses its original v2 `m1-integrity.json` and is also bound by
the formal capacity manifest. Its reference hash is recorded in the
[result snapshot](docs/results/longctx-v5.md).

## 3. Regenerate checked-in figures without a GPU

This command reads only three sealed `summary.json` files and writes the public PNGs:

```bash
.venv/bin/python scripts/generate_longctx_v5_figures.py \
  --planner-artifact /root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m1-planner-init-002 \
  --apc-artifact /root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m3-apc-formal-001 \
  --m5-artifact /root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m5-decode-tail-engineering-001 \
  --output-dir docs/results
```

Expected outputs:

```text
docs/results/longctx-v5-capacity-planner.png
docs/results/longctx-v5-apc.png
docs/results/longctx-v5-decode-tail.png
```

To verify reproducibility without changing the checked-in files, render to a temporary directory
and compare:

```bash
M6_FIGURES_TMP="$(mktemp -d)"
.venv/bin/python scripts/generate_longctx_v5_figures.py --output-dir "${M6_FIGURES_TMP}"
cmp docs/results/longctx-v5-capacity-planner.png \
  "${M6_FIGURES_TMP}/longctx-v5-capacity-planner.png"
cmp docs/results/longctx-v5-apc.png "${M6_FIGURES_TMP}/longctx-v5-apc.png"
cmp docs/results/longctx-v5-decode-tail.png \
  "${M6_FIGURES_TMP}/longctx-v5-decode-tail.png"
```

## 4. Fresh M1 reproduction

These are GPU commands. Use new IDs exactly as shown or replace them with other previously absent
names.

### Planner calibration and unseen validation

```bash
./scripts/run_longctx_m1_init.sh \
  --config experiments/long_context/v5/m1-initialization.yaml \
  --experiment-id longctx-v5-m1-planner-init-repro-001
```

Acceptance requires the in-profile held-out point and both context-extrapolation points to remain
within the configured 10% block/cached-token/concurrency error target.

### Formal 8K/16K/32K service-capacity sweep

```bash
./scripts/run_longctx_m1_capacity.sh \
  --config experiments/long_context/v5/m1-capacity-formal.yaml \
  --experiment-id longctx-v5-m1-capacity-formal-repro-001
```

Derive the v2 service/saturation boundaries from the newly sealed root without running the GPU:

```bash
.venv/bin/vllm-tuner longctx-m1-capacity-boundaries \
  --artifact-root /root/autodl-tmp/longctx-v5-artifacts \
  --source-experiment-id longctx-v5-m1-capacity-formal-repro-001 \
  --experiment-id longctx-v5-m1-capacity-formal-repro-001-boundaries-v2
```

The v2 command must report `GPU runs executed: 0`; it separates SLO service and joint saturation
boundaries without modifying numeric thresholds or the source root.

## 5. M2 is inspect-only on this stack

The final smoke proved that the selected `fp8_e5m2 + TRITON_ATTN` path fails before engine
readiness at vLLM's attention dtype assertion. Do not launch `m2-fp8-formal.yaml`, change backend,
or rerun FP8 under the frozen v5 identity. Inspect the retained failure instead:

```bash
sed -n '1,220p' \
  /root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m2-fp8-smoke-004/report/m2-fp8.md
sed -n '80,115p' \
  /root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m2-fp8-smoke-004/trials/\
fp8-fp8-e5m2-triton-context-8k-repeat-0/server.log
```

A future FP8 investigation requires a new project/runtime lock and a new plan; it cannot be mixed
with the existing v5 claims.

## 6. Fresh M3 APC reproduction

```bash
./scripts/run_longctx_m3_apc.sh \
  --config experiments/long_context/v5/m3-apc-formal.yaml \
  --experiment-id longctx-v5-m3-apc-formal-repro-001
```

The formal configuration is fixed at 18 core runs plus two 4K-prefix pool-boundary runs. It binds
the real RAG corpus, APC off/on, 2K/4K prefixes, 0/50/100% reuse, cold/warm protocol, exact request
hit tokens, and the M1/M2 evidence. Do not turn this into a repeated-string microbenchmark.

Expected acceptance includes:

- 20/20 jobs complete with zero unsafe cleanup;
- APC-off and zero-reuse hits equal zero;
- shared-reuse hits are observed;
- warm TTFT improves and Goodput is not lower in all 50%/100% cells;
- the smaller prefix pool retains a higher hit ratio and the larger pool exposes a miss.

## 7. Fresh M4 Chunked Prefill reproduction

```bash
./scripts/run_longctx_m4_chunked.sh \
  --config experiments/long_context/v5/m4-chunked-formal.yaml \
  --experiment-id longctx-v5-m4-chunked-formal-repro-001
```

This is exactly 18 runs: production default, native threshold 1024, and native threshold 512 at
4K/8K, each with three balanced repeats. The expected protocol result is still a
`production-default` selection under the original zero-tolerance Goodput-direction rule. A fresh
run must not relabel the calibration as a deployment win merely because the ITL direction is
positive.

## 8. Fresh M5 target/held-out reproduction

```bash
./scripts/run_longctx_m5_decode_tail.sh \
  --config experiments/long_context/v5/m5-decode-tail-formal.yaml \
  --experiment-id longctx-v5-m5-decode-tail-formal-repro-001
```

The fixed matrix contains only `production-default` and `decode-tail-1024`, with three target and
three held-out pairs (12 runs total). It must not add threshold 512, APC-off, FP8, a custom
Scheduler, or post-held-out retuning.

The formal artifact preserves the original phase-sensitive KV gate. Create a separate engineering
view from the newly sealed 12-run root:

```bash
.venv/bin/python -m vllm_tuner.longctx.m5_decode_tail_engineering \
  --source /root/autodl-tmp/longctx-v5-artifacts/\
longctx-v5-m5-decode-tail-formal-repro-001 \
  --output /root/autodl-tmp/longctx-v5-artifacts/\
longctx-v5-m5-decode-tail-engineering-repro-001 \
  --repository /root/autodl-tmp/vllm-tuner
```

The derived command requires a clean committed repository, validates the source integrity seal,
executes zero GPU runs, refuses an existing output path, and never mutates the source.

## 9. Acceptance and reporting checklist

- Confirm model/runtime locks before every GPU matrix.
- Use only fresh artifact IDs; never overwrite a reference root.
- Verify balanced execution order and same-trace pairing within every cohort.
- Keep warmup out of the measured reducer.
- Report every repeat, paired median, and range; do not select the best run.
- Keep target and held-out separate, and do not retune after held-out.
- Inspect raw request records, server logs, Prometheus/NVML, cleanup, and integrity—not only the
  Markdown report.
- Keep M2 incompatibility, M4 production-default selection, and the original M5 formal negative.
- State that APC/Chunked Prefill are upstream native features and that the Planner/evidence system
  are the independent contributions.
- Restrict `decode-tail-1024` claims to the measured Long Prefill interference workload and locked
  environment.

## Reference artifact roots

```text
/root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m1-planner-init-002
/root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m1-capacity-formal-001
/root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m1-capacity-formal-001-boundaries-v2
/root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m2-fp8-smoke-004
/root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m3-apc-formal-001
/root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m4-chunked-formal-001
/root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m5-decode-tail-formal-001
/root/autodl-tmp/longctx-v5-artifacts/longctx-v5-m5-decode-tail-engineering-001
```

For exact values and integrity hashes, see the
[long-context v5 result snapshot](docs/results/longctx-v5.md). The legacy 3B search/TPE evidence
remains available in [its original snapshot](docs/results/qwen25-3b-34a25a2.md) and must not be
merged with v5.
