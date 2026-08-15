# Reproducing SLOTune

This guide separates three evidence tiers that must not be merged:

1. the **legacy bring-up artifact**, retained only as a historical boundary;
2. the **current Qwen3-0.6B smoke**, used to verify wiring and cleanup;
3. the **Qwen2.5-3B formal protocol**, used for benchmark evidence only after a complete,
   immutable artifact has been recorded.

No 3B performance result is currently recorded in this repository. Checked-in configuration,
unit tests, and a deterministic scheduler simulation do not substitute for a GPU measurement.

## Evidence boundary

| Tier | Model | Status | Permitted claim |
|---|---|---|---|
| Legacy bring-up | Qwen3-0.6B | Preserved as `reproduction_gpu_20260815_a` | The pre-refactor chain completed two requests |
| Current smoke | Qwen3-0.6B | Current-format `smoke-20260815-b` exists; each new run still needs its own artifact | Model load, SSE requests, telemetry, cleanup, and artifact wiring |
| Current 3B preflight | Qwen2.5-3B-Instruct | `qwen25-3b-preflight-20260815-a` completed a two-request default run and one repeat | 3B model and current pipeline wiring only |
| Formal chat | Qwen2.5-3B-Instruct | Protocol ready; result not yet recorded | None until the immutable artifact is linked |
| Formal RAG | Qwen2.5-3B-Instruct | Protocol ready; result not yet recorded | None until the immutable artifact is linked |
| Scheduler ablation | Synthetic or supplied deterministic trace | CPU mechanism experiment | Simulator budget, fairness, starvation, and downside behavior only |

The legacy artifact and its frozen environment are documented in
[`docs/BASELINE_20260815.md`](docs/BASELINE_20260815.md). Its study data live under
`/root/autodl-tmp/vllm-tuner-output/studies/reproduction_gpu_20260815_a`, with the historical
HTML report under `/root/autodl-tmp/vllm-tuner-output/reports/reproduction_gpu_20260815_a`.
Those paths use the upstream-era study/report layout. They are not inputs to current formal
comparisons and must not be relabeled as SLOTune benchmark evidence.

## Data-disk layout

Keep the repository, environment, model weights, caches, traces, and generated artifacts on the
data disk:

| Purpose | Path |
|---|---|
| Repository and virtual environment | `/root/autodl-tmp/vllm-tuner` and `.venv/` |
| Local formal model | `/root/autodl-tmp/models/Qwen2.5-3B-Instruct` |
| Local smoke model | `/root/autodl-tmp/models/Qwen3-0.6B` |
| Hugging Face cache | `/root/autodl-tmp/huggingface` |
| uv and pip caches | `/root/autodl-tmp/uv-cache`, `/root/autodl-tmp/pip-cache` |
| CUDA, Triton, Torch, and vLLM caches | `/root/autodl-tmp/{cuda-cache,triton,torchinductor,vllm-cache}` |
| Current smoke experiments | `/root/autodl-tmp/vllm-tuner-output/slotune-results/<study-name>` |
| Formal experiments | `/root/autodl-tmp/slotune-results/<study-name>` |
| Reviewed external traces | `/root/autodl-tmp/traces/*.jsonl` |
| CPU scheduler demo | `/root/autodl-tmp/slotune-demo/<run-name>` |

Do not put model weights, compiler caches, or generated experiment directories in Git.

## 1. Rebuild the environment

From the fixed repository path:

```bash
cd /root/autodl-tmp/vllm-tuner
./scripts/setup_data_disk_reproduction.sh
```

The setup script places package and compiler caches under `/root/autodl-tmp`, installs the locked
development environment plus the pinned reproduction requirements, and runs `pip check`.
The shared lock is constrained to `transformers<5` and `numpy<2.3`, matching vLLM 0.16;
the overlay additionally pins `idna==3.18` for its `httpx2` dependency.
Frozen sync runs in inexact mode so a repeat setup keeps the host-specific GPU overlay while
still enforcing every project package recorded in `uv.lock`.

Shell exports made by a setup subprocess cannot modify its caller. Therefore, run every GPU
command through the checked-in wrapper, which sources the same cache/TMP environment and directly
executes the installed CLI without an implicit `uv run` resync:

```bash
./scripts/run_reproduction_command.sh --help
```

For interactive inspection only, use `source scripts/data_disk_reproduction_env.sh` followed by
`source .venv/bin/activate`. The wrapper is the canonical path for evidence-producing commands.

Record the repository revision and dirty state before collecting evidence:

```bash
source scripts/data_disk_reproduction_env.sh
source .venv/bin/activate
git rev-parse HEAD
git status --short
python --version
python -m pip freeze > /root/autodl-tmp/slotune-environment-pip-freeze.txt
nvidia-smi
vllm-tuner --help
vllm-tuner tune --help
```

The experiment manifest captures the run identity, but an externally archived environment dump
is useful when the checkout is intentionally dirty during development.

## 2. Validate current configurations without a GPU run

```bash
python - <<'PY'
from pathlib import Path
from vllm_tuner.config.validation import load_yaml_config

for path in sorted(Path("config").glob("*.yaml")):
    config = load_yaml_config(path)
    print(path, config.model, config.workload.name, config.workload.benchmark_backend)
PY
```

Current executable examples reject the old weighted `objectives` block and the ineffective
`search_space.batch_size` field. The formal configs fix TP/PP to one, use `benchmark_backend:
sse`, declare capacity rates `1/2/4/8/16/32`, and repeat each capacity point three times.

## 3. Run the current GPU smoke

Confirm that `/root/autodl-tmp/models/Qwen3-0.6B` exists and that the GPU is otherwise idle, then
use a new, descriptive study name:

```bash
cd /root/autodl-tmp/vllm-tuner
./scripts/run_data_disk_reproduction.sh slotune_smoke_001
```

Omit the argument only when a timestamp-generated name is desired. The script uses
[`config/reproduction_smoke.yaml`](config/reproduction_smoke.yaml) and writes the current artifact
tree to:

```text
/root/autodl-tmp/vllm-tuner-output/slotune-results/slotune_smoke_001/
```

Before calling the smoke successful, verify all of the following:

- the server reached READY and was stopped without a residual process;
- request-level records contain nonzero token evidence and explicit status;
- unavailable Prometheus or NVML data are marked unavailable rather than zero-filled;
- `manifest.json`, trace files and checksums, environment files, trial status/log/raw files,
  aggregates, and reports are present;
- failed, infeasible, or pruned attempts remain visible.

This run is deliberately tiny. Do not quote it as throughput, tail-latency, memory, energy, or
optimization evidence.

## 4. Run the formal 3B protocols

The formal model must exist at `/root/autodl-tmp/models/Qwen2.5-3B-Instruct`. Start with unique
experiment names and keep the chat and RAG artifacts separate:

```bash
./scripts/run_reproduction_command.sh tune \
  --config config/formal_3b_chat.yaml \
  --study-name qwen25_3b_chat_001 \
  --results-root /root/autodl-tmp/slotune-results

./scripts/run_reproduction_command.sh tune \
  --config config/formal_3b_rag.yaml \
  --study-name qwen25_3b_rag_001 \
  --results-root /root/autodl-tmp/slotune-results
```

Both protocols use equal measured budgets for default, seeded random, and constrained TPE; three
repeats; held-out validation; and a three-repeat capacity sweep at 1, 2, 4, 8, 16, and 32
requests/s. The SSE backend is the formal backend because it replays persisted arrival offsets
exactly. The official `vllm bench serve` adapter is a live reference/cross-validation backend;
its independently generated arrival process must be labeled and compared, not silently merged
with frozen-trace results.

To replay externally reviewed, frozen traces:

```bash
./scripts/run_reproduction_command.sh tune \
  --config config/formal_3b_chat.yaml \
  --study-name qwen25_3b_chat_fixed_trace_001 \
  --trace /root/autodl-tmp/traces/chat-search.jsonl \
  --holdout-trace /root/autodl-tmp/traces/chat-holdout.jsonl \
  --results-root /root/autodl-tmp/slotune-results
```

Every formal claim must identify the model and tokenizer revision, GPU, vLLM/PyTorch/CUDA/driver,
repository revision and dirty state, trace checksum, SLO, backend, search budget, repeats,
held-out outcome, and failures. Follow
[`docs/FORMAL_EXPERIMENTS.md`](docs/FORMAL_EXPERIMENTS.md) for the complete run and reporting
checklists.

## 5. Run the deterministic scheduler ablation

This CPU-only command writes JSON and Markdown to an explicit, initially absent directory:

```bash
./scripts/run_demo.sh /root/autodl-tmp/slotune-demo/scheduler-001
```

It compares fixed budgets 512/1024/2048/4096/8192 with adaptive on calibration and held-out
traces. It preserves no-benefit and regression conditions. These outputs validate simulator
mechanics; they do not demonstrate a vLLM runtime speedup.

## Current real-results register

### Formal Qwen2.5-3B results

**Status: NOT YET RECORDED.** Do not fill the table from console output, a smoke run, a simulator,
or a best search observation without repeats and held-out validation.

| Workload | Artifact root or archive | Commit | Trace checksum | Repeats | Held-out result | Failures | Result summary |
|---|---|---|---|---:|---|---|---|
| Chat | Pending | Pending final commit | Pending | 0 | Pending | Pending | No claim |
| RAG | Pending | Pending final commit | Pending | 0 | Pending | Pending | No claim |

To update a row, first validate the immutable artifact tree, then link the archived report and
record exact medians/ranges or confidence intervals. Retain regressions, infeasible trials,
missing telemetry, and cases where adaptive scheduling has no benefit.

## Reproduction checklist

- Use a unique experiment name; resume only against a manifest-compatible experiment.
- Preserve search and held-out trace files plus SHA-256 checksums.
- Keep warmup records out of the measurement reducer.
- Inspect raw request data, server logs, telemetry availability, and structured status before
  accepting aggregates.
- Compare default/random/TPE with equal measured budgets.
- Randomize formal run order where practical and report all three repeats.
- Keep simulator results separate from runtime GPU results.
- Link every reported number to its immutable artifact and environment fingerprint.
- Preserve negative/no-benefit conditions rather than selecting only favorable trials.

The milestone-by-milestone implementation/evidence gap is tracked in
[`docs/PLAN_AUDIT.md`](docs/PLAN_AUDIT.md).
