# Configuration

SLOTune validates YAML with Pydantic and rejects unknown fields. Use the files under
[`config/`](../../config/) as executable examples.

## Complete schema

```yaml
model: "/root/autodl-tmp/models/Qwen2.5-3B-Instruct"
model_revision: null
tokenizer: null

gpu:
  device_ids: [0]
  count: 1

slo:
  ttft_ms: 1000
  tpot_ms: 100
  e2e_ms: null

constraints:
  max_error_rate: 0.01
  max_peak_vram_mb: 31000
  max_memory_utilization: 0.95
  require_no_oom: true
  require_server_alive: true

search_space:
  gpu_memory_utilization: [0.60, 0.95]
  max_num_seqs: [8, 16, 32, 64, 128]
  max_num_batched_tokens: [1024, 2048, 4096, 8192]
  tensor_parallel_size: 1
  pipeline_parallel_size: 1

workload:
  name: "mixed"
  dataset_name: "synthetic:mixed"
  sample_size: 500
  prompt_length_distribution: "weighted"
  warmup_requests: 30
  max_concurrency: 32
  request_rate: 8
  capacity_request_rates: [1, 2, 4, 8, 16, 32]
  capacity_repeats: 3
  burstiness: 1.0
  max_tokens: 128
  ignore_eos: false
  seed: 2026
  request_timeout_seconds: 300
  benchmark_backend: "sse"

telemetry:
  enabled: true
  interval_ms: 200
  metrics_path: "/metrics"
  collect_nvml: true
  collect_energy: true

study:
  trial_budget: 16
  timeout_minutes: 1440
  prune_enabled: false
  n_startup_trials: 5
  seed: 2026
  methods: ["default", "random", "tpe"]
  repeat_count: 3
  top_candidates: 3
  holdout_enabled: true
  holdout_min_goodput_ratio: 0.8
  resume: false

baseline:
  enabled: false

vllm_args:
  max-model-len: 8192
```

## Objective and SLO

There is no weighted `objectives` block. SLOTune maximizes request SLO goodput under hard
constraints. A request is good only if it succeeds and meets every non-null TTFT, TPOT, and E2E
threshold. Thresholds are milliseconds.

`max_error_rate`, no-OOM/server-alive requirements, peak VRAM, memory utilization, and p99
latency checks determine feasibility. An infeasible result is retained but cannot be selected.

## Effective search space

`gpu_memory_utilization`, `max_num_seqs`, and `max_num_batched_tokens` are tunable.
`tensor_parallel_size` and `pipeline_parallel_size` must both be exactly one.

`batch_size` is intentionally invalid: it is not an effective vLLM server knob. Do not repeat a
searched key inside `vllm_args`; the collision validator rejects ambiguous ownership.

## Workload

Named deterministic profiles are `chat`, `rag`, `mixed`, and `codegen`. A fixed seed controls
lengths and open-loop interarrival times. `request_rate: null` represents immediate load;
positive values generate open-loop arrivals. `burstiness: 1.0` is Poisson-like and larger values
are burstier.

`max_concurrency` limits in-flight clients. `concurrent_requests` is accepted as a legacy alias
but new files should use `max_concurrency`. Warmup requests are never part of measured metrics.
`sse` selects fixed-trace streaming replay and is used by the formal configs so frozen arrival
offsets remain exact. `official` selects `vllm bench serve` as a live reference/cross-validation
backend.

`capacity_request_rates` declares the formal offered-load sweep and `capacity_repeats` declares
the repeats per rate. The checked-in 3B protocols use rates 1/2/4/8/16/32 and three repeats.

If `workload.name` is not a named profile, `dataset_name` must resolve to a supported local
JSON/JSONL prompt file or dataset name. Fixed trace files passed through CLI take precedence over
generation.

## Equal search budget and validation

`trial_budget` applies independently to every method in `methods`. Formal files use all three
methods, repeat candidates three times, and enable held-out validation. `resume: false` prevents
accidental reuse; when enabled, manifest/search-space compatibility is checked.

Validate without starting vLLM:

```bash
python - <<'PY'
from vllm_tuner.config.validation import load_yaml_config

for path in ("config/formal_3b_chat.yaml", "config/formal_3b_rag.yaml"):
    config = load_yaml_config(path)
    print(path, config.workload.name, config.study.methods)
PY
```
