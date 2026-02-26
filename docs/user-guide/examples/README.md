# Example Configurations

This directory contains ready-to-use example YAML configurations for vLLM-Tuner.

## Example Configurations

### 1. Simple Tune (simple_tune.yaml)

**Use Case:** First-time user, basic tuning study

```yaml
# Model to tune
model: "Qwen/Qwen1.5-0.5B-Chat-AWQ"

# GPU configuration
gpu:
  device_ids: [0]
  count: 1

# Multi-objective optimization weights (balanced)
objectives:
  throughput: 60
  latency: 30
  memory: 10

# Search space (moderate range)
search_space:
  batch_size: [1, 256]
  max_num_seqs: [16, 256]
  gpu_memory_utilization: [0.6, 0.99]
  max_num_batched_tokens: [1024, 16384]

# Workload (Alpaca dataset)
workload:
  dataset_name: "tatsu-lab/alpaca"
  sample_size: 100
  concurrent_requests: 10
  warmup_requests: 5
  max_tokens: 256

# Study settings
study:
  min_trials: 20
  timeout_minutes: 60
  prune_enabled: true
  n_startup_trials: 5

# Baseline generation
baseline:
  enabled: true
  num_requests: 1000
  max_tokens: 256
```

**To run:**
```bash
vllm-tuner tune --config docs/user-guide/examples/simple_tune.yaml --study-name simple_test
```

**Expected Results:**
- Baseline throughput: ~10-15 req/s (depends on model)
- Optimized throughput: +20-50% improvement
- Moderate latency impact (if any)

---

### 2. Throughput Optimization (throughput-opt.yaml)

**Use Case:** Maximize requests/second for batch processing, API backends

```yaml
# Model
model: "Qwen/Qwen1.5-0.5B-Chat-AWQ"

# GPU
gpu:
  device_ids: [0]
  count: 1

# Maximize throughput
objectives:
  throughput: 80
  latency: 10
  memory: 10

# Wider search space for throughput
search_space:
  batch_size: [64, 512]
  max_num_seqs: [128, 512]
  gpu_memory_utilization: [0.85, 0.99]
  max_num_batched_tokens: [2048, 32768]

# Workload (more concurrent requests)
workload:
  dataset_name: "tatsu-lab/alpaca"
  sample_size: 200
  concurrent_requests: 20
  warmup_requests: 10
  max_tokens: 256

# Longer study to find optimal
study:
  min_trials: 30
  timeout_minutes: 120
  prune_enabled: true
  n_startup_trials: 5
```

**To run:**
```bash
vllm-tuner tune --config docs/user-guide/examples/throughput-opt.yaml --study-name throughput_test
```

---

### 3. Latency Optimization (latency-opt.yaml)

**Use Case:** Minimize response time for real-time applications, chatbots, APIs

```yaml
# Model
model: "Qwen/Qwen1.5-0.5B-Chat-AWQ"

# GPU
gpu:
  device_ids: [0]
  count: 1

# Prioritize low latency
objectives:
  throughput: 20
  latency: 80
  memory: 0

# Constraints
constraints:
  max_latency_ms: 50  # Max allowed latency

# Smaller batches for low latency
search_space:
  batch_size: [1, 32]
  max_num_seqs: [4, 32]
  gpu_memory_utilization: [0.7, 0.9]

# Lower concurrency
workload:
  dataset_name: "tatsu-lab/alpaca"
  sample_size: 100
  concurrent_requests: 5
  warmup_requests: 10
  max_tokens: 256

# Strict timeout and more trials
study:
  min_trials: 25
  timeout_minutes: 60
  prune_enabled: true
  n_startup_trials: 5
```

**To run:**
```bash
vllm-tuner tune --config docs/user-guide/examples/latency-opt.yaml --study-name latency_test
```

---

### 4. Multi-GPU Tuning (multi-gpu-tune.yaml)

**Use Case:** Large models requiring multiple GPUs, model parallelism

```yaml
# Large model requiring multiple GPUs
model: "meta-llama/Llama-2-7b-chat-hf"

# Multi-GPU setup
gpu:
  device_ids: [0, 1]
  count: 2

# Balanced multi-objective
objectives:
  throughput: 60
  latency: 30
  memory: 10

# Multi-GPU search space
search_space:
  batch_size: [32, 256]
  max_num_seqs: [64, 256]
  gpu_memory_utilization: [0.85, 0.99]
  tensor_parallel_size: [1, 2]
  pipeline_parallel_size: [1, 2]

# Workload (larger sample)
workload:
  dataset_name: "tatsu-lab/alpaca"
  sample_size: 200
  concurrent_requests: 15
  warmup_requests: 10
  max_tokens: 256

# More trials for multi-GPO
study:
  min_trials: 30
  timeout_minutes: 120
  prune_enabled: true
  n_startup_trials: 5
```

**To run:**
```bash
vllm-tune --config docs/user-guide/examples/multi-gpu-tune.yaml --study-name multi_gpu_test
```

---

## Configuration Patterns

### Pattern 1: Fast Exploration

```yaml
study:
  min_trials: 10     # Quick exploration
  timeout_minutes: 30  # Short timeout
  n_startup_trials: 2   # Quick startup
```

### Pattern 2: Refining Best Configuration

```bash
# Run study with wider search space
vllm-tuner tune --config wide_search.yaml --study-name exploration

# Use best config for refinement study
vllm-tune --config studies/exploration/configs/best_config.yaml \\
  --study-name refinement
```

### Pattern 3: Memory-Constrained

```yaml
constraints:
  max_memory_utilization: 0.90

search_space:
  batch_size: [1, 64]      # Small batches
  max_num_seqs: [8, 64]    # Limit concurrency
```

### Pattern 4: Production Deployment

```yaml
study:
  min_trials: 50     # More trials for reliability
  timeout_minutes: 240  # Longer timeout

objectives:
  throughput: 70
  latency: 30
  memory: 0

search_space:
  batch_size: [128, 512]
  max_num_seqs: [256, 512]
```

## Adapting Examples

### Change Model

Replace the `model` field:
```yaml
model: "your-model-name"
```

### Adjust GPU Count

Update `gpu.device_ids` and `gpu.count`:
```yaml
gpu:
  device_ids: [0, 1, 2, 3]
  count: 4
```

### Change Dataset

Update `workload.dataset_name`:
```yaml
workload:
  dataset_name: "your-dataset-name"
  sample_size: 100
```

### Adjust Objectives

Change weights to suit your use case:
```yaml
objectives:
  throughput: 80    # Throughput priority
  latency: 20     # Low latency
```

See [Configuration Guide](../configuration.md) for all configuration options.

## Results Interpretation

### After Running Studies

Check baseline vs best trial comparison in the HTML report:
1. Open `reports/<study_name>/report.html`
2. Look at "Baseline vs Best Trial Comparison" section
3. Check improvement percentages

### Export Best Configuration

```bash
vllm-tuner export --study-name my_study --format yaml
```

This opens or saves to `studies/my_study/configs/best_config.yaml`

Reuse configuration:
```bash
vllm-tuner tune --config studies/my_study/configs/best_config.yaml --study-name refine
```

## Best Practices

1. Start with simple_tune.yaml
2. Verify baseline generation succeeds
3. Check vLLM server logs if errors occur
4. Increase min_trials gradually for refinement
5. Review Pareto Front plot for trade-offs
6. Export and document best configuration

## Troubleshooting

### Study stops early?

Check timeout and constraints:
- Increase timeout_minutes
- Relax max_latency_ms constraint
- Lower min_trials requirement

### All trials fail?

Check GPU resources:
```bash
nvidia-smi
```

Reduce workload or search space.

### Poor throughput/latency?

Check:
- GPU utilization with `nvidia-smi`
- Model fits in GPU memory
- Dataset loading works correctly

## See Also

- [Configuration Guide](../configuration.md) - Configuration options
- [Reports](../reports/) - Understanding reports
- [Troubleshooting](../../troubleshooting/) - Common issues
- [Installation](../installation.md) - Installation guide
