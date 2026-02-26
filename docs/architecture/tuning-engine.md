# Tuning Engine

## How Optimization Works

### 1. Study Creation

```python
study = optuna.create_study(
    direction="maximize",  # or "minimize"
    sampler=TPESampler(),
    pruner=MedianPruner()
)
```

### 2. Optimization Loop

For each trial:
1. **Generate parameters** from search space
2. **Launch vLLM server** with trial parameters
3. **Run benchmarks** with workload
4. **Collect metrics** (throughput, latency, memory)
5. **Update Optuna** with trial result
6. **Prune unpromising trials** if enabled

### 3. Multi-Objective Optimization

```python
directions = ["maximize", "minimize", "minimize"]  # throughput, latency, memory
```

## Search Space

All vLLM-tunable parameters:
- `batch_size`: [1, 256]
- `max_num_seqs`: [16, 512]
- `gpu_memory_utilization`: [0.6, 0.99]
- `max_num_batched_tokens`: [1024, 32768]
- `tensor_parallel_size`: [1, 2, 4]
- `pipeline_parallel_size`: [1, 2]

## Objective Functions

### Throughput (Maximize)

```python
throughput_requests_per_sec = num_completed / duration_seconds
```

### Latency (Minimize)

```python
avg_latency_ms = total_latency_ms / num_completed
```

### Memory (Minimize)

```python
memory_utilization = avg_memory_mb / total_gpu_memory_mb
```

## See Also

- [Architecture Overview](index.md) - System design
- [Configuration](../user-guide/configuration.md) - Search space definition
