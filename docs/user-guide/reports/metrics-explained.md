# Metrics Explained

## Primary Metrics

### Throughput (Requests Per Second)

**Definition:** Number of requests completed per second

**Higher is better** for batch processing.

**Example:**
- Baseline: 10 req/s
- Optimized: 15 req/s
- Improvement: +50%

### Average Latency (ms)

**Definition:** Average time to complete one request

**Lower is better** for real-time applications.

**Example:**
- Baseline: 50ms
- Optimized: 35ms
- Improvement: -30% (30% faster)

### P95/P99 Latency (ms)

**Definition:** 95th/99th percentile of request times

**Lower is better** for predictable response times.

**Example:**
- P95: 80ms (95% of requests < 80ms)
- P99: 120ms (99% of requests < 120ms)

## Secondary Metrics

### Peak Memory (MB)

**Definition:** Maximum GPU memory used during benchmark

Lower values indicate more room for larger batches or concurrent requests.

### Average Memory (MB)

**Definition:** Memory usage averaged over entire benchmark

Shows typical memory usage during normal operation.

### Average Memory Utilization

**Definition:** Percentage of total GPU memory used

Formula: `avg_memory_mb / total_gpu_memory_mb * 100`

Lower utilization leaves room for scaling.

### GPU Utilization (%)

**Definition:** GPU compute utilization percentage

Higher values indicate GPU is working efficiently.

## Calculations

### Throughput Improvement

```
(best_throughput - baseline_throughput) / baseline_throughput * 100%
```

### Latency Improvement

```
(baseline_latency - best_latency) / baseline_latency * 100%
```

**Note:** Negative value means better (faster)

### Memory Delta

```
(best_memory - baseline_memory) / baseline_memory * 100%
```

**Note:** Negative value means better (less memory)

## See Also

- [HTML Reports](html-reports.md) - Interactive report features
- [Baseline Comparison](baseline-comparison.md) - Using baseline comparisons
- [Configuration](../configuration.md) - Tuning metrics
