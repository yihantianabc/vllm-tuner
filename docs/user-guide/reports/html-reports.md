# HTML Reports

## Report Features

The HTML report provides an interactive dashboard with:

### Key Information

Best Configuration Metrics
- Throughput (requests/sec)
- Average Latency (ms)
- Total Trials run
- Best Trial number

### Parameters Table

- All vLLM parameters from best trial
- Batch size, max_num_seqs, gpu_memory_utilization, etc.
- Easy to copy-paste for reuse

### Baseline vs Best Trial Comparison

Shows comparison between:
- Baseline: Default vLLM parameters
- Best Trial: Optimized parameters found

Metrics compared:
- Throughput improvement (%)
- Latency improvement (%)
- P95/P99 latency improvement (%)
- Memory delta (%)

**Color coding:**
- 🟢 Positive improvement: Better than baseline
- 🔴 Negative change: Worse than baseline
- ⚪ No change: Same as baseline

### Interactive Charts

#### Throughput Progression
Shows throughput across trials with baseline reference line.

#### Latency Distribution
Shows average latency over trials.

#### Pareto Front (Throughput vs. Locality)
Trade-off visualization: higher throughput vs lower latency.

#### GPU Memory Utilization
Memory usage over trials, shows optimization impact.

#### Combined View
Multi-panel chart with all metrics in one dashboard.

## Viewing Reports

```bash
# Generate report
vllm-tuner report --study-name my_study --format html

# View report (opens in browser)
open reports/my_study/report.html
```

Report location: `reports/<study_name>/report.html`

## Report Data

Metrics are saved in:
- `studies/<study_name>/configs/summary.json` - Summary and best trial
- `studies/<study_name>/configs/trials.json` - All trial data

## See Also

- [Metrics Explained](metrics-explained.md) - What each metric means
- [Baseline Comparison](baseline-comparison.md) - How baseline comparison works
- [Architecture](../architecture/) - How reports are generated
