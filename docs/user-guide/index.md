# User Guide

Welcome to the vLLM-Tuner user guide. This guide helps you get started with tuning vLLM models for optimal performance.

## What is vLLM-Tuner?

vLLM-Tuner is a tool that automatically optimizes vLLM model configurations for:
- **Maximizing throughput** - Process more requests per second
- **Minimizing latency** - Reduce time-to-first-token and generation time
- **Balancing memory** - Keep GPU memory usage efficient

Using Optuna's Bayesian optimization, it automatically searches the parameter space to find the best configuration for your specific workload and hardware.

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
pip install vllm
```

See [Installation](installation.md) for detailed instructions.

### 2. Create a Configuration

```yaml
model: "Qwen/Qwen1.5-0.5B-Chat-AWQ"

gpu:
  device_ids: [0]
  count: 1

objectives:
  throughput: 60
  latency: 30
  memory: 10

workload:
  dataset_name: "tatsu-lab/alpaca"
  sample_size: 100
  max_tokens: 256

search_space:
  batch_size: [1, 256]
  max_num_seqs: [16, 256]
  gpu_memory_utilization: [0.6, 0.99]

study:
  min_trials: 25
  timeout_minutes: 1440
```

Save this as `my_config.yaml`.

### 3. Run a Tuning Study

```bash
vllm-tuner tune --config my_config.yaml --study-name my_first_study
```

### 4. Generate a Report

```bash
vllm-tuner report --study-name my_first_study --format html
```

Open the generated HTML report to see the optimization results.

## Core Concepts

### Configuration Files

YAML files define your tuning objectives, constraints, and search spaces. See [Configuration](configuration.md).

### Objectives and Constraints

- **Objectives**: What you want to optimize (throughput, latency, memory)
- **Constraints**: Limits that must not be exceeded (max latency, memory cap)

### Workload Definition

Your workload describes the typical requests your model will handle:
- Dataset for prompt generation
- Number of concurrent requests
- Request token lengths

### Search Space

The parameter ranges that vLLM-Tuner will explore:
- `batch_size`: How many tokens to process together
- `max_num_seqs`: Maximum concurrent sequences per batch
- `gpu_memory_utilization`: Fraction of GPU memory to use

### Baseline Generation

Before tuning, vLLM-Tuner can generate baseline metrics using default vLLM settings. This provides a reference point for comparison.

### Study and Trials

- **Study**: A complete optimization run
- **Trial**: One iteration testing a specific configuration
- Optuna automatically learns from trials to improve over time

## Common Workflows

### Workflow 1: Maximize Throughput

```yaml
objectives:
  throughput: 100
  latency: 0
  memory: 0
```

Best for: Batch processing, high-volume inference.

### Workflow 2: Minimize Latency

```yaml
objectives:
  throughput: 0
  latency: 100
  memory: 0

constraints:
  max_latency_ms: 500
```

Best for: Real-time chatbots, interactive applications.

### Workflow 3: Balanced Performance

```yaml
objectives:
  throughput: 50
  latency: 40
  memory: 10
```

Best for: General-purpose deployment with multiple priorities.

## Output Artifacts

### Study Data

Stored in `studies/<study_name>/`:
- `optuna.db`: Optuna study database
- `best_params.json`: Best found configuration
- `study_summary.json`: Run statistics

### HTML Reports

Interactive reports (`<study_name>_report.html`) with:
- Optimization progress plots
- Trial comparison table
- Baseline vs. optimized metrics
- Parameter importance analysis

### Exported Configuration

Export the best configuration:

```bash
vllm-tuner export --study-name my Study --format yaml > best_config.yaml
```

## Next Steps

### For Running Studies

- [Configuration](configuration.md) - Learn all configuration options
- [CLI Commands](cli-commands.md) - Command-line reference
- [Examples](examples/) - Ready-to-use examples

### For Understanding Results

- [HTML Reports](reports/html-reports.md) - Interactive report features
- [Metrics Explained](reports/metrics-explained.md) - What each metric means
- [Baseline Comparison](reports/baseline-comparison.md) - Comparing with defaults

### For Custom Workloads

- [Custom Workload Guide](examples/custom-workload.md) - Use your own dataset

## Best Practices

### Start Small

- Use `sample_size: 100` for initial experiments
- Set `min_trials: 25` for quick feedback
- Use small models (e.g., Qwen1.5-0.5B) for testing

### Choose Realistic Constraints

- Set `max_memory_utilization` based on your actual GPU memory
- Use realistic `max_tokens` values based on your workload
- Consider your GPU model when setting constraints

### Monitor GPU

- Ensure no other processes are using the target GPU
- Monitor GPU memory during study with `nvidia-smi dmon`
- Use `gpu_memory_utilization: 0.8` initially, not 0.99

### Save Your Work

Study data persists in `studies/<study_name>/optuna.db`. You can:

- Resume interrupted studies
- Compare multiple studies
- Export best configurations

## Troubleshooting

### Study Running Slow?

- Reduce `sample_size` and `min_trials`
- Use fewer concurrent requests
- Reduce dataset size

### Trials Failing?

- Check vLLM logs in the study directory
- Verify your model is accessible
- Ensure sufficient GPU memory (try lower `gpu_memory_utilization`)

### Poor Results?

- Verify workload matches your real usage
- Check that constraints are realistic
- Increase number of trials

## Getting Help

- [Troubleshooting](../troubleshooting/common-issues.md) - Common problems
- [Examples](examples/) - Working examples
- [Developer Guide](../developer-guide/) - For advanced users

---

**Next:** [Installation](installation.md)