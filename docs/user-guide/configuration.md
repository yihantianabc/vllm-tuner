# Configuration

vLLM-Tuner uses YAML configuration files to specify tuning objectives, constraints, and search spaces.

## Configuration File Location

Place your config files anywhere:

- **Current directory**: `./my_config.yaml`
- **Project config directory**: `config/default.yaml`
- **Custom location**: Use `--config <path>` with CLI

## Configuration File Format

```yaml
# Model
model: "Qwen/Qwen1.5-0.5B-Chat-AWQ"

# GPU Configuration
gpu:
  device_ids: [0]
  count: 1

# Multi-Objective Weights
objectives:
  throughput: 60
  latency: 30
  memory: 10

# Constraints
constraints:
  max_latency_ms: null
  max_memory_utilization: null
  throughput_min: null

# Search Space
search_space:
  batch_size: [1, 256]
  max_num_seqs: [16, 256]
  gpu_memory_utilization: [0.6, 0.99]
  tensor_parallel_size: null
  pipeline_parallel_size: null

# Workload
workload:
  dataset_name: "tatsu-lab/alpaca"
  sample_size: 100
  concurrent_requests: 10
  warmup_requests: 5
  max_tokens: 256

# Baseline Generation
baseline:
  enabled: true
  num_requests: 1000
  max_tokens: 256

# Study Settings
study:
  min_trials: 25
  timeout_minutes: 1440
  storage_backend: "sqlite:///studies/optuna.db"
  prune_enabled: true
  n_startup_trials: 2

# Additional vLLM Args
vllm_args:
  max-model-len: 1024
```
