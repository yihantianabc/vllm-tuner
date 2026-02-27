# OOM Errors (Out of Memory)

## Understanding OOM

OOM occurs when the GPU runs out of memory while running vLLM.

## Causes

1. **Model too large** - Model requires more VRAM than available
2. **High batch size** - Processing too many concurrent requests
3. **Large sequence length** - max-model-len too large
4 - **High memory utilization** - gpu_memory_utilization too high
5. **Memory fragmentation** - Repeated allocations/deallocations

## Immediate Solutions

### Reduce Memory Utilization

```yaml
search_space:
  gpu_memory_utilization: [0.5, 0.85]  # Lower upper bound
```

### Reduce Batch Size

```yaml
search_space:
  batch_size: [1, 64]  # Smaller batches
  max_num_seqs: [8, 64]
```

### Reduce Sequence Length

```yaml
vllm_args:
  max-model-len: 512  # Shorter sequences
```

## Prevention

### Set Realistic Constraints

```yaml
constraints:
  max_memory_utilization: 0.90
```

### Start Conservative

```yaml
search_space:
  batch_size: [1, 32]    # Start small
  max_num_seqs: [4, 32]
  gpu_memory_utilization: [0.6, 0.80]
```

## Monitoring

Check GPU during tuning:

```bash
# In another terminal:
watch -n 1 nvidia-smi
```

Look for:
- Memory usage approaching limit
- High GPU utilization
- Power spikes

## System Configuration

 Ubuntu/Debian settings

Check swap:
```bash
swapon --show
free -h
```

Increase swap if needed:
```bash
# Create 16GB swap
sudo dd if=/dev/zero bs=1M count=16384 of=/swapfile
sudo chmod 600 /swapfile
sudo swapon /swapfile
```
