# API Reference

## Main Components

### src.config.models

`TuningConfig` - Main configuration model
`GPUConfig` - GPU configuration
`WeightedObjectives` - Optimization objectives
`Constraints` - Tuning constraints
`WorkloadConfig` - Benchmark workload

### src.tuner

`StudyManager` - Manages Optuna studies
`VLLMOptimizer` - Optimization logic

### src.baseline

`VLLMBaselineRunner` - Baseline generation

See [Architecture](../architecture/) for details.
