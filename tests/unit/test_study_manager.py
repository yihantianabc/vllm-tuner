"""Compatibility facade tests for the SLOTune study manager."""

from unittest.mock import Mock

from vllm_tuner.config.models import GPUConfig, TuningConfig, WorkloadConfig
from vllm_tuner.tuner.study_manager import StudyManager


def _config() -> TuningConfig:
    return TuningConfig(
        model="gpt2",
        gpu=GPUConfig(device_ids=[0]),
        workload=WorkloadConfig(sample_size=2, max_tokens=17),
    )


def test_study_manager_initialization(tmp_path) -> None:
    manager = StudyManager(_config(), "test_study", tmp_path)
    assert manager.study_name == "test_study"
    assert manager.output_dir == tmp_path
    assert manager.config.workload.max_tokens == 17


def test_get_study_summary_for_compatibility_optuna(tmp_path) -> None:
    manager = StudyManager(_config(), "test_study", tmp_path)
    manager.optimizer.study = Mock()
    manager.optimizer.study.trials = []
    manager.optimizer.get_best_result = Mock(return_value={"trial_number": 1})
    manager.optimizer.get_top_n_results = Mock(return_value=[])
    summary = manager.get_study_summary()
    assert summary["num_trials"] == 0
    assert summary["best_trial"] == {"trial_number": 1}


def test_get_best_config_uses_slo_goodput(tmp_path) -> None:
    manager = StudyManager(_config(), "test_study", tmp_path)
    manager.optimizer.get_best_result = Mock(
        return_value={
            "parameters": {"max_num_seqs": 8},
            "metrics": {"goodput_requests_per_sec": 2.0},
        }
    )
    best = manager.get_best_config()
    assert best is not None
    assert best["vllm_params"] == {"max_num_seqs": 8}
    assert best["metrics"]["goodput_requests_per_sec"] == 2.0


def test_get_best_config_without_feasible_result(tmp_path) -> None:
    manager = StudyManager(_config(), "test_study", tmp_path)
    manager.optimizer.get_best_result = Mock(return_value={})
    assert manager.get_best_config() is None
