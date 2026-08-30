import pytest

from image_classification.config import ExperimentConfig, load_config


def test_hybrid_experiment_id():
    config = ExperimentConfig(
        experiment_name="hybrid",
        model_type="hybrid",
        se_positions=(1, 2),
        cbam_positions=(15, 16),
    )
    assert config.experiment_id == "hybrid_hybrid_se1-2_cbam15-16_cifar10"


def test_cli_positions_are_parsed():
    config = load_config(["--model_type", "cbam", "--aux_positions", "1,2"])
    assert config.aux_positions == (1, 2)


def test_cifar100_config_sets_class_count_and_split():
    config = load_config(["--dataset", "cifar100", "--validation_size", "5000"])

    assert config.dataset == "cifar100"
    assert config.num_classes == 100
    assert config.validation_size == 5000
    assert config.experiment_id.endswith("_cifar100")


def test_validation_only_flag_is_parsed():
    config = load_config(["--evaluate_test", "false"])

    assert config.evaluate_test is False


def test_shared_gpu_runtime_options_are_explicit_and_validated():
    config = load_config(["--torch_num_threads", "1", "--measure_inference", "false"])
    assert config.torch_num_threads == 1
    assert config.measure_inference is False
    assert ExperimentConfig().torch_num_threads == 0
    assert ExperimentConfig().measure_inference is True
    with pytest.raises(ValueError, match="torch_num_threads"):
        ExperimentConfig(torch_num_threads=-1)


def test_csgha_config_records_guidance_in_experiment_id():
    config = ExperimentConfig(
        experiment_name="guided",
        model_type="csgha",
        se_positions=(1, 2),
        cbam_positions=(7, 8),
        guidance_position=2,
    )

    assert config.experiment_id == "guided_csgha_se1-2_guide2_cbam7-8_cifar10"


def test_csgha_rejects_targets_before_guidance_source():
    with pytest.raises(ValueError, match="must follow"):
        ExperimentConfig(
            model_type="csgha",
            se_positions=(1, 2),
            cbam_positions=(1, 2),
            guidance_position=2,
        )
