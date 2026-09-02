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


def test_stage_sparse_positions_and_id_are_explicit():
    config = load_config(
        [
            "--experiment_name", "selected",
            "--model_type", "stage_sparse",
            "--eca_positions", "1,2",
            "--se_positions", "7,8",
            "--cbam_positions", "15,16",
        ]
    )

    assert config.eca_positions == (1, 2)
    assert config.experiment_id == (
        "selected_stage_sparse_se7-8_eca1-2_cbam15-16_cifar10"
    )
    assert config.architecture_version == "stage_sparse_v1_independent_se_eca_cbam"


def test_stage_sparse_rejects_overlapping_attention_positions():
    with pytest.raises(ValueError, match="must be disjoint"):
        ExperimentConfig(
            model_type="stage_sparse",
            eca_positions=(1, 2),
            se_positions=(2, 3),
        )


def test_eca_positions_are_not_silently_ignored_by_other_models():
    with pytest.raises(ValueError, match="only supported"):
        ExperimentConfig(model_type="mobilenetv2", eca_positions=(1, 2))


def test_cifar100_config_sets_class_count_and_split():
    config = load_config(["--dataset", "cifar100", "--validation_size", "5000"])

    assert config.dataset == "cifar100"
    assert config.num_classes == 100
    assert config.validation_size == 5000
    assert config.experiment_id.endswith("_cifar100")


def test_validation_only_flag_is_parsed():
    config = load_config(["--evaluate_test", "false"])

    assert config.evaluate_test is False


def test_multi_exit_config_records_training_contract():
    config = load_config(
        [
            "--experiment_name", "exit_p0",
            "--model_type", "multi_exit",
            "--exit_positions", "8,16",
            "--exit_loss_weights", "0.2,0.3",
            "--exit_distillation_alpha", "0.5",
            "--exit_temperature", "3",
        ]
    )

    assert config.exit_positions == (8, 16)
    assert config.exit_loss_weights == (0.2, 0.3)
    assert config.experiment_id == "exit_p0_multi_exit_pos8-16_cifar10"
    assert config.to_dict()["exit_positions"] == [8, 16]
    assert "exit_positions" not in ExperimentConfig().to_dict()


def test_formal_development_split_is_explicit_without_changing_historical_schema():
    config = load_config(
        [
            "--validation_size", "5000",
            "--calibration_size", "5000",
            "--split_seed", "20260902",
        ]
    )

    assert config.calibration_size == 5000
    assert config.split_seed == 20260902
    assert config.to_dict()["calibration_size"] == 5000
    assert config.to_dict()["split_seed"] == 20260902
    assert "calibration_size" not in ExperimentConfig().to_dict()
    assert "split_seed" not in ExperimentConfig().to_dict()


@pytest.mark.parametrize(
    "values",
    [
        {"calibration_size": -1},
        {"validation_size": 45_000, "calibration_size": 5000},
        {"calibration_size": 9},
        {"split_seed": -1},
    ],
)
def test_development_split_contract_is_validated(values):
    with pytest.raises(ValueError):
        ExperimentConfig(**values)


@pytest.mark.parametrize(
    ("positions", "weights", "message"),
    [
        ((), (), "requires at least one"),
        ((16, 8), (0.2, 0.3), "unique and increasing"),
        ((8, 18), (0.2, 0.3), "between 1 and 17"),
        ((8, 16), (0.2,), "one exit_loss_weight"),
        ((8, 16), (0.2, 0.0), "must be positive"),
    ],
)
def test_multi_exit_config_rejects_ambiguous_contract(positions, weights, message):
    with pytest.raises(ValueError, match=message):
        ExperimentConfig(
            model_type="multi_exit",
            exit_positions=positions,
            exit_loss_weights=weights,
        )


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
