from image_classification.config import ExperimentConfig, load_config


def test_hybrid_experiment_id():
    config = ExperimentConfig(
        experiment_name="hybrid",
        model_type="hybrid",
        se_positions=(1, 2),
        cbam_positions=(15, 16),
    )
    assert config.experiment_id == "hybrid_hybrid_se1-2_cbam15-16"


def test_cli_positions_are_parsed():
    config = load_config(["--model_type", "cbam", "--aux_positions", "1,2"])
    assert config.aux_positions == (1, 2)
