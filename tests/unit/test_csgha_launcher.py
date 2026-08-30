from scripts.launch_csgha_validation import (
    CONFIG_PATH,
    build_command,
    load_target_config,
    target_run_directory,
)


def test_csgha_launcher_targets_validation_only_middle_candidate():
    config = load_target_config()

    assert build_command()[-2:] == ["--config", str(CONFIG_PATH)]
    assert config.experiment_name == "csgha_v3_se1-2_cbam7-8"
    assert config.model_type == "csgha"
    assert config.evaluate_test is False
    assert config.se_positions == (1, 2)
    assert config.cbam_positions == (7, 8)
    assert config.guidance_position == 2
    assert target_run_directory().name == config.experiment_id
