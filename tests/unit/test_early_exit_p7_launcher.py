from scripts.launch_early_exit_p7_source import validated_plan


def test_p7_source_plan_is_exact_and_test_locked():
    plan = validated_plan()
    assert len(plan) == 6
    assert {run["seed"] for run in plan} == {77, 78, 79}
    assert {run["resolved_config"]["model_type"] for run in plan} == {
        "mobilenetv2",
        "multi_exit",
    }
    for run in plan:
        config = run["resolved_config"]
        assert config["dataset"] == "imagenet100"
        assert config["evaluate_test"] is False
        assert config["split_seed"] == 20_261_003
        if config["model_type"] == "multi_exit":
            assert config["exit_positions"] == [8, 15]
            assert config["exit_distillation_alpha"] == 0.0
