from scripts.launch_early_exit_p7_target import load_protocol, validated_plan


def test_p7_target_plan_is_exact_and_uses_frozen_source_lock():
    protocol = load_protocol()
    assert protocol["source_gate"]["selected_shared_threshold"] == 0.85
    assert protocol["official_test"]["prepared"] is False
    assert protocol["official_test"]["accessed"] is False

    plan = validated_plan()
    assert len(plan) == 6
    assert {run["seed"] for run in plan} == {80, 81, 82}
    assert {run["resolved_config"]["model_type"] for run in plan} == {
        "mobilenetv2",
        "multi_exit",
    }
    for run in plan:
        config = run["resolved_config"]
        assert config["dataset"] == "imagenet100"
        assert config["evaluate_test"] is False
        assert config["split_seed"] == 20_261_003
        assert config["num_workers"] == 12
        if config["model_type"] == "multi_exit":
            assert config["exit_positions"] == [8, 15]
            assert config["exit_distillation_alpha"] == 0.0
