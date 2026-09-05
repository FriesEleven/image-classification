from collections import Counter

from scripts.launch_early_exit_p6_source import load_protocol, validated_plans


def test_p6_source_launcher_freezes_exact_matrix_and_no_test_access():
    protocol = load_protocol()
    plans = validated_plans()
    assert len(plans) == 12
    assert len({plan["experiment_id"] for plan in plans}) == 12
    counts = Counter(
        (plan["resolved_config"]["dataset"], plan["resolved_config"]["model_type"], plan["seed"])
        for plan in plans
    )
    assert set(counts.values()) == {1}
    assert {plan["seed"] for plan in plans} == {71, 72, 73}
    assert all(not plan["resolved_config"]["evaluate_test"] for plan in plans)
    assert protocol["run_accounting"]["target_runs_after_source_gate"] == 12
