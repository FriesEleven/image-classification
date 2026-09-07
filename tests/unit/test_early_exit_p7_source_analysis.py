import numpy as np

from scripts.analysis.analyze_early_exit_p7_source import (
    corrected_cost_fractions,
    p7_risk_budget,
    unpack_p7_logits,
)


def test_p7_risk_budget_uses_frozen_protocol_names():
    protocol = {
        "risk_budget": {
            "overall_accuracy_drop": 0.005,
            "balanced_accuracy_drop": 0.005,
            "worst_class_accuracy_drop": 0.04,
        }
    }
    assert p7_risk_budget(protocol) == {
        "overall_drop": 0.005,
        "balanced_drop": 0.005,
        "worst_class_drop": 0.04,
    }


def test_corrected_cost_includes_only_deployable_head_on_fallback():
    profile = {
        "final_path_macs": 299_622_272,
        "selected": {
            "deployable": {
                "position": 8,
                "channels": 64,
                "path_macs_including_exit_head": 131_141_376,
            },
            "training_only_auxiliary": {"position": 15},
        },
    }
    costs = corrected_cost_fractions(profile, num_classes=100)
    assert costs["exit_head_macs"] == 6_400
    assert costs["exit_cost_fraction"] == 131_141_376 / 299_622_272
    assert costs["fallback_cost_fraction"] == (299_622_272 + 6_400) / 299_622_272


def test_unpack_p7_logits_uses_exit8_and_retains_training_only_exit15():
    values = [np.full((2, 100), index) for index in range(3)]
    final_logits, deployable_logits, auxiliary_logits = unpack_p7_logits(values)
    assert np.all(final_logits == 0)
    assert np.all(deployable_logits == 1)
    assert np.all(auxiliary_logits == 2)
