import copy
import json
import unittest

from scripts.analysis.audit_experiments import analyze, digest, stats


def make_snapshot():
    runs = []
    definitions = [
        ("baseline_mobilenetv2_seed", "mobilenetv2", [], [0.8786, 0.8768, 0.8754]),
        ("position_se1-2_cbam1-2_seed", "hybrid", [1, 2], [0.8878, 0.8816, 0.8788]),
        ("position_se1-2_cbam7-8_seed", "hybrid", [7, 8], [0.8846, 0.8796, 0.8740]),
        ("csgha_v3_seed", "csgha", [7, 8], [0.8868, 0.8762, 0.8774]),
    ]
    for prefix, model, positions, values in definitions:
        for seed, value in zip((42, 43, 44), values):
            run_id = f"{prefix}{seed}"
            config = {"seed": seed, "epochs": 2, "dataset": "cifar10", "model_type": model, "cbam_positions": positions, "evaluate_test": False}
            summary = {"experiment_id": run_id, "dataset": "cifar10", "best_validation_accuracy": value,
                       "best_epoch": 2, "test_evaluated": False, "train_samples": 45000,
                       "validation_samples": 5000, "test_samples": 10000}
            texts = {
                "config.yaml": json.dumps(config), "summary.json": json.dumps(summary),
                "metrics.json": '{"parameters_total": 100}', "benchmark.json": "{}",
                "logs/training.csv": f"epoch,train_acc,val_acc\n1,0.7,0.6\n2,0.9,{value}\n",
            }
            files = {name: {"path": f"artifacts/runs/{run_id}/{name}", "exists": True,
                            "text": text, "sha256": digest(text.encode())} for name, text in texts.items()}
            runs.append({"run_id": run_id, "config": config, "files": files,
                         "best_checkpoint": {"exists": True}, "other_checkpoints": {"final.pth": True}})
    return {"runs": runs, "manifests": []}


class ExperimentAuditTests(unittest.TestCase):
    def test_sample_standard_deviation(self):
        result = stats([88.78, 88.16, 87.88])
        self.assertAlmostEqual(result["mean_percent"], 88.2733333333333)
        self.assertAlmostEqual(result["sample_sd_percent"], 0.4605793453177579)
        self.assertIsNone(stats([88.68])["sample_sd_percent"])

    def test_invalid_statistical_inputs_are_rejected(self):
        for values in ([], [float("nan")], [float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                stats(values)

    def test_seed_pairing_is_independent_of_input_order(self):
        snapshot = make_snapshot()
        snapshot["runs"].reverse()
        result = analyze(snapshot)
        pair = next(p for p in result["paired"] if p["comparison"] == "CSGHA v3 - Independent shallow")
        self.assertEqual(pair["wins"], 0)
        self.assertAlmostEqual(pair["mean_delta"], -0.26)
        self.assertTrue(all(not row["issues"] for row in result["runs"]))

    def test_tampered_source_is_rejected(self):
        snapshot = make_snapshot()
        snapshot["runs"][0]["files"]["summary.json"]["text"] += " "
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            analyze(snapshot)

    def test_missing_paired_seed_is_rejected(self):
        snapshot = make_snapshot()
        snapshot["runs"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete paired seed"):
            analyze(snapshot)

    def test_duplicate_seed_is_rejected(self):
        snapshot = make_snapshot()
        snapshot["runs"].append(copy.deepcopy(snapshot["runs"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate seed"):
            analyze(snapshot)

    def test_noncontiguous_epochs_are_flagged(self):
        snapshot = make_snapshot()
        evidence = snapshot["runs"][0]["files"]["logs/training.csv"]
        text = evidence["text"].replace("1,0.7,0.6", "0,0.7,0.6")
        evidence.update(text=text, sha256=digest(text.encode()))
        self.assertIn("epoch_sequence_mismatch", analyze(snapshot)["runs"][0]["issues"])

    def test_summary_and_log_disagreement_is_flagged(self):
        snapshot = make_snapshot()
        evidence = snapshot["runs"][0]["files"]["summary.json"]
        summary = json.loads(evidence["text"])
        summary["best_epoch"] = 1
        evidence.update(text=json.dumps(summary), sha256=digest(json.dumps(summary).encode()))
        self.assertIn("summary_vs_log_best_epoch_mismatch", analyze(snapshot)["runs"][0]["issues"])

    def test_legacy_test_metrics_are_not_confused_with_untouched_test(self):
        snapshot = make_snapshot()
        run = snapshot["runs"][0]
        run["config"].pop("evaluate_test")
        evidence = run["files"]["summary.json"]
        summary = json.loads(evidence["text"])
        summary.pop("test_evaluated")
        summary["test_accuracy"] = 0.86
        evidence.update(text=json.dumps(summary), sha256=digest(json.dumps(summary).encode()))
        row = analyze(snapshot)["runs"][0]
        self.assertIs(row["test_evaluated"], True)
        self.assertEqual(row["test_evaluated_source"], "legacy_saved_test_metrics")
        self.assertNotIn("test_flag_mismatch", row["issues"])

    def test_manifest_config_mismatch_is_not_silent(self):
        snapshot = make_snapshot()
        run = snapshot["runs"][0]
        wrong = copy.deepcopy(run["config"])
        wrong["seed"] = 999
        manifest = {"runtime": {"git_commit": "test"}, "runs": [{
            "experiment_id": run["run_id"], "status": "completed", "return_code": 0,
            "summary": json.loads(run["files"]["summary.json"]["text"]), "resolved_config": wrong,
        }]}
        text = json.dumps(manifest)
        snapshot["manifests"] = [{"path": "manifest.json", "text": text, "sha256": digest(text.encode())}]
        self.assertIn("manifest_config_mismatch", analyze(snapshot)["runs"][0]["issues"])


if __name__ == "__main__":
    unittest.main()
