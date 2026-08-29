from image_classification.paths import RunPaths


def test_run_paths_keep_one_experiment_together():
    paths = RunPaths("smoke")
    assert paths.checkpoints.parent == paths.root
    assert paths.predictions.parent == paths.root
    assert paths.training_log.parents[1] == paths.root
