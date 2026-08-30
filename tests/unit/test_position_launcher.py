from scripts.launch_position_screening import (
    SWEEP_PATH,
    build_command,
    position_run_directories,
)


def test_position_launcher_targets_three_unique_validation_runs():
    command = build_command("--dry-run")
    directories = position_run_directories()

    assert command[-3:] == ["--sweep", str(SWEEP_PATH), "--dry-run"]
    assert len(directories) == 3
    assert len(set(directories)) == 3
    assert all("position_se1-2_cbam" in path.name for path in directories)
