from image_classification.training import provenance


def test_source_snapshot_records_untracked_source_and_preserves_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "PROJECT_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("# test\n")
    (tmp_path / "src").mkdir()
    source = tmp_path / "src/new.py"
    source.write_text("VALUE = 1\n")
    fingerprint = provenance.source_fingerprint()
    assert "src/new.py" in fingerprint
    destination = tmp_path / "snapshot"
    provenance.snapshot_sources(destination, fingerprint)
    assert (destination / "src/new.py").read_bytes() == source.read_bytes()
    source.write_text("VALUE = 2\n")
    assert provenance.source_fingerprint() != fingerprint
