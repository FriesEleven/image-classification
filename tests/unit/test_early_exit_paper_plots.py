from __future__ import annotations

from pathlib import Path

from scripts.visualization import plot_early_exit_paper as paper_plots


def test_method_overview_is_pdf_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(paper_plots, "FIGURES", tmp_path)
    paper_plots.configure()
    paper_plots.method_overview()
    assert (tmp_path / "method_overview.pdf").is_file()
    assert not list(tmp_path.glob("*.png"))
