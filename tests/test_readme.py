"""README and the committed example report are generated from real runs, not typed numbers."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
REPORT = ROOT / "examples" / "reports" / "compas_like.md"
SCREENSHOT = ROOT / "docs" / "images" / "compas_like_groups.png"
CITATION = ROOT / "CITATION.cff"

# Copied from the seeded COMPAS-like CLI run (n_boot=1000, seed 0). If the
# library's numbers move, the README must move with them.
_FROM_THE_RUN = (
    "B's TPR (recall) is 73 percentage points higher than A's (95% CI 70 to 75 pp).",
    "0.280 [0.267, 0.294]",
    "0.121 [0.111, 0.132]",
    "0.015",
    "0.017",
    "Screening tool. Not a legal or causal conclusion.",
)


def test_readme_and_citation_exist() -> None:
    assert README.is_file()
    assert REPORT.is_file()
    assert SCREENSHOT.is_file()
    assert CITATION.is_file()


def test_readme_numbers_come_from_the_committed_report() -> None:
    readme = README.read_text(encoding="utf-8")
    report = REPORT.read_text(encoding="utf-8")
    for snippet in _FROM_THE_RUN:
        assert snippet in report, snippet
        assert snippet in readme, snippet


def test_readme_does_not_overclaim_a_demo_or_a_grade() -> None:
    raw = README.read_text(encoding="utf-8")
    readme = raw.lower()
    assert "live demo" not in readme
    assert "streamlit.app" not in readme
    assert "huggingface.co/spaces" not in readme
    assert "fairness grade" in readme
    assert "not a legal" in readme
    assert "not real compas" in readme
    assert "n/a" in readme
