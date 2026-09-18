"""Report builder: every section, no NaN leak, numbers match the computed objects."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from faircheck.guidance import (
    DecisionKind,
    GuidanceAnswers,
    LabelTrust,
    ScoreReading,
    format_interval,
    format_rate,
)
from faircheck.report import (
    DISCLAIMER,
    render_markdown,
    run_audit,
    write_markdown,
    write_pdf,
)
from faircheck.types import ColumnMapping
from tests.conftest import cells_to_frame

REQUIRED_HEADINGS = (
    "# FairCheck report",
    "## Plain-language summary",
    "## Per-group rates",
    "## Summary gaps",
    "## Which metric fits this decision?",
    "## Scores",
    "## Why not all at once",
    "## Methods",
)


def _mapping(*, score: str | None = None) -> ColumnMapping:
    return ColumnMapping(
        label="y_true",
        prediction="y_pred",
        sensitive=("group",),
        positive_label=1,
        score=score,
    )


def test_report_contains_every_section(hand_computed_frame: pd.DataFrame) -> None:
    audit = run_audit(hand_computed_frame, _mapping(), n_boot=200, seed=0)
    markdown = render_markdown(audit)
    for heading in REQUIRED_HEADINGS:
        assert heading in markdown, f"missing {heading!r}"
    assert DISCLAIMER in markdown
    assert "four-fifths" in markdown.lower()
    assert "not a legal" in markdown.lower()


def test_report_numbers_match_the_computed_objects(hand_computed_frame: pd.DataFrame) -> None:
    audit = run_audit(hand_computed_frame, _mapping(), n_boot=200, seed=0)
    markdown = render_markdown(audit)
    boot = audit.bootstrap

    for group_index, _label in enumerate(boot.group_labels):
        for key in ("selection_rate", "tpr", "fpr", "precision"):
            formatted = format_interval(*boot.rate_interval(key, group_index))
            assert formatted in markdown

    di_low, di_point, di_high = boot.gap_interval("disparate_impact_ratio")
    assert di_point == pytest.approx(0.40)
    assert format_interval(di_low, di_point, di_high) in markdown
    assert format_rate(di_point) in markdown
    assert "below 0.80" in markdown

    eo_low, eo_point, eo_high = boot.gap_interval("equal_opportunity_difference")
    assert eo_point == pytest.approx(0.20)
    assert format_interval(eo_low, eo_point, eo_high) in markdown


def test_report_does_not_leak_nan_when_a_rate_is_undefined() -> None:
    frame = cells_to_frame({"A": (40, 10, 10, 40), "B": (0, 5, 0, 45)})
    audit = run_audit(frame, _mapping(), n_boot=100, seed=0)
    markdown = render_markdown(audit)
    assert re.search(r"\bnan\b", markdown, flags=re.IGNORECASE) is None
    assert "n/a" in markdown
    assert "not computable" in markdown


def test_guidance_answers_appear_in_the_report(hand_computed_frame: pd.DataFrame) -> None:
    answers = GuidanceAnswers(
        decision=DecisionKind.PUNITIVE,
        labels=LabelTrust.TRUSTED,
        scores=ScoreReading.NOT_USED,
    )
    audit = run_audit(hand_computed_frame, _mapping(), n_boot=50, seed=0, answers=answers)
    markdown = render_markdown(audit)
    assert "Equalized odds" in markdown
    assert audit.recommendation.families == ("equalized_odds_difference",)


def test_scores_section_explains_the_skip_when_no_score_is_mapped(
    hand_computed_frame: pd.DataFrame,
) -> None:
    audit = run_audit(hand_computed_frame, _mapping(), n_boot=50, seed=0)
    assert audit.calibration is None
    assert "No score column was mapped" in render_markdown(audit)


def test_scores_section_reports_ece_when_a_score_is_mapped(
    hand_computed_frame: pd.DataFrame,
) -> None:
    frame = hand_computed_frame.copy()
    frame["score"] = 0.5
    audit = run_audit(frame, _mapping(score="score"), n_boot=50, seed=0)
    assert audit.calibration is not None
    markdown = render_markdown(audit)
    assert "positive-class" in markdown
    assert "ROC-AUC" in markdown
    assert "ECE" in markdown
    for group in audit.calibration.groups:
        assert format_rate(group.ece) in markdown


def test_write_markdown_round_trips(hand_computed_frame: pd.DataFrame, tmp_path: Path) -> None:
    audit = run_audit(hand_computed_frame, _mapping(), n_boot=50, seed=0)
    dest = tmp_path / "report.md"
    write_markdown(audit, dest)
    assert dest.read_text(encoding="utf-8") == render_markdown(audit)


def test_write_pdf_produces_a_pdf_file(hand_computed_frame: pd.DataFrame, tmp_path: Path) -> None:
    audit = run_audit(hand_computed_frame, _mapping(), n_boot=50, seed=0)
    dest = tmp_path / "report.pdf"
    write_pdf(audit, dest)
    data = dest.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(data) > 500


def test_four_fifths_screen_is_not_raised_when_selection_rates_match() -> None:
    frame = cells_to_frame({"A": (40, 10, 10, 40), "B": (40, 10, 10, 40)})
    markdown = render_markdown(run_audit(frame, _mapping(), n_boot=50, seed=0))
    assert "four-fifths screen is not raised" in markdown
    assert "do not meaningfully differ" in markdown


def test_four_fifths_screen_is_undefined_when_nobody_is_selected() -> None:
    frame = cells_to_frame({"A": (0, 0, 40, 40), "B": (0, 0, 12, 72)})
    markdown = render_markdown(run_audit(frame, _mapping(), n_boot=50, seed=0))
    assert "four-fifths screen was not computable" in markdown


def test_perfect_classifier_is_named_as_an_impossibility_exception() -> None:
    frame = cells_to_frame({"A": (50, 0, 0, 50), "B": (20, 0, 0, 80)})
    markdown = render_markdown(run_audit(frame, _mapping(), n_boot=50, seed=0))
    assert "classified perfectly" in markdown


def test_one_vs_rest_is_named_in_the_header() -> None:
    frame = cells_to_frame({"A": (40, 10, 10, 40), "B": (12, 8, 8, 72)})
    # Keep some 0s so the label column has three values (0, 1, 2).
    zeros = frame.index[frame["y_true"] == 0][:5]
    frame.loc[zeros, "y_true"] = 2
    markdown = render_markdown(run_audit(frame, _mapping(), n_boot=50, seed=0))
    assert "one-vs-rest" in markdown


def test_small_n_groups_are_counted_in_the_header() -> None:
    frame = cells_to_frame({"A": (4, 1, 1, 4), "B": (40, 10, 10, 40)})
    markdown = render_markdown(run_audit(frame, _mapping(), n_boot=50, seed=0))
    assert "Small-n groups" in markdown


def test_auc_skip_reason_is_rendered_when_a_group_has_one_class() -> None:
    frame = cells_to_frame({"A": (40, 10, 10, 40), "B": (0, 5, 0, 45)})
    frame["score"] = 0.5
    markdown = render_markdown(run_audit(frame, _mapping(score="score"), n_boot=50, seed=0))
    assert "n/a (" in markdown
    assert "ROC-AUC" in markdown
