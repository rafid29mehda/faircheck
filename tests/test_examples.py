"""Bundled examples: committed CSVs, synthetic generators, and the two pedagogical claims.

Tests read the files under ``examples/data/``. They must not download anything.
The synthetic generators are re-run here so a drift between script and CSV is a failure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from examples.catalog import EXAMPLES, EXAMPLES_BY_KEY, ExampleSpec
from examples.make_compas_like import build_frame as build_compas_like
from examples.make_near_fair import build_frame as build_near_fair
from faircheck.__main__ import main
from faircheck.report import run_audit
from faircheck.validate import read_csv


@pytest.mark.parametrize("spec", EXAMPLES, ids=lambda spec: spec.key)
def test_example_csv_exists_and_audits(spec: ExampleSpec) -> None:
    assert spec.path.is_file(), f"missing {spec.path}; run the matching examples/make_*.py"
    frame = read_csv(spec.path)
    assert len(frame) >= 1_000
    for column in spec.mapping.columns:
        assert column in frame.columns
    assert spec.mapping.score is not None
    scores = frame[spec.mapping.score].to_numpy(dtype=float)
    assert float(np.nanmin(scores)) >= 0.0
    assert float(np.nanmax(scores)) <= 1.0
    audit = run_audit(frame, spec.mapping, n_boot=50, seed=0)
    assert audit.counts.n_groups >= 2
    assert audit.calibration is not None


def test_compas_like_csv_matches_the_generator() -> None:
    spec = EXAMPLES_BY_KEY["compas_like"]
    built = build_compas_like()
    committed = pd.read_csv(spec.path)
    assert list(committed.columns) == list(built.columns)
    assert (committed["y_true"].to_numpy() == built["y_true"].to_numpy()).all()
    assert (committed["y_pred"].to_numpy() == built["y_pred"].to_numpy()).all()
    assert (committed["group"].astype(str).to_numpy() == built["group"].to_numpy()).all()
    np.testing.assert_allclose(committed["score"], built["score"], atol=1e-6, rtol=0)


def test_near_fair_csv_matches_the_generator() -> None:
    spec = EXAMPLES_BY_KEY["near_fair"]
    built = build_near_fair()
    committed = pd.read_csv(spec.path)
    assert (committed["y_true"].to_numpy() == built["y_true"].to_numpy()).all()
    np.testing.assert_allclose(committed["score"], built["score"], atol=1e-6, rtol=0)


def test_compas_like_is_calibrated_with_unequal_error_rates() -> None:
    spec = EXAMPLES_BY_KEY["compas_like"]
    frame = read_csv(spec.path)
    audit = run_audit(frame, spec.mapping, n_boot=400, seed=0)
    assert audit.calibration is not None
    for group in audit.calibration.groups:
        assert group.ece < 0.05, f"{group.label} ECE {group.ece} is not calibrated"
    assert audit.impossibility.base_rate_spread > 0.20
    assert audit.impossibility.tradeoff_applies is True
    other = next(i for i in range(audit.counts.n_groups) if i != audit.bootstrap.reference_index)
    assert audit.bootstrap.has_clear_contrast("tpr", other) is True
    assert audit.bootstrap.has_clear_contrast("fpr", other) is True


def test_near_fair_contrasts_cover_zero() -> None:
    spec = EXAMPLES_BY_KEY["near_fair"]
    frame = read_csv(spec.path)
    audit = run_audit(frame, spec.mapping, n_boot=400, seed=0)
    other = next(i for i in range(audit.counts.n_groups) if i != audit.bootstrap.reference_index)
    for key in ("selection_rate", "tpr", "fpr", "precision"):
        assert audit.bootstrap.has_clear_contrast(key, other) is False, key
    # Sampling noise still makes max-min sit away from zero (D11).
    gap_low, _, _ = audit.bootstrap.gap_interval("demographic_parity_difference")
    assert gap_low > 0.0


@pytest.mark.parametrize("key", ["acs_income", "adult"])
def test_real_examples_keep_sex_and_race(key: str) -> None:
    frame = read_csv(EXAMPLES_BY_KEY[key].path)
    sexes = set(frame["sex"].astype(str).unique())
    assert {"Female", "Male"}.issubset(sexes)
    assert frame["race"].nunique() >= 2


def test_cli_smokes_on_the_bundled_compas_like_example(tmp_path: Path) -> None:
    spec = EXAMPLES_BY_KEY["compas_like"]
    out = tmp_path / "compas_like.md"
    assert (
        main(
            [
                "report",
                str(spec.path),
                "--label",
                "y_true",
                "--pred",
                "y_pred",
                "--group",
                "group",
                "--score",
                "score",
                "--positive",
                "1",
                "--n-boot",
                "50",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    text = out.read_text(encoding="utf-8")
    assert "# FairCheck report" in text
    assert "8000" in text or "8,000" in text


def test_committed_compas_like_report_matches_the_cli(tmp_path: Path) -> None:
    """The README quotes this file. If the library moves, regenerate it; do not edit numbers."""
    spec = EXAMPLES_BY_KEY["compas_like"]
    out = tmp_path / "compas_like.md"
    assert (
        main(
            [
                "report",
                str(spec.path),
                "--label",
                "y_true",
                "--pred",
                "y_pred",
                "--group",
                "group",
                "--score",
                "score",
                "--positive",
                "1",
                "--n-boot",
                "1000",
                "--seed",
                "0",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    committed = Path(__file__).resolve().parent.parent / "examples" / "reports" / "compas_like.md"
    assert committed.read_text(encoding="utf-8") == out.read_text(encoding="utf-8")
