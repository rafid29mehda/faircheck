"""Calibration: ROC-AUC skip, reliability bins, ECE direction, threshold sweep."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from faircheck.calibration import (
    auc_skip_reason,
    calibrate,
    expected_calibration_error,
    reliability_bins,
    roc_auc,
    threshold_sweep,
)
from faircheck.metrics import confusion_counts, per_group_rates
from faircheck.types import ColumnMapping
from faircheck.validate import validate

sklearn_metrics = pytest.importorskip("sklearn.metrics")


def test_roc_auc_perfect_ranking_is_one() -> None:
    y = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert roc_auc(y, s) == pytest.approx(1.0)


def test_roc_auc_reversed_ranking_is_zero() -> None:
    y = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert roc_auc(y, s) == pytest.approx(0.0)


def test_roc_auc_all_ties_is_one_half() -> None:
    y = np.array([0, 0, 1, 1], dtype=bool)
    s = np.array([0.4, 0.4, 0.4, 0.4])
    assert roc_auc(y, s) == pytest.approx(0.5)


def test_roc_auc_is_nan_when_a_class_is_missing() -> None:
    s = np.array([0.2, 0.5, 0.9])
    assert np.isnan(roc_auc(np.zeros(3, dtype=bool), s))
    assert np.isnan(roc_auc(np.ones(3, dtype=bool), s))


@pytest.mark.parametrize("seed", range(6))
def test_roc_auc_matches_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=80).astype(bool)
    if y.all() or not y.any():
        y[0] = False
        y[-1] = True
    s = rng.random(80)
    ours = roc_auc(y, s)
    theirs = sklearn_metrics.roc_auc_score(y, s)
    assert ours == pytest.approx(theirs)


def test_hand_computed_ece_two_bins() -> None:
    """Five points, two equal-width bins: ECE = 0.14 by hand.

    Bin [0, 0.5): scores 0.1, 0.2, 0.4 with labels 0, 0, 1
        mean score 0.7/3, positive rate 1/3, gap 0.1
    Bin [0.5, 1]: scores 0.7, 0.9 with labels 1, 1
        mean score 0.8, positive rate 1.0, gap 0.2
    ECE = (3/5)·0.1 + (2/5)·0.2 = 0.14
    """
    y = np.array([0, 0, 1, 1, 1], dtype=bool)
    s = np.array([0.1, 0.2, 0.4, 0.7, 0.9])
    bins = reliability_bins(y, s, n_bins=2)

    assert bins[0].count == 3
    assert bins[0].mean_score == pytest.approx(0.7 / 3)
    assert bins[0].positive_rate == pytest.approx(1 / 3)
    assert bins[1].count == 2
    assert bins[1].mean_score == pytest.approx(0.8)
    assert bins[1].positive_rate == pytest.approx(1.0)
    assert expected_calibration_error(y, s, n_bins=2) == pytest.approx(0.14)


def test_empty_bin_contributes_zero_to_ece() -> None:
    y = np.array([0, 1], dtype=bool)
    s = np.array([0.05, 0.95])
    bins = reliability_bins(y, s, n_bins=10)
    empty = [b for b in bins if b.count == 0]
    assert len(empty) == 8
    assert all(b.gap == 0.0 and np.isnan(b.mean_score) for b in empty)
    # Occupied bins: |0-0.05| and |1-0.95|, equal weight.
    assert expected_calibration_error(y, s, n_bins=10) == pytest.approx(0.05)


def test_perfectly_calibrated_scores_have_near_zero_ece() -> None:
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.05, 0.95, size=20_000)
    y = rng.random(scores.size) < scores
    assert expected_calibration_error(y, scores, n_bins=10) < 0.02


def test_ece_grows_as_scores_are_shifted() -> None:
    """Direction, not a number: a larger systematic bias produces a larger ECE.

    Scores stay inside [0.2, 0.8] so clipping cannot fold the shift back on itself.
    """
    rng = np.random.default_rng(1)
    scores = rng.uniform(0.2, 0.8, size=8_000)
    y = rng.random(scores.size) < scores
    ece = [
        expected_calibration_error(y, np.clip(scores + shift, 0.0, 1.0), n_bins=10)
        for shift in (0.0, 0.10, 0.20)
    ]
    assert ece[0] < ece[1] < ece[2]


def test_calibrate_skips_auc_for_a_single_class_group_and_keeps_the_other() -> None:
    y = np.array([1, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.3, 0.1, 0.55])
    groups = np.array(["A", "A", "A", "B", "B", "B", "B", "B"], dtype=object)
    result = calibrate(y, s, (groups,), ("group",), n_bins=5)

    skipped = result.by_label("A")
    kept = result.by_label("B")
    assert skipped.auc_skip_reason is not None
    assert "no negative" in skipped.auc_skip_reason
    assert np.isnan(skipped.roc_auc)
    assert kept.auc_skip_reason is None
    assert np.isfinite(kept.roc_auc)
    assert np.isfinite(skipped.ece)


def test_threshold_sweep_matches_confusion_counts_at_the_cut() -> None:
    rng = np.random.default_rng(2)
    n = 400
    score = rng.random(n)
    y = rng.random(n) < score
    group = rng.choice(["A", "B"], n)
    cut = 0.4
    pred = score >= cut

    mapping = ColumnMapping(
        label="y", prediction="pred", sensitive=("g",), positive_label=True, score="s"
    )
    data = validate(
        pd.DataFrame({"y": y.astype(int), "pred": pred.astype(int), "g": group, "s": score}),
        mapping,
    )
    counts = confusion_counts(data.y_true, data.y_pred, data.group_values, mapping.sensitive)
    rates = per_group_rates(counts.cells)

    sweep = threshold_sweep(y, score, (group,), ("g",), thresholds=[cut])
    assert sweep.group_labels == counts.labels()
    assert sweep.tpr[0].tolist() == pytest.approx(rates["tpr"].tolist())
    assert sweep.fpr[0].tolist() == pytest.approx(rates["fpr"].tolist())
    assert sweep.selection_rate[0].tolist() == pytest.approx(rates["selection_rate"].tolist())


def test_threshold_sweep_endpoints() -> None:
    y = np.array([1, 1, 0, 0], dtype=bool)
    s = np.array([0.9, 0.6, 0.4, 0.1])
    g = np.array(["A", "A", "A", "A"], dtype=object)
    sweep = threshold_sweep(y, s, (g,), ("g",), thresholds=[0.0, 0.5, 1.0])

    assert sweep.tpr[:, 0].tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert sweep.fpr[:, 0].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert sweep.selection_rate[:, 0].tolist() == pytest.approx([1.0, 0.5, 0.0])


def test_threshold_sweep_undefined_when_a_class_is_absent() -> None:
    y = np.zeros(6, dtype=bool)
    s = np.linspace(0.1, 0.9, 6)
    g = np.array(["C"] * 6, dtype=object)
    sweep = threshold_sweep(y, s, (g,), ("g",), thresholds=[0.3, 0.7])

    assert np.all(np.isnan(sweep.tpr))
    assert np.all(np.isfinite(sweep.fpr))


def test_auc_skip_reason_names_the_missing_class() -> None:
    assert "empty" in (auc_skip_reason(np.array([], dtype=bool)) or "")
    assert "no positive" in (auc_skip_reason(np.zeros(3, dtype=bool)) or "")
    assert "no negative" in (auc_skip_reason(np.ones(3, dtype=bool)) or "")
    assert auc_skip_reason(np.array([True, False])) is None


def test_calibration_rejects_inconsistent_inputs() -> None:
    y = np.array([0, 1], dtype=bool)
    s = np.array([0.2, 0.8])
    g = np.array(["A", "B"], dtype=object)

    with pytest.raises(ValueError, match="n_bins"):
        reliability_bins(y, s, n_bins=0)
    with pytest.raises(ValueError, match="equal length"):
        reliability_bins(y, s[:1])
    with pytest.raises(ValueError, match="equal length"):
        calibrate(y, s[:1], (g,), ("g",))
    with pytest.raises(ValueError, match="sensitive column"):
        calibrate(y, s, (g,), ())
    with pytest.raises(ValueError, match="equal length"):
        threshold_sweep(y, s[:1], (g,), ("g",))
    with pytest.raises(ValueError, match="sensitive column"):
        threshold_sweep(y, s, (g,), ())

    assert np.isnan(expected_calibration_error(np.array([], dtype=bool), np.array([])))
    result = calibrate(y, s, (g,), ("g",))
    with pytest.raises(KeyError):
        result.by_label("missing")
