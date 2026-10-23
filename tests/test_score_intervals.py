"""Row-level bootstrap intervals for the score metrics (ROC-AUC and ECE), D13.

These metrics are not functions of the four confusion cells, so the multinomial fast path
of D4 cannot cover them. The checks mirror ``test_bootstrap.py``: the fast implementation
must agree with a naive row-resampling reference, runs must be reproducible, and an
interval must narrow as data grows.
"""

from __future__ import annotations

import numpy as np
import pytest

from faircheck.calibration import calibrate, expected_calibration_error, roc_auc


def _one_group(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrated scores (Y ~ Bernoulli(score)) for a single group."""
    rng = np.random.default_rng(seed)
    score = rng.beta(2.0, 3.0, size=n)
    y = rng.random(n) < score
    group = np.array(["A"] * n, dtype=object)
    return y, score, group


def _naive_interval(
    metric: str, y: np.ndarray, score: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float]:
    """Resample rows with replacement and recompute the metric from scratch."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        rows = rng.integers(0, y.size, size=y.size)
        if metric == "auc":
            values.append(roc_auc(y[rows], score[rows]))
        else:
            values.append(expected_calibration_error(y[rows], score[rows], n_bins=10))
    low, high = np.nanquantile(values, [0.025, 0.975])
    return float(low), float(high)


def test_auc_interval_brackets_the_point_estimate() -> None:
    y, s, g = _one_group(3_000, seed=0)
    group = calibrate(y, s, (g,), ("g",), n_boot=300, seed=0).groups[0]
    low, high = group.roc_auc_ci
    assert low < group.roc_auc < high


def test_auc_interval_matches_naive_row_resampling() -> None:
    y, s, g = _one_group(1_500, seed=1)
    ours = calibrate(y, s, (g,), ("g",), n_boot=800, seed=0).groups[0].roc_auc_ci
    naive = _naive_interval("auc", y, s, n_boot=800, seed=123)
    assert ours[0] == pytest.approx(naive[0], abs=0.006)
    assert ours[1] == pytest.approx(naive[1], abs=0.006)


def test_auc_interval_handles_scores_tied_across_classes() -> None:
    """Rounded scores put positives and negatives on identical values; ties count 0.5."""
    y, s, g = _one_group(1_500, seed=11)
    s = np.round(s, 1)
    group = calibrate(y, s, (g,), ("g",), n_boot=800, seed=0).groups[0]
    naive = _naive_interval("auc", y, s, n_boot=800, seed=123)
    assert group.roc_auc == pytest.approx(roc_auc(y, s))
    assert group.roc_auc_ci[0] == pytest.approx(naive[0], abs=0.006)
    assert group.roc_auc_ci[1] == pytest.approx(naive[1], abs=0.006)


def test_ece_interval_matches_naive_row_resampling() -> None:
    y, s, g = _one_group(1_500, seed=2)
    ours = calibrate(y, s, (g,), ("g",), n_boot=800, seed=0).groups[0].ece_ci
    naive = _naive_interval("ece", y, s, n_boot=800, seed=123)
    assert ours[0] == pytest.approx(naive[0], abs=0.004)
    assert ours[1] == pytest.approx(naive[1], abs=0.004)


def test_ece_interval_is_ordered_and_non_negative() -> None:
    y, s, g = _one_group(800, seed=3)
    low, high = calibrate(y, s, (g,), ("g",), n_boot=200, seed=0).groups[0].ece_ci
    assert 0.0 <= low <= high


def test_same_seed_gives_identical_intervals_and_a_new_seed_changes_them() -> None:
    y, s, g = _one_group(600, seed=4)
    first = calibrate(y, s, (g,), ("g",), n_boot=200, seed=7).groups[0]
    again = calibrate(y, s, (g,), ("g",), n_boot=200, seed=7).groups[0]
    other = calibrate(y, s, (g,), ("g",), n_boot=200, seed=8).groups[0]
    assert first.roc_auc_ci == again.roc_auc_ci
    assert first.ece_ci == again.ece_ci
    assert first.roc_auc_ci != other.roc_auc_ci


def test_intervals_narrow_as_the_group_grows() -> None:
    small = calibrate(*_reorder(_one_group(400, seed=5)), n_boot=300, seed=0).groups[0]
    large = calibrate(*_reorder(_one_group(6_400, seed=5)), n_boot=300, seed=0).groups[0]
    small_width = small.roc_auc_ci[1] - small.roc_auc_ci[0]
    large_width = large.roc_auc_ci[1] - large.roc_auc_ci[0]
    assert large_width < 0.6 * small_width


def _reorder(
    triple: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[str, ...]]:
    y, s, g = triple
    return y, s, (g,), ("g",)


def test_single_class_group_has_no_auc_interval_but_keeps_an_ece_interval() -> None:
    rng = np.random.default_rng(6)
    s = rng.random(60)
    y = np.ones(60, dtype=bool)
    g = np.array(["A"] * 60, dtype=object)
    group = calibrate(y, s, (g,), ("g",), n_boot=100, seed=0).groups[0]
    assert all(np.isnan(v) for v in group.roc_auc_ci)
    assert group.auc_undefined_fraction == 1.0
    assert all(np.isfinite(v) for v in group.ece_ci)


def test_resamples_that_lose_a_class_are_skipped_and_counted() -> None:
    """One positive in twelve rows: about (11/12)^12 = 35% of resamples have no positive."""
    y = np.array([1] + [0] * 11, dtype=bool)
    s = np.linspace(0.05, 0.95, 12)
    g = np.array(["A"] * 12, dtype=object)
    group = calibrate(y, s, (g,), ("g",), n_boot=2_000, seed=0).groups[0]
    assert 0.25 < group.auc_undefined_fraction < 0.45
    assert all(np.isfinite(v) for v in group.roc_auc_ci)


def test_n_boot_zero_turns_intervals_off_without_changing_point_estimates() -> None:
    y, s, g = _one_group(500, seed=7)
    off = calibrate(y, s, (g,), ("g",), n_boot=0).groups[0]
    on = calibrate(y, s, (g,), ("g",), n_boot=100, seed=0).groups[0]
    assert all(np.isnan(v) for v in off.roc_auc_ci + off.ece_ci)
    assert off.roc_auc == on.roc_auc
    assert off.ece == on.ece


def test_negative_n_boot_is_rejected() -> None:
    y, s, g = _one_group(50, seed=8)
    with pytest.raises(ValueError, match="n_boot"):
        calibrate(y, s, (g,), ("g",), n_boot=-1)


def test_every_group_and_the_overall_row_get_intervals() -> None:
    rng = np.random.default_rng(9)
    n = 900
    s = rng.random(n)
    y = rng.random(n) < s
    g = rng.choice(np.array(["A", "B", "C"], dtype=object), n)
    result = calibrate(y, s, (g,), ("g",), n_boot=100, seed=0)
    for row in (*result.groups, result.overall):
        assert all(np.isfinite(v) for v in row.roc_auc_ci)
        assert all(np.isfinite(v) for v in row.ece_ci)
    assert result.n_boot == 100
    assert result.seed == 0
