"""Bootstrap behaviour: exactness of the fast path, reproducibility, and honest verdicts.

The multinomial shortcut in :mod:`faircheck.bootstrap` is the one piece of cleverness in the
project, so it is checked three independent ways: against a naive row-resampling reference,
against a closed-form interval for a single proportion, and by simulated coverage.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy import stats

from faircheck import metrics
from faircheck.bootstrap import (
    DEFAULT_QUANTILES,
    FULL,
    STRATIFIED,
    bootstrap_audit,
    bootstrap_counts,
)
from faircheck.types import GroupCounts

Z_95 = 1.959963984540054


def counts_from(cells: list[tuple[int, int, int, int]]) -> GroupCounts:
    names = tuple(chr(ord("A") + index) for index in range(len(cells)))
    return GroupCounts(
        group_names=("group",),
        groups=tuple((name,) for name in names),
        cells=np.array(cells, dtype=np.int64),
    )


def rows_from(counts: GroupCounts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand a count table back into the row-level arrays it summarises."""
    y_true: list[bool] = []
    y_pred: list[bool] = []
    codes: list[int] = []
    for index, (tp, fp, fn, tn) in enumerate(counts.cells):
        for truth, prediction, repeats in (
            (True, True, tp),
            (False, True, fp),
            (True, False, fn),
            (False, False, tn),
        ):
            y_true.extend([truth] * int(repeats))
            y_pred.extend([prediction] * int(repeats))
            codes.extend([index] * int(repeats))
    return np.array(y_true), np.array(y_pred), np.array(codes)


def naive_bootstrap_counts(counts: GroupCounts, *, n_boot: int, seed: int) -> np.ndarray:
    """Reference implementation: literally resample rows within each group.

    Deliberately the slow, obvious version -- O(n_boot x n_rows) -- so that agreement with
    the multinomial fast path is evidence about the fast path rather than a restatement of it.
    """
    y_true, y_pred, codes = rows_from(counts)
    rng = np.random.default_rng(seed)
    resampled = np.empty((n_boot, counts.n_groups, 4), dtype=np.int64)

    for group in range(counts.n_groups):
        members = np.flatnonzero(codes == group)
        for draw in range(n_boot):
            rows = members[rng.integers(0, len(members), len(members))]
            truth, prediction = y_true[rows], y_pred[rows]
            resampled[draw, group] = (
                int(np.sum(truth & prediction)),
                int(np.sum(~truth & prediction)),
                int(np.sum(truth & ~prediction)),
                int(np.sum(~truth & ~prediction)),
            )
    return resampled


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Closed-form score interval for a single proportion (Wilson 1927)."""
    phat = successes / total
    denominator = 1.0 + Z_95**2 / total
    centre = (phat + Z_95**2 / (2 * total)) / denominator
    halfwidth = Z_95 * np.sqrt(phat * (1 - phat) / total + Z_95**2 / (4 * total**2)) / denominator
    return centre - halfwidth, centre + halfwidth


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------


def test_same_seed_gives_identical_resamples() -> None:
    counts = counts_from([(40, 10, 10, 40), (12, 8, 8, 72)])

    first = bootstrap_counts(counts, n_boot=200, seed=11)
    second = bootstrap_counts(counts, n_boot=200, seed=11)

    assert np.array_equal(first, second)


def test_different_seed_gives_different_resamples() -> None:
    counts = counts_from([(40, 10, 10, 40), (12, 8, 8, 72)])

    assert not np.array_equal(
        bootstrap_counts(counts, n_boot=200, seed=11),
        bootstrap_counts(counts, n_boot=200, seed=12),
    )


def test_whole_audit_is_reproducible() -> None:
    counts = counts_from([(40, 10, 10, 40), (12, 8, 8, 72)])

    first = bootstrap_audit(counts, n_boot=500, seed=3)
    second = bootstrap_audit(counts, n_boot=500, seed=3)

    for key in first.rate_ci:
        assert np.array_equal(first.rate_ci[key], second.rate_ci[key], equal_nan=True)
    for key in first.gap_ci:
        assert np.array_equal(first.gap_ci[key], second.gap_ci[key], equal_nan=True)


def test_resampled_counts_preserve_group_sizes_when_stratified() -> None:
    counts = counts_from([(40, 10, 10, 40), (12, 8, 8, 72)])

    resampled = bootstrap_counts(counts, n_boot=300, seed=0, scheme=STRATIFIED)

    assert np.array_equal(resampled.sum(axis=-1), np.tile(counts.sizes, (300, 1)))


def test_full_scheme_preserves_the_total_but_not_group_sizes() -> None:
    counts = counts_from([(40, 10, 10, 40), (12, 8, 8, 72)])

    resampled = bootstrap_counts(counts, n_boot=300, seed=0, scheme=FULL)

    assert np.all(resampled.sum(axis=(1, 2)) == counts.sizes.sum())
    assert not np.array_equal(resampled.sum(axis=-1), np.tile(counts.sizes, (300, 1)))


# --------------------------------------------------------------------------------------
# The fast path is exact
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_multinomial_fast_path_matches_naive_row_resampling(seed: int) -> None:
    """Same distribution, checked per group and per rate with a two-sample KS test."""
    counts = counts_from([(40, 10, 10, 40), (12, 8, 8, 72)])
    n_boot = 3000

    fast = metrics.per_group_rates(bootstrap_counts(counts, n_boot=n_boot, seed=seed))
    slow = metrics.per_group_rates(naive_bootstrap_counts(counts, n_boot=n_boot, seed=seed + 500))

    for key in ("selection_rate", "tpr", "fpr", "precision", "accuracy"):
        for group in range(counts.n_groups):
            result = stats.ks_2samp(fast[key][:, group], slow[key][:, group])
            assert result.pvalue > 1e-4, f"{key} group {group}: KS p={result.pvalue:.2e}"


def test_fast_path_matches_naive_on_means_and_spreads() -> None:
    counts = counts_from([(40, 10, 10, 40), (12, 8, 8, 72)])
    n_boot = 4000

    fast = metrics.per_group_rates(bootstrap_counts(counts, n_boot=n_boot, seed=7))
    slow = metrics.per_group_rates(naive_bootstrap_counts(counts, n_boot=n_boot, seed=99))

    for key in ("selection_rate", "tpr", "fpr"):
        assert fast[key].mean(axis=0) == pytest.approx(slow[key].mean(axis=0), abs=0.01)
        assert fast[key].std(axis=0) == pytest.approx(slow[key].std(axis=0), abs=0.01)


def test_single_proportion_interval_agrees_with_wilson() -> None:
    """A selection rate is just a proportion, so a closed form exists to check against."""
    for selected, total in [(50, 100), (20, 100), (5, 100), (200, 1000)]:
        counts = counts_from([(0, selected, 0, total - selected)])
        result = bootstrap_audit(counts, n_boot=4000, seed=1)

        low, point, high = result.rate_interval("selection_rate", 0)
        expected_low, expected_high = wilson_interval(selected, total)

        assert point == pytest.approx(selected / total)
        assert low == pytest.approx(expected_low, abs=0.03)
        assert high == pytest.approx(expected_high, abs=0.03)


def test_simulated_coverage_is_close_to_the_nominal_level() -> None:
    """Across repeated draws from a known truth, ~95% of intervals should contain it.

    This is the property a confidence interval is *for*, and it is cheap to check here
    because a resampled confusion table is itself a multinomial draw.
    """
    truth = np.array([0.30, 0.20, 0.10, 0.40])  # TP, FP, FN, TN
    true_selection_rate = truth[0] + truth[1]
    rng = np.random.default_rng(20260921)
    n_rows, trials = 500, 300

    contained = 0
    for trial in range(trials):
        observed = rng.multinomial(n_rows, truth)
        counts = counts_from([tuple(observed.tolist())])
        low, _, high = bootstrap_audit(counts, n_boot=400, seed=trial).rate_interval(
            "selection_rate", 0
        )
        contained += int(low <= true_selection_rate <= high)

    coverage = contained / trials
    assert 0.90 <= coverage <= 1.0, f"coverage {coverage:.3f} is not close to 0.95"


# --------------------------------------------------------------------------------------
# Undefined metrics
# --------------------------------------------------------------------------------------


def test_permanently_undefined_rate_has_an_undefined_interval() -> None:
    counts = counts_from([(40, 10, 10, 40), (0, 5, 0, 45)])

    result = bootstrap_audit(counts, n_boot=500, seed=0)

    low, point, high = result.rate_interval("tpr", 1)
    assert np.isnan(point) and np.isnan(low) and np.isnan(high)
    # No resample can invent positives for a group that has none.
    assert result.rate_undefined_fraction["tpr"][1] == pytest.approx(1.0)
    assert np.isnan(result.gap_interval("equalized_odds_difference")[0])


def test_occasionally_undefined_rate_keeps_its_interval_and_reports_the_fraction() -> None:
    """A group with two positives loses some resamples to zero positives.

    Throwing the whole interval away would be less informative than reporting it beside the
    proportion of resamples in which the metric did not exist.
    """
    counts = counts_from([(40, 10, 10, 40), (1, 5, 1, 45)])

    result = bootstrap_audit(counts, n_boot=2000, seed=0)

    low, point, high = result.rate_interval("tpr", 1)
    assert np.isfinite(low) and np.isfinite(point) and np.isfinite(high)
    fraction = result.rate_undefined_fraction["tpr"][1]
    assert 0.0 < fraction < 1.0


def test_defined_rates_report_no_undefined_resamples() -> None:
    counts = counts_from([(40, 10, 10, 40), (12, 8, 8, 72)])

    result = bootstrap_audit(counts, n_boot=500, seed=0)

    for key in ("selection_rate", "tpr", "fpr", "precision", "accuracy"):
        assert np.all(result.rate_undefined_fraction[key] == 0.0)


# --------------------------------------------------------------------------------------
# Verdicts: why max-min gaps cannot support an evidence claim
# --------------------------------------------------------------------------------------


def test_max_minus_min_gap_is_positive_even_when_all_groups_are_identical() -> None:
    """The trap this module exists to avoid (docs/DECISIONS.md D11).

    Three groups drawn from one distribution have no true disparity, yet ``max - min`` is
    strictly positive and its interval sits away from zero. A tool that read "CI excludes 0"
    as evidence would report a disparity here -- and in every dataset it ever saw.
    """
    truth = np.array([0.25, 0.25, 0.25, 0.25])
    rng = np.random.default_rng(5)
    counts = counts_from([tuple(rng.multinomial(2000, truth).tolist()) for _ in range(3)])

    result = bootstrap_audit(counts, n_boot=2000, seed=0)

    gap_low, gap_point, _ = result.gap_interval("demographic_parity_difference")
    assert gap_point > 0.0
    assert gap_low > 0.0  # non-negative by construction, so zero is unreachable

    # The signed contrasts, which is what the verdict actually uses, correctly find nothing.
    for group in range(counts.n_groups):
        if group == result.reference_index:
            continue
        low, _, high = result.contrast_interval("selection_rate", group)
        assert low < 0.0 < high
        assert result.has_clear_contrast("selection_rate", group) is False


def test_contrast_detects_a_real_difference() -> None:
    counts = counts_from([(400, 100, 100, 400), (120, 80, 80, 720)])

    result = bootstrap_audit(counts, n_boot=2000, seed=0)

    other = 1 - result.reference_index
    low, point, high = result.contrast_interval("selection_rate", other)
    assert result.has_clear_contrast("selection_rate", other) is True
    assert not (low <= 0.0 <= high)
    assert point == pytest.approx(-0.30) or point == pytest.approx(0.30)


def test_contrast_against_the_reference_group_is_exactly_zero() -> None:
    counts = counts_from([(40, 10, 10, 40), (12, 8, 8, 72)])

    result = bootstrap_audit(counts, n_boot=200, seed=0)

    reference = result.reference_index
    assert result.contrasts["selection_rate"][reference] == 0.0
    assert result.has_clear_contrast("selection_rate", reference) is False


def test_reference_group_defaults_to_the_largest() -> None:
    counts = counts_from([(4, 1, 1, 4), (120, 80, 80, 720)])

    result = bootstrap_audit(counts, n_boot=100, seed=0)

    assert result.reference_index == 1
    assert result.reference_label == "B"


def test_undefined_contrast_reports_none_not_false() -> None:
    counts = counts_from([(40, 10, 10, 40), (0, 5, 0, 45)])

    result = bootstrap_audit(counts, n_boot=200, seed=0)

    assert result.has_clear_contrast("tpr", 1) is None


# --------------------------------------------------------------------------------------
# Intervals bracket the point estimate; arguments are validated
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", [STRATIFIED, FULL])
def test_intervals_bracket_the_point_estimate(scheme: str) -> None:
    counts = counts_from([(400, 100, 100, 400), (120, 80, 80, 720)])

    result = bootstrap_audit(counts, n_boot=2000, seed=0, scheme=scheme)

    for spec in metrics.RATES:
        for group in range(counts.n_groups):
            low, point, high = result.rate_interval(spec.key, group)
            assert low <= point <= high, f"{spec.key} group {group}"


def test_quantile_axis_is_ordered_as_requested() -> None:
    counts = counts_from([(400, 100, 100, 400), (120, 80, 80, 720)])

    result = bootstrap_audit(counts, n_boot=500, seed=0, quantiles=(0.1, 0.5, 0.9))

    assert result.quantiles == (0.1, 0.5, 0.9)
    lower = result.rate_ci["selection_rate"][0]
    upper = result.rate_ci["selection_rate"][2]
    assert np.all(lower <= upper)
    # A 80% interval must sit inside the 95% one.
    wide = bootstrap_audit(counts, n_boot=500, seed=0, quantiles=DEFAULT_QUANTILES)
    assert np.all(wide.rate_ci["selection_rate"][0] <= lower)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_boot": 0}, "at least 1"),
        ({"scheme": "jackknife"}, "scheme must be one of"),
        ({"quantiles": ()}, "At least one quantile"),
        ({"quantiles": (0.5, 1.5)}, "strictly between 0 and 1"),
        ({"quantiles": (0.9, 0.1)}, "ascending"),
        ({"reference_index": 9}, "outside the"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict[str, object], message: str) -> None:
    counts = counts_from([(40, 10, 10, 40), (12, 8, 8, 72)])

    with pytest.raises(ValueError, match=message):
        bootstrap_audit(counts, **kwargs)  # type: ignore[arg-type]


def test_cost_does_not_grow_with_the_number_of_rows() -> None:
    """The point of the multinomial shortcut: 100x the rows, same work.

    A loose bound rather than a tight one, because CI machines are noisy -- this is here to
    catch a regression to row-level resampling, which would be ~100x slower.
    """
    small = counts_from([(400, 100, 100, 400), (120, 80, 80, 720)])
    large = counts_from([(40_000, 10_000, 10_000, 40_000), (12_000, 8_000, 8_000, 72_000)])

    def elapsed(counts: GroupCounts) -> float:
        start = time.perf_counter()
        bootstrap_audit(counts, n_boot=1000, seed=0)
        return time.perf_counter() - start

    elapsed(small)  # warm up numpy
    small_time, large_time = elapsed(small), elapsed(large)

    assert large_time < max(4.0 * small_time, 0.5)
