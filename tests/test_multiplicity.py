"""Simultaneous (max-t) verdicts across the headline contrasts, D14.

Each group-versus-reference contrast is a separate 95% claim. Across several groups and four
rates, some "clear" results appear by chance, so verdicts are judged against a band that
covers every contrast in the family at once.
"""

from __future__ import annotations

import numpy as np
import pytest

from faircheck import bootstrap
from faircheck.bootstrap import VERDICT_RATE_KEYS, bootstrap_audit
from faircheck.types import GroupCounts
from tests.conftest import HAND_COMPUTED_CELLS


def _counts(cells: list[list[int]]) -> GroupCounts:
    names = tuple((chr(ord("A") + i),) for i in range(len(cells)))
    return GroupCounts(group_names=("g",), groups=names, cells=np.array(cells, dtype=np.int64))


def _any_clear(result: bootstrap.BootstrapResult, n_groups: int) -> bool:
    return any(
        result.has_clear_contrast(key, group) is True
        for key in VERDICT_RATE_KEYS
        for group in range(n_groups)
        if group != result.reference_index
    )


def test_the_family_is_four_headline_rates_for_every_non_reference_group() -> None:
    result = bootstrap_audit(_counts([[60, 30, 40, 70]] * 3), n_boot=200, seed=0)
    assert result.adjustment == "max_t"
    assert result.family_size == len(VERDICT_RATE_KEYS) * 2
    assert np.isfinite(result.critical_value)


def test_adjusted_bands_are_at_least_as_wide_as_the_marginal_intervals() -> None:
    result = bootstrap_audit(_counts([[300, 150, 200, 350], [280, 160, 220, 340]]), seed=1)
    for key in VERDICT_RATE_KEYS:
        low, _, high = result.contrast_interval(key, 1)
        adj_low, adj_high = result.adjusted_contrast_ci[key][:, 1]
        assert adj_low <= low + 1e-9
        assert adj_high >= high - 1e-9


def test_critical_value_grows_with_the_number_of_comparisons() -> None:
    few = bootstrap_audit(_counts([[60, 30, 40, 70]] * 2), n_boot=1_000, seed=0)
    many = bootstrap_audit(_counts([[60, 30, 40, 70]] * 7), n_boot=1_000, seed=0)
    assert many.family_size > few.family_size
    assert many.critical_value > few.critical_value


def test_under_no_difference_the_adjusted_rule_keeps_false_findings_near_nominal() -> None:
    """Three identical groups, 8 contrasts per audit: the headline case for D14."""
    rng = np.random.default_rng(1)
    trials = 300
    adjusted_hits = unadjusted_hits = 0
    for trial in range(trials):
        cells = [rng.multinomial(200, [0.30, 0.15, 0.20, 0.35]).tolist() for _ in range(3)]
        counts = _counts(cells)
        adjusted_hits += _any_clear(bootstrap_audit(counts, n_boot=300, seed=trial), 3)
        unadjusted_hits += _any_clear(
            bootstrap_audit(counts, n_boot=300, seed=trial, adjust="none"), 3
        )
    assert adjusted_hits / trials <= 0.10
    assert unadjusted_hits / trials >= 0.15


def test_a_large_planted_gap_survives_the_adjustment() -> None:
    compas_like = [[237, 197, 884, 2682], [2696, 887, 181, 236]]
    result = bootstrap_audit(_counts(compas_like), n_boot=500, seed=0)
    for key in ("selection_rate", "tpr", "fpr"):
        assert result.has_clear_contrast(key, 1) is True, key


def test_hand_computed_case_keeps_its_verdicts() -> None:
    cells = [list(HAND_COMPUTED_CELLS["A"]), list(HAND_COMPUTED_CELLS["B"])]
    result = bootstrap_audit(_counts(cells), n_boot=1_000, seed=0)
    assert result.has_clear_contrast("selection_rate", 1) is True
    assert result.has_clear_contrast("tpr", 1) is False


def test_a_borderline_contrast_is_clear_unadjusted_but_not_after_adjustment() -> None:
    """Three groups of 1,000. Group C's TPR contrast has a marginal interval excluding zero."""
    counts = _counts([[300, 150, 200, 350], [280, 160, 220, 340], [330, 140, 170, 360]])
    plain = bootstrap_audit(counts, n_boot=2_000, seed=0, adjust="none")
    adjusted = bootstrap_audit(counts, n_boot=2_000, seed=0)
    assert plain.has_clear_contrast("tpr", 2) is True
    assert adjusted.has_clear_contrast("tpr", 2) is False
    assert adjusted.has_clear_unadjusted_contrast("tpr", 2) is True


def test_adjust_none_is_the_marginal_rule_and_records_no_family() -> None:
    counts = _counts([[60, 30, 40, 70], [50, 32, 50, 68], [66, 28, 34, 72]])
    result = bootstrap_audit(counts, n_boot=300, seed=0, adjust="none")
    assert result.adjustment == "none"
    assert result.family_size == 0
    assert np.isnan(result.critical_value)
    for key in VERDICT_RATE_KEYS:
        for group in range(3):
            low, _, high = result.contrast_interval(key, group)
            assert result.has_clear_contrast(key, group) is bool(low > 0 or high < 0)


def test_verdict_equals_the_adjusted_band_excluding_zero() -> None:
    counts = _counts([[60, 30, 40, 70], [50, 32, 50, 68], [66, 28, 34, 72]])
    result = bootstrap_audit(counts, n_boot=300, seed=2)
    for key in VERDICT_RATE_KEYS:
        for group in (1, 2):
            low, high = result.adjusted_contrast_ci[key][:, group]
            assert result.has_clear_contrast(key, group) is bool(low > 0 or high < 0)


def test_the_reference_group_is_never_clear() -> None:
    result = bootstrap_audit(_counts([[60, 30, 40, 70], [10, 5, 40, 120]]), n_boot=200, seed=0)
    for key in VERDICT_RATE_KEYS:
        assert result.has_clear_contrast(key, result.reference_index) is False


def test_an_undefined_contrast_has_no_verdict_and_does_not_break_the_others() -> None:
    # Group B has no positives, so its TPR (and the contrast) is undefined.
    counts = _counts([[60, 30, 40, 70], [0, 6, 0, 34]])
    result = bootstrap_audit(counts, n_boot=300, seed=0)
    assert result.has_clear_contrast("tpr", 1) is None
    assert result.has_clear_contrast("selection_rate", 1) is not None
    assert np.isfinite(result.critical_value)


def test_adjusted_bands_are_reproducible_and_follow_the_seed() -> None:
    counts = _counts([[60, 30, 40, 70], [50, 32, 50, 68], [66, 28, 34, 72]])
    first = bootstrap_audit(counts, n_boot=300, seed=5)
    again = bootstrap_audit(counts, n_boot=300, seed=5)
    other = bootstrap_audit(counts, n_boot=300, seed=6)
    assert first.critical_value == again.critical_value
    assert first.critical_value != other.critical_value


def test_unknown_adjustment_is_rejected() -> None:
    with pytest.raises(ValueError, match="adjust"):
        bootstrap_audit(_counts([[60, 30, 40, 70]] * 2), adjust="bonferroni")


def test_marginal_verdict_is_none_when_the_contrast_is_undefined() -> None:
    result = bootstrap_audit(_counts([[60, 30, 40, 70], [0, 6, 0, 34]]), n_boot=200, seed=0)
    assert result.has_clear_unadjusted_contrast("tpr", 1) is None


def test_a_single_group_has_no_family_and_no_verdict_to_adjust() -> None:
    result = bootstrap_audit(_counts([[60, 30, 40, 70]]), n_boot=200, seed=0)
    assert result.family_size == 0
    assert np.isnan(result.critical_value)
    assert result.has_clear_contrast("tpr", 0) is False


def test_constant_contrasts_keep_their_sign_instead_of_being_widened() -> None:
    """A catches every positive and B misses every one: the TPR contrast is -1 in every resample."""
    result = bootstrap_audit(_counts([[50, 0, 0, 50], [0, 0, 50, 50]]), n_boot=300, seed=0)
    assert result.has_clear_contrast("tpr", 1) is True
    assert list(result.adjusted_contrast_ci["tpr"][:, 1]) == [-1.0, -1.0]
    same = bootstrap_audit(_counts([[50, 0, 0, 50], [50, 0, 0, 50]]), n_boot=300, seed=0)
    assert same.has_clear_contrast("tpr", 1) is False
