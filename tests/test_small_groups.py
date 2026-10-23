"""Size-adaptive intervals for small groups, D15.

The resampling bootstrap can only reproduce cells it has seen, so a group with a zero (or
tiny) cell gets a degenerate or far-too-narrow interval. Such groups use a Dirichlet
posterior over the four cells instead. Large groups keep the D4 bootstrap unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from faircheck.bootstrap import (
    MIN_CELL_FOR_BOOTSTRAP,
    POSTERIOR_PRIOR,
    bootstrap_audit,
    choose_methods,
    posterior_counts,
)
from faircheck.types import GroupCounts


def _counts(cells: list[list[int]]) -> GroupCounts:
    names = tuple((chr(ord("A") + i),) for i in range(len(cells)))
    return GroupCounts(group_names=("g",), groups=names, cells=np.array(cells, dtype=np.int64))


LARGE = [300, 150, 200, 350]


def test_the_rule_switches_on_the_smallest_cell_not_the_group_size() -> None:
    below = MIN_CELL_FOR_BOOTSTRAP - 1
    counts = _counts(
        [
            LARGE,
            [900, 800, below, 700],
            [900, 800, MIN_CELL_FOR_BOOTSTRAP, 700],
            [3, 40, 60, 200],
        ]
    )
    assert choose_methods(counts, "adaptive") == (
        "bootstrap",
        "dirichlet",
        "bootstrap",
        "dirichlet",
    )


def test_a_big_group_with_one_tiny_cell_still_gets_the_posterior() -> None:
    counts = _counts([[5000, 2000, 2, 3000]])
    assert choose_methods(counts, "adaptive") == ("dirichlet",)


def test_forced_methods_apply_to_every_group() -> None:
    counts = _counts([LARGE, [3, 40, 60, 200]])
    assert choose_methods(counts, "bootstrap") == ("bootstrap", "bootstrap")
    assert choose_methods(counts, "dirichlet") == ("dirichlet", "dirichlet")


def test_unknown_interval_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="interval_method"):
        bootstrap_audit(_counts([LARGE, LARGE]), interval_method="wilson")
    with pytest.raises(ValueError, match="interval_method"):
        choose_methods(_counts([LARGE]), "wilson")


def test_posterior_draws_are_valid_cell_proportions() -> None:
    counts = _counts([LARGE, [0, 20, 15, 65]])
    draws = posterior_counts(counts, n_boot=500, seed=0)
    assert draws.shape == (500, 2, 4)
    assert np.all(draws > 0.0)
    assert np.allclose(draws.sum(axis=-1), 1.0)
    # The prior only matters at the margin: draws centre on the observed proportions.
    expected = (np.array(LARGE) + POSTERIOR_PRIOR) / (sum(LARGE) + 4 * POSTERIOR_PRIOR)
    assert draws[:, 0, :].mean(axis=0) == pytest.approx(expected, abs=0.01)


def test_posterior_draws_follow_the_seed() -> None:
    counts = _counts([[3, 40, 60, 200]])
    assert np.array_equal(
        posterior_counts(counts, n_boot=50, seed=4), posterior_counts(counts, n_boot=50, seed=4)
    )
    assert not np.array_equal(
        posterior_counts(counts, n_boot=50, seed=4), posterior_counts(counts, n_boot=50, seed=5)
    )


def test_all_large_groups_give_exactly_the_old_numbers() -> None:
    counts = _counts([LARGE, [280, 160, 220, 340], [330, 140, 170, 360]])
    adaptive = bootstrap_audit(counts, n_boot=400, seed=3)
    classic = bootstrap_audit(counts, n_boot=400, seed=3, interval_method="bootstrap")
    assert adaptive.interval_methods == ("bootstrap",) * 3
    for key in adaptive.rate_ci:
        assert np.array_equal(adaptive.rate_ci[key], classic.rate_ci[key], equal_nan=True)
        assert np.array_equal(adaptive.contrast_ci[key], classic.contrast_ci[key], equal_nan=True)
    assert adaptive.critical_value == classic.critical_value


def test_a_large_group_is_unaffected_by_a_small_neighbour() -> None:
    mixed = bootstrap_audit(_counts([LARGE, [3, 40, 60, 200]]), n_boot=400, seed=3)
    classic = bootstrap_audit(
        _counts([LARGE, [3, 40, 60, 200]]), n_boot=400, seed=3, interval_method="bootstrap"
    )
    assert mixed.interval_methods == ("bootstrap", "dirichlet")
    for key in mixed.rate_ci:
        assert np.array_equal(mixed.rate_ci[key][:, 0], classic.rate_ci[key][:, 0], equal_nan=True)


def test_a_zero_cell_gets_a_real_interval_instead_of_a_point() -> None:
    zero_tp = [0, 20, 15, 65]
    classic = bootstrap_audit(_counts([zero_tp]), n_boot=1_000, seed=0, interval_method="bootstrap")
    adaptive = bootstrap_audit(_counts([zero_tp]), n_boot=1_000, seed=0)
    low, _, high = classic.rate_ci["tpr"][:, 0]
    assert low == high == 0.0
    low, _, high = adaptive.rate_ci["tpr"][:, 0]
    assert low < 0.001
    assert 0.05 < high < 0.35


def test_small_group_coverage_beats_the_plain_bootstrap() -> None:
    """True TPR 0.95 in a group of 40: the case where the old interval fails most."""
    rng = np.random.default_rng(0)
    cell_p = [0.3 * 0.95, 0.7 * 0.2, 0.3 * 0.05, 0.7 * 0.8]
    hits = {"adaptive": 0, "bootstrap": 0}
    used = 0
    for trial in range(500):
        cells = rng.multinomial(40, cell_p).tolist()
        if cells[0] + cells[2] == 0:
            continue
        used += 1
        counts = _counts([cells])
        for method, name in (("adaptive", "adaptive"), ("bootstrap", "bootstrap")):
            result = bootstrap_audit(counts, n_boot=300, seed=trial, interval_method=method)
            low, _, high = result.rate_ci["tpr"][:, 0]
            hits[name] += bool(low <= 0.95 <= high)
    assert hits["adaptive"] / used >= 0.93
    assert hits["bootstrap"] / used < 0.60


def test_undefined_rates_stay_undefined_under_the_posterior() -> None:
    """A group with no positives has no TPR; the prior must not invent one (D3)."""
    result = bootstrap_audit(_counts([LARGE, [0, 6, 0, 34]]), n_boot=300, seed=0)
    assert result.interval_methods[1] == "dirichlet"
    assert np.isnan(result.rates["tpr"][1])
    assert np.all(np.isnan(result.rate_ci["tpr"][:, 1]))
    assert result.rate_undefined_fraction["tpr"][1] == 1.0
    assert result.has_clear_contrast("tpr", 1) is None
    assert result.has_clear_contrast("selection_rate", 1) is not None


def test_posterior_groups_feed_the_simultaneous_verdicts() -> None:
    """A small group missing every positive is a clear TPR shortfall against a large one."""
    counts = _counts([[800, 200, 200, 800], [0, 10, 12, 60]])
    result = bootstrap_audit(counts, n_boot=1_000, seed=0)
    assert result.interval_methods == ("bootstrap", "dirichlet")
    assert result.has_clear_contrast("tpr", 0) is False  # reference group
    reference = result.reference_index
    other = 1 - reference
    assert result.has_clear_contrast("tpr", other) is True
