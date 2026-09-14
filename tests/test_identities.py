"""Algebraic identities that must hold between the rates.

These are stronger than per-metric tests: they constrain the rates *jointly*, so a
plausible-looking error in one denominator shows up immediately. The Chouldechova
identity doubles as the engine behind the impossibility explainer.
"""

from __future__ import annotations

import numpy as np
import pytest

from faircheck import metrics


def random_cells(seed: int, n_groups: int = 4, low: int = 5, high: int = 500) -> np.ndarray:
    """Random confusion tables with every cell non-empty, so all rates are defined."""
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, size=(n_groups, 4), dtype=np.int64)


@pytest.mark.parametrize("seed", range(8))
def test_fnr_is_one_minus_tpr(seed: int) -> None:
    rates = metrics.per_group_rates(random_cells(seed))
    assert rates["fnr"].tolist() == pytest.approx((1.0 - rates["tpr"]).tolist())


@pytest.mark.parametrize("seed", range(8))
def test_chouldechova_identity(seed: int) -> None:
    """FPR = p/(1-p) * (1-PPV)/PPV * (1-FNR)  --  Chouldechova 2017, arXiv:1610.07524.

    This is why calibration/predictive parity and equal error rates cannot all hold when
    base rates differ: fix PPV across groups and vary p, and FPR is forced to move.
    """
    rates = metrics.per_group_rates(random_cells(seed))
    p = rates["base_rate"]
    ppv = rates["precision"]

    implied = (p / (1.0 - p)) * ((1.0 - ppv) / ppv) * (1.0 - rates["fnr"])

    assert implied.tolist() == pytest.approx(rates["fpr"].tolist(), abs=1e-12)


def test_chouldechova_identity_on_the_hand_computed_case(
    hand_computed_cells: np.ndarray,
) -> None:
    """The identity reproduces the specification's FPRs of 0.20 and 0.10 exactly."""
    rates = metrics.per_group_rates(hand_computed_cells)
    p, ppv, fnr = rates["base_rate"], rates["precision"], rates["fnr"]

    implied = (p / (1.0 - p)) * ((1.0 - ppv) / ppv) * (1.0 - fnr)

    assert implied.tolist() == pytest.approx([0.20, 0.10])


@pytest.mark.parametrize("seed", range(4))
def test_rates_are_shape_generic(seed: int) -> None:
    """A (n_boot, n_groups, 4) array must give the same answer as looping.

    ``faircheck.bootstrap`` depends on this: it reuses these formulas on a stack of
    resampled count tables rather than reimplementing them.
    """
    rng = np.random.default_rng(seed)
    stack = rng.integers(5, 200, size=(11, 3, 4), dtype=np.int64)

    stacked_rates = metrics.per_group_rates(stack)
    stacked_gaps = metrics.compute_gaps(stacked_rates)

    for index, table in enumerate(stack):
        single_rates = metrics.per_group_rates(table)
        single_gaps = metrics.compute_gaps(single_rates)
        for key, values in single_rates.items():
            assert stacked_rates[key][index].tolist() == pytest.approx(values.tolist())
        for key, value in single_gaps.items():
            assert float(stacked_gaps[key][index]) == pytest.approx(float(value))


def test_gap_registry_inputs_match_the_rate_registry() -> None:
    """Every gap declares which per-group rates it is built from, and they must exist.

    The UI relies on this to show per-group values next to each gap.
    """
    for spec in metrics.GAPS:
        for key in spec.inputs:
            assert key in metrics.RATES_BY_KEY, f"{spec.key} references unknown rate {key!r}"
        assert spec.kind in metrics.PARITY_VALUE
