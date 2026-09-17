"""Chouldechova identity as an explainer, with the user's numbers substituted in."""

from __future__ import annotations

import numpy as np
import pytest

from faircheck.impossibility import (
    chouldechova_fpr,
    chouldechova_precision,
    explain_impossibility,
)
from faircheck.metrics import per_group_rates
from faircheck.types import GroupCounts
from tests.test_bootstrap import counts_from


def test_identity_reproduces_observed_fpr_on_the_hand_computed_case(
    hand_computed_cells: np.ndarray,
) -> None:
    rates = per_group_rates(hand_computed_cells)
    implied = chouldechova_fpr(rates["base_rate"], rates["precision"], rates["fnr"])
    assert implied.tolist() == pytest.approx([0.20, 0.10])


def test_hand_computed_case_triggers_the_tradeoff() -> None:
    result = explain_impossibility(
        GroupCounts(
            group_names=("group",),
            groups=(("A",), ("B",)),
            cells=np.array([[40, 10, 10, 40], [12, 8, 8, 72]], dtype=np.int64),
        )
    )

    assert result.base_rates_differ is True
    assert result.is_perfect is False
    assert result.tradeoff_applies is True
    assert result.base_rate.tolist() == pytest.approx([0.50, 0.20])
    assert result.implied_fpr.tolist() == pytest.approx(result.fpr.tolist(), abs=1e-12)


def test_equalising_ppv_forces_group_b_fpr_to_move() -> None:
    """Hold A's PPV = 0.8 and B's own prevalence and FNR: B's FPR must become 0.0375.

    FPR = (0.2/0.8) · (0.2/0.8) · 0.6 = 0.0375, not the observed 0.10. That is the
    concrete cost of demanding predictive parity on this table.
    """
    result = explain_impossibility(counts_from([(40, 10, 10, 40), (12, 8, 8, 72)]))
    # Reference is the largest group; both n=100, argmax returns 0 (A).
    assert result.reference_label == "A"
    assert result.counterfactual_fpr_equal_ppv[0] == pytest.approx(0.20)
    assert result.counterfactual_fpr_equal_ppv[1] == pytest.approx(0.0375)
    assert result.fpr[1] == pytest.approx(0.10)
    assert result.equal_ppv_feasible.tolist() == [True, True]


def test_equalising_error_rates_forces_ppv_apart() -> None:
    """Hold A's TPR and FPR: B's PPV would have to be 0.5, not A's 0.8.

    PPV = 1 / (1 + FPR·(1-p)/(TPR·p)) = 1 / (1 + 0.2·0.8 / (0.8·0.2)) = 0.5.
    """
    result = explain_impossibility(counts_from([(40, 10, 10, 40), (12, 8, 8, 72)]))
    assert result.counterfactual_ppv_equal_errors[0] == pytest.approx(0.80)
    assert result.counterfactual_ppv_equal_errors[1] == pytest.approx(0.50)
    assert result.precision[1] == pytest.approx(0.60)


def test_perfect_classifier_is_the_other_exception() -> None:
    """Different base rates, zero errors: the identity does not force a trade-off."""
    result = explain_impossibility(counts_from([(30, 0, 0, 70), (80, 0, 0, 20)]))
    assert result.base_rates_differ is True
    assert result.is_perfect is True
    assert result.tradeoff_applies is False
    assert result.fpr.tolist() == pytest.approx([0.0, 0.0])
    assert result.fnr.tolist() == pytest.approx([0.0, 0.0])


def test_equal_base_rates_do_not_trigger_the_tradeoff() -> None:
    result = explain_impossibility(counts_from([(40, 10, 10, 40), (35, 15, 15, 35)]))
    assert result.base_rate.tolist() == pytest.approx([0.50, 0.50])
    assert result.base_rates_differ is False
    assert result.tradeoff_applies is False
    assert result.is_perfect is False


def test_solved_identity_recovers_observed_ppv(hand_computed_cells: np.ndarray) -> None:
    rates = per_group_rates(hand_computed_cells)
    recovered = chouldechova_precision(rates["base_rate"], rates["fpr"], rates["tpr"])
    assert recovered.tolist() == pytest.approx(rates["precision"].tolist(), abs=1e-12)


def test_bad_reference_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        explain_impossibility(counts_from([(40, 10, 10, 40), (12, 8, 8, 72)]), reference_index=9)


def test_infeasible_equal_ppv_is_flagged_not_clipped() -> None:
    """Holding A's modest PPV while keeping B's high TPR and high prevalence asks for FPR > 1.

    That is the identity refusing the counterfactual, not a bug in the algebra. The
    report must say "impossible" rather than print 1.3 as if it were a rate.
    """
    result = explain_impossibility(
        counts_from([(20, 10, 30, 140), (200, 80, 20, 20)]), reference_index=0
    )
    assert result.counterfactual_fpr_equal_ppv[1] > 1.0
    assert bool(result.equal_ppv_feasible[0]) is True
    assert bool(result.equal_ppv_feasible[1]) is False


def test_undefined_inputs_stay_undefined() -> None:
    implied = chouldechova_fpr([0.0], [np.nan], [np.nan])
    assert np.isnan(implied[0])
    recovered = chouldechova_precision([0.0], [0.1], [np.nan])
    assert np.isnan(recovered[0])
