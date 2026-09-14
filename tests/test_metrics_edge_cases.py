"""Edge cases: the places where fairness tooling usually reports something false.

The central claim under test is that an *undefined* rate stays undefined all the way
through the gaps, instead of being coerced to 0 and read as "this group always fails".
"""

from __future__ import annotations

import numpy as np
import pytest

from faircheck import metrics
from faircheck.types import ColumnMapping
from faircheck.validate import validate
from tests.conftest import cells_to_frame


def _rates(cells: list[tuple[int, int, int, int]]) -> dict[str, np.ndarray]:
    return metrics.per_group_rates(np.array(cells, dtype=np.int64))


def test_group_with_no_positives_has_undefined_tpr_not_zero() -> None:
    # Group B has TP + FN = 0, so "what fraction of positives were found?" has no answer.
    rates = _rates([(40, 10, 10, 40), (0, 5, 0, 45)])

    assert np.isnan(rates["tpr"][1])
    assert np.isnan(rates["fnr"][1])
    # Rates that do not condition on the positive class remain perfectly well defined.
    assert rates["fpr"].tolist() == pytest.approx([0.20, 0.10])
    assert rates["precision"][1] == pytest.approx(0.0)
    assert rates["base_rate"][1] == pytest.approx(0.0)


def test_undefined_rate_propagates_into_gaps() -> None:
    rates = _rates([(40, 10, 10, 40), (0, 5, 0, 45)])
    gaps = metrics.compute_gaps(rates)

    # A TPR gap cannot be computed against a group with no positives.
    assert np.isnan(gaps["equal_opportunity_difference"])
    assert np.isnan(gaps["equalized_odds_difference"])
    # ... but selection-rate comparisons are unaffected: 50/100 against 5/50.
    assert float(gaps["demographic_parity_difference"]) == pytest.approx(0.40)


def test_group_with_no_negatives_has_undefined_fpr() -> None:
    rates = _rates([(40, 10, 10, 40), (30, 0, 20, 0)])

    assert np.isnan(rates["fpr"][1])
    assert np.isnan(metrics.compute_gaps(rates)["equalized_odds_difference"])
    assert rates["tpr"].tolist() == pytest.approx([0.80, 0.60])


def test_group_never_selected_has_undefined_precision() -> None:
    rates = _rates([(40, 10, 10, 40), (0, 0, 20, 80)])

    assert rates["selection_rate"][1] == pytest.approx(0.0)
    assert np.isnan(rates["precision"][1])
    assert np.isnan(metrics.compute_gaps(rates)["predictive_parity_difference"])
    # min/max selection rate is 0/0.5, a legitimate ratio of zero.
    assert float(metrics.compute_gaps(rates)["disparate_impact_ratio"]) == pytest.approx(0.0)


def test_single_group_reports_parity_not_an_error() -> None:
    gaps = metrics.compute_gaps(_rates([(40, 10, 10, 40)]))

    assert float(gaps["demographic_parity_difference"]) == pytest.approx(0.0)
    assert float(gaps["disparate_impact_ratio"]) == pytest.approx(1.0)
    assert float(gaps["equalized_odds_difference"]) == pytest.approx(0.0)


def test_nobody_selected_anywhere_leaves_the_ratio_undefined() -> None:
    gaps = metrics.compute_gaps(_rates([(0, 0, 50, 50), (0, 0, 20, 80)]))

    assert np.isnan(gaps["disparate_impact_ratio"])
    # The four-fifths screen must not claim a pass when it never ran.
    assert metrics.fails_four_fifths(float(gaps["disparate_impact_ratio"])) is None


def test_all_zero_cells_for_one_metric_do_not_crash() -> None:
    rates = _rates([(0, 0, 0, 100), (5, 5, 5, 5)])
    assert np.isnan(rates["tpr"][0])
    assert np.isnan(rates["precision"][0])
    assert rates["accuracy"][0] == pytest.approx(1.0)


def test_flipping_the_positive_label_swaps_the_confusion_matrix() -> None:
    """Choosing the other label value must relabel the matrix, not merely rename columns.

    With positive = 0, yesterday's TN become TP and yesterday's FP become FN, so the new
    TPR is the old TNR. This is the check that ``positive_label`` is honoured everywhere
    rather than a hard-coded 1 surviving somewhere downstream.
    """
    frame = cells_to_frame({"A": (40, 10, 10, 40), "B": (12, 8, 8, 72)})

    def cells_for(positive: int) -> np.ndarray:
        mapping = ColumnMapping(
            label="y_true", prediction="y_pred", sensitive=("group",), positive_label=positive
        )
        data = validate(frame, mapping)
        return metrics.confusion_counts(
            data.y_true, data.y_pred, data.group_values, mapping.sensitive
        ).cells

    original, flipped = cells_for(1), cells_for(0)

    # (TP, FP, FN, TN) -> (TN, FN, FP, TP)
    assert flipped[:, 0].tolist() == original[:, 3].tolist()
    assert flipped[:, 1].tolist() == original[:, 2].tolist()
    assert flipped[:, 2].tolist() == original[:, 1].tolist()
    assert flipped[:, 3].tolist() == original[:, 0].tolist()

    flipped_rates = metrics.per_group_rates(flipped)
    original_rates = metrics.per_group_rates(original)
    # New TPR == old TNR == 1 - old FPR.
    assert flipped_rates["tpr"].tolist() == pytest.approx((1.0 - original_rates["fpr"]).tolist())
    assert flipped_rates["selection_rate"].tolist() == pytest.approx(
        (1.0 - original_rates["selection_rate"]).tolist()
    )
    # Accuracy is symmetric under relabelling.
    assert flipped_rates["accuracy"].tolist() == pytest.approx(original_rates["accuracy"].tolist())


def test_intersectional_groups_are_crossed_and_ordered() -> None:
    frame = cells_to_frame({"A": (4, 1, 1, 4), "B": (2, 1, 1, 6)})
    frame["region"] = ["north", "south"] * (len(frame) // 2)

    mapping = ColumnMapping(
        label="y_true",
        prediction="y_pred",
        sensitive=("group", "region"),
        positive_label=1,
    )
    data = validate(frame, mapping)
    counts = metrics.confusion_counts(
        data.y_true, data.y_pred, data.group_values, mapping.sensitive
    )

    assert counts.groups == (("A", "north"), ("A", "south"), ("B", "north"), ("B", "south"))
    assert counts.labels() == ("A × north", "A × south", "B × north", "B × south")
    assert int(counts.sizes.sum()) == len(frame)


def test_fairlearn_disagrees_on_undefined_rates_and_that_is_deliberate() -> None:
    """Documents the one place FairCheck intentionally differs from Fairlearn.

    ``fairlearn.metrics.true_positive_rate`` returns 0.0 for a group with no positives
    (it inherits scikit-learn's ``zero_division=0`` default). FairCheck returns NaN and
    renders "n/a". If a future Fairlearn release changes this, this test fails and the
    oracle comparison in ``test_metrics_vs_fairlearn`` can be widened.
    """
    fairlearn_metrics = pytest.importorskip("fairlearn.metrics")

    y_true = np.zeros(50, dtype=bool)
    y_pred = np.zeros(50, dtype=bool)
    y_pred[:5] = True

    assert fairlearn_metrics.true_positive_rate(y_true, y_pred) == 0.0
    assert np.isnan(metrics.per_group_rates(np.array([[0, 5, 0, 45]]))["tpr"][0])
