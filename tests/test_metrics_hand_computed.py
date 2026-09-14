"""The anchor test: a case whose every value was computed by hand.

These numbers were independently confirmed against Fairlearn's ``MetricFrame`` during
planning (see PLAN.md 2.3a), so this file pins *intended* behaviour rather than merely
asserting that FairCheck agrees with itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from faircheck import metrics
from faircheck.types import ColumnMapping
from faircheck.validate import validate
from tests.conftest import HAND_COMPUTED_CELLS


@pytest.fixture
def rates(hand_computed_cells: np.ndarray) -> dict[str, np.ndarray]:
    return metrics.per_group_rates(hand_computed_cells)


def test_group_sizes(hand_computed_cells: np.ndarray) -> None:
    assert metrics.n_total(hand_computed_cells).tolist() == [100.0, 100.0]


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("base_rate", [0.50, 0.20]),
        ("selection_rate", [0.50, 0.20]),
        ("tpr", [0.80, 0.60]),
        ("fpr", [0.20, 0.10]),
        ("fnr", [0.20, 0.40]),
        ("precision", [0.80, 0.60]),
        ("accuracy", [0.80, 0.84]),
    ],
)
def test_per_group_rates(rates: dict[str, np.ndarray], key: str, expected: list[float]) -> None:
    assert rates[key].tolist() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("demographic_parity_difference", 0.30),
        ("disparate_impact_ratio", 0.40),
        ("equal_opportunity_difference", 0.20),
        ("equalized_odds_difference", 0.20),
        ("predictive_parity_difference", 0.20),
    ],
)
def test_gaps(rates: dict[str, np.ndarray], key: str, expected: float) -> None:
    assert float(metrics.compute_gaps(rates)[key]) == pytest.approx(expected)


def test_four_fifths_flag_is_raised(rates: dict[str, np.ndarray]) -> None:
    ratio = float(metrics.compute_gaps(rates)["disparate_impact_ratio"])
    assert ratio == pytest.approx(0.40)
    assert metrics.fails_four_fifths(ratio) is True


def test_end_to_end_from_a_dataframe(hand_computed_frame: pd.DataFrame) -> None:
    """The same numbers must come out of the full validate -> count path."""
    mapping = ColumnMapping(
        label="y_true", prediction="y_pred", sensitive=("group",), positive_label=1
    )
    data = validate(hand_computed_frame, mapping)
    counts = metrics.confusion_counts(
        data.y_true, data.y_pred, data.group_values, mapping.sensitive
    )

    assert counts.groups == (("A",), ("B",))
    assert counts.labels() == ("A", "B")
    expected = [list(HAND_COMPUTED_CELLS["A"]), list(HAND_COMPUTED_CELLS["B"])]
    assert counts.cells.tolist() == expected

    rates = metrics.per_group_rates(counts.cells)
    assert rates["tpr"].tolist() == pytest.approx([0.80, 0.60])
    assert float(metrics.compute_gaps(rates)["equalized_odds_difference"]) == pytest.approx(0.20)
