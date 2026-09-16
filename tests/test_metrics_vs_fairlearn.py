"""Cross-validation against Fairlearn, the reference implementation.

FairCheck computes its own metrics (docs/DECISIONS.md D2), so the value of that choice rests
on this file: random data, several seeds, varying group counts, every rate and gap compared
against ``fairlearn.metrics``.

The one intended divergence -- undefined rates, where Fairlearn returns 0.0 and FairCheck
returns NaN (D3) -- is pinned separately in ``test_metrics_edge_cases.py``. Here the
comparison is made on groups where every rate is defined, plus one case that mixes in an
undefined group to show the defined groups still agree exactly.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest

fairlearn_metrics = pytest.importorskip("fairlearn.metrics")
sklearn_metrics = pytest.importorskip("sklearn.metrics")

from faircheck import metrics  # noqa: E402
from faircheck.types import ColumnMapping  # noqa: E402
from faircheck.validate import validate  # noqa: E402
from tests.conftest import cells_to_frame  # noqa: E402

MAPPING = ColumnMapping(label="y_true", prediction="y_pred", sensitive=("group",), positive_label=1)

# pos_label is pinned explicitly: Fairlearn's default of "largest unique value" is resolved
# per group, which would silently mean different things in different groups.
ORACLE_METRICS = {
    "count": fairlearn_metrics.count,
    "selection_rate": partial(fairlearn_metrics.selection_rate, pos_label=1),
    "tpr": partial(fairlearn_metrics.true_positive_rate, pos_label=1),
    "fpr": partial(fairlearn_metrics.false_positive_rate, pos_label=1),
    "fnr": partial(fairlearn_metrics.false_negative_rate, pos_label=1),
    "precision": partial(sklearn_metrics.precision_score, pos_label=1, zero_division=np.nan),
    "accuracy": sklearn_metrics.accuracy_score,
}


def random_cells(seed: int, n_groups: int, *, minimum: int = 5) -> dict[str, list[int]]:
    """Confusion tables with every cell non-empty, so every rate is defined in every group."""
    rng = np.random.default_rng(seed)
    return {
        chr(ord("A") + index): rng.integers(minimum, 400, size=4).tolist()
        for index in range(n_groups)
    }


def faircheck_results(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    data = validate(frame, MAPPING)
    counts = metrics.confusion_counts(
        data.y_true, data.y_pred, data.group_values, MAPPING.sensitive
    )
    rates = metrics.per_group_rates(counts.cells)
    return rates, metrics.compute_gaps(rates)


def oracle_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return fairlearn_metrics.MetricFrame(
        metrics=ORACLE_METRICS,
        y_true=frame["y_true"].to_numpy(),
        y_pred=frame["y_pred"].to_numpy(),
        sensitive_features=frame["group"].to_numpy(),
    ).by_group


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("n_groups", [2, 3, 5])
def test_per_group_rates_match_fairlearn(seed: int, n_groups: int) -> None:
    frame = cells_to_frame(random_cells(seed, n_groups))
    rates, _ = faircheck_results(frame)
    expected = oracle_frame(frame)

    for key in ("selection_rate", "tpr", "fpr", "fnr", "precision", "accuracy"):
        assert rates[key].tolist() == pytest.approx(expected[key].to_numpy().tolist()), key


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("n_groups", [2, 3, 5])
def test_gaps_match_fairlearn(seed: int, n_groups: int) -> None:
    frame = cells_to_frame(random_cells(seed, n_groups))
    _, gaps = faircheck_results(frame)

    y_true = frame["y_true"].to_numpy()
    y_pred = frame["y_pred"].to_numpy()
    groups = frame["group"].to_numpy()

    assert float(gaps["demographic_parity_difference"]) == pytest.approx(
        fairlearn_metrics.demographic_parity_difference(y_true, y_pred, sensitive_features=groups)
    )
    assert float(gaps["disparate_impact_ratio"]) == pytest.approx(
        fairlearn_metrics.demographic_parity_ratio(y_true, y_pred, sensitive_features=groups)
    )
    assert float(gaps["equalized_odds_difference"]) == pytest.approx(
        fairlearn_metrics.equalized_odds_difference(
            y_true, y_pred, sensitive_features=groups, agg="worst_case"
        )
    )


@pytest.mark.parametrize("seed", range(4))
def test_equal_opportunity_matches_a_tpr_metricframe(seed: int) -> None:
    """Fairlearn has no ``equal_opportunity_difference``; the TPR spread is its definition."""
    frame = cells_to_frame(random_cells(seed, 4))
    _, gaps = faircheck_results(frame)

    # A dict rather than a bare callable: Fairlearn takes the metric name from ``__name__``,
    # which a functools.partial does not have, and warns when it cannot find one.
    oracle = fairlearn_metrics.MetricFrame(
        metrics={"tpr": partial(fairlearn_metrics.true_positive_rate, pos_label=1)},
        y_true=frame["y_true"].to_numpy(),
        y_pred=frame["y_pred"].to_numpy(),
        sensitive_features=frame["group"].to_numpy(),
    )

    assert float(gaps["equal_opportunity_difference"]) == pytest.approx(
        oracle.difference(method="between_groups")["tpr"]
    )


@pytest.mark.parametrize("seed", range(4))
def test_group_sizes_match_fairlearn_counts(seed: int) -> None:
    frame = cells_to_frame(random_cells(seed, 3))
    data = validate(frame, MAPPING)
    counts = metrics.confusion_counts(
        data.y_true, data.y_pred, data.group_values, MAPPING.sensitive
    )

    assert counts.sizes.tolist() == oracle_frame(frame)["count"].to_numpy().tolist()


def test_intersectional_rates_match_a_two_column_metricframe() -> None:
    """Fairlearn crosses multiple sensitive columns the same way FairCheck does."""
    rng = np.random.default_rng(0)
    frame = cells_to_frame(random_cells(0, 3))
    frame["region"] = rng.choice(["north", "south"], len(frame))

    mapping = ColumnMapping(
        label="y_true", prediction="y_pred", sensitive=("group", "region"), positive_label=1
    )
    data = validate(frame, mapping)
    counts = metrics.confusion_counts(
        data.y_true, data.y_pred, data.group_values, mapping.sensitive
    )
    rates = metrics.per_group_rates(counts.cells)

    oracle = fairlearn_metrics.MetricFrame(
        metrics=ORACLE_METRICS,
        y_true=frame["y_true"].to_numpy(),
        y_pred=frame["y_pred"].to_numpy(),
        sensitive_features=frame[["group", "region"]],
    ).by_group

    assert counts.groups == tuple(oracle.index)
    for key in ("selection_rate", "tpr", "fpr", "precision"):
        assert rates[key].tolist() == pytest.approx(oracle[key].to_numpy().tolist()), key


def test_defined_groups_still_match_when_another_group_is_undefined() -> None:
    """Our NaN policy is localised: it changes the undefined group only."""
    cells = {"A": [40, 10, 10, 40], "B": [12, 8, 8, 72], "C": [0, 5, 0, 45]}
    frame = cells_to_frame(cells)
    rates, _ = faircheck_results(frame)
    expected = oracle_frame(frame)

    for key in ("selection_rate", "tpr", "fpr", "precision", "accuracy"):
        assert rates[key][:2].tolist() == pytest.approx(expected[key].to_numpy()[:2].tolist()), key

    # Group C is where the two libraries deliberately differ.
    assert np.isnan(rates["tpr"][2])
    assert expected["tpr"].to_numpy()[2] == 0.0
