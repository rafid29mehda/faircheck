"""Shared fixtures.

Tests are built from *confusion-matrix cells* rather than from random rows wherever
possible: stating TP/FP/FN/TN directly means the expected metric values can be worked
out by hand, so a failure points at the formula rather than at the fixture.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

#: The case from the project specification, in FairCheck cell order (TP, FP, FN, TN).
#: Group A: TP=40 FN=10 FP=10 TN=40. Group B: TP=12 FN=8 FP=8 TN=72.
HAND_COMPUTED_CELLS: dict[str, tuple[int, int, int, int]] = {
    "A": (40, 10, 10, 40),
    "B": (12, 8, 8, 72),
}


def cells_to_frame(
    groups: Mapping[str, Sequence[int]],
    *,
    positive: object = 1,
    negative: object = 0,
    group_column: str = "group",
) -> pd.DataFrame:
    """Build a DataFrame whose per-group confusion counts are exactly ``groups``.

    ``groups`` maps a group name to (TP, FP, FN, TN).
    """
    rows: list[dict[str, object]] = []
    for name, (tp, fp, fn, tn) in groups.items():
        for _ in range(tp):
            rows.append({"y_true": positive, "y_pred": positive, group_column: name})
        for _ in range(fp):
            rows.append({"y_true": negative, "y_pred": positive, group_column: name})
        for _ in range(fn):
            rows.append({"y_true": positive, "y_pred": negative, group_column: name})
        for _ in range(tn):
            rows.append({"y_true": negative, "y_pred": negative, group_column: name})
    return pd.DataFrame(rows)


@pytest.fixture
def hand_computed_frame() -> pd.DataFrame:
    return cells_to_frame(HAND_COMPUTED_CELLS)


@pytest.fixture
def hand_computed_cells() -> np.ndarray:
    return np.array([HAND_COMPUTED_CELLS["A"], HAND_COMPUTED_CELLS["B"]], dtype=np.int64)
