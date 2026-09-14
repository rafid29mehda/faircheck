"""Typed containers shared by the FairCheck core library.

Nothing under ``faircheck/`` may import a UI framework. The core library has to be
usable identically from the Streamlit app, the CLI and a notebook, and
``tests/test_core_has_no_ui.py`` enforces that mechanically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

#: Position of each confusion-matrix cell along the last axis of every count array.
#: Fixing this order once is what lets :mod:`faircheck.metrics` accept both a
#: ``(n_groups, 4)`` array of observed counts and a ``(n_boot, n_groups, 4)`` array
#: of bootstrap counts without changing a single formula.
TP: Final = 0
FP: Final = 1
FN: Final = 2
TN: Final = 3
N_CELLS: Final = 4

CELL_NAMES: Final = ("TP", "FP", "FN", "TN")

#: One entry per sensitive column, so a single type covers both the marginal view
#: ("female",) and the intersectional view ("female", "Black").
GroupKey = tuple[str, ...]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ColumnMapping:
    """Which CSV column plays which role, and which label value is favorable.

    ``positive_label`` holds a value taken from the label column itself rather than a
    hard-coded ``1``: the favorable outcome is a modelling choice the user makes, and
    for a multi-class column it also selects the one-vs-rest split (DECISIONS.md D5).
    """

    label: str
    prediction: str
    sensitive: tuple[str, ...]
    positive_label: Any
    score: str | None = None

    def __post_init__(self) -> None:
        if not self.sensitive:
            raise ValueError("At least one sensitive column is required.")

    @property
    def columns(self) -> tuple[str, ...]:
        """Every column the audit will read, in role order."""
        extra = (self.score,) if self.score is not None else ()
        return (self.label, self.prediction, *self.sensitive, *extra)


@dataclass(frozen=True, slots=True)
class ValidatedData:
    """The audit's input after validation: binarised labels and resolved group keys.

    ``y_true``/``y_pred`` are already one-vs-rest against
    :attr:`ColumnMapping.positive_label`, so downstream code never has to know what
    the original label values were.
    """

    y_true: BoolArray
    y_pred: BoolArray
    group_values: tuple[npt.NDArray[np.object_], ...]
    mapping: ColumnMapping
    score: FloatArray | None = None
    warnings: tuple[str, ...] = ()
    n_rows_input: int = 0
    n_rows_used: int = 0
    label_values: tuple[str, ...] = ()

    @property
    def is_one_vs_rest(self) -> bool:
        """True when the label column had more than two distinct values."""
        return len(self.label_values) > 2

    @property
    def was_subsampled(self) -> bool:
        return self.n_rows_used < self.n_rows_input


@dataclass(frozen=True, slots=True)
class GroupCounts:
    """Confusion-matrix counts for every observed combination of sensitive values.

    Groups are the combinations actually present in the data, in lexicographic order
    of the sensitive columns; combinations that never occur are absent rather than
    present-with-zero, because a zero-row group has no defined rates at all.
    """

    group_names: tuple[str, ...]
    groups: tuple[GroupKey, ...]
    cells: IntArray

    def __post_init__(self) -> None:
        if self.cells.shape != (len(self.groups), N_CELLS):
            raise ValueError(
                f"cells must have shape {(len(self.groups), N_CELLS)}, got {self.cells.shape}"
            )
        for key in self.groups:
            if len(key) != len(self.group_names):
                raise ValueError(
                    f"group key {key!r} does not match sensitive columns {self.group_names!r}"
                )

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def sizes(self) -> IntArray:
        """Row count per group."""
        return self.cells.sum(axis=-1)

    def labels(self, separator: str = " × ") -> tuple[str, ...]:
        """Human-readable group labels, e.g. ``("female × Black", ...)``."""
        return tuple(separator.join(key) for key in self.groups)
