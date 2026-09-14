"""Disaggregated confusion-matrix rates and the group gaps built from them.

Two properties of this module are load-bearing elsewhere:

1. **Shape-generic.** Every rate takes an array whose last axis is the four
   confusion cells, and every gap takes arrays whose last axis is the group axis.
   So ``faircheck.bootstrap`` can feed a ``(n_boot, n_groups, 4)`` array through the
   exact same formulas used for the point estimates -- there is no second
   implementation of any metric to keep in sync.
2. **Undefined means NaN.** A rate whose denominator is zero (TPR for a group with no
   positives, precision for a group that is never selected) is ``NaN``, and any gap
   involving a ``NaN`` is ``NaN``. Reporting ``0.0`` there would assert something false
   about the group. Fairlearn returns ``0.0`` in this situation, which is why FairCheck
   owns these formulas rather than wrapping it -- see docs/DECISIONS.md D2 and D3.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from faircheck.types import (
    FN,
    FP,
    N_CELLS,
    TN,
    TP,
    BoolArray,
    FloatArray,
    GroupCounts,
    GroupKey,
    IntArray,
)


def _divide(numerator: object, denominator: object) -> FloatArray:
    """Elementwise division yielding NaN -- not 0, not a warning -- where the divisor is 0."""
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=den != 0)
    return out


# --------------------------------------------------------------------------------------
# Building the confusion counts
# --------------------------------------------------------------------------------------


def _group_codes(
    group_values: Sequence[np.ndarray],
) -> tuple[IntArray, tuple[GroupKey, ...]]:
    """Map each row to a group index, returning codes and the observed group keys.

    Each sensitive column is factorised separately and the per-column codes are packed
    into one integer, which is then factorised again. Factorising the packed integer
    keeps only combinations that actually occur while preserving lexicographic order of
    the (sorted) per-column values -- so the group order is stable across runs and
    across datasets, which matters because group order drives colour assignment.
    """
    per_column_codes: list[IntArray] = []
    per_column_values: list[np.ndarray] = []
    for column in group_values:
        column_codes, column_uniques = pd.factorize(pd.Series(column).astype(str), sort=True)
        per_column_codes.append(np.asarray(column_codes, dtype=np.int64))
        per_column_values.append(np.asarray(column_uniques, dtype=object))

    packed = np.zeros(len(per_column_codes[0]), dtype=np.int64)
    for codes, values in zip(per_column_codes, per_column_values, strict=True):
        packed = packed * len(values) + codes

    group_codes, observed = pd.factorize(packed, sort=True)

    keys: list[GroupKey] = []
    for value in np.asarray(observed, dtype=np.int64):
        parts: list[str] = []
        remainder = int(value)
        # Unpack in reverse, because the last column varies fastest in the packed integer.
        for column_values in reversed(per_column_values):
            parts.append(str(column_values[remainder % len(column_values)]))
            remainder //= len(column_values)
        keys.append(tuple(reversed(parts)))

    return np.asarray(group_codes, dtype=np.int64), tuple(keys)


def confusion_counts(
    y_true: BoolArray,
    y_pred: BoolArray,
    group_values: Sequence[np.ndarray],
    group_names: Sequence[str],
) -> GroupCounts:
    """Count TP/FP/FN/TN per observed group in one vectorised pass."""
    if not (len(y_true) == len(y_pred) == len(group_values[0])):
        raise ValueError("y_true, y_pred and the sensitive columns must have equal length.")

    codes, keys = _group_codes(group_values)

    # Each row lands in exactly one of the four cells, so the whole table is a single
    # bincount over (group, cell) pairs rather than a groupby per metric.
    cell = np.where(y_true, np.where(y_pred, TP, FN), np.where(y_pred, FP, TN))
    flat = np.bincount(codes * N_CELLS + cell, minlength=len(keys) * N_CELLS)

    return GroupCounts(
        group_names=tuple(group_names),
        groups=keys,
        cells=flat.reshape(len(keys), N_CELLS).astype(np.int64),
    )


# --------------------------------------------------------------------------------------
# Per-group rates
# --------------------------------------------------------------------------------------


def n_total(cells: object) -> FloatArray:
    return np.asarray(cells, dtype=np.float64).sum(axis=-1)


def _positives(cells: np.ndarray) -> FloatArray:
    return cells[..., TP] + cells[..., FN]


def _negatives(cells: np.ndarray) -> FloatArray:
    return cells[..., FP] + cells[..., TN]


def _selected(cells: np.ndarray) -> FloatArray:
    return cells[..., TP] + cells[..., FP]


def base_rate(cells: object) -> FloatArray:
    """P(Y = positive). Context for every other metric, and the driver of the
    impossibility result: when base rates differ, some parities cannot co-exist."""
    c = np.asarray(cells, dtype=np.float64)
    return _divide(_positives(c), n_total(c))


def selection_rate(cells: object) -> FloatArray:
    """P(Yhat = positive) -- the decision's actual footprint, and the quantity the
    four-fifths rule is about."""
    c = np.asarray(cells, dtype=np.float64)
    return _divide(_selected(c), n_total(c))


def true_positive_rate(cells: object) -> FloatArray:
    """TP / positives. Undefined (NaN) for a group with no positive-class rows."""
    c = np.asarray(cells, dtype=np.float64)
    return _divide(c[..., TP], _positives(c))


def false_positive_rate(cells: object) -> FloatArray:
    """FP / negatives. Undefined (NaN) for a group with no negative-class rows."""
    c = np.asarray(cells, dtype=np.float64)
    return _divide(c[..., FP], _negatives(c))


def false_negative_rate(cells: object) -> FloatArray:
    """FN / positives = 1 - TPR. Undefined (NaN) for a group with no positives."""
    c = np.asarray(cells, dtype=np.float64)
    return _divide(c[..., FN], _positives(c))


def precision(cells: object) -> FloatArray:
    """TP / selected (PPV). Undefined (NaN) for a group that is never selected."""
    c = np.asarray(cells, dtype=np.float64)
    return _divide(c[..., TP], _selected(c))


def accuracy(cells: object) -> FloatArray:
    """(TP + TN) / n. Reported for completeness; misleading under class imbalance."""
    c = np.asarray(cells, dtype=np.float64)
    return _divide(c[..., TP] + c[..., TN], n_total(c))


@dataclass(frozen=True, slots=True)
class RateSpec:
    """Metadata for one per-group rate, so the report and UI never restate a formula."""

    key: str
    label: str
    formula: str
    undefined_when: str | None
    fn: Callable[[object], FloatArray]


NO_POSITIVES = "the group has no positives"
NO_NEGATIVES = "the group has no negatives"
NEVER_SELECTED = "the group is never selected"

RATES: tuple[RateSpec, ...] = (
    RateSpec("base_rate", "Base rate", "(TP + FN) / n", None, base_rate),
    RateSpec("selection_rate", "Selection rate", "(TP + FP) / n", None, selection_rate),
    RateSpec("tpr", "TPR (recall)", "TP / (TP + FN)", NO_POSITIVES, true_positive_rate),
    RateSpec("fpr", "FPR", "FP / (FP + TN)", NO_NEGATIVES, false_positive_rate),
    RateSpec("fnr", "FNR", "FN / (TP + FN)", NO_POSITIVES, false_negative_rate),
    RateSpec("precision", "Precision (PPV)", "TP / (TP + FP)", NEVER_SELECTED, precision),
    RateSpec("accuracy", "Accuracy", "(TP + TN) / n", None, accuracy),
)

RATES_BY_KEY: dict[str, RateSpec] = {spec.key: spec for spec in RATES}


def per_group_rates(cells: object) -> dict[str, FloatArray]:
    """Every rate in :data:`RATES`, keyed by rate name.

    Works unchanged on a ``(n_groups, 4)`` observed table and on a
    ``(n_boot, n_groups, 4)`` bootstrap array.
    """
    c = np.asarray(cells, dtype=np.float64)
    return {spec.key: spec.fn(c) for spec in RATES}


# --------------------------------------------------------------------------------------
# Gaps between groups
# --------------------------------------------------------------------------------------


def _spread(values: FloatArray) -> FloatArray:
    """max - min across groups. NaN propagates, which is the intended behaviour: a gap
    computed while ignoring an undefined group would understate the true spread."""
    return np.max(values, axis=-1) - np.min(values, axis=-1)


def demographic_parity_difference(rates: Mapping[str, FloatArray]) -> FloatArray:
    return _spread(rates["selection_rate"])


def disparate_impact_ratio(rates: Mapping[str, FloatArray]) -> FloatArray:
    """Lower selection rate / higher selection rate, so 1.0 is parity and smaller is
    worse. Undefined when no group is ever selected."""
    sr = rates["selection_rate"]
    return _divide(np.min(sr, axis=-1), np.max(sr, axis=-1))


def equal_opportunity_difference(rates: Mapping[str, FloatArray]) -> FloatArray:
    return _spread(rates["tpr"])


def equalized_odds_difference(rates: Mapping[str, FloatArray]) -> FloatArray:
    """max(|dTPR|, |dFPR|) -- the worst of the two error directions, matching
    Fairlearn's ``agg='worst_case'``."""
    return np.maximum(_spread(rates["tpr"]), _spread(rates["fpr"]))


def predictive_parity_difference(rates: Mapping[str, FloatArray]) -> FloatArray:
    return _spread(rates["precision"])


@dataclass(frozen=True, slots=True)
class GapSpec:
    """Metadata for one summary gap, including the per-group rates it came from.

    ``inputs`` is what lets the UI honour "always show per-group values next to gaps"
    without hard-coding which rates belong to which gap.
    """

    key: str
    label: str
    formula: str
    reference: str
    inputs: tuple[str, ...]
    kind: str  # "difference" (0 = parity) or "ratio" (1 = parity)
    fn: Callable[[Mapping[str, FloatArray]], FloatArray]


GAPS: tuple[GapSpec, ...] = (
    GapSpec(
        "demographic_parity_difference",
        "Demographic parity difference",
        "max_a SR_a - min_a SR_a",
        "Hardt et al. 2016 (arXiv:1610.02413), for contrast",
        ("selection_rate",),
        "difference",
        demographic_parity_difference,
    ),
    GapSpec(
        "disparate_impact_ratio",
        "Disparate impact ratio",
        "min_a SR_a / max_a SR_a",
        "29 CFR 1607.4(D) (four-fifths rule of thumb)",
        ("selection_rate",),
        "ratio",
        disparate_impact_ratio,
    ),
    GapSpec(
        "equal_opportunity_difference",
        "Equal opportunity difference",
        "max_a TPR_a - min_a TPR_a",
        "Hardt et al. 2016, Definition 2.2",
        ("tpr",),
        "difference",
        equal_opportunity_difference,
    ),
    GapSpec(
        "equalized_odds_difference",
        "Equalized odds difference",
        "max(|dTPR|, |dFPR|)",
        "Hardt et al. 2016, Definition 2.1",
        ("tpr", "fpr"),
        "difference",
        equalized_odds_difference,
    ),
    GapSpec(
        "predictive_parity_difference",
        "Predictive parity difference",
        "max_a PPV_a - min_a PPV_a",
        "Chouldechova 2017 (arXiv:1610.07524)",
        ("precision",),
        "difference",
        predictive_parity_difference,
    ),
)

GAPS_BY_KEY: dict[str, GapSpec] = {spec.key: spec for spec in GAPS}

#: Parity value for each gap kind, i.e. the value meaning "no disparity".
PARITY_VALUE: dict[str, float] = {"difference": 0.0, "ratio": 1.0}


def compute_gaps(rates: Mapping[str, FloatArray]) -> dict[str, FloatArray]:
    """Every gap in :data:`GAPS`, keyed by gap name."""
    return {spec.key: spec.fn(rates) for spec in GAPS}


FOUR_FIFTHS_THRESHOLD: float = 0.8


def fails_four_fifths(disparate_impact: float) -> bool | None:
    """Whether the disparate impact ratio falls below 0.8.

    Deliberately returns ``None`` rather than ``False`` when the ratio is undefined, so
    that callers cannot render "passes" for a dataset where the check never ran. The
    four-fifths rule is a screening heuristic that federal enforcement agencies use as a
    rule of thumb, not a legal finding -- 29 CFR 1607.4(D) and the EEOC's own Q&A say so
    explicitly, and the UI wording must too.
    """
    if not np.isfinite(disparate_impact):
        return None
    return bool(disparate_impact < FOUR_FIFTHS_THRESHOLD)
