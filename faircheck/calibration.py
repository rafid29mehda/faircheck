"""Score-based metrics: ROC-AUC, reliability diagrams, ECE, and a threshold sweep.

These cannot be computed from the four confusion cells, so they only run when the user
maps a score column. Two design choices are load-bearing:

1. **ECE is on the positive-class probability, not top-label confidence.** Guo et al.
   (2017) define ECE over ``max_k p̂(y=k)``. For a binary task with one score column
   ``s = p̂(Y=1)``, that is a different quantity -- and the wrong one for "will a human
   read this as a probability of the favorable outcome?" FairCheck bins ``s`` and
   compares the mean score in each bin to the observed positive rate. See D12.
2. **A group that lacks both classes has no ROC-AUC.** The value is ``NaN`` with a
   reason, never ``0.5`` or ``0.0``. sklearn warns and returns NaN in the same situation;
   we skip without a warning so ``filterwarnings = ["error"]`` stays honest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

import numpy as np

from faircheck.metrics import _divide, _group_codes
from faircheck.types import BoolArray, FloatArray, IntArray

DEFAULT_N_BINS: Final = 10
DEFAULT_N_THRESHOLDS: Final = 101

NO_POSITIVES: Final = "this group has no positive labels, so ROC-AUC is not defined"
NO_NEGATIVES: Final = "this group has no negative labels, so ROC-AUC is not defined"


def _as_bool(y_true: object) -> BoolArray:
    return np.asarray(y_true, dtype=bool)


def _as_score(score: object) -> FloatArray:
    return np.asarray(score, dtype=np.float64)


def roc_auc(y_true: object, score: object) -> float:
    """Mann-Whitney ROC-AUC of ``score`` as a ranking of the positive class.

    Ties contribute 0.5, matching sklearn. Returns ``NaN`` when either class is absent
    rather than inventing a number.
    """
    y = _as_bool(y_true)
    s = _as_score(score)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)

    # Average ranks of tied scores so a run of equal values does not prefer one class.
    sorted_scores = s[order]
    tie_starts = np.flatnonzero(np.diff(sorted_scores, prepend=np.nan, append=np.nan) != 0)
    for start, end in pairwise(tie_starts):
        if end - start > 1:
            ranks[order[start:end]] = 0.5 * (start + end + 1)

    auc = (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def auc_skip_reason(y_true: object) -> str | None:
    y = _as_bool(y_true)
    if y.size == 0:
        return "this group is empty, so ROC-AUC is not defined"
    if not y.any():
        return NO_POSITIVES
    if y.all():
        return NO_NEGATIVES
    return None


def _bin_index(score: FloatArray, n_bins: int) -> IntArray:
    """Equal-width bins on [0, 1]. Score 1.0 lands in the last bin, not off the end."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    return np.minimum((score * n_bins).astype(np.int64), n_bins - 1)


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One equal-width bin of a reliability diagram.

    Empty bins keep ``mean_score`` and ``positive_rate`` as ``NaN`` and contribute 0 to ECE.
    """

    lower: float
    upper: float
    count: int
    mean_score: float
    positive_rate: float

    @property
    def gap(self) -> float:
        if self.count == 0:
            return 0.0
        return float(abs(self.positive_rate - self.mean_score))


def reliability_bins(
    y_true: object,
    score: object,
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> tuple[ReliabilityBin, ...]:
    """Equal-width reliability bins for ``score = P̂(Y = 1)``."""
    y = _as_bool(y_true)
    s = _as_score(score)
    if y.size != s.size:
        raise ValueError("y_true and score must have equal length.")

    index = _bin_index(s, n_bins)
    counts = np.bincount(index, minlength=n_bins).astype(np.int64)
    score_sums = np.bincount(index, weights=s, minlength=n_bins)
    pos_sums = np.bincount(index, weights=y.astype(np.float64), minlength=n_bins)
    mean_score = _divide(score_sums, counts)
    pos_rate = _divide(pos_sums, counts)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return tuple(
        ReliabilityBin(
            lower=float(edges[i]),
            upper=float(edges[i + 1]),
            count=int(counts[i]),
            mean_score=float(mean_score[i]),
            positive_rate=float(pos_rate[i]),
        )
        for i in range(n_bins)
    )


def expected_calibration_error(
    y_true: object,
    score: object,
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> float:
    """Positive-class ECE (Naeini et al. 2015; Guo et al. 2017 eq. 3, adapted -- D12).

    ``Σ_m (|B_m|/n) · |ȳ(B_m) - s̄(B_m)|``. Empty bins contribute 0. ``NaN`` if ``n = 0``.
    """
    y = _as_bool(y_true)
    if y.size == 0:
        return float("nan")
    bins = reliability_bins(y, score, n_bins=n_bins)
    return float(sum(bin_.count * bin_.gap for bin_ in bins) / y.size)


@dataclass(frozen=True, slots=True)
class GroupCalibration:
    """Per-group score diagnostics. ``roc_auc`` is ``NaN`` when ``auc_skip_reason`` is set."""

    label: str
    n: int
    n_positive: int
    n_negative: int
    roc_auc: float
    auc_skip_reason: str | None
    ece: float
    n_bins: int
    bins: tuple[ReliabilityBin, ...]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    n_bins: int
    groups: tuple[GroupCalibration, ...]
    overall: GroupCalibration

    def by_label(self, label: str) -> GroupCalibration:
        for group in self.groups:
            if group.label == label:
                return group
        raise KeyError(label)


def _one_group(label: str, y: BoolArray, score: FloatArray, n_bins: int) -> GroupCalibration:
    n_positive = int(y.sum())
    return GroupCalibration(
        label=label,
        n=int(y.size),
        n_positive=n_positive,
        n_negative=int(y.size - n_positive),
        roc_auc=roc_auc(y, score),
        auc_skip_reason=auc_skip_reason(y),
        ece=expected_calibration_error(y, score, n_bins=n_bins),
        n_bins=n_bins,
        bins=reliability_bins(y, score, n_bins=n_bins),
    )


def calibrate(
    y_true: object,
    score: object,
    group_values: Sequence[np.ndarray],
    group_names: Sequence[str],
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> CalibrationResult:
    """Per-group and overall ROC-AUC, reliability bins and ECE."""
    y = _as_bool(y_true)
    s = _as_score(score)
    if y.size != s.size:
        raise ValueError("y_true and score must have equal length.")
    if not group_names:
        raise ValueError("At least one sensitive column name is required.")
    codes, keys = _group_codes(group_values)
    labels = tuple(" × ".join(key) for key in keys)
    groups = tuple(
        _one_group(labels[i], y[codes == i], s[codes == i], n_bins) for i in range(len(keys))
    )
    overall_names = " × ".join(group_names)
    return CalibrationResult(
        n_bins=n_bins,
        groups=groups,
        overall=_one_group(f"overall ({overall_names})", y, s, n_bins),
    )


@dataclass(frozen=True, slots=True)
class ThresholdSweep:
    """Per-group TPR, FPR and selection rate at each threshold, with ``score >= t``.

    A group that has no positives (or no negatives) at the *data* level has ``NaN`` TPR
    (FPR) at every threshold -- moving ``t`` cannot invent a class that is not there.
    """

    thresholds: FloatArray
    group_labels: tuple[str, ...]
    tpr: FloatArray
    fpr: FloatArray
    selection_rate: FloatArray


def threshold_sweep(
    y_true: object,
    score: object,
    group_values: Sequence[np.ndarray],
    group_names: Sequence[str],
    *,
    thresholds: object | None = None,
) -> ThresholdSweep:
    """How per-group TPR/FPR change as the score cut-off moves.

    ``searchsorted`` on the sorted scores of each class makes this ``O(n log n)`` per
    group rather than ``O(n × T)``. Thresholds default to 101 equally spaced points on
    [0, 1], which is what a slider needs.
    """
    y = _as_bool(y_true)
    s = _as_score(score)
    if y.size != s.size:
        raise ValueError("y_true and score must have equal length.")
    if not group_names:
        raise ValueError("At least one sensitive column name is required.")
    t = (
        np.linspace(0.0, 1.0, DEFAULT_N_THRESHOLDS)
        if thresholds is None
        else np.asarray(thresholds, dtype=np.float64)
    )
    codes, keys = _group_codes(group_values)
    n_groups = len(keys)
    tpr = np.full((t.size, n_groups), np.nan)
    fpr = np.full((t.size, n_groups), np.nan)
    selection = np.full((t.size, n_groups), np.nan)

    for group in range(n_groups):
        mask = codes == group
        n = int(mask.sum())
        if n == 0:
            continue
        pos = np.sort(s[mask & y])
        neg = np.sort(s[mask & ~y])
        n_pos, n_neg = pos.size, neg.size
        # score >= t  <=>  index of first score >= t, counting from the right.
        if n_pos:
            tpr[:, group] = (n_pos - np.searchsorted(pos, t, side="left")) / n_pos
        if n_neg:
            fpr[:, group] = (n_neg - np.searchsorted(neg, t, side="left")) / n_neg
        selection[:, group] = (
            n_pos
            - (np.searchsorted(pos, t, side="left") if n_pos else 0)
            + n_neg
            - (np.searchsorted(neg, t, side="left") if n_neg else 0)
        ) / n

    return ThresholdSweep(
        thresholds=t,
        group_labels=tuple(" × ".join(key) for key in keys),
        tpr=tpr,
        fpr=fpr,
        selection_rate=selection,
    )
