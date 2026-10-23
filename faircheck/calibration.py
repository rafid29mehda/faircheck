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

#: Row-level resamples for the score-metric intervals. Lower than the confusion-cell
#: default (1000) because each resample costs O(rows), unlike the multinomial fast path.
DEFAULT_SCORE_N_BOOT: Final = 200
SCORE_QUANTILES: Final = (0.025, 0.975)

_NAN_PAIR: Final = (float("nan"), float("nan"))

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
    #: 95% percentile row-level bootstrap intervals (D13). ``NaN`` pairs when intervals
    #: are switched off (``n_boot=0``) or the metric is undefined in every resample.
    roc_auc_ci: tuple[float, float] = _NAN_PAIR
    ece_ci: tuple[float, float] = _NAN_PAIR
    #: Share of resamples in which a class went missing, so ROC-AUC did not exist.
    auc_undefined_fraction: float = float("nan")


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    n_bins: int
    groups: tuple[GroupCalibration, ...]
    overall: GroupCalibration
    n_boot: int = 0
    seed: int = 0

    def by_label(self, label: str) -> GroupCalibration:
        for group in self.groups:
            if group.label == label:
                return group
        raise KeyError(label)


def _percentile_pair(values: FloatArray) -> tuple[float, float]:
    """95% percentile interval of the finite values, or NaN if none are finite.

    Guarded explicitly because ``nanquantile`` on an all-NaN slice warns, and
    ``filterwarnings = ["error"]`` is part of this project's contract.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return _NAN_PAIR
    low, high = np.quantile(finite, SCORE_QUANTILES)
    return float(low), float(high)


def _score_bootstrap(
    y: BoolArray,
    score: FloatArray,
    *,
    n_bins: int,
    n_boot: int,
    seed: Sequence[int],
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Row-level bootstrap of ROC-AUC and ECE for one group (D13).

    Each resample draws ``n`` rows with replacement. Only the per-row multiplicity matters,
    so scores are sorted and binned once and every resample is a weighted count:

    * AUC: pooled positive/negative weights per distinct score, then
      ``sum(pos_w * (neg_below + 0.5 * neg_tied)) / (W_pos * W_neg)`` -- the same
      Mann-Whitney statistic as :func:`roc_auc`, including ties at 0.5.
    * ECE: ``|B_m| * |positive rate - mean score|`` equals ``|sum(y) - sum(s)|`` in a bin,
      so ECE is ``sum_m |sum(y) - sum(s)| / n`` with weighted sums.

    Resamples that lose a class are skipped for AUC and counted, as the confusion-cell
    bootstrap does. ECE is biased upward in sparse bins, so its interval describes the
    sampling variability of the estimate and need not contain the point estimate.
    """
    n = int(y.size)
    if n_boot == 0 or n == 0:
        return _NAN_PAIR, _NAN_PAIR, float("nan")

    rng = np.random.default_rng(list(seed))
    distinct, inverse = np.unique(score, return_inverse=True)
    bins = _bin_index(score, n_bins)
    positive = y.astype(np.float64)
    negative = 1.0 - positive

    aucs = np.full(n_boot, np.nan)
    eces = np.empty(n_boot)
    for b in range(n_boot):
        weight = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(np.float64)
        pos_w = np.bincount(inverse, weights=weight * positive, minlength=distinct.size)
        neg_w = np.bincount(inverse, weights=weight * negative, minlength=distinct.size)
        total_pos, total_neg = float(pos_w.sum()), float(neg_w.sum())
        if total_pos > 0.0 and total_neg > 0.0:
            below = np.cumsum(neg_w) - neg_w
            aucs[b] = float((pos_w * (below + 0.5 * neg_w)).sum() / (total_pos * total_neg))
        score_sum = np.bincount(bins, weights=weight * score, minlength=n_bins)
        pos_sum = np.bincount(bins, weights=weight * positive, minlength=n_bins)
        eces[b] = float(np.abs(pos_sum - score_sum).sum() / n)

    undefined = float(1.0 - np.isfinite(aucs).mean())
    return _percentile_pair(aucs), _percentile_pair(eces), undefined


def _one_group(
    label: str,
    y: BoolArray,
    score: FloatArray,
    n_bins: int,
    *,
    n_boot: int = 0,
    seed: Sequence[int] = (0,),
) -> GroupCalibration:
    n_positive = int(y.sum())
    auc_ci, ece_ci, undefined = _score_bootstrap(y, score, n_bins=n_bins, n_boot=n_boot, seed=seed)
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
        roc_auc_ci=auc_ci,
        ece_ci=ece_ci,
        auc_undefined_fraction=undefined,
    )


def calibrate(
    y_true: object,
    score: object,
    group_values: Sequence[np.ndarray],
    group_names: Sequence[str],
    *,
    n_bins: int = DEFAULT_N_BINS,
    n_boot: int = DEFAULT_SCORE_N_BOOT,
    seed: int = 0,
) -> CalibrationResult:
    """Per-group and overall ROC-AUC, reliability bins and ECE, each with a bootstrap interval.

    ``n_boot=0`` skips the intervals and leaves them ``NaN``.
    """
    if n_boot < 0:
        raise ValueError(f"n_boot must be at least 0, got {n_boot}")
    y = _as_bool(y_true)
    s = _as_score(score)
    if y.size != s.size:
        raise ValueError("y_true and score must have equal length.")
    if not group_names:
        raise ValueError("At least one sensitive column name is required.")
    codes, keys = _group_codes(group_values)
    labels = tuple(" × ".join(key) for key in keys)
    # One independent stream per group (and one for the overall row), all fixed by ``seed``.
    groups = tuple(
        _one_group(labels[i], y[codes == i], s[codes == i], n_bins, n_boot=n_boot, seed=(seed, i))
        for i in range(len(keys))
    )
    overall_names = " × ".join(group_names)
    return CalibrationResult(
        n_bins=n_bins,
        groups=groups,
        overall=_one_group(
            f"overall ({overall_names})", y, s, n_bins, n_boot=n_boot, seed=(seed, len(keys))
        ),
        n_boot=n_boot,
        seed=seed,
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
