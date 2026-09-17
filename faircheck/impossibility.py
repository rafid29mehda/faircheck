"""Why calibration and equal error rates cannot all hold when base rates differ.

Chouldechova (2017, arXiv:1610.07524) gives the identity that every confusion matrix
obeys:

    FPR = p / (1 - p)  ·  (1 - PPV) / PPV  ·  (1 - FNR)

So if two groups have different prevalence ``p`` and an imperfect classifier, they cannot
simultaneously have equal predictive parity (equal PPV) *and* equal error rates (equal
FPR and FNR). Kleinberg, Mullainathan and Raghavan (2016, arXiv:1609.05807) prove the
score-space analogue: calibration within groups plus balance for both classes holds only
under perfect prediction or equal base rates.

This module does not produce a verdict. It substitutes the user's own rates into the
identity and into two counterfactuals -- equalise PPV, or equalise error rates -- so the
report can show *which* of those parities the numbers have already given up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from faircheck.metrics import _divide, per_group_rates
from faircheck.types import BoolArray, FloatArray, GroupCounts

#: Base rates that differ by less than this are treated as equal for the explainer.
#: Smaller than any gap a screening tool would care about; larger than float noise.
BASE_RATE_EQUAL_TOL: Final = 1e-6


def chouldechova_fpr(base_rate: object, precision: object, fnr: object) -> FloatArray:
    """FPR implied by prevalence, PPV and FNR. ``NaN`` where a denominator is 0."""
    p = np.asarray(base_rate, dtype=np.float64)
    ppv = np.asarray(precision, dtype=np.float64)
    miss = np.asarray(fnr, dtype=np.float64)
    return _divide(p, 1.0 - p) * _divide(1.0 - ppv, ppv) * (1.0 - miss)


def chouldechova_precision(base_rate: object, fpr: object, tpr: object) -> FloatArray:
    """PPV implied by prevalence, FPR and TPR (the identity solved for PPV).

    ``PPV = 1 / (1 + FPR · (1-p) / (TPR · p))``. ``NaN`` when TPR or p is 0.
    """
    p = np.asarray(base_rate, dtype=np.float64)
    fpr_ = np.asarray(fpr, dtype=np.float64)
    tpr_ = np.asarray(tpr, dtype=np.float64)
    ratio = _divide(fpr_ * (1.0 - p), tpr_ * p)
    return _divide(1.0, 1.0 + ratio)


@dataclass(frozen=True, slots=True)
class ImpossibilityResult:
    """Numbers the report needs to explain the trade-off from the user's data.

    ``tradeoff_applies`` is the only boolean the UI should branch on. When it is false,
    either the base rates match (so the identity does not force a split) or every group
    is classified perfectly (the other exception in Chouldechova / Kleinberg).
    """

    group_labels: tuple[str, ...]
    reference_index: int
    base_rate: FloatArray
    precision: FloatArray
    tpr: FloatArray
    fpr: FloatArray
    fnr: FloatArray
    implied_fpr: FloatArray
    base_rate_spread: float
    base_rates_differ: bool
    is_perfect: bool
    tradeoff_applies: bool
    counterfactual_fpr_equal_ppv: FloatArray
    counterfactual_ppv_equal_errors: FloatArray
    #: False where equalising PPV would require an FPR outside [0, 1] -- i.e. the
    #: combination is not a possible classifier, so the report should say so.
    equal_ppv_feasible: BoolArray

    @property
    def reference_label(self) -> str:
        return self.group_labels[self.reference_index]


def explain_impossibility(
    counts: GroupCounts,
    *,
    reference_index: int | None = None,
) -> ImpossibilityResult:
    """Substitute observed rates into Chouldechova's identity and the two counterfactuals.

    ``reference_index`` defaults to the largest group, matching :func:`bootstrap_audit`,
    so the "hold this group's PPV / error rates fixed" story is about the same group the
    rest of the report uses as the comparison point.
    """
    rates = per_group_rates(counts.cells)
    reference = int(np.argmax(counts.sizes)) if reference_index is None else reference_index
    if not 0 <= reference < counts.n_groups:
        raise ValueError(
            f"reference_index {reference} is outside the {counts.n_groups} observed groups"
        )

    p = rates["base_rate"]
    ppv = rates["precision"]
    tpr = rates["tpr"]
    fpr = rates["fpr"]
    fnr = rates["fnr"]
    implied = chouldechova_fpr(p, ppv, fnr)

    spread = float(np.nanmax(p) - np.nanmin(p)) if counts.n_groups else 0.0
    differ = bool(np.isfinite(spread) and spread > BASE_RATE_EQUAL_TOL)
    # Perfect: every *defined* error rate is 0. A group with no positives has undefined
    # FNR; that is not perfection, it is a missing class.
    defined_fnr = fnr[np.isfinite(fnr)]
    defined_fpr = fpr[np.isfinite(fpr)]
    perfect = (
        defined_fnr.size > 0
        and defined_fpr.size > 0
        and bool(np.all(defined_fnr == 0.0) and np.all(defined_fpr == 0.0))
        and defined_fnr.size == counts.n_groups
        and defined_fpr.size == counts.n_groups
    )

    # Hold the reference group's PPV and each group's own prevalence and FNR.
    equal_ppv_fpr = chouldechova_fpr(p, np.full_like(ppv, ppv[reference]), fnr)
    # A value outside [0, 1] is not a rounding error: it means those (p, PPV, FNR)
    # cannot occur together, so equalising PPV at the reference value is impossible
    # without also moving error rates.
    equal_ppv_feasible = (
        np.isfinite(equal_ppv_fpr) & (equal_ppv_fpr >= 0.0) & (equal_ppv_fpr <= 1.0)
    )
    # Hold the reference group's TPR and FPR; prevalence still differs, so PPV must move.
    equal_err_ppv = chouldechova_precision(
        p, np.full_like(fpr, fpr[reference]), np.full_like(tpr, tpr[reference])
    )

    return ImpossibilityResult(
        group_labels=counts.labels(),
        reference_index=reference,
        base_rate=p,
        precision=ppv,
        tpr=tpr,
        fpr=fpr,
        fnr=fnr,
        implied_fpr=implied,
        base_rate_spread=spread,
        base_rates_differ=differ,
        is_perfect=perfect,
        tradeoff_applies=bool(differ and not perfect),
        counterfactual_fpr_equal_ppv=equal_ppv_fpr,
        counterfactual_ppv_equal_errors=equal_err_ppv,
        equal_ppv_feasible=np.asarray(equal_ppv_feasible, dtype=bool),
    )
