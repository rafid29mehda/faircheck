"""Metric-selection helper and the plain-language summary.

Both are deterministic templates. There is no model and no overall fairness grade: the
helper only highlights a *family* of metrics given the decision context, and the summary
only restates numbers that :class:`BootstrapResult` already computed. Verdicts use signed
contrasts (D11), never max-min gap CIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from faircheck.bootstrap import BootstrapResult
from faircheck.impossibility import ImpossibilityResult
from faircheck.metrics import RATES_BY_KEY, GapSpec
from faircheck.types import GroupCounts
from faircheck.validate import SMALL_GROUP_THRESHOLD


class DecisionKind(Enum):
    """Is the favorable outcome a benefit being granted, or a penalty being imposed?"""

    ASSISTIVE = "assistive"
    PUNITIVE = "punitive"


class LabelTrust(Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


class ScoreReading(Enum):
    """How will a human use the score, if one is present?"""

    AS_PROBABILITY = "as_probability"
    AS_RANK_ONLY = "as_rank_only"
    NOT_USED = "not_used"


@dataclass(frozen=True, slots=True)
class GuidanceAnswers:
    decision: DecisionKind | None = None
    labels: LabelTrust | None = None
    scores: ScoreReading | None = None

    @property
    def is_complete(self) -> bool:
        return self.decision is not None and self.labels is not None and self.scores is not None


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Which gap family the helper highlights. ``families`` are keys into :data:`GAPS`."""

    families: tuple[str, ...]
    headline: str
    reason: str
    caveats: tuple[str, ...]


def recommend(answers: GuidanceAnswers | None) -> Recommendation:
    """Map the three questions to a metric family. No number is produced here."""
    if answers is None or not answers.is_complete:
        return Recommendation(
            families=(),
            headline="Answer the three questions to highlight a metric family.",
            reason=(
                "Until then every metric is shown. None of them is a fairness grade -- "
                "different criteria can disagree, and that disagreement is the point."
            ),
            caveats=(),
        )

    assert answers.decision is not None
    assert answers.labels is not None
    assert answers.scores is not None

    caveats: list[str] = []
    if answers.labels is LabelTrust.UNTRUSTED and answers.scores is ScoreReading.AS_PROBABILITY:
        caveats.append(
            "Reading a score as P(Y) while also treating Y as untrustworthy is a tension: "
            "predictive parity conditions on the flag, not on Y, but calibration still "
            "needs Y. Treat the calibration numbers as descriptive only."
        )

    families: tuple[str, ...]
    if answers.labels is LabelTrust.UNTRUSTED:
        families = ("demographic_parity_difference", "disparate_impact_ratio")
        headline = "Selection-rate parity (demographic parity / disparate impact)"
        reason = (
            "Error rates condition on the true label. If that label is not trusted, the "
            "honest comparison is who is selected, not who is correctly classified."
        )
    elif answers.decision is DecisionKind.PUNITIVE:
        families = ("equalized_odds_difference",)
        headline = "Equalized odds (TPR and FPR together)"
        reason = (
            "A punitive decision harms people who are flagged by mistake, so false "
            "positives carry real cost; equal opportunity (TPR only) would ignore them."
        )
    else:
        families = ("equal_opportunity_difference",)
        headline = "Equal opportunity (TPR / recall)"
        reason = (
            "An assistive decision is about who deserved the favorable outcome and got "
            "it. Equal opportunity equalises that rate; it does not constrain false "
            "positives."
        )

    if answers.scores is ScoreReading.AS_PROBABILITY:
        families = (*families, "predictive_parity_difference")
        caveats.append(
            "Because scores will be read as probabilities, also look at per-group "
            "precision (predictive parity) and ECE. Those can disagree with equal "
            "error rates when base rates differ -- see 'Why not all at once'."
        )

    return Recommendation(
        families=families,
        headline=headline,
        reason=reason,
        caveats=tuple(caveats),
    )


def _fmt_rate(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


def _fmt_pp(delta: float) -> str:
    """Format a rate difference in percentage points, never as a raw 'nan'."""
    if not np.isfinite(delta):
        return "n/a"
    points = abs(delta) * 100.0
    if points >= 1.0:
        return f"{points:.0f} percentage points"
    return f"{points:.1f} percentage points"


def _fmt_ci_pp(low: float, high: float) -> str:
    if not (np.isfinite(low) and np.isfinite(high)):
        return "n/a"
    return f"{low * 100:.0f} to {high * 100:.0f} pp"


def _largest_clear_contrast(
    boot: BootstrapResult, rate_key: str
) -> tuple[int, float, float, float] | None:
    """Return (group_index, low, point, high) for the largest |contrast| that excludes 0."""
    best: tuple[int, float, float, float] | None = None
    best_abs = -1.0
    for index in range(len(boot.group_labels)):
        if index == boot.reference_index:
            continue
        if boot.has_clear_contrast(rate_key, index) is not True:
            continue
        low, point, high = boot.contrast_interval(rate_key, index)
        if abs(point) > best_abs:
            best_abs = abs(point)
            best = (index, low, point, high)
    return best


def summarize(
    boot: BootstrapResult,
    counts: GroupCounts,
    impossibility: ImpossibilityResult,
) -> tuple[str, ...]:
    """Three to five sentences. Hedged wherever a contrast CI includes zero (D11)."""
    sentences: list[str] = []
    ref = boot.reference_label

    for rate_key in ("selection_rate", "tpr", "fpr", "precision"):
        spec = RATES_BY_KEY[rate_key]
        found = _largest_clear_contrast(boot, rate_key)
        if found is None:
            # Only say "no clear evidence" when the contrast was actually computable.
            computable = any(
                boot.has_clear_contrast(rate_key, i) is False
                for i in range(len(boot.group_labels))
                if i != boot.reference_index
            )
            if computable:
                sentences.append(
                    f"There is **no clear evidence of a gap** in {spec.label} relative to "
                    f"{ref} (the 95% CI includes zero)."
                )
            continue
        index, low, point, high = found
        other = boot.group_labels[index]
        direction = "higher" if point > 0 else "lower"
        sentences.append(
            f"{other}'s {spec.label} is {_fmt_pp(point)} {direction} than {ref}'s "
            f"(95% CI {_fmt_ci_pp(low, high)})."
        )

    small = [
        f"{label} (n = {int(n)})"
        for label, n in zip(counts.labels(), counts.sizes, strict=True)
        if int(n) < SMALL_GROUP_THRESHOLD
    ]
    if small:
        sentences.append(
            "Estimates for "
            + ", ".join(small)
            + f" rest on fewer than {SMALL_GROUP_THRESHOLD} rows and are unstable."
        )

    if impossibility.tradeoff_applies:
        labels = " vs ".join(
            f"{lab} {impossibility.base_rate[i]:.0%}"
            for i, lab in enumerate(impossibility.group_labels)
        )
        sentences.append(
            f"Base rates differ ({labels}), so calibration/predictive parity and equal "
            "error rates cannot all hold for an imperfect classifier -- see "
            "'Why not all at once'."
        )

    return tuple(sentences[:5])


def verdict_for_gap(spec: GapSpec, boot: BootstrapResult) -> str:
    """Contrast-based chip for a gap card. Does *not* look at the max-min CI (D11)."""
    flags = [
        boot.has_clear_contrast(rate_key, index)
        for rate_key in spec.inputs
        for index in range(len(boot.group_labels))
        if index != boot.reference_index
    ]
    if any(flag is True for flag in flags):
        return "clear difference vs reference"
    if flags and all(flag is None for flag in flags):
        return "not computable"
    return "no clear evidence vs reference"


def format_rate(value: float) -> str:
    """Public formatter so the report and CLI never invent a 'NaN' string."""
    return _fmt_rate(value)


def format_interval(low: float, point: float, high: float) -> str:
    return f"{_fmt_rate(point)} [{_fmt_rate(low)}, {_fmt_rate(high)}]"
