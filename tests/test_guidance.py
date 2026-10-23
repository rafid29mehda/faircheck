"""Metric-selection helper: family highlighting and hedged summary wording (D7, D11)."""

from __future__ import annotations

import numpy as np

from faircheck.bootstrap import bootstrap_audit
from faircheck.guidance import (
    DecisionKind,
    GuidanceAnswers,
    LabelTrust,
    ScoreReading,
    format_interval,
    format_rate,
    interval_method_note,
    recommend,
    summarize,
    verdict_for_gap,
)
from faircheck.impossibility import explain_impossibility
from faircheck.metrics import GAPS_BY_KEY
from tests.test_bootstrap import counts_from


def test_incomplete_answers_highlight_nothing() -> None:
    rec = recommend(None)
    assert rec.families == ()
    assert "three questions" in rec.headline.lower()

    partial = recommend(GuidanceAnswers(decision=DecisionKind.ASSISTIVE))
    assert partial.families == ()


def test_untrusted_labels_highlight_selection_rate_family() -> None:
    rec = recommend(
        GuidanceAnswers(
            decision=DecisionKind.ASSISTIVE,
            labels=LabelTrust.UNTRUSTED,
            scores=ScoreReading.NOT_USED,
        )
    )
    assert rec.families == ("demographic_parity_difference", "disparate_impact_ratio")


def test_punitive_trusted_labels_highlight_equalized_odds() -> None:
    rec = recommend(
        GuidanceAnswers(
            decision=DecisionKind.PUNITIVE,
            labels=LabelTrust.TRUSTED,
            scores=ScoreReading.NOT_USED,
        )
    )
    assert rec.families == ("equalized_odds_difference",)


def test_assistive_trusted_labels_highlight_equal_opportunity() -> None:
    rec = recommend(
        GuidanceAnswers(
            decision=DecisionKind.ASSISTIVE,
            labels=LabelTrust.TRUSTED,
            scores=ScoreReading.NOT_USED,
        )
    )
    assert rec.families == ("equal_opportunity_difference",)


def test_scores_as_probability_adds_predictive_parity() -> None:
    rec = recommend(
        GuidanceAnswers(
            decision=DecisionKind.ASSISTIVE,
            labels=LabelTrust.TRUSTED,
            scores=ScoreReading.AS_PROBABILITY,
        )
    )
    assert rec.families == ("equal_opportunity_difference", "predictive_parity_difference")
    assert any("predictive parity" in caveat.lower() for caveat in rec.caveats)


def test_untrusted_labels_and_probability_scores_flag_the_tension() -> None:
    rec = recommend(
        GuidanceAnswers(
            decision=DecisionKind.PUNITIVE,
            labels=LabelTrust.UNTRUSTED,
            scores=ScoreReading.AS_PROBABILITY,
        )
    )
    assert "demographic_parity_difference" in rec.families
    assert "predictive_parity_difference" in rec.families
    assert any("tension" in caveat.lower() for caveat in rec.caveats)


def test_formatters_never_print_nan() -> None:
    assert format_rate(float("nan")) == "n/a"
    assert format_rate(np.inf) == "n/a"
    assert format_rate(0.4) == "0.400"
    text = format_interval(float("nan"), float("nan"), float("nan"))
    assert "n/a" in text
    assert "nan" not in text.lower()


def test_summary_hedges_when_contrast_ci_includes_zero() -> None:
    """The honesty requirement as an assertion (plan test 9; D11).

    On the spec's hand-computed case the equal-opportunity *max-min* CI excludes
    zero, but the signed TPR contrast includes it. The summary must hedge.
    """
    counts = counts_from([(40, 10, 10, 40), (12, 8, 8, 72)])
    boot = bootstrap_audit(counts, n_boot=1000, seed=0)
    other = next(i for i in range(counts.n_groups) if i != boot.reference_index)
    assert boot.has_clear_contrast("tpr", other) is False
    eo_low, _, _ = boot.gap_interval("equal_opportunity_difference")
    assert eo_low > 0.0

    impossibility = explain_impossibility(counts, reference_index=boot.reference_index)
    joined = " ".join(summarize(boot, counts, impossibility)).lower()
    assert "no clear evidence of a gap" in joined
    assert "tpr" in joined
    assert "ci includes zero" in joined


def test_summary_states_a_clear_selection_rate_gap() -> None:
    counts = counts_from([(400, 100, 100, 400), (120, 80, 80, 720)])
    boot = bootstrap_audit(counts, n_boot=500, seed=0)
    impossibility = explain_impossibility(counts, reference_index=boot.reference_index)
    joined = " ".join(summarize(boot, counts, impossibility))
    assert "Selection rate" in joined
    assert "percentage points" in joined
    assert "95% CI" in joined


def test_verdict_uses_signed_contrasts_not_max_min_ci() -> None:
    truth = np.array([0.25, 0.25, 0.25, 0.25])
    rng = np.random.default_rng(5)
    counts = counts_from([tuple(rng.multinomial(2000, truth).tolist()) for _ in range(3)])
    boot = bootstrap_audit(counts, n_boot=2000, seed=0)
    spec = GAPS_BY_KEY["demographic_parity_difference"]
    gap_low, _, _ = boot.gap_interval(spec.key)
    assert gap_low > 0.0
    assert verdict_for_gap(spec, boot) == "no clear evidence vs reference"


def test_verdict_is_not_computable_when_the_rate_is_undefined() -> None:
    counts = counts_from([(40, 10, 10, 40), (0, 5, 0, 45)])
    boot = bootstrap_audit(counts, n_boot=200, seed=0)
    spec = GAPS_BY_KEY["equal_opportunity_difference"]
    assert verdict_for_gap(spec, boot) == "not computable"


def test_small_n_groups_are_called_unstable() -> None:
    counts = counts_from([(4, 1, 1, 4), (40, 10, 10, 40)])
    boot = bootstrap_audit(counts, n_boot=100, seed=0)
    impossibility = explain_impossibility(counts, reference_index=boot.reference_index)
    joined = " ".join(summarize(boot, counts, impossibility))
    assert "n = 10" in joined
    assert "unstable" in joined


def test_summary_says_when_a_clear_result_survived_the_adjustment() -> None:
    counts = counts_from([(237, 197, 884, 2682), (2696, 887, 181, 236)])
    boot = bootstrap_audit(counts, n_boot=500, seed=0)
    impossibility = explain_impossibility(counts, reference_index=boot.reference_index)
    joined = " ".join(summarize(boot, counts, impossibility))
    assert "still clear after adjusting for 4 comparisons" in joined


def test_summary_explains_a_contrast_that_only_the_adjustment_removes() -> None:
    counts = counts_from([(300, 150, 200, 350), (280, 160, 220, 340), (330, 140, 170, 360)])
    boot = bootstrap_audit(counts, n_boot=2000, seed=0)
    impossibility = explain_impossibility(counts, reference_index=boot.reference_index)
    joined = " ".join(summarize(boot, counts, impossibility))
    assert "excludes zero on its own" in joined
    assert "after adjusting for 8 comparisons" in joined
    assert "no clear evidence of a gap" in joined


def test_summary_makes_no_adjustment_claim_when_adjustment_is_off() -> None:
    counts = counts_from([(237, 197, 884, 2682), (2696, 887, 181, 236)])
    boot = bootstrap_audit(counts, n_boot=500, seed=0, adjust="none")
    impossibility = explain_impossibility(counts, reference_index=boot.reference_index)
    joined = " ".join(summarize(boot, counts, impossibility))
    assert "adjusting" not in joined


def test_interval_method_note_names_only_the_posterior_groups() -> None:
    boot = bootstrap_audit(counts_from([(300, 150, 200, 350), (3, 40, 60, 200)]), n_boot=200)
    note = interval_method_note(boot)
    assert note is not None
    assert "B" in note
    assert "Dirichlet" in note
    assert "fewer than 5 rows" in note
    assert "A and" not in note


def test_interval_method_note_is_silent_when_every_group_uses_the_bootstrap() -> None:
    boot = bootstrap_audit(counts_from([(300, 150, 200, 350), (280, 160, 220, 340)]), n_boot=200)
    assert interval_method_note(boot) is None
