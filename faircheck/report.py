"""Markdown (and derived PDF) report. The single source of truth for the CLI and the app.

Every number printed here comes from an :class:`AuditResult` that already ran the core
library. This module formats; it does not recompute a rate. ``NaN`` is rendered ``n/a``
-- pandas' default ``NaN`` string is a rendering bug, not a value (D3).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from faircheck.bootstrap import DEFAULT_N_BOOT, BootstrapResult, bootstrap_audit
from faircheck.calibration import DEFAULT_N_BINS, CalibrationResult, calibrate
from faircheck.guidance import (
    GuidanceAnswers,
    Recommendation,
    format_interval,
    format_rate,
    recommend,
    summarize,
    verdict_for_gap,
)
from faircheck.impossibility import ImpossibilityResult, explain_impossibility
from faircheck.metrics import GAPS, RATES, confusion_counts, fails_four_fifths
from faircheck.types import ColumnMapping, GroupCounts, ValidatedData
from faircheck.validate import SMALL_GROUP_THRESHOLD, validate


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Build a GitHub-flavoured table without depending on the ``tabulate`` extra."""

    def cell(value: object) -> str:
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(cell(h) for h in headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(cell(c) for c in row) + " |" for row in rows)
    return header + "\n" + rule + "\n" + body


DISCLAIMER = "Screening tool. Not a legal or causal conclusion."
FOUR_FIFTHS_NOTE = (
    "The four-fifths (80%) rule is a screening heuristic that US federal enforcement "
    "agencies use as a rule of thumb, not a legal finding. Smaller gaps can still "
    "matter, and larger gaps may not when they rest on small numbers "
    "(29 CFR 1607.4(D); EEOC Q&A)."
)
PRIVACY_NOTE = "Uploaded data is processed in memory only and is never stored or logged."

RATES_LABEL = {spec.key: spec.label for spec in RATES}
GAPS_BY_LABEL = {spec.key: spec.label for spec in GAPS}


@dataclass(frozen=True, slots=True)
class AuditResult:
    """Everything the report needs, computed once."""

    data: ValidatedData
    counts: GroupCounts
    bootstrap: BootstrapResult
    impossibility: ImpossibilityResult
    calibration: CalibrationResult | None
    recommendation: Recommendation
    summary: tuple[str, ...]
    answers: GuidanceAnswers | None


def run_audit(
    frame: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    n_bins: int = DEFAULT_N_BINS,
    answers: GuidanceAnswers | None = None,
) -> AuditResult:
    """Validate, count, bootstrap, and (if scores are present) calibrate."""
    data = validate(frame, mapping)
    counts = confusion_counts(data.y_true, data.y_pred, data.group_values, mapping.sensitive)
    boot = bootstrap_audit(counts, n_boot=n_boot, seed=seed)
    impossibility = explain_impossibility(counts, reference_index=boot.reference_index)
    calibration = None
    if data.score is not None:
        calibration = calibrate(
            data.y_true, data.score, data.group_values, mapping.sensitive, n_bins=n_bins
        )
    recommendation = recommend(answers)
    summary = summarize(boot, counts, impossibility)
    return AuditResult(
        data=data,
        counts=counts,
        bootstrap=boot,
        impossibility=impossibility,
        calibration=calibration,
        recommendation=recommendation,
        summary=summary,
        answers=answers,
    )


def render_markdown(audit: AuditResult) -> str:
    """Full report as Markdown. Byte-identical between the CLI and the app."""
    sections = [
        _header(audit),
        _summary(audit),
        _groups(audit),
        _gaps(audit),
        _guidance(audit),
        _scores(audit),
        _impossibility(audit),
        _methods(audit),
        f"\n---\n\n*{DISCLAIMER}*\n",
    ]
    text = "\n\n".join(section for section in sections if section)
    if re.search(r"\bnan\b", text, flags=re.IGNORECASE):
        raise RuntimeError("report leaked a NaN token; formatters must use n/a")
    return text


def write_markdown(audit: AuditResult, path: str | Path) -> Path:
    dest = Path(path)
    dest.write_text(render_markdown(audit), encoding="utf-8")
    return dest


def write_pdf(audit: AuditResult, path: str | Path) -> Path:
    """Derived PDF of the Markdown report. Pure-Python ``fpdf2``, no system libraries (D6)."""
    try:
        from fpdf import FPDF
        from fpdf.enums import WrapMode
    except ImportError as exc:
        raise RuntimeError(
            "PDF export needs fpdf2 (a pure-Python wheel). "
            "Install it with: pip install 'faircheck[pdf]'"
        ) from exc

    dest = Path(path)
    markdown = render_markdown(audit)
    pdf = FPDF(format="Letter")
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)

    def emit(text: str, *, height: float) -> None:
        # w=0 means "from current x to the right margin". After a wrapped table
        # row, x can sit near the right edge, so a later w=0 call has no room
        # for even one Courier character. Always write from the left margin.
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(
            w=pdf.epw,
            h=height,
            text=text if text else " ",
            wrapmode=WrapMode.CHAR,
        )

    for raw_line in markdown.splitlines():
        line = _pdf_safe(raw_line)
        if line.startswith("# "):
            pdf.set_font("Helvetica", style="B", size=16)
            emit(line[2:], height=8)
            pdf.ln(2)
            pdf.set_font("Helvetica", size=10)
        elif line.startswith("## "):
            pdf.set_font("Helvetica", style="B", size=12)
            emit(line[3:], height=7)
            pdf.ln(1)
            pdf.set_font("Helvetica", size=10)
        elif line.startswith("### "):
            pdf.set_font("Helvetica", style="B", size=11)
            emit(line[4:], height=6)
            pdf.set_font("Helvetica", size=10)
        elif line.startswith("|"):
            pdf.set_font("Courier", size=8)
            emit(line, height=4)
            pdf.set_font("Helvetica", size=10)
        else:
            emit(line, height=5)
    pdf.output(str(dest))
    return dest


def _pdf_safe(text: str) -> str:
    """Helvetica is Latin-1; fold markdown emphasis and non-Latin glyphs rather than crash."""
    folded = (
        text.replace("**", "")
        .replace("\u00d7", " x ")
        .replace("\u2212", "-")
        .replace("\u2014", "--")
        .replace("\u2013", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    return folded.encode("latin-1", errors="replace").decode("latin-1")


def _header(audit: AuditResult) -> str:
    data, boot, counts = audit.data, audit.bootstrap, audit.counts
    mapping = data.mapping
    small = int((counts.sizes < SMALL_GROUP_THRESHOLD).sum())
    ovr = ""
    if data.is_one_vs_rest:
        ovr = f" (one-vs-rest over {len(data.label_values)} label values)"
    lines = [
        "# FairCheck report",
        "",
        f"{DISCLAIMER} {PRIVACY_NOTE}",
        "",
        f"- Rows audited: **{data.n_rows_used:,}** of {data.n_rows_input:,} in the file",
        f"- Groups: **{counts.n_groups}** ({', '.join(counts.labels())})",
        f"- Sensitive columns: {', '.join(mapping.sensitive)}",
        f"- Positive (favorable) outcome: `{mapping.positive_label}`{ovr}",
        (
            f"- Bootstrap: {boot.n_boot:,} stratified resamples, seed {boot.seed}, "
            f"reference group **{boot.reference_label}**"
        ),
    ]
    if small:
        lines.append(f"- Small-n groups (n < {SMALL_GROUP_THRESHOLD}): **{small}**")
    lines.extend(f"- Warning: {w}" for w in data.warnings)
    return "\n".join(lines)


def _summary(audit: AuditResult) -> str:
    bullets = "\n".join(f"- {s}" for s in audit.summary)
    return f"## Plain-language summary\n\n{bullets}"


def _groups(audit: AuditResult) -> str:
    boot, counts = audit.bootstrap, audit.counts
    headers = ["group", "n", *[spec.label for spec in RATES]]
    rows: list[list[str]] = []
    for i, label in enumerate(counts.labels()):
        row = [label, str(int(counts.sizes[i]))]
        for spec in RATES:
            row.append(format_interval(*boot.rate_interval(spec.key, i)))
        rows.append(row)
    body = _markdown_table(headers, rows)
    return f"""## Per-group rates

Each cell is the point estimate and a 95% percentile bootstrap interval. `n/a` means the
rate's denominator was zero (for example TPR when a group has no positives).

{body}

Accuracy is reported for completeness; it is misleading under class imbalance."""


def _gaps(audit: AuditResult) -> str:
    boot = audit.bootstrap
    rows: list[dict[str, str]] = []
    for spec in GAPS:
        low, point, high = boot.gap_interval(spec.key)
        per_group = []
        for key in spec.inputs:
            bits = ", ".join(
                f"{lab} {format_rate(float(boot.rates[key][i]))}"
                for i, lab in enumerate(boot.group_labels)
            )
            per_group.append(f"{RATES_LABEL[key]}: {bits}")
        rows.append(
            {
                "Gap": spec.label,
                "Value [95% CI]": format_interval(low, point, high),
                "Per-group": "; ".join(per_group),
                "Verdict": verdict_for_gap(spec, boot),
                "Reference": spec.reference,
            }
        )
    table = _markdown_table(
        ["Gap", "Value [95% CI]", "Per-group", "Verdict", "Reference"],
        [
            [
                row["Gap"],
                row["Value [95% CI]"],
                row["Per-group"],
                row["Verdict"],
                row["Reference"],
            ]
            for row in rows
        ],
    )

    di = float(boot.gaps["disparate_impact_ratio"])
    flag = fails_four_fifths(di)
    if flag is True:
        screen = (
            f"Disparate impact ratio {format_rate(di)} is below 0.80, so the four-fifths "
            f"screen is raised. {FOUR_FIFTHS_NOTE}"
        )
    elif flag is False:
        screen = (
            f"Disparate impact ratio {format_rate(di)} is at or above 0.80, so the "
            f"four-fifths screen is not raised. {FOUR_FIFTHS_NOTE}"
        )
    else:
        screen = (
            "The four-fifths screen was not computable (the ratio is undefined). "
            f"{FOUR_FIFTHS_NOTE}"
        )

    return f"""## Summary gaps

Max-min gaps are the literature's headline numbers (and match Fairlearn). They are
non-negative by construction, so a CI that excludes zero is **not** evidence of a
disparity -- sampling noise alone pushes the maximum above the minimum. The **Verdict**
column uses signed contrasts against the reference group ({boot.reference_label}), which
can contain zero. See docs/DECISIONS.md D11.

{table}

{screen}"""


def _guidance(audit: AuditResult) -> str:
    rec = audit.recommendation
    caveats = "".join(f"\n\n- {c}" for c in rec.caveats)
    highlighted = ", ".join(GAPS_BY_LABEL.get(key, key) for key in rec.families) or "(none yet)"
    questions = (
        "The helper asks three questions and highlights a family. "
        "It does not score the model.\n\n"
        "1. Is the decision *assistive* (granting a benefit) or "
        "*punitive* (imposing a penalty)?\n"
        "2. Are the labels trustworthy enough that error rates "
        "(which condition on Y) are meaningful?\n"
        "3. Will a human read the score as a probability of the favorable outcome?"
    )
    return f"""## Which metric fits this decision?

{questions}

**Highlighted:** {highlighted}

**{rec.headline}** {rec.reason}{caveats}"""


def _scores(audit: AuditResult) -> str:
    cal = audit.calibration
    if cal is None:
        return """## Scores

No score column was mapped, so ROC-AUC, reliability diagrams, ECE and the threshold
sweep are skipped. Re-run with a score in [0, 1] to see calibration by group."""

    rows = []
    for group in cal.groups:
        auc = format_rate(group.roc_auc)
        if group.auc_skip_reason:
            auc = f"n/a ({group.auc_skip_reason})"
        rows.append(
            {
                "Group": group.label,
                "n": str(group.n),
                "ROC-AUC": auc,
                f"ECE ({cal.n_bins} bins)": format_rate(group.ece),
            }
        )
    table = _markdown_table(
        ["Group", "n", "ROC-AUC", f"ECE ({cal.n_bins} bins)"],
        [
            [
                row["Group"],
                row["n"],
                row["ROC-AUC"],
                row[f"ECE ({cal.n_bins} bins)"],
            ]
            for row in rows
        ],
    )
    return f"""## Scores

ECE here is the **positive-class** expected calibration error (mean score vs. observed
positive rate in equal-width bins on [0, 1]), not Guo et al.'s top-label ECE. Empty bins
contribute 0. The bin count is part of the number; treat ECE as descriptive.

{table}

Moving the decision threshold trades TPR against FPR and cannot equalise both when base
rates differ. Use the CLI/UI slider for the sweep; the export records ECE and AUC only."""


def _impossibility(audit: AuditResult) -> str:
    imp = audit.impossibility
    rates = _markdown_table(
        ["group", "Base rate", "TPR", "FPR", "PPV"],
        [
            [
                lab,
                format_rate(float(imp.base_rate[i])),
                format_rate(float(imp.tpr[i])),
                format_rate(float(imp.fpr[i])),
                format_rate(float(imp.precision[i])),
            ]
            for i, lab in enumerate(imp.group_labels)
        ],
    )

    identity = (
        "Chouldechova (2017) identity: "
        "`FPR = p/(1-p) · (1-PPV)/PPV · (1-FNR)`. "
        "On this table the implied FPR matches the observed FPR to numerical precision."
    )

    if not imp.tradeoff_applies:
        if imp.is_perfect:
            why = (
                "Every group is classified perfectly (FPR = FNR = 0), which is one of "
                "the two exceptions where the criteria can co-exist."
            )
        else:
            why = (
                "Base rates do not meaningfully differ, so the identity does not force "
                "predictive parity and equal error rates apart."
            )
        return f"""## Why not all at once

{identity}

{why}

{rates}

Kleinberg, Mullainathan and Raghavan (2016, arXiv:1609.05807) prove the score-space
analogue: calibration within groups plus balance for both classes holds only under
perfect prediction or equal base rates."""

    ref = imp.reference_label
    counterfactual_bits = []
    for i, lab in enumerate(imp.group_labels):
        fpr_star = float(imp.counterfactual_fpr_equal_ppv[i])
        if not imp.equal_ppv_feasible[i]:
            counterfactual_bits.append(
                f"{lab}: equalising PPV to {ref}'s is **impossible** without also moving "
                "error rates (the identity would require an FPR outside [0, 1])"
            )
        else:
            counterfactual_bits.append(
                f"{lab}: FPR would be {format_rate(fpr_star)} if PPV matched {ref} "
                f"while prevalence and FNR stayed as observed (observed FPR "
                f"{format_rate(float(imp.fpr[i]))})"
            )
    ppv_bits = ", ".join(
        f"{lab} {format_rate(float(imp.counterfactual_ppv_equal_errors[i]))}"
        for i, lab in enumerate(imp.group_labels)
    )
    return (
        f"""## Why not all at once

{identity}

Base rates differ by {imp.base_rate_spread:.1%} on this file, and the classifier is not
perfect. Predictive parity (equal PPV) and equal error rates therefore cannot hold
together.

- """
        + "\n- ".join(counterfactual_bits)
        + f"""
- If TPR and FPR were equalised to {ref}'s, PPV would have to be {ppv_bits}.

{rates}

Sources: Chouldechova 2017 (arXiv:1610.07524); Kleinberg, Mullainathan and Raghavan 2016
(arXiv:1609.05807)."""
    )


def _methods(audit: AuditResult) -> str:
    boot = audit.bootstrap
    rate_lines = "\n".join(
        f"- {s.label}: `{s.formula}`"
        + (f" (`n/a` when {s.undefined_when})" if s.undefined_when else "")
        for s in RATES
    )
    gap_lines = "\n".join(f"- {s.label}: `{s.formula}` -- {s.reference}" for s in GAPS)
    return f"""## Methods

Stratified percentile bootstrap with {boot.n_boot:,} resamples and seed {boot.seed}.
Because every confusion-matrix rate is a function of the four cell counts, a within-group
row resample is a multinomial draw over those cells (D4). Verdicts use signed contrasts
against {boot.reference_label}, not max-min gap CIs (D11).

{rate_lines}

{gap_lines}"""
