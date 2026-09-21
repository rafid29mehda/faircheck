"""FairCheck Streamlit app. Thin caller of the core library; no metric is computed here."""

from __future__ import annotations

import io
import tempfile
from dataclasses import replace
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from examples.catalog import EXAMPLES, EXAMPLES_BY_KEY
from faircheck.bootstrap import DEFAULT_N_BOOT
from faircheck.calibration import threshold_sweep
from faircheck.guidance import (
    DecisionKind,
    GuidanceAnswers,
    LabelTrust,
    ScoreReading,
    format_interval,
    format_rate,
    recommend,
    verdict_for_gap,
)
from faircheck.metrics import GAPS, RATES, fails_four_fifths
from faircheck.report import (
    DISCLAIMER,
    FOUR_FIFTHS_NOTE,
    PRIVACY_NOTE,
    AuditResult,
    render_markdown,
    run_audit,
    write_pdf,
)
from faircheck.types import ColumnMapping
from faircheck.validate import SMALL_GROUP_THRESHOLD, ValidationError, label_options

# Okabe-Ito qualitative palette (https://jfly.uni-koeln.de/color/). Colour is never
# the only encoding: every series also has a distinct marker and a direct label.
OKABE_ITO: tuple[str, ...] = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)
MARKERS: tuple[str, ...] = (
    "circle",
    "square",
    "triangle-up",
    "diamond",
    "cross",
    "triangle-down",
    "triangle-right",
    "triangle-left",
)
NONE = "(none)"
MAX_UPLOAD_MB = 50


def _palette(n: int) -> list[str]:
    return [OKABE_ITO[i % len(OKABE_ITO)] for i in range(n)]


def _markers(n: int) -> list[str]:
    return [MARKERS[i % len(MARKERS)] for i in range(n)]


def _index(options: list[str], preferred: str | None) -> int:
    if preferred is not None and preferred in options:
        return options.index(preferred)
    return 0


@st.cache_data(show_spinner=False)
def _load_example(key: str) -> pd.DataFrame:
    return pd.read_csv(EXAMPLES_BY_KEY[key].path)


@st.cache_data(show_spinner=False)
def _load_upload(data: bytes, name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))


@st.cache_data(show_spinner="Running the audit...")
def _cached_audit(
    frame: pd.DataFrame,
    label: str,
    prediction: str,
    sensitive: tuple[str, ...],
    positive: str,
    score: str | None,
    n_boot: int,
    seed: int,
) -> AuditResult:
    mapping = ColumnMapping(
        label=label,
        prediction=prediction,
        sensitive=sensitive,
        positive_label=positive,
        score=score,
    )
    return run_audit(frame, mapping, n_boot=n_boot, seed=seed)


def _pdf_bytes(audit: AuditResult) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "faircheck-report.pdf"
        write_pdf(audit, path)
        return path.read_bytes()


def _selection_chart(audit: AuditResult) -> alt.Chart | None:
    boot = audit.bootstrap
    rows: list[dict[str, object]] = []
    for i, label in enumerate(boot.group_labels):
        low, point, high = boot.rate_interval("selection_rate", i)
        if not np.isfinite(point):
            continue
        rows.append(
            {
                "group": label,
                "rate": float(point),
                "low": float(low) if np.isfinite(low) else float(point),
                "high": float(high) if np.isfinite(high) else float(point),
            }
        )
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    labels = [str(r["group"]) for r in rows]
    colors = _palette(len(labels))
    shapes = _markers(len(labels))
    color = alt.Color(
        "group:N",
        scale=alt.Scale(domain=labels, range=colors),
        legend=None,
    )
    shape = alt.Shape(
        "group:N",
        scale=alt.Scale(domain=labels, range=shapes),
        legend=None,
    )
    bars = (
        alt.Chart(frame)
        .mark_bar(size=18)
        .encode(
            x=alt.X("rate:Q", title="Selection rate", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("group:N", sort=labels, title=""),
            color=color,
            tooltip=["group:N", "rate:Q", "low:Q", "high:Q"],
        )
    )
    whiskers = (
        alt.Chart(frame)
        .mark_rule(strokeWidth=2)
        .encode(x="low:Q", x2="high:Q", y=alt.Y("group:N", sort=labels), color=color)
    )
    points = (
        alt.Chart(frame)
        .mark_point(filled=True, size=90)
        .encode(x="rate:Q", y=alt.Y("group:N", sort=labels), color=color, shape=shape)
    )
    text = (
        alt.Chart(frame)
        .mark_text(align="left", dx=10, fontSize=12)
        .encode(x="high:Q", y=alt.Y("group:N", sort=labels), text="group:N")
    )
    return (bars + whiskers + points + text).properties(height=max(120, 36 * len(labels)))


def _sparkline(values: list[tuple[str, float]]) -> alt.Chart | None:
    frame = pd.DataFrame(
        [{"group": name, "value": value} for name, value in values if np.isfinite(value)]
    )
    if frame.empty:
        return None
    labels = list(frame["group"])
    return (
        alt.Chart(frame)
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X("value:Q", title="", scale=alt.Scale(zero=False)),
            y=alt.Y("group:N", sort=labels, title=""),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(domain=labels, range=_palette(len(labels))),
                legend=None,
            ),
            shape=alt.Shape(
                "group:N",
                scale=alt.Scale(domain=labels, range=_markers(len(labels))),
                legend=None,
            ),
            tooltip=["group:N", "value:Q"],
        )
        .properties(height=max(80, 24 * len(labels)), width=220)
    )


def _reliability_chart(audit: AuditResult) -> alt.Chart | None:
    cal = audit.calibration
    if cal is None:
        return None
    rows: list[dict[str, object]] = []
    for group in cal.groups:
        for bin_ in group.bins:
            if bin_.count == 0 or not np.isfinite(bin_.mean_score):
                continue
            rows.append(
                {
                    "group": group.label,
                    "mean_score": bin_.mean_score,
                    "positive_rate": bin_.positive_rate,
                    "count": bin_.count,
                }
            )
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    labels = [g.label for g in cal.groups]
    color = alt.Color(
        "group:N",
        scale=alt.Scale(domain=labels, range=_palette(len(labels))),
        legend=alt.Legend(title="Group"),
    )
    shape = alt.Shape(
        "group:N",
        scale=alt.Scale(domain=labels, range=_markers(len(labels))),
        legend=alt.Legend(title="Group"),
    )
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
        .mark_line(strokeDash=[4, 4], color="#888")
        .encode(x="x:Q", y="y:Q")
    )
    points = (
        alt.Chart(frame)
        .mark_point(filled=True)
        .encode(
            x=alt.X("mean_score:Q", title="Mean score", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(
                "positive_rate:Q",
                title="Observed positive rate",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=color,
            shape=shape,
            size=alt.Size("count:Q", legend=alt.Legend(title="Bin n")),
            tooltip=["group:N", "mean_score:Q", "positive_rate:Q", "count:Q"],
        )
    )
    return (diagonal + points).properties(height=360)


def _threshold_lines(sweep_frame: pd.DataFrame, labels: list[str]) -> alt.Chart:
    color = alt.Color(
        "group:N",
        scale=alt.Scale(domain=labels, range=_palette(len(labels))),
        legend=alt.Legend(title="Group"),
    )
    stroke = alt.StrokeDash(
        "metric:N",
        scale=alt.Scale(domain=["TPR", "FPR"], range=[[1, 0], [6, 3]]),
        legend=alt.Legend(title="Metric"),
    )
    return (
        alt.Chart(sweep_frame)
        .mark_line()
        .encode(
            x=alt.X("threshold:Q", title="Threshold t  (score ≥ t)"),
            y=alt.Y("value:Q", title="", scale=alt.Scale(domain=[0, 1])),
            color=color,
            strokeDash=stroke,
            tooltip=["group:N", "metric:N", "threshold:Q", "value:Q"],
        )
        .properties(height=320)
    )


def _intersectional_chart(audit: AuditResult) -> alt.Chart | None:
    if len(audit.counts.group_names) != 2:
        return None
    row_name, col_name = audit.counts.group_names
    rows: list[dict[str, object]] = []
    for i, key in enumerate(audit.counts.groups):
        low, point, high = audit.bootstrap.rate_interval("selection_rate", i)
        n = int(audit.counts.sizes[i])
        rows.append(
            {
                row_name: key[0],
                col_name: key[1],
                "rate": float(point) if np.isfinite(point) else None,
                "label": format_interval(low, point, high),
                "n": n,
                "small": n < SMALL_GROUP_THRESHOLD,
            }
        )
    frame = pd.DataFrame(rows)
    heat = (
        alt.Chart(frame)
        .mark_rect(stroke="#fff", strokeWidth=1)
        .encode(
            x=alt.X(f"{col_name}:N", title=col_name),
            y=alt.Y(f"{row_name}:N", title=row_name),
            color=alt.Color(
                "rate:Q",
                scale=alt.Scale(scheme="blues", domain=[0, 1]),
                legend=alt.Legend(title="Selection rate"),
            ),
            opacity=alt.condition("datum.small", alt.value(0.35), alt.value(1.0)),
            tooltip=[row_name, col_name, "label:N", "n:Q"],
        )
    )
    text = (
        alt.Chart(frame)
        .mark_text(fontSize=11)
        .encode(
            x=f"{col_name}:N",
            y=f"{row_name}:N",
            text="label:N",
            color=alt.value("#111"),
        )
    )
    return (heat + text).properties(height=max(200, 48 * frame[row_name].nunique()))


def _guidance_answers() -> GuidanceAnswers:
    st.markdown(
        "The helper highlights a **family** of metrics. It does not score the model, "
        "and it is not a fairness grade."
    )
    decision_raw = st.radio(
        "1. Is the decision assistive (granting a benefit) or punitive (imposing a penalty)?",
        ["Not answered", "Assistive", "Punitive"],
        horizontal=True,
    )
    labels_raw = st.radio(
        "2. Are the labels trustworthy enough that error rates "
        "(which condition on Y) are meaningful?",
        ["Not answered", "Trusted", "Untrusted"],
        horizontal=True,
    )
    scores_raw = st.radio(
        "3. Will a human read the score as a probability of the favorable outcome?",
        ["Not answered", "As a probability", "As a rank only", "Score not used"],
        horizontal=True,
    )
    decision = {
        "Assistive": DecisionKind.ASSISTIVE,
        "Punitive": DecisionKind.PUNITIVE,
    }.get(decision_raw)
    labels = {"Trusted": LabelTrust.TRUSTED, "Untrusted": LabelTrust.UNTRUSTED}.get(labels_raw)
    scores = {
        "As a probability": ScoreReading.AS_PROBABILITY,
        "As a rank only": ScoreReading.AS_RANK_ONLY,
        "Score not used": ScoreReading.NOT_USED,
    }.get(scores_raw)
    return GuidanceAnswers(decision=decision, labels=labels, scores=scores)


def _tab_groups(audit: AuditResult) -> None:
    st.markdown(
        "Each cell is the point estimate and a 95% percentile bootstrap interval. "
        "`n/a` means the rate's denominator was zero (for example TPR when a group "
        "has no positives). Accuracy is reported for completeness; it is misleading "
        "under class imbalance."
    )
    boot, counts = audit.bootstrap, audit.counts
    table: dict[str, list[str]] = {
        "group": list(counts.labels()),
        "n": [str(int(n)) for n in counts.sizes],
    }
    for spec in RATES:
        table[spec.label] = [
            format_interval(*boot.rate_interval(spec.key, i)) for i in range(counts.n_groups)
        ]
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    chart = _selection_chart(audit)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
        st.caption(
            "Colour follows the Okabe-Ito palette; each group also has a distinct marker "
            "and a written label, so colour is never load-bearing."
        )


def _tab_gaps(audit: AuditResult, highlighted: tuple[str, ...]) -> None:
    st.markdown(
        "Max-min gaps are the literature's headline numbers. They are non-negative by "
        "construction, so a CI that excludes zero is **not** evidence of a disparity. "
        "The verdict uses signed contrasts against the reference group "
        f"**{audit.bootstrap.reference_label}** (D11)."
    )
    boot = audit.bootstrap
    for spec in GAPS:
        with st.container(border=True):
            if spec.key in highlighted:
                st.caption("Highlighted by the metric-selection helper")
            st.markdown(f"**{spec.label}**")
            low, point, high = boot.gap_interval(spec.key)
            left, right = st.columns((2, 1))
            with left:
                st.markdown(f"{format_interval(low, point, high)}")
                st.markdown(f"`{verdict_for_gap(spec, boot)}`")
                st.caption(spec.reference)
                st.caption(f"Formula: `{spec.formula}`")
                if spec.key == "disparate_impact_ratio":
                    di = float(boot.gaps[spec.key])
                    flag = fails_four_fifths(di)
                    if flag is True:
                        st.warning(
                            f"Disparate impact ratio {format_rate(di)} is below 0.80, "
                            f"so the four-fifths screen is raised. {FOUR_FIFTHS_NOTE}"
                        )
                    elif flag is False:
                        st.info(
                            f"Disparate impact ratio {format_rate(di)} is at or above 0.80, "
                            f"so the four-fifths screen is not raised. {FOUR_FIFTHS_NOTE}"
                        )
                    else:
                        st.info(
                            "The four-fifths screen was not computable (the ratio is undefined). "
                            f"{FOUR_FIFTHS_NOTE}"
                        )
            with right:
                spark = []
                for key in spec.inputs:
                    for i, lab in enumerate(boot.group_labels):
                        spark.append((f"{lab} {key}", float(boot.rates[key][i])))
                spark_chart = _sparkline(spark)
                if spark_chart is not None:
                    st.altair_chart(spark_chart, width="stretch")


def _tab_scores(audit: AuditResult) -> None:
    if audit.calibration is None or audit.data.score is None:
        st.info(
            "No score column was mapped, so ROC-AUC, reliability diagrams, ECE and the "
            "threshold sweep are skipped. Map a score in [0, 1] in the sidebar."
        )
        return
    cal = audit.calibration
    st.markdown(
        f"ECE here is the **positive-class** expected calibration error ({cal.n_bins} "
        "equal-width bins on [0, 1]), not Guo et al.'s top-label ECE. Empty bins "
        "contribute 0. Treat ECE as descriptive."
    )
    rows = []
    for group in cal.groups:
        auc = format_rate(group.roc_auc)
        if group.auc_skip_reason:
            auc = f"n/a ({group.auc_skip_reason})"
        rows.append(
            {
                "Group": group.label,
                "n": group.n,
                "ROC-AUC": auc,
                f"ECE ({cal.n_bins} bins)": format_rate(group.ece),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    chart = _reliability_chart(audit)
    if chart is not None:
        st.altair_chart(chart, width="stretch")

    st.markdown("#### Threshold sweep")
    st.caption(
        "Moving the decision threshold trades TPR against FPR and cannot equalise both "
        "when base rates differ."
    )
    sweep = threshold_sweep(
        audit.data.y_true,
        audit.data.score,
        audit.data.group_values,
        audit.data.mapping.sensitive,
    )
    t = st.slider("Decision threshold t  (predict positive when score ≥ t)", 0.0, 1.0, 0.5, 0.01)
    idx = int(np.argmin(np.abs(sweep.thresholds - t)))
    table_rows = []
    long_rows = []
    for g, label in enumerate(sweep.group_labels):
        tpr = float(sweep.tpr[idx, g])
        fpr = float(sweep.fpr[idx, g])
        sel = float(sweep.selection_rate[idx, g])
        table_rows.append(
            {
                "Group": label,
                "TPR": format_rate(tpr),
                "FPR": format_rate(fpr),
                "Selection rate": format_rate(sel),
            }
        )
        for metric, series in (("TPR", sweep.tpr), ("FPR", sweep.fpr)):
            for k, thr in enumerate(sweep.thresholds):
                value = float(series[k, g])
                if np.isfinite(value):
                    long_rows.append(
                        {"group": label, "metric": metric, "threshold": float(thr), "value": value}
                    )
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")
    if long_rows:
        st.altair_chart(
            _threshold_lines(pd.DataFrame(long_rows), list(sweep.group_labels)),
            width="stretch",
        )


def _tab_intersectional(audit: AuditResult) -> None:
    st.warning(
        "Intersectional cells are small by construction. Cells with "
        f"n < {SMALL_GROUP_THRESHOLD} are faded. Do not over-read a quiet cell."
    )
    chart = _intersectional_chart(audit)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    _tab_groups(audit)


def _tab_impossibility(audit: AuditResult) -> None:
    imp = audit.impossibility
    st.markdown(
        "Chouldechova (2017) identity: "
        "`FPR = p/(1-p) · (1-PPV)/PPV · (1-FNR)`. "
        "On this table the implied FPR matches the observed FPR to numerical precision."
    )
    rates = pd.DataFrame(
        {
            "group": list(imp.group_labels),
            "Base rate": [format_rate(float(v)) for v in imp.base_rate],
            "TPR": [format_rate(float(v)) for v in imp.tpr],
            "FPR": [format_rate(float(v)) for v in imp.fpr],
            "PPV": [format_rate(float(v)) for v in imp.precision],
        }
    )
    st.dataframe(rates, hide_index=True, width="stretch")
    if not imp.tradeoff_applies:
        if imp.is_perfect:
            st.success(
                "Every group is classified perfectly (FPR = FNR = 0), which is one of "
                "the two exceptions where the criteria can co-exist."
            )
        else:
            st.info(
                "Base rates do not meaningfully differ, so the identity does not force "
                "predictive parity and equal error rates apart."
            )
        return
    st.markdown(
        f"Base rates differ by {imp.base_rate_spread:.1%} on this file, and the "
        "classifier is not perfect. Predictive parity (equal PPV) and equal error "
        "rates therefore cannot hold together."
    )
    ref = imp.reference_label
    for i, lab in enumerate(imp.group_labels):
        if not imp.equal_ppv_feasible[i]:
            st.markdown(
                f"- **{lab}**: equalising PPV to {ref}'s is **impossible** without also "
                "moving error rates (the identity would require an FPR outside [0, 1])."
            )
        else:
            st.markdown(
                f"- **{lab}**: FPR would be "
                f"{format_rate(float(imp.counterfactual_fpr_equal_ppv[i]))} "
                f"if PPV matched {ref} while prevalence and FNR stayed as observed "
                f"(observed FPR {format_rate(float(imp.fpr[i]))})."
            )
    ppv_bits = ", ".join(
        f"{lab} {format_rate(float(imp.counterfactual_ppv_equal_errors[i]))}"
        for i, lab in enumerate(imp.group_labels)
    )
    st.markdown(f"- If TPR and FPR were equalised to {ref}'s, PPV would have to be {ppv_bits}.")
    st.caption(
        "Sources: Chouldechova 2017 (arXiv:1610.07524); Kleinberg, Mullainathan and "
        "Raghavan 2016 (arXiv:1609.05807)."
    )


def _sidebar_mapping(frame: pd.DataFrame, preferred: ColumnMapping | None) -> ColumnMapping:
    columns = list(map(str, frame.columns))
    pref_label = preferred.label if preferred and preferred.label in columns else None
    pref_pred = preferred.prediction if preferred and preferred.prediction in columns else None
    pref_score = preferred.score if preferred and preferred.score in columns else None
    pref_sens = list(preferred.sensitive) if preferred else []

    label = st.sidebar.selectbox("True label", columns, index=_index(columns, pref_label))
    prediction = st.sidebar.selectbox("Predicted label", columns, index=_index(columns, pref_pred))
    score_options = [NONE, *columns]
    score_raw = st.sidebar.selectbox(
        "Score (optional)",
        score_options,
        index=_index(score_options, pref_score),
    )
    score = None if score_raw == NONE else score_raw

    first_pref = pref_sens[0] if pref_sens else None
    sensitive_1 = st.sidebar.selectbox(
        "Sensitive attribute", columns, index=_index(columns, first_pref)
    )
    second_options = [NONE, *[c for c in columns if c != sensitive_1]]
    second_pref = pref_sens[1] if len(pref_sens) > 1 else NONE
    sensitive_2 = st.sidebar.selectbox(
        "Second sensitive attribute (intersectional)",
        second_options,
        index=_index(second_options, second_pref),
        help="Optional. Maps Tab 4 (Intersectional). Cells will be small.",
    )
    sensitive = (sensitive_1,) if sensitive_2 == NONE else (sensitive_1, sensitive_2)

    try:
        options = label_options(frame, label)
    except ValidationError as exc:
        st.sidebar.error(str(exc))
        st.stop()
        raise
    positive_pref = (
        str(preferred.positive_label) if preferred is not None else (options[0] if options else "1")
    )
    if len(options) <= 6:
        positive = st.sidebar.radio(
            "Positive (favorable) outcome",
            options,
            index=_index(list(options), positive_pref),
        )
    else:
        positive = st.sidebar.selectbox(
            "Positive (favorable) outcome",
            options,
            index=_index(list(options), positive_pref),
        )
    return ColumnMapping(
        label=label,
        prediction=prediction,
        sensitive=sensitive,
        positive_label=positive,
        score=score,
    )


def main() -> None:
    st.set_page_config(page_title="FairCheck", layout="wide")
    st.title("FairCheck")
    st.caption(f"{DISCLAIMER} {PRIVACY_NOTE}")

    st.sidebar.header("FairCheck")
    source = st.sidebar.radio("Data", ["Example", "Upload CSV"])
    preferred: ColumnMapping | None = None
    if source == "Example":
        keys = [spec.key for spec in EXAMPLES]
        titles = {spec.key: spec.title for spec in EXAMPLES}
        example_key = st.sidebar.selectbox(
            "Example",
            keys,
            index=keys.index("compas_like"),
            format_func=lambda key: titles[key],
        )
        spec = EXAMPLES_BY_KEY[example_key]
        st.sidebar.caption(spec.description)
        frame = _load_example(example_key)
        preferred = spec.mapping
    else:
        uploaded = st.sidebar.file_uploader(
            "CSV of predictions",
            type=["csv"],
            max_upload_size=MAX_UPLOAD_MB,
            help=f"Maximum {MAX_UPLOAD_MB} MB. {PRIVACY_NOTE}",
        )
        if uploaded is None:
            st.info("Upload a CSV in the sidebar, or switch to a bundled example.")
            st.stop()
            return
        frame = _load_upload(uploaded.getvalue(), uploaded.name)

    st.sidebar.subheader("Column mapping")
    mapping = _sidebar_mapping(frame, preferred)

    n_boot = int(
        st.sidebar.number_input(
            "Bootstrap resamples",
            min_value=50,
            max_value=5_000,
            value=DEFAULT_N_BOOT,
            step=50,
        )
    )
    seed = int(st.sidebar.number_input("Seed", min_value=0, max_value=10_000, value=0, step=1))
    st.sidebar.caption(f"🔒 {PRIVACY_NOTE}")

    try:
        audit = _cached_audit(
            frame,
            mapping.label,
            mapping.prediction,
            mapping.sensitive,
            str(mapping.positive_label),
            mapping.score,
            n_boot,
            seed,
        )
    except ValidationError as exc:
        st.error(str(exc))
        st.stop()
        return

    small = int((audit.counts.sizes < SMALL_GROUP_THRESHOLD).sum())
    header = (
        f"{audit.data.n_rows_used:,} rows · {audit.counts.n_groups} groups · "
        f"positive outcome = `{audit.data.mapping.positive_label}` · seed {seed} · "
        f"{n_boot:,} bootstrap resamples · reference **{audit.bootstrap.reference_label}**"
    )
    if small:
        header += f" · ⚠ {small} group(s) with n < {SMALL_GROUP_THRESHOLD}"
    st.markdown(header)
    for warning in audit.data.warnings:
        st.warning(warning)
    for sentence in audit.summary:
        st.markdown(f"- {sentence}")

    tab_labels = ["Groups", "Gaps", "Scores"]
    if len(mapping.sensitive) >= 2:
        tab_labels.append("Intersectional")
    tab_labels.extend(["Which metric fits?", "Why not all at once"])
    tabs = st.tabs(tab_labels)
    tab_map = dict(zip(tab_labels, tabs, strict=True))

    with tab_map["Which metric fits?"]:
        answers = _guidance_answers()
        rec = recommend(answers)
        st.markdown(f"**{rec.headline}**")
        st.write(rec.reason)
        if rec.families:
            names = ", ".join(next(s.label for s in GAPS if s.key == key) for key in rec.families)
            st.success(f"Look at: {names}  (Gaps tab).")
        for caveat in rec.caveats:
            st.warning(caveat)
        st.caption(DISCLAIMER)

    with tab_map["Groups"]:
        _tab_groups(audit)
        st.caption(DISCLAIMER)
    with tab_map["Gaps"]:
        _tab_gaps(audit, rec.families)
        st.caption(DISCLAIMER)
    with tab_map["Scores"]:
        _tab_scores(audit)
        st.caption(DISCLAIMER)
    if "Intersectional" in tab_map:
        with tab_map["Intersectional"]:
            _tab_intersectional(audit)
            st.caption(DISCLAIMER)
    with tab_map["Why not all at once"]:
        _tab_impossibility(audit)
        st.caption(DISCLAIMER)

    export = replace(audit, answers=answers, recommendation=rec)
    markdown = render_markdown(export)
    st.sidebar.download_button(
        "Download report (.md)",
        data=markdown,
        file_name="faircheck-report.md",
        mime="text/markdown",
    )
    try:
        st.sidebar.download_button(
            "Download report (.pdf)",
            data=_pdf_bytes(export),
            file_name="faircheck-report.pdf",
            mime="application/pdf",
        )
    except RuntimeError as exc:
        st.sidebar.caption(str(exc))


if __name__ == "__main__":
    main()
