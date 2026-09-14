"""Input validation and column mapping.

Error messages here are user-facing and are treated as part of the product: each one
says what is wrong, where, and what to do about it. Warnings are returned rather than
raised so the UI can show them alongside a report that is still worth reading.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any, Final

import numpy as np
import pandas as pd

from faircheck.types import ColumnMapping, ValidatedData

#: Above this many rows the audit still runs, but the UI warns and offers a subsample:
#: Streamlit Community Cloud starts throttling around 690 MB of memory.
LARGE_ROW_WARNING: Final = 200_000

#: Hard ceiling. Beyond this a seeded subsample is taken rather than risking the host.
DEFAULT_MAX_ROWS: Final = 1_000_000

#: A sensitive column with more distinct values than this is almost certainly an
#: identifier rather than a group, and would produce an unreadable table of n=1 groups.
MAX_GROUPS: Final = 100

#: Groups smaller than this get a visible small-sample flag throughout the report.
SMALL_GROUP_THRESHOLD: Final = 30


class ValidationError(ValueError):
    """Raised when the data cannot be audited. The message is shown to the user."""


def read_csv(source: str | Path | IO[bytes] | IO[str]) -> pd.DataFrame:
    """Read a CSV into a DataFrame, converting parser failures into clear messages."""
    try:
        frame = pd.read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise ValidationError("That file is empty -- there are no rows to audit.") from exc
    except UnicodeDecodeError as exc:
        raise ValidationError(
            "That file is not UTF-8 text. FairCheck reads CSV files; if this is an Excel "
            "workbook, export it as CSV first."
        ) from exc
    except pd.errors.ParserError as exc:
        raise ValidationError(f"That file could not be parsed as CSV: {exc}") from exc

    if frame.empty:
        raise ValidationError("That file has a header but no data rows.")
    return frame


def label_options(frame: pd.DataFrame, column: str) -> tuple[str, ...]:
    """Distinct values of a label column, as display strings, for the positive-label picker."""
    _require_columns(frame, (column,))
    values = pd.Series(frame[column]).dropna().unique()
    return tuple(sorted(str(value) for value in values))


def resolve_positive_label(frame: pd.DataFrame, column: str, chosen: Any) -> Any:
    """Match the user's chosen value back to the column's own dtype.

    The UI and the CLI both hand us text ("1", ">50K"), but the column may hold ints or
    floats. Comparing ``"1" == 1`` silently yields an all-negative label vector, so the
    choice is resolved against the column's actual values instead.
    """
    values = pd.Series(frame[column]).dropna().unique()
    for value in values:
        if value == chosen or str(value) == str(chosen):
            return value
    available = ", ".join(repr(str(v)) for v in sorted(values, key=str)[:10])
    raise ValidationError(
        f"{chosen!r} does not appear in column {column!r}. "
        f"Values present: {available}. Pick the value that means the favorable outcome."
    )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        available = ", ".join(repr(str(c)) for c in frame.columns[:20])
        raise ValidationError(
            f"Column(s) not found in the file: {', '.join(repr(c) for c in missing)}. "
            f"Columns available: {available}."
        )


def validate(
    frame: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    seed: int = 0,
) -> ValidatedData:
    """Check the mapped columns and return binarised arrays ready to audit.

    Raises :class:`ValidationError` for anything that makes the audit wrong, and collects
    everything that merely makes it *harder to interpret* into ``ValidatedData.warnings``.
    """
    warnings: list[str] = []
    n_rows_input = len(frame)

    _require_columns(frame, mapping.columns)

    if mapping.label == mapping.prediction:
        warnings.append(
            f"The true label and the prediction are both mapped to {mapping.label!r}, so the "
            "classifier will look perfect. Map the model's output to 'predicted label'."
        )
    overlap = set(mapping.sensitive) & {mapping.label, mapping.prediction}
    if overlap:
        warnings.append(
            f"Column(s) {', '.join(repr(c) for c in sorted(overlap))} are used both as a "
            "sensitive attribute and as a label or prediction. That is unusual -- check "
            "the mapping."
        )

    used = list(dict.fromkeys(mapping.columns))
    data = frame.loc[:, used]

    counts = {column: int(data[column].isna().sum()) for column in used}
    if any(counts.values()):
        detail = ", ".join(f"{column!r}: {n}" for column, n in counts.items() if n)
        raise ValidationError(
            f"Missing values in mapped column(s) -- {detail}. FairCheck will not guess what a "
            "missing label, prediction or group means. Drop or fill those rows and re-upload."
        )

    positive = resolve_positive_label(data, mapping.label, mapping.positive_label)
    label_values = label_options(data, mapping.label)

    if len(label_values) == 1:
        warnings.append(
            f"Column {mapping.label!r} has only one distinct value ({label_values[0]!r}), so every "
            "row is in the same class and rates that condition on the other class are undefined."
        )
    elif len(label_values) > 2:
        warnings.append(
            f"Column {mapping.label!r} has {len(label_values)} distinct values, so this is a "
            f"one-vs-rest audit: {str(positive)!r} against everything else. Re-run with a "
            "different positive value to audit another class."
        )

    prediction_values = label_options(data, mapping.prediction)
    if str(positive) not in prediction_values:
        warnings.append(
            f"The model never predicts {str(positive)!r} in column {mapping.prediction!r}, so its "
            "selection rate is 0 for every group and precision is undefined."
        )

    y_true = (data[mapping.label] == positive).to_numpy(dtype=bool)
    y_pred = (data[mapping.prediction] == positive).to_numpy(dtype=bool)

    score = _validate_score(data, mapping, y_pred, warnings)
    group_values = _validate_groups(data, mapping, warnings)

    if n_rows_input > max_rows:
        keep = np.sort(np.random.default_rng(seed).choice(n_rows_input, max_rows, replace=False))
        y_true, y_pred = y_true[keep], y_pred[keep]
        group_values = tuple(column[keep] for column in group_values)
        score = None if score is None else score[keep]
        warnings.append(
            f"The file has {n_rows_input:,} rows, above the {max_rows:,}-row limit. A seeded "
            f"random subsample of {max_rows:,} rows was audited (seed {seed}); confidence "
            "intervals widen accordingly."
        )
    elif n_rows_input > LARGE_ROW_WARNING:
        warnings.append(
            f"{n_rows_input:,} rows is large for the hosted demo. If it feels slow, run FairCheck "
            "locally or via the CLI."
        )

    return ValidatedData(
        y_true=y_true,
        y_pred=y_pred,
        group_values=group_values,
        mapping=mapping,
        score=score,
        warnings=tuple(warnings),
        n_rows_input=n_rows_input,
        n_rows_used=len(y_true),
        label_values=label_values,
    )


def _validate_score(
    data: pd.DataFrame,
    mapping: ColumnMapping,
    y_pred: np.ndarray,
    warnings: list[str],
) -> np.ndarray | None:
    if mapping.score is None:
        return None

    column = data[mapping.score]
    if not pd.api.types.is_numeric_dtype(column):
        raise ValidationError(
            f"Score column {mapping.score!r} is not numeric (dtype {column.dtype}). Scores must be "
            "probabilities or risk scores in [0, 1]."
        )

    score = column.to_numpy(dtype=np.float64)
    low, high = float(np.min(score)), float(np.max(score))
    if low < 0.0 or high > 1.0:
        raise ValidationError(
            f"Score column {mapping.score!r} ranges from {low:g} to {high:g}, outside [0, 1]. "
            "Calibration and the threshold sweep assume probabilities. If these are logits or "
            "arbitrary risk scores, convert them to probabilities first."
        )

    # If predictions are a plain threshold on this score, every selected row scores at least
    # as high as every unselected one. When that fails, the threshold slider still shows what
    # *would* happen at each cut-off, but it no longer reproduces the supplied predictions.
    mixed_predictions = bool(y_pred.any() and not y_pred.all())
    if mixed_predictions and float(np.max(score[~y_pred])) > float(np.min(score[y_pred])):
        warnings.append(
            f"The predictions in {mapping.prediction!r} are not a single threshold applied to "
            f"{mapping.score!r} (some unselected rows score higher than some selected ones). "
            "The threshold sweep is therefore illustrative and will not reproduce the supplied "
            "predictions at any cut-off."
        )
    return score


def _validate_groups(
    data: pd.DataFrame,
    mapping: ColumnMapping,
    warnings: list[str],
) -> tuple[np.ndarray, ...]:
    columns: list[np.ndarray] = []
    for name in mapping.sensitive:
        values = pd.Series(data[name]).astype(str)
        n_unique = int(values.nunique())
        if n_unique > MAX_GROUPS:
            raise ValidationError(
                f"Sensitive column {name!r} has {n_unique:,} distinct values, more than the "
                f"{MAX_GROUPS} FairCheck will group by. That usually means an ID or a continuous "
                "variable was mapped by mistake -- bucket it into categories first."
            )
        if n_unique == 1:
            warnings.append(
                f"Sensitive column {name!r} has a single value ({values.iloc[0]!r}), so there is "
                "nothing to compare and every gap is 0 by construction."
            )
        columns.append(values.to_numpy(dtype=object))

    if len(mapping.sensitive) > 1:
        warnings.append(
            "Intersectional groups are formed by crossing the sensitive columns, so cells are "
            f"smaller than the marginal groups; any cell under n = {SMALL_GROUP_THRESHOLD} is "
            "flagged as unstable."
        )
    return tuple(columns)
