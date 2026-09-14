"""Validation behaviour, including the exact wording of the failure modes.

Error messages are asserted on, not just exception types: a message that does not say
which column is wrong is a bug in a tool whose users are uploading unfamiliar CSVs.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from faircheck.types import ColumnMapping
from faircheck.validate import (
    ValidationError,
    label_options,
    read_csv,
    resolve_positive_label,
    validate,
)
from tests.conftest import cells_to_frame


def base_mapping(**overrides: object) -> ColumnMapping:
    defaults: dict[str, object] = {
        "label": "y_true",
        "prediction": "y_pred",
        "sensitive": ("group",),
        "positive_label": 1,
    }
    defaults.update(overrides)
    return ColumnMapping(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def frame() -> pd.DataFrame:
    return cells_to_frame({"A": (40, 10, 10, 40), "B": (12, 8, 8, 72)})


def test_valid_input_produces_no_warnings(frame: pd.DataFrame) -> None:
    data = validate(frame, base_mapping())

    assert data.warnings == ()
    assert data.n_rows_used == data.n_rows_input == len(frame)
    assert int(data.y_true.sum()) == 70  # positives: A = 40 + 10, B = 12 + 8
    assert data.is_one_vs_rest is False
    assert data.was_subsampled is False


def test_missing_column_names_the_column_and_lists_what_exists(frame: pd.DataFrame) -> None:
    with pytest.raises(ValidationError, match="not found") as excinfo:
        validate(frame, base_mapping(prediction="predicted"))

    message = str(excinfo.value)
    assert "'predicted'" in message
    assert "y_pred" in message  # tells the user what is available


def test_missing_values_are_rejected_with_counts(frame: pd.DataFrame) -> None:
    frame = frame.copy()
    frame.loc[0:2, "group"] = None

    with pytest.raises(ValidationError, match="Missing values") as excinfo:
        validate(frame, base_mapping())

    assert "'group': 3" in str(excinfo.value)


def test_unknown_positive_label_lists_the_available_values(frame: pd.DataFrame) -> None:
    with pytest.raises(ValidationError, match="does not appear") as excinfo:
        validate(frame, base_mapping(positive_label="yes"))

    assert "'0'" in str(excinfo.value)
    assert "'1'" in str(excinfo.value)


def test_positive_label_is_resolved_across_dtypes(frame: pd.DataFrame) -> None:
    """The UI hands us text; the column holds ints. "1" must not silently mean no positives."""
    assert resolve_positive_label(frame, "y_true", "1") == 1
    data = validate(frame, base_mapping(positive_label="1"))
    assert int(data.y_true.sum()) == 70


def test_label_options_are_sorted_display_strings(frame: pd.DataFrame) -> None:
    assert label_options(frame, "y_true") == ("0", "1")


def test_multiclass_label_is_audited_one_vs_rest_with_a_warning(frame: pd.DataFrame) -> None:
    frame = frame.copy()
    frame.loc[frame.index[:10], "y_true"] = 2

    data = validate(frame, base_mapping())

    assert data.is_one_vs_rest is True
    assert any("one-vs-rest" in warning for warning in data.warnings)
    # The 10 relabelled rows were positives; they are now part of "rest".
    assert int(data.y_true.sum()) == 60


def test_single_valued_label_warns(frame: pd.DataFrame) -> None:
    frame = frame.copy()
    frame["y_true"] = 1

    data = validate(frame, base_mapping())
    assert any("only one distinct value" in warning for warning in data.warnings)


def test_model_that_never_predicts_positive_warns(frame: pd.DataFrame) -> None:
    frame = frame.copy()
    frame["y_pred"] = 0

    data = validate(frame, base_mapping())
    assert any("never predicts" in warning for warning in data.warnings)


def test_non_numeric_score_is_rejected(frame: pd.DataFrame) -> None:
    frame = frame.copy()
    frame["score"] = "high"

    with pytest.raises(ValidationError, match="not numeric"):
        validate(frame, base_mapping(score="score"))


def test_score_outside_unit_interval_is_rejected_with_the_range(frame: pd.DataFrame) -> None:
    frame = frame.copy()
    frame["score"] = np.linspace(-2.0, 3.0, len(frame))

    with pytest.raises(ValidationError, match=r"outside \[0, 1\]") as excinfo:
        validate(frame, base_mapping(score="score"))

    assert "-2" in str(excinfo.value)
    assert "3" in str(excinfo.value)


def test_score_consistent_with_predictions_produces_no_threshold_warning(
    frame: pd.DataFrame,
) -> None:
    frame = frame.copy()
    # Every selected row scores above every unselected one, i.e. a genuine threshold rule.
    frame["score"] = np.where(frame["y_pred"] == 1, 0.9, 0.1)

    data = validate(frame, base_mapping(score="score"))
    assert not any("single threshold" in warning for warning in data.warnings)
    assert data.score is not None


def test_predictions_that_are_not_a_threshold_on_the_score_warn(frame: pd.DataFrame) -> None:
    frame = frame.copy()
    frame["score"] = np.where(frame["y_pred"] == 1, 0.4, 0.6)

    data = validate(frame, base_mapping(score="score"))
    assert any("single threshold" in warning for warning in data.warnings)


def test_identifier_like_sensitive_column_is_rejected(frame: pd.DataFrame) -> None:
    frame = frame.copy()
    frame["person_id"] = np.arange(len(frame))

    with pytest.raises(ValidationError, match="distinct values") as excinfo:
        validate(frame, base_mapping(sensitive=("person_id",)))

    assert "person_id" in str(excinfo.value)


def test_constant_sensitive_column_warns(frame: pd.DataFrame) -> None:
    frame = frame.copy()
    frame["group"] = "everyone"

    data = validate(frame, base_mapping())
    assert any("single value" in warning for warning in data.warnings)


def test_label_and_prediction_mapped_to_the_same_column_warns(frame: pd.DataFrame) -> None:
    data = validate(frame, base_mapping(prediction="y_true"))
    assert any("look perfect" in warning for warning in data.warnings)


def test_two_sensitive_columns_warn_about_small_cells(frame: pd.DataFrame) -> None:
    frame = frame.copy()
    frame["region"] = ["north", "south"] * (len(frame) // 2)

    data = validate(frame, base_mapping(sensitive=("group", "region")))
    assert any("Intersectional" in warning for warning in data.warnings)


def test_oversized_input_is_subsampled_reproducibly(frame: pd.DataFrame) -> None:
    first = validate(frame, base_mapping(), max_rows=50, seed=7)
    second = validate(frame, base_mapping(), max_rows=50, seed=7)
    different_seed = validate(frame, base_mapping(), max_rows=50, seed=8)

    assert first.n_rows_used == 50
    assert first.was_subsampled is True
    assert any("subsample" in warning for warning in first.warnings)
    assert first.y_true.tolist() == second.y_true.tolist()
    assert first.y_true.tolist() != different_seed.y_true.tolist()


def test_read_csv_rejects_an_empty_file() -> None:
    with pytest.raises(ValidationError, match="empty"):
        read_csv(io.StringIO(""))


def test_read_csv_rejects_a_header_only_file() -> None:
    with pytest.raises(ValidationError, match="no data rows"):
        read_csv(io.StringIO("y_true,y_pred,group\n"))


def test_read_csv_round_trips_a_normal_file(frame: pd.DataFrame) -> None:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)

    assert len(read_csv(buffer)) == len(frame)


def test_mapping_requires_a_sensitive_column() -> None:
    with pytest.raises(ValueError, match="sensitive column"):
        ColumnMapping(label="y", prediction="p", sensitive=(), positive_label=1)
