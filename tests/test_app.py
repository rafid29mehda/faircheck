"""The Streamlit app is a thin caller: no metric formulas, no Fairlearn."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"

# AppTest emits this from @st.cache_data when no browser session is attached.
_APPTEST_WARNINGS = (
    "ignore:.*missing ScriptRunContext:UserWarning",
    "ignore:.*No runtime found:UserWarning",
)


def test_app_exists() -> None:
    assert APP.is_file()


def test_app_does_not_import_fairlearn() -> None:
    source = APP.read_text(encoding="utf-8")
    assert "import fairlearn" not in source
    assert "from fairlearn" not in source


def test_app_delegates_metrics_to_the_library() -> None:
    source = APP.read_text(encoding="utf-8")
    assert "run_audit" in source
    assert "format_interval" in source
    assert "verdict_for_gap" in source
    assert "recommend" in source
    assert "threshold_sweep" in source
    assert "fails_four_fifths" in source
    assert "format_score_estimate" in source
    # Formulas live in the RATES/GAPS registries, not restated in the UI.
    assert "TP / (TP + FN)" not in source
    assert "max_a SR_a" not in source


def _markdown_text(at: Any) -> str:
    return "\n".join(str(m.value) for m in at.markdown)


@pytest.mark.filterwarnings(_APPTEST_WARNINGS[0])
@pytest.mark.filterwarnings(_APPTEST_WARNINGS[1])
def test_app_runs_the_bundled_compas_like_example() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    assert len(at.exception) == 0
    assert at.title[0].value == "FairCheck"
    text = _markdown_text(at)
    assert "8,000 rows" in text
    assert "Each cell is the point estimate" in text
    assert "Base rates differ" in text
    captions = " ".join(c.value for c in at.caption)
    assert "Screening tool. Not a legal or causal conclusion." in captions
    assert "held in server memory" in captions
    labels = [d.label for d in at.download_button]
    assert "Download report (.md)" in labels
    assert "Download report (.pdf)" in labels
    assert not any("NaN" in str(m.value) for m in at.markdown)


@pytest.mark.filterwarnings(_APPTEST_WARNINGS[0])
@pytest.mark.filterwarnings(_APPTEST_WARNINGS[1])
def test_app_grows_an_intersectional_tab_when_two_attributes_are_mapped() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    example = next(s for s in at.selectbox if s.label == "Example")
    example.select("acs_income").run()
    assert len(at.exception) == 0
    second = next(s for s in at.selectbox if s.label.startswith("Second sensitive"))
    second.select("race").run()
    assert len(at.exception) == 0
    header = str(at.markdown[0].value)
    assert "16 groups" in header
    assert "Male × White" in header
    warnings = " ".join(str(w.value) for w in at.warning)
    assert "Intersectional cells are small by construction" in warnings
    assert "n < 30" in warnings


@pytest.mark.filterwarnings(_APPTEST_WARNINGS[0])
@pytest.mark.filterwarnings(_APPTEST_WARNINGS[1])
def test_app_scores_tab_shows_intervals_for_auc_and_ece() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    assert len(at.exception) == 0
    scores = next(d.value for d in at.dataframe if "ROC-AUC" in d.value.columns)
    ece_column = next(c for c in scores.columns if str(c).startswith("ECE"))
    assert scores["ROC-AUC"].str.contains(r"\[", regex=True).all()
    assert scores[ece_column].str.contains(r"\[", regex=True).all()


@pytest.mark.filterwarnings(_APPTEST_WARNINGS[0])
@pytest.mark.filterwarnings(_APPTEST_WARNINGS[1])
def test_app_gaps_tab_says_verdicts_are_simultaneous() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    assert len(at.exception) == 0
    assert "simultaneous" in _markdown_text(at)


@pytest.mark.filterwarnings(_APPTEST_WARNINGS[0])
@pytest.mark.filterwarnings(_APPTEST_WARNINGS[1])
def test_app_says_which_small_groups_use_a_posterior() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    assert not any("Dirichlet" in c.value for c in at.caption)
    example = next(s for s in at.selectbox if s.label == "Example")
    example.select("acs_income").run()
    second = next(s for s in at.selectbox if s.label.startswith("Second sensitive"))
    second.select("race").run()
    assert len(at.exception) == 0
    assert any("Dirichlet" in c.value for c in at.caption)


def _decorator_above(source: str, function: str) -> str:
    match = re.search(rf"((?:@[^\n]*\n)+)def {function}\(", source)
    assert match is not None, f"no decorator found above {function}"
    return match.group(1)


def test_upload_caches_are_bounded() -> None:
    """A public host must not keep every upload in memory until it restarts."""
    source = APP.read_text(encoding="utf-8")
    for name in ("_load_upload", "_cached_audit"):
        decorator = _decorator_above(source, name)
        assert "ttl=" in decorator, f"{name} cache has no time limit"
        assert "max_entries=" in decorator, f"{name} cache has no entry cap"
    assert "max_entries=" in _decorator_above(source, "_load_example")


def test_config_upload_cap_matches_the_app_and_stays_small() -> None:
    """Streamlit enforces the config value, the widget shows the app's; they must agree.

    25 MB keeps a single worst-case upload near 300 MB, leaving room for a few visitors
    sharing one process on the free host.
    """
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    match = re.search(r"^MAX_UPLOAD_MB = (\d+)$", APP.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None
    assert config["server"]["maxUploadSize"] == int(match.group(1))
    assert int(match.group(1)) <= 25


def _hand_case_csv() -> bytes:
    """The reference case in the README's documented column names."""
    rows = ["y_true,y_pred,group"]
    for group, (tp, fp, fn, tn) in {"A": (40, 10, 10, 40), "B": (12, 8, 8, 72)}.items():
        rows += [f"1,1,{group}"] * tp + [f"0,1,{group}"] * fp
        rows += [f"1,0,{group}"] * fn + [f"0,0,{group}"] * tn
    return ("\n".join(rows) + "\n").encode()


def _run_bundled_app() -> Any:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    return at


def _upload(content: bytes, name: str = "upload.csv") -> Any:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    next(r for r in at.radio if r.label == "Data").set_value("Upload CSV").run()
    at.file_uploader[0].set_value((name, content, "text/csv")).run()
    return at


def _selected(at: Any, label: str) -> str:
    widgets = [*at.selectbox, *at.radio]
    return str(next(w for w in widgets if w.label == label).value)


@pytest.mark.filterwarnings(_APPTEST_WARNINGS[0])
@pytest.mark.filterwarnings(_APPTEST_WARNINGS[1])
def test_an_empty_upload_gets_the_clear_message_not_a_traceback() -> None:
    at = _upload(b"", "empty.csv")
    assert len(at.exception) == 0
    assert any("That file is empty" in str(e.value) for e in at.error)


@pytest.mark.filterwarnings(_APPTEST_WARNINGS[0])
@pytest.mark.filterwarnings(_APPTEST_WARNINGS[1])
def test_a_header_only_upload_gets_the_clear_message() -> None:
    at = _upload(b"y_true,y_pred,group\n", "header_only.csv")
    assert len(at.exception) == 0
    assert any("no data rows" in str(e.value) for e in at.error)


@pytest.mark.filterwarnings(_APPTEST_WARNINGS[0])
@pytest.mark.filterwarnings(_APPTEST_WARNINGS[1])
def test_an_upload_in_the_documented_format_arrives_correctly_mapped() -> None:
    at = _upload(_hand_case_csv())
    assert len(at.exception) == 0
    assert _selected(at, "True label") == "y_true"
    assert _selected(at, "Predicted label") == "y_pred"
    assert _selected(at, "Sensitive attribute") == "group"
    assert _selected(at, "Positive (favorable) outcome") == "1"
    warnings = " ".join(str(w.value) for w in at.warning)
    assert "both mapped" not in warnings
    assert "positive outcome = `1`" in _markdown_text(at)


@pytest.mark.filterwarnings(_APPTEST_WARNINGS[0])
@pytest.mark.filterwarnings(_APPTEST_WARNINGS[1])
def test_yes_no_labels_default_to_yes_as_the_positive_outcome() -> None:
    rows = ["actual,predicted,sex"] + ["yes,yes,F"] * 30 + ["no,yes,F"] * 10
    rows += ["yes,no,M"] * 20 + ["no,no,M"] * 40
    at = _upload(("\n".join(rows) + "\n").encode())
    assert len(at.exception) == 0
    assert _selected(at, "True label") == "actual"
    assert _selected(at, "Predicted label") == "predicted"
    assert _selected(at, "Positive (favorable) outcome") == "yes"


@pytest.mark.filterwarnings(_APPTEST_WARNINGS[0])
@pytest.mark.filterwarnings(_APPTEST_WARNINGS[1])
def test_the_app_says_the_upload_defaults_are_guesses() -> None:
    at = _upload(_hand_case_csv())
    assert any("guessed" in c.value for c in at.sidebar.caption)
    bundled = _run_bundled_app()
    assert not any("guessed" in c.value for c in bundled.sidebar.caption)
