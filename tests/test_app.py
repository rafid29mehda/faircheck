"""The Streamlit app is a thin caller: no metric formulas, no Fairlearn."""

from __future__ import annotations

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
    assert "processed in memory" in captions
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
