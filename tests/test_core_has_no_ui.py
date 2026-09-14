"""Architectural guard: the core library must stay free of UI code.

"No metric logic in the UI" is a promise that is easy to make and easy to break during a
late change. Enforcing the dependency direction mechanically is cheaper than reviewing
for it, and it is what keeps the CLI and the app guaranteed to agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parent.parent / "faircheck"
FORBIDDEN = ("streamlit", "gradio", "altair", "matplotlib")


def core_modules() -> list[Path]:
    return sorted(CORE.glob("*.py"))


def test_the_core_package_exists() -> None:
    assert core_modules(), f"no modules found under {CORE}"


@pytest.mark.parametrize("module", core_modules(), ids=lambda path: path.name)
def test_no_ui_imports_in_core(module: Path) -> None:
    source = module.read_text(encoding="utf-8")
    for name in FORBIDDEN:
        assert f"import {name}" not in source, (
            f"{module.name} imports {name}; UI code belongs in app.py, not in faircheck/"
        )


def test_importing_faircheck_does_not_pull_in_a_ui_framework() -> None:
    for name in list(sys.modules):
        if name.split(".")[0] in FORBIDDEN:
            pytest.skip(f"{name} already imported by another test")

    import faircheck  # noqa: F401

    assert not any(name.split(".")[0] in FORBIDDEN for name in sys.modules)


@pytest.mark.parametrize("module", core_modules(), ids=lambda path: path.name)
def test_core_does_not_import_fairlearn(module: Path) -> None:
    """Fairlearn is a test oracle, not a runtime dependency (docs/DECISIONS.md D2).

    Prose references to it in docstrings are expected and welcome; imports are not.
    """
    source = module.read_text(encoding="utf-8")
    assert "import fairlearn" not in source
    assert "from fairlearn" not in source
