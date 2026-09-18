"""CLI smoke test (plan test 11). The CLI is a thin caller of the report builder."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from faircheck.__main__ import main
from faircheck.report import render_markdown, run_audit
from faircheck.types import ColumnMapping
from tests.conftest import cells_to_frame


def _write_csv(path: Path) -> Path:
    frame = cells_to_frame({"A": (40, 10, 10, 40), "B": (12, 8, 8, 72)})
    frame.to_csv(path, index=False)
    return path


def test_cli_stdout_matches_the_library_report(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "preds.csv")
    argv = [
        "report",
        str(csv_path),
        "--label",
        "y_true",
        "--pred",
        "y_pred",
        "--group",
        "group",
        "--positive",
        "1",
        "--n-boot",
        "50",
        "--seed",
        "0",
    ]
    # Capture via a file so we can compare byte-for-byte with the library.
    out = tmp_path / "cli.md"
    assert main([*argv, "--out", str(out)]) == 0
    frame = cells_to_frame({"A": (40, 10, 10, 40), "B": (12, 8, 8, 72)})
    audit = run_audit(
        frame,
        ColumnMapping(
            label="y_true",
            prediction="y_pred",
            sensitive=("group",),
            positive_label=1,
        ),
        n_boot=50,
        seed=0,
    )
    assert out.read_text(encoding="utf-8") == render_markdown(audit)


def test_python_module_invocation_writes_markdown_and_pdf(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "preds.csv")
    md_path = tmp_path / "report.md"
    pdf_path = tmp_path / "report.pdf"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "faircheck",
            "report",
            str(csv_path),
            "--label",
            "y_true",
            "--pred",
            "y_pred",
            "--group",
            "group",
            "--positive",
            "1",
            "--n-boot",
            "50",
            "--out",
            str(md_path),
            "--pdf",
            str(pdf_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert md_path.is_file()
    text = md_path.read_text(encoding="utf-8")
    assert "# FairCheck report" in text
    assert "n/a" in text or "0.500" in text
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_cli_validation_error_is_actionable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = _write_csv(tmp_path / "preds.csv")
    result = main(
        [
            "report",
            str(csv_path),
            "--label",
            "missing_column",
            "--pred",
            "y_pred",
            "--group",
            "group",
            "--positive",
            "1",
        ]
    )
    assert result == 2
    err = capsys.readouterr().err
    assert "missing_column" in err
    assert "not found" in err.lower() or "Column" in err


def test_cli_stdout_and_guidance_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv_path = _write_csv(tmp_path / "preds.csv")
    assert (
        main(
            [
                "report",
                str(csv_path),
                "--label",
                "y_true",
                "--pred",
                "y_pred",
                "--group",
                "group",
                "--positive",
                "1",
                "--n-boot",
                "50",
                "--decision",
                "punitive",
                "--labels",
                "trusted",
                "--scores-as",
                "not_used",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "# FairCheck report" in out
    assert "Equalized odds" in out


def test_cli_write_pdf_via_main(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "preds.csv")
    pdf_path = tmp_path / "report.pdf"
    assert (
        main(
            [
                "report",
                str(csv_path),
                "--label",
                "y_true",
                "--pred",
                "y_pred",
                "--group",
                "group",
                "--positive",
                "1",
                "--n-boot",
                "50",
                "--out",
                str(tmp_path / "report.md"),
                "--pdf",
                str(pdf_path),
            ]
        )
        == 0
    )
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_cli_missing_file_returns_2(tmp_path: Path) -> None:
    missing = tmp_path / "no-such.csv"
    result = main(
        [
            "report",
            str(missing),
            "--label",
            "y_true",
            "--pred",
            "y_pred",
            "--group",
            "group",
            "--positive",
            "1",
        ]
    )
    assert result == 2
