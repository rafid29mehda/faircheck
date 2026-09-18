"""CLI: ``python -m faircheck report data.csv --label ... --pred ... --group ...``.

Thin caller of :mod:`faircheck.report`. No metric is computed here.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from faircheck.guidance import DecisionKind, GuidanceAnswers, LabelTrust, ScoreReading
from faircheck.report import render_markdown, run_audit, write_markdown, write_pdf
from faircheck.types import ColumnMapping
from faircheck.validate import ValidationError, read_csv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faircheck",
        description=("Correct, uncertainty-aware fairness reports for classification predictions."),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser(
        "report",
        help="Audit a CSV of predictions and write a Markdown (and optional PDF) report.",
    )
    report.add_argument("csv", help="Path to the predictions CSV.")
    report.add_argument("--label", required=True, help="Column of true labels.")
    report.add_argument("--pred", required=True, help="Column of predicted labels.")
    report.add_argument(
        "--group",
        required=True,
        action="append",
        dest="groups",
        metavar="COLUMN",
        help="Sensitive-attribute column. Repeat for an intersectional audit.",
    )
    report.add_argument("--score", default=None, help="Optional score column in [0, 1].")
    report.add_argument(
        "--positive",
        required=True,
        help="Value of --label that is the favorable outcome.",
    )
    report.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed (default 0).")
    report.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        dest="n_boot",
        help="Number of stratified bootstrap resamples (default 1000).",
    )
    report.add_argument(
        "--out",
        default=None,
        help="Markdown destination. Default: stdout.",
    )
    report.add_argument(
        "--pdf",
        default=None,
        help="Optional PDF destination, derived from the same Markdown (D6).",
    )
    report.add_argument("--decision", choices=["assistive", "punitive"])
    report.add_argument("--labels", choices=["trusted", "untrusted"])
    report.add_argument(
        "--scores-as",
        dest="scores_as",
        choices=["as_probability", "as_rank_only", "not_used"],
        help="How a human will read the score, if one is mapped.",
    )
    return parser


def _answers(args: argparse.Namespace) -> GuidanceAnswers | None:
    if args.decision is None and args.labels is None and args.scores_as is None:
        return None
    return GuidanceAnswers(
        decision=None if args.decision is None else DecisionKind(args.decision),
        labels=None if args.labels is None else LabelTrust(args.labels),
        scores=None if args.scores_as is None else ScoreReading(args.scores_as),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "report":
        parser.error(f"unknown command {args.command}")
    try:
        frame = read_csv(args.csv)
        mapping = ColumnMapping(
            label=args.label,
            prediction=args.pred,
            sensitive=tuple(args.groups),
            positive_label=args.positive,
            score=args.score,
        )
        audit = run_audit(
            frame,
            mapping,
            n_boot=args.n_boot,
            seed=args.seed,
            answers=_answers(args),
        )
        if args.out:
            write_markdown(audit, args.out)
        else:
            markdown = render_markdown(audit)
            sys.stdout.write(markdown if markdown.endswith("\n") else markdown + "\n")
        if args.pdf:
            write_pdf(audit, args.pdf)
    except ValidationError as exc:
        print(exc, file=sys.stderr)
        return 2
    except FileNotFoundError:
        print(f"File not found: {args.csv}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
