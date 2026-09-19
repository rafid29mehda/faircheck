"""Named example CSVs and the column mapping each one expects.

The Streamlit app (M6) will iterate this catalog for the example dropdown.
Tests use it so a renamed column in a generator cannot silently desync the audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from examples._common import DATA_DIR
from faircheck.types import ColumnMapping


@dataclass(frozen=True, slots=True)
class ExampleSpec:
    key: str
    filename: str
    title: str
    mapping: ColumnMapping
    description: str

    @property
    def path(self) -> Path:
        return DATA_DIR / self.filename


EXAMPLES: tuple[ExampleSpec, ...] = (
    ExampleSpec(
        key="compas_like",
        filename="compas_like.csv",
        title="COMPAS-like (synthetic)",
        mapping=ColumnMapping(
            label="y_true",
            prediction="y_pred",
            sensitive=("group",),
            positive_label=1,
            score="score",
        ),
        description=(
            "Calibrated scores, unequal base rates, unequal error rates. "
            "Synthetic; not real COMPAS records."
        ),
    ),
    ExampleSpec(
        key="near_fair",
        filename="near_fair.csv",
        title="Near-fair (synthetic)",
        mapping=ColumnMapping(
            label="y_true",
            prediction="y_pred",
            sensitive=("group",),
            positive_label=1,
            score="score",
        ),
        description="Same data-generating process in both groups; contrasts cover zero.",
    ),
    ExampleSpec(
        key="acs_income",
        filename="acs_income.csv",
        title="ACS Income (Washington)",
        mapping=ColumnMapping(
            label="y_true",
            prediction="y_pred",
            sensitive=("sex",),
            positive_label=1,
            score="score",
        ),
        description=(
            "ACS PUMS via Ding et al. 2021 / Fairlearn. y_true = 1 means income >= $50,000."
        ),
    ),
    ExampleSpec(
        key="adult",
        filename="adult.csv",
        title="UCI Adult",
        mapping=ColumnMapping(
            label="y_true",
            prediction="y_pred",
            sensitive=("sex",),
            positive_label=1,
            score="score",
        ),
        description="UCI Adult. y_true = 1 means income > $50,000. Prefer ACS Income (D9).",
    ),
)

EXAMPLES_BY_KEY: dict[str, ExampleSpec] = {spec.key: spec for spec in EXAMPLES}
