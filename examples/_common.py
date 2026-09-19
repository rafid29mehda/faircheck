"""Shared paths and constants for the bundled example CSVs.

The committed files under :data:`DATA_DIR` are what tests and the app read.
Regenerating ACS Income or Adult hits the network once (Fairlearn/OpenML);
the two synthetic examples never do.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Matches every generator and the tests that re-run the synthetic ones.
SEED: int = 0

#: Cap on exported real-data rows so the CSVs stay demo-sized and git-friendly.
N_EXPORT: int = 8_000

DATA_DIR: Path = Path(__file__).resolve().parent / "data"


def write_example(frame: pd.DataFrame, filename: str) -> Path:
    """Write ``frame`` under :data:`DATA_DIR` with a stable float format."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    frame.to_csv(path, index=False, float_format="%.6f")
    return path
