"""UCI Adult predictions.

Hits the network the first time Fairlearn/OpenML has no cache. Tests never import
this module. y_true = 1 means income > $50,000. Prefer ACS Income as the real-data
example (D9; Ding et al. 2021).
"""

from __future__ import annotations

import pandas as pd

from examples._common import SEED, write_example
from examples._supervised import holdout_logistic_predictions

FILENAME = "adult.csv"


def _is_over_50k(value: object) -> bool:
    return ">" in str(value).replace(" ", "")


def build_frame(*, seed: int = SEED) -> pd.DataFrame:
    from fairlearn.datasets import fetch_adult

    bunch = fetch_adult(as_frame=True)
    frame = bunch.frame.copy()
    # Fairlearn/OpenML may name the target ``class`` or ``income``.
    target_col = "class" if "class" in frame.columns else bunch.target.name
    y = frame[target_col].map(_is_over_50k).to_numpy(dtype=bool)
    extra = pd.DataFrame(
        {
            "sex": frame["sex"].astype(str).str.strip(),
            "race": frame["race"].astype(str).str.strip(),
        }
    )
    features = frame.loc[:, ["age", "education-num", "hours-per-week", "sex", "race"]].copy()
    features["sex"] = features["sex"].astype(str)
    features["race"] = features["race"].astype(str)
    return holdout_logistic_predictions(
        features,
        y,
        extra,
        numeric=("age", "education-num", "hours-per-week"),
        categorical=("sex", "race"),
        seed=seed,
    )


def main() -> None:
    path = write_example(build_frame(), FILENAME)
    print(f"wrote {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
