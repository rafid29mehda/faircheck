"""ACS Income predictions for Washington state.

Hits the network the first time Fairlearn/OpenML has no cache (Ding et al. 2021,
OpenML id 43141). Tests never import this module. y_true = 1 means PINCP >= $50,000.
"""

from __future__ import annotations

import pandas as pd

from examples._common import SEED, write_example
from examples._supervised import holdout_logistic_predictions

FILENAME = "acs_income.csv"
STATE = "WA"
INCOME_THRESHOLD = 50_000

# Census ACS PUMS RAC1P codes.
_RACE: dict[int, str] = {
    1: "White",
    2: "Black",
    3: "American Indian",
    4: "Alaska Native",
    5: "American Indian",
    6: "Asian",
    7: "NHPI",
    8: "Other",
    9: "Two or more",
}
_SEX: dict[int, str] = {1: "Male", 2: "Female"}


def _map_race(code: object) -> str:
    try:
        return _RACE[int(float(str(code)))]
    except (TypeError, ValueError, KeyError):
        return "Other"


def _map_sex(code: object) -> str:
    try:
        return _SEX[int(float(str(code)))]
    except (TypeError, ValueError, KeyError):
        return str(code)


def build_frame(*, seed: int = SEED) -> pd.DataFrame:
    from fairlearn.datasets import fetch_acs_income

    bunch = fetch_acs_income(states=[STATE], as_frame=True)
    frame = bunch.frame.copy()
    y = frame["PINCP"].to_numpy(dtype=float) >= INCOME_THRESHOLD
    extra = pd.DataFrame(
        {
            "sex": frame["SEX"].map(_map_sex),
            "race": frame["RAC1P"].map(_map_race),
        }
    )
    features = frame.loc[:, ["AGEP", "SCHL", "WKHP", "SEX", "RAC1P"]].copy()
    # OneHotEncoder wants strings for the categoricals; numerics stay numeric.
    features["SEX"] = features["SEX"].astype(str)
    features["RAC1P"] = features["RAC1P"].astype(str)
    features["SCHL"] = features["SCHL"].astype(str)
    return holdout_logistic_predictions(
        features,
        y,
        extra,
        numeric=("AGEP", "WKHP"),
        categorical=("SCHL", "SEX", "RAC1P"),
        seed=seed,
    )


def main() -> None:
    path = write_example(build_frame(), FILENAME)
    print(f"wrote {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
