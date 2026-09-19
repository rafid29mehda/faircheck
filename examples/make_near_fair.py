"""Seeded synthetic: identical DGP in both groups, so contrasts cover zero.

The contrast case for the COMPAS-like example. A max−min gap CI will still
exclude zero (D11); signed contrasts against the reference group will not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from examples._common import SEED, write_example

FILENAME = "near_fair.csv"
N_PER_GROUP = 5_000
THRESHOLD = 0.5
# Symmetric Beta, mean 0.5, same in both groups.
SHAPE = (3.0, 3.0)


def build_frame(*, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    score_a = rng.beta(*SHAPE, size=N_PER_GROUP)
    score_b = rng.beta(*SHAPE, size=N_PER_GROUP)
    y_a = rng.random(N_PER_GROUP) < score_a
    y_b = rng.random(N_PER_GROUP) < score_b
    score = np.concatenate([score_a, score_b])
    y_true = np.concatenate([y_a, y_b])
    group = np.array(["A"] * N_PER_GROUP + ["B"] * N_PER_GROUP)
    return pd.DataFrame(
        {
            "y_true": y_true.astype(np.int64),
            "y_pred": (score >= THRESHOLD).astype(np.int64),
            "score": score,
            "group": group,
        }
    )


def main() -> None:
    path = write_example(build_frame(), FILENAME)
    print(f"wrote {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
