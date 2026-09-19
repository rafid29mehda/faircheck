"""Seeded synthetic: calibrated scores, unequal base rates, unequal error rates.

This is the pedagogical stand-in for the ProPublica-vs-Northpointe dispute.
Real COMPAS records are not bundled (D9). Both groups' labels are drawn as
``Y ~ Bernoulli(score)``, so the score *is* P(Y=1) and ECE stays small; base
rates differ because the score distributions differ, and a shared threshold
therefore cannot equalise TPR and FPR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from examples._common import SEED, write_example

FILENAME = "compas_like.csv"
N_PER_GROUP = 4_000
THRESHOLD = 0.5
# Beta(2, 5) mean 2/7 ≈ 0.29; Beta(5, 2) mean 5/7 ≈ 0.71.
LOW_P_SHAPE = (2.0, 5.0)
HIGH_P_SHAPE = (5.0, 2.0)


def build_frame(*, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    score_a = rng.beta(*LOW_P_SHAPE, size=N_PER_GROUP)
    score_b = rng.beta(*HIGH_P_SHAPE, size=N_PER_GROUP)
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
