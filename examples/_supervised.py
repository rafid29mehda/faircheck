"""Hold-out logistic predictions for the two real-data examples.

sklearn lives here, not in ``faircheck/``. The committed CSVs are the artifact;
this module is only needed when regenerating them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from examples._common import N_EXPORT, SEED


def holdout_logistic_predictions(
    features: pd.DataFrame,
    y: np.ndarray,
    extra: pd.DataFrame,
    *,
    numeric: tuple[str, ...],
    categorical: tuple[str, ...],
    seed: int = SEED,
    n_export: int = N_EXPORT,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Fit on a train split, export test-set labels, scores and extra columns."""
    y_bool = np.asarray(y, dtype=bool)

    index = np.arange(len(features))
    train_idx, test_idx = train_test_split(
        index, test_size=0.30, random_state=seed, stratify=y_bool
    )

    transform = ColumnTransformer(
        [
            ("num", StandardScaler(), list(numeric)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical),
            ),
        ]
    )
    model = Pipeline(
        [
            ("prep", transform),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    model.fit(features.iloc[train_idx], y_bool[train_idx])

    scores = model.predict_proba(features.iloc[test_idx])[:, 1]
    keep = test_idx
    if len(keep) > n_export:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(keep, size=n_export, replace=False))
        scores = model.predict_proba(features.iloc[keep])[:, 1]

    out = extra.iloc[keep].copy()
    out.insert(0, "y_true", y_bool[keep].astype(np.int64))
    out.insert(1, "y_pred", (scores >= threshold).astype(np.int64))
    out.insert(2, "score", scores)
    return out.reset_index(drop=True)
