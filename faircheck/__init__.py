"""FairCheck -- correct, uncertainty-aware fairness reports for classification predictions.

The public surface is the pure-Python core: the Streamlit app and the CLI are both thin
callers of these functions and contain no metric logic of their own.
"""

from faircheck.bootstrap import BootstrapResult, bootstrap_audit, bootstrap_counts
from faircheck.metrics import (
    GAPS,
    RATES,
    compute_gaps,
    confusion_counts,
    fails_four_fifths,
    per_group_rates,
)
from faircheck.types import ColumnMapping, GroupCounts, ValidatedData
from faircheck.validate import ValidationError, read_csv, validate

__version__ = "0.1.0"

__all__ = [
    "GAPS",
    "RATES",
    "BootstrapResult",
    "ColumnMapping",
    "GroupCounts",
    "ValidatedData",
    "ValidationError",
    "__version__",
    "bootstrap_audit",
    "bootstrap_counts",
    "compute_gaps",
    "confusion_counts",
    "fails_four_fifths",
    "per_group_rates",
    "read_csv",
    "validate",
]
