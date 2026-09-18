"""FairCheck -- correct, uncertainty-aware fairness reports for classification predictions.

The public surface is the pure-Python core: the Streamlit app and the CLI are both thin
callers of these functions and contain no metric logic of their own.
"""

from faircheck.bootstrap import BootstrapResult, bootstrap_audit, bootstrap_counts
from faircheck.calibration import CalibrationResult, calibrate, threshold_sweep
from faircheck.guidance import (
    DecisionKind,
    GuidanceAnswers,
    LabelTrust,
    Recommendation,
    ScoreReading,
    recommend,
    summarize,
)
from faircheck.impossibility import ImpossibilityResult, explain_impossibility
from faircheck.metrics import (
    GAPS,
    RATES,
    compute_gaps,
    confusion_counts,
    fails_four_fifths,
    per_group_rates,
)
from faircheck.report import (
    AuditResult,
    render_markdown,
    run_audit,
    write_markdown,
    write_pdf,
)
from faircheck.types import ColumnMapping, GroupCounts, ValidatedData
from faircheck.validate import ValidationError, read_csv, validate

__version__ = "0.1.0"

__all__ = [
    "GAPS",
    "RATES",
    "AuditResult",
    "BootstrapResult",
    "CalibrationResult",
    "ColumnMapping",
    "DecisionKind",
    "GroupCounts",
    "GuidanceAnswers",
    "ImpossibilityResult",
    "LabelTrust",
    "Recommendation",
    "ScoreReading",
    "ValidatedData",
    "ValidationError",
    "__version__",
    "bootstrap_audit",
    "bootstrap_counts",
    "calibrate",
    "compute_gaps",
    "confusion_counts",
    "explain_impossibility",
    "fails_four_fifths",
    "per_group_rates",
    "read_csv",
    "recommend",
    "render_markdown",
    "run_audit",
    "summarize",
    "threshold_sweep",
    "validate",
    "write_markdown",
    "write_pdf",
]
