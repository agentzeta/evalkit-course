"""evalkit -- a teaching library for LLM and agent evals.

Pure standard library. No installation, no dependencies, nothing to break on a
workshop laptop. Every function is short enough to read aloud.

    from evalkit import make_dataset, tune_cv, evaluate_calibrator

    ds = make_dataset()
    report = tune_cv(ds.scores("judge_strong"), ds.labels,
                     min_precision=0.80, min_recall=0.80, model="judge_strong")
    print(report.summary())

Design rule for the whole package: a function that cannot honestly answer
returns None or reports the shortfall. It never silently relaxes a constraint
to produce a number. That rule is the course's main argument, expressed as code.
"""

from .calibration import (
    Bin,
    CalibrationReport,
    IsotonicCalibrator,
    PlattScaler,
    brier,
    brier_decomposition,
    ece,
    evaluate_calibrator,
    log_loss,
    reliability_bins,
)
from .compare import (
    Candidate,
    DeltaResult,
    McNemarResult,
    agreement,
    bootstrap_metric,
    cohens_kappa,
    cost_per_month,
    mcnemar,
    paired_bootstrap_delta,
    pareto_frontier,
)
from .data import (
    EXAMPLE_COSTS,
    FAILURE_MODES,
    JUDGES,
    Dataset,
    Trace,
    make_dataset,
    subsample,
)
from .leakage import (
    Baseline,
    GroupingGap,
    LeakageReport,
    Multiplicity,
    ceiling_violation,
    degenerate_baselines,
    group_folds,
    grouping_gap,
    naive_bayes_fit_predict,
    row_folds,
    template_multiplicity,
)
from .metrics import (
    Confusion,
    MetricRow,
    accuracy,
    balanced_accuracy,
    confusion_at,
    f1,
    f_beta,
    mcc,
    metric_row,
    precision,
    precision_interval,
    recall,
    recall_interval,
    specificity,
    wilson_interval,
)
from .thresholds import (
    CVReport,
    FoldResult,
    ThresholdChoice,
    best_threshold,
    candidate_thresholds,
    feasible_region,
    stratified_folds,
    sweep,
    threshold_sensitivity,
    tune_cv,
)

__version__ = "1.0.0"

__all__ = [
    "Baseline", "Bin", "CVReport", "CalibrationReport", "Candidate", "Confusion", "Dataset",
    "EXAMPLE_COSTS", "FAILURE_MODES", "FoldResult", "IsotonicCalibrator", "JUDGES",
    "DeltaResult", "GroupingGap", "LeakageReport", "McNemarResult", "MetricRow", "PlattScaler", "ThresholdChoice", "Trace",
    "Multiplicity", "accuracy", "agreement", "balanced_accuracy", "best_threshold", "bootstrap_metric",
    "brier", "brier_decomposition", "candidate_thresholds", "cohens_kappa",
    "ceiling_violation", "confusion_at", "cost_per_month", "degenerate_baselines", "ece", "evaluate_calibrator", "f1", "f_beta",
    "feasible_region", "group_folds", "grouping_gap", "log_loss", "make_dataset", "mcc", "mcnemar", "metric_row",
    "paired_bootstrap_delta", "naive_bayes_fit_predict", "pareto_frontier", "precision", "precision_interval", "recall", "recall_interval",
    "reliability_bins", "row_folds", "specificity", "stratified_folds", "subsample", "sweep",
    "template_multiplicity", "threshold_sensitivity", "tune_cv", "wilson_interval",
]
