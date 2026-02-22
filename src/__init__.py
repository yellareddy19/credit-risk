# src package — Loan Default Risk Modeling
# Import key functions at package level for convenience

from .data_generation import generate_loan_dataset, get_feature_metadata
from .feature_engineering import engineer_features, get_feature_columns
from .modeling import (
    apply_smote,
    train_logistic_regression,
    train_gradient_boosting,
    calibrate_model,
    save_model,
    load_model,
)
from .evaluation import (
    compute_roc_auc,
    compute_ks_statistic,
    compute_gini_coefficient,
    compute_brier_score,
    build_metrics_summary_table,
)
from .business_simulation import (
    compute_expected_profit_per_loan,
    threshold_sweep,
    find_optimal_threshold,
    compute_portfolio_summary,
)

__all__ = [
    "generate_loan_dataset",
    "get_feature_metadata",
    "engineer_features",
    "get_feature_columns",
    "apply_smote",
    "train_logistic_regression",
    "train_gradient_boosting",
    "calibrate_model",
    "save_model",
    "load_model",
    "compute_roc_auc",
    "compute_ks_statistic",
    "compute_gini_coefficient",
    "compute_brier_score",
    "build_metrics_summary_table",
    "compute_expected_profit_per_loan",
    "threshold_sweep",
    "find_optimal_threshold",
    "compute_portfolio_summary",
]
