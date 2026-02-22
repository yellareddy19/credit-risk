"""
evaluation.py
=============
Model evaluation metrics and visualisation for credit risk models.

All functions are pure (no side effects beyond optional file saving).
They accept numpy arrays so they're agnostic to the specific model used.

Metrics implemented
-------------------
ROC-AUC:
    Area Under the Receiver Operating Characteristic curve. Measures
    discrimination ability — the probability that the model ranks a
    randomly chosen defaulter higher than a randomly chosen non-defaulter.
    Range: [0.5 (random), 1.0 (perfect)]. Industry target for credit risk: > 0.75.

KS Statistic (Kolmogorov-Smirnov):
    Maximum separation between the CDF of predicted scores for defaults
    vs. non-defaults. Widely used in retail banking model validation.
    Rule of thumb: KS > 0.40 = good model, KS > 0.55 = excellent.

Gini Coefficient:
    Gini = 2 × AUC − 1. Equivalent to Somers' D. Required metric in
    Basel III/IV internal ratings-based (IRB) model documentation.
    Range: [0 (random), 1.0 (perfect)].

Brier Score:
    Mean squared error between predicted probabilities and true labels.
    Measures calibration quality. Lower is better. A random model on a
    20% default rate dataset scores 0.20 × 0.80 = 0.16.

Average Precision:
    Area under the Precision-Recall curve. More informative than ROC-AUC
    when the positive class (defaults) is rare and false positives are costly.

Example
-------
>>> from src.evaluation import compute_roc_auc, compute_ks_statistic
>>> auc, fpr, tpr, thresh = compute_roc_auc(y_test, y_prob)
>>> ks, ks_thresh = compute_ks_statistic(y_test, y_prob)
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")  # Non-interactive backend; switch to 'TkAgg' if running locally


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def compute_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC-AUC score and curve components.

    Returns
    -------
    auc_score : float
    fpr : np.ndarray
        False positive rates at each threshold.
    tpr : np.ndarray
        True positive rates at each threshold.
    thresholds : np.ndarray
        Decision thresholds corresponding to each (fpr, tpr) point.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    return auc_score, fpr, tpr, thresholds


def compute_ks_statistic(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, float]:
    """
    Compute the Kolmogorov-Smirnov statistic for a binary classifier.

    KS = max_t | CDF_{defaults}(t) - CDF_{non-defaults}(t) |

    where CDF_{class}(t) is the cumulative fraction of that class's
    predicted scores that fall at or below threshold t.

    The KS statistic measures the maximum discrimination the model
    achieves at any single threshold. Unlike AUC, it focuses on the
    single best separating threshold rather than averaging across all.

    Returns
    -------
    ks_stat : float
        The KS statistic (0 = no discrimination, 1 = perfect separation).
    ks_threshold : float
        The predicted score threshold at which max separation occurs.
    """
    # Sort by predicted score
    sort_idx = np.argsort(y_score)
    y_sorted = y_true[sort_idx]
    scores_sorted = y_score[sort_idx]

    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos

    # Cumulative fractions
    cum_defaults = np.cumsum(y_sorted) / max(n_pos, 1)
    cum_non_defaults = np.cumsum(1 - y_sorted) / max(n_neg, 1)

    ks_at_each_threshold = np.abs(cum_defaults - cum_non_defaults)
    ks_idx = np.argmax(ks_at_each_threshold)

    return float(ks_at_each_threshold[ks_idx]), float(scores_sorted[ks_idx])


def compute_gini_coefficient(auc_score: float) -> float:
    """
    Compute the Gini coefficient from ROC-AUC.

    Gini = 2 × AUC − 1

    The Gini coefficient is the standard credit risk discrimination metric
    in Basel regulatory frameworks. It equals Somers' D for binary outcomes.

    A model with AUC = 0.83 has Gini = 0.66, meaning it correctly
    orders 66% more pairs than a random model.
    """
    return 2.0 * auc_score - 1.0


def compute_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    Compute the Brier Score: mean squared error of probability predictions.

    Lower is better. Baseline (predicting the base rate for all):
        Brier_baseline = base_rate × (1 − base_rate)
        For 20% default rate: 0.20 × 0.80 = 0.160

    A well-calibrated model should score substantially below this baseline.
    """
    return float(brier_score_loss(y_true, y_prob))


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_metrics_summary_table(
    models_results: dict,
) -> pd.DataFrame:
    """
    Aggregate all evaluation metrics into a comparison DataFrame.

    Parameters
    ----------
    models_results : dict
        Mapping: {model_display_name: {'y_true': array, 'y_prob': array}}

    Returns
    -------
    pd.DataFrame
        Columns: Model, ROC-AUC, Gini, KS-Stat, Avg-Precision, Brier, Log-Loss
        One row per model. Sorted by ROC-AUC descending.

    Example
    -------
    >>> results = {
    ...     'Logistic Regression': {'y_true': y_test, 'y_prob': lr_probs},
    ...     'Gradient Boosting':   {'y_true': y_test, 'y_prob': gb_probs},
    ... }
    >>> df_metrics = build_metrics_summary_table(results)
    """
    rows = []
    for model_name, data in models_results.items():
        y_true = data["y_true"]
        y_prob = data["y_prob"]

        auc, _, _, _ = compute_roc_auc(y_true, y_prob)
        ks, _ = compute_ks_statistic(y_true, y_prob)
        gini = compute_gini_coefficient(auc)
        brier = compute_brier_score(y_true, y_prob)
        avg_prec = average_precision_score(y_true, y_prob)
        ll = log_loss(y_true, y_prob)

        rows.append({
            "Model": model_name,
            "ROC-AUC": round(auc, 4),
            "Gini": round(gini, 4),
            "KS-Stat": round(ks, 4),
            "Avg-Precision": round(avg_prec, 4),
            "Brier Score": round(brier, 4),
            "Log-Loss": round(ll, 4),
        })

    df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_roc_curves(
    models_dict: dict,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
    figsize: tuple = (8, 6),
) -> plt.Axes:
    """
    Plot overlaid ROC curves for multiple models.

    Parameters
    ----------
    models_dict : dict
        {model_name: (y_true, y_score)} — tuples of arrays.
    ax : matplotlib Axes, optional
        If None, a new figure is created.
    save_path : str, optional
        If provided, saves the figure at this path.
    figsize : tuple

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    for (model_name, (y_true, y_score)), color in zip(models_dict.items(), colors):
        auc, fpr, tpr, _ = compute_roc_auc(y_true, y_score)
        ax.plot(fpr, tpr, color=color, lw=2.0,
                label=f"{model_name}  (AUC = {auc:.4f})")

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random (AUC = 0.50)", alpha=0.6)

    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("ROC Curves — Discrimination Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_precision_recall_curves(
    models_dict: dict,
    baseline_positive_rate: float,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
    figsize: tuple = (8, 6),
) -> plt.Axes:
    """
    Plot Precision-Recall curves with average precision annotation.

    The horizontal baseline line represents the Precision of a random
    model — it equals the positive class prevalence (default rate).
    Any model above this line adds value over random guessing.

    Parameters
    ----------
    models_dict : dict
        {model_name: (y_true, y_score)}
    baseline_positive_rate : float
        Positive class prevalence (default rate), drawn as a baseline.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    for (model_name, (y_true, y_score)), color in zip(models_dict.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, color=color, lw=2.0,
                label=f"{model_name}  (AP = {ap:.4f})")

    # Baseline
    ax.axhline(
        y=baseline_positive_rate,
        color="k",
        linestyle="--",
        lw=1.2,
        label=f"Random (AP = {baseline_positive_rate:.2f})",
        alpha=0.6,
    )

    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (PPV)", fontsize=12)
    ax.set_title("Precision-Recall Curves — Minority Class Performance", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_ks_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str = "",
    ax: plt.Axes | None = None,
    save_path: str | None = None,
    figsize: tuple = (9, 6),
) -> plt.Axes:
    """
    Plot the KS separation chart.

    Shows the cumulative distribution of predicted scores for default and
    non-default loans. The KS statistic is the maximum vertical distance
    between the two curves, annotated with a vertical line.

    This is one of the most standard charts in credit risk model validation
    reports (often called the "KS chart" or "separation chart").
    """
    sort_idx = np.argsort(y_score)
    y_sorted = y_true[sort_idx]
    scores_sorted = y_score[sort_idx]

    n = len(y_sorted)
    n_pos = y_sorted.sum()
    n_neg = n - n_pos

    cum_defaults = np.cumsum(y_sorted) / max(n_pos, 1)
    cum_non_defaults = np.cumsum(1 - y_sorted) / max(n_neg, 1)
    separation = np.abs(cum_defaults - cum_non_defaults)

    ks_idx = np.argmax(separation)
    ks_stat = separation[ks_idx]
    ks_score = scores_sorted[ks_idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(scores_sorted, cum_non_defaults, color="#2196F3", lw=2.0, label="Non-Default (Good loans)")
    ax.plot(scores_sorted, cum_defaults, color="#FF5722", lw=2.0, label="Default (Bad loans)")

    # Annotate KS
    ax.axvline(x=ks_score, color="gray", linestyle="--", lw=1.5, alpha=0.8)
    ax.annotate(
        f"KS = {ks_stat:.3f}\n@ score = {ks_score:.3f}",
        xy=(ks_score, (cum_defaults[ks_idx] + cum_non_defaults[ks_idx]) / 2),
        xytext=(ks_score + 0.05, 0.5),
        fontsize=10,
        color="black",
        arrowprops=dict(arrowstyle="->", color="black"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )

    title = f"KS Separation Chart — {model_name}" if model_name else "KS Separation Chart"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Default Probability", fontsize=12)
    ax.set_ylabel("Cumulative Distribution", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_calibration_curve(
    models_dict: dict,
    n_bins: int = 10,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
    figsize: tuple = (8, 6),
) -> plt.Axes:
    """
    Plot reliability diagrams (calibration curves).

    A well-calibrated model's curve lies close to the diagonal y=x line.
    Points above the diagonal = model under-predicts probabilities.
    Points below the diagonal = model over-predicts (overconfident).

    Tree ensembles typically produce sigmoid-shaped calibration curves
    (too confident at the extremes), which Platt scaling corrects.

    Parameters
    ----------
    models_dict : dict
        {model_name: (y_true, y_prob)}
    n_bins : int
        Number of bins for the reliability diagram.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    for (model_name, (y_true, y_prob)), color in zip(models_dict.items(), colors):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        brier = brier_score_loss(y_true, y_prob)
        ax.plot(
            mean_predicted_value,
            fraction_of_positives,
            marker="o",
            markersize=5,
            color=color,
            lw=2.0,
            label=f"{model_name}  (Brier = {brier:.4f})",
        )

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration", alpha=0.7)

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives (Actual Rate)", fontsize=12)
    ax.set_title("Reliability Diagram — Probability Calibration", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax
