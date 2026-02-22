"""
visualization.py
================
Consistent, publication-quality plotting utilities for the loan default
risk modeling project.

All plots follow a unified visual style defined by set_portfolio_style():
- Colorblind-safe palette (blue, orange, green, red)
- DejaVu Sans font, 11pt base size
- Light gray grid, top/right spines removed
- 150 DPI for file output

The functions here wrap matplotlib/seaborn/SHAP to produce the EDA charts
used in the notebook. Evaluation and business simulation charts live in
their own modules (evaluation.py, business_simulation.py) since those
functions are also called directly from those modules' logic.

Example
-------
>>> from src.visualization import set_portfolio_style, plot_class_distribution
>>> set_portfolio_style()
>>> ax = plot_class_distribution(y_train, save_path='reports/figures/class_dist.png')
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

# Colorblind-safe categorical palette (Paul Tol's bright scheme)
PALETTE = {
    "blue":   "#2196F3",
    "orange": "#FF5722",
    "green":  "#4CAF50",
    "red":    "#F44336",
    "purple": "#9C27B0",
    "gray":   "#607D8B",
}

PALETTE_LIST = list(PALETTE.values())


def set_portfolio_style() -> None:
    """
    Set matplotlib rcParams for consistent professional styling across all plots.

    Call this once at the top of the notebook or script before any plotting.
    """
    plt.rcParams.update({
        # Figure
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "figure.facecolor": "white",
        "figure.edgecolor": "none",

        # Font
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,

        # Grid and spines
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Lines and patches
        "lines.linewidth": 2.0,
        "patch.edgecolor": "none",

        # Legend
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#BDBDBD",
    })


# ---------------------------------------------------------------------------
# EDA visualisations
# ---------------------------------------------------------------------------

def plot_class_distribution(
    y: pd.Series | np.ndarray,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
    figsize: tuple = (6, 5),
) -> plt.Axes:
    """
    Bar chart of class counts with percentage labels.

    Shows the degree of class imbalance. The chart includes both raw
    counts and percentages to give a complete picture.

    Parameters
    ----------
    y : array-like
        Binary labels (0 = performing, 1 = default).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    y = np.asarray(y)
    counts = np.bincount(y)
    labels = ["Performing (0)", "Default (1)"]
    colors = [PALETTE["blue"], PALETTE["orange"]]
    total = len(y)

    bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor="none")

    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title("Loan Default — Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Loans", fontsize=12)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.set_xlabel("")

    imbalance_ratio = counts[0] / max(counts[1], 1)
    ax.text(
        0.98, 0.96,
        f"Imbalance ratio: {imbalance_ratio:.1f}:1",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=PALETTE["gray"],
        style="italic",
    )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_feature_distributions(
    df: pd.DataFrame,
    features: list,
    hue_col: str = "default",
    n_cols: int = 3,
    save_path: str | None = None,
    figsize_per_plot: tuple = (4.5, 3.5),
) -> plt.Figure:
    """
    Grid of distribution plots, one per feature, split by default status.

    Uses KDE (kernel density estimate) for continuous features and bar
    charts for discrete features. Comparing distributions by default status
    reveals which features are most predictive before any modelling.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all columns in `features` plus `hue_col`.
    features : list
        List of column names to plot.
    hue_col : str
        Binary column used to colour-split distributions.
    """
    n_features = len(features)
    n_rows = int(np.ceil(n_features / n_cols))
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    hue_labels = {0: "Performing", 1: "Default"}
    hue_colors = {0: PALETTE["blue"], 1: PALETTE["orange"]}

    for ax, feature in zip(axes, features):
        for label_val, label_name in hue_labels.items():
            subset = df[df[hue_col] == label_val][feature].dropna()
            if df[feature].dtype in ["object", "category"]:
                # Categorical: normalised bar chart
                counts = subset.value_counts(normalize=True).sort_index()
                ax.bar(
                    [str(x) for x in counts.index],
                    counts.values,
                    alpha=0.6,
                    color=hue_colors[label_val],
                    label=label_name,
                )
            else:
                # Continuous: KDE
                sns.kdeplot(
                    subset,
                    ax=ax,
                    color=hue_colors[label_val],
                    label=label_name,
                    fill=True,
                    alpha=0.25,
                    linewidth=2.0,
                    warn_singular=False,
                )

        ax.set_title(feature.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=8, framealpha=0.8)

    # Hide unused subplot axes
    for extra_ax in axes[n_features:]:
        extra_ax.set_visible(False)

    fig.suptitle("Feature Distributions by Default Status", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    figsize: tuple = (14, 10),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Correlation heatmap of all numeric features.

    Uses Pearson correlation. The diagonal is masked (trivially 1.0).
    Strong positive correlations (> 0.7) between features can indicate
    multicollinearity issues for Logistic Regression.

    Parameters
    ----------
    df : pd.DataFrame
        Should include only numeric columns. String and boolean columns
        are automatically excluded.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    mask = np.eye(len(corr), dtype=bool)  # mask diagonal

    fig, ax = plt.subplots(figsize=figsize)

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.5,
        linecolor="#E0E0E0",
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
    )

    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_smote_comparison(
    y_before: np.ndarray,
    y_after: np.ndarray,
    save_path: str | None = None,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Side-by-side bar charts showing class distribution before and after SMOTE.

    Visually confirms that SMOTE increased the minority class without
    distorting the majority class distribution excessively.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, y, title in [
        (ax1, y_before, "Before SMOTE"),
        (ax2, y_after, "After SMOTE"),
    ]:
        counts = np.bincount(y)
        labels = ["Performing (0)", "Default (1)"]
        colors = [PALETTE["blue"], PALETTE["orange"]]
        bars = ax.bar(labels, counts, color=colors, width=0.5)

        for bar, count in zip(bars, counts):
            pct = count / len(y) * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + len(y) * 0.005,
                f"{count:,}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_ylim(0, max(counts) * 1.2)

    fig.suptitle("Class Distribution: Before vs. After SMOTE", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# SHAP plots
# ---------------------------------------------------------------------------

def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    max_display: int = 15,
    save_path: str | None = None,
) -> None:
    """
    SHAP beeswarm summary plot.

    Shows the distribution of SHAP values for each feature across all
    test-set predictions. Each dot is one loan. The colour represents
    the feature value (red = high, blue = low). The x-position shows
    the magnitude and direction of impact on the model output.

    Features are ordered by mean absolute SHAP value (overall importance).
    """
    try:
        import shap
    except ImportError:
        print("shap is not installed. Run: pip install shap")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Feature Impact — Gradient Boosting Model", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_shap_waterfall(
    explanation,
    sample_index: int,
    save_path: str | None = None,
) -> None:
    """
    SHAP waterfall plot for a single loan prediction.

    Shows how each feature pushes the model output above or below the
    expected value (base rate). Red bars push toward default; blue bars
    push toward performing. This chart is the basis for adverse action
    reason codes required under ECOA / FCRA.

    Parameters
    ----------
    explanation : shap.Explanation
        Full explanation object from shap.TreeExplainer(model)(X_test).
    sample_index : int
        Index into the explanation object to plot.
    """
    try:
        import shap
    except ImportError:
        print("shap is not installed. Run: pip install shap")
        return

    shap.waterfall_plot(explanation[sample_index], show=False)
    plt.title(f"SHAP Waterfall — Loan #{sample_index} (High-Risk)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_shap_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: str = "auto",
    save_path: str | None = None,
    figsize: tuple = (9, 6),
) -> plt.Figure:
    """
    SHAP dependence plot for a single feature.

    X-axis: feature value. Y-axis: SHAP value for that feature.
    Colour: value of interaction feature (if 'auto', SHAP picks the
    feature with the strongest interaction automatically).

    A non-linear SHAP-vs-feature relationship reveals that the feature
    has a non-linear or threshold-based effect on default risk.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array, shape (n_samples, n_features).
    X : pd.DataFrame
        Feature matrix corresponding to shap_values.
    feature : str
        Feature name to plot.
    interaction_feature : str
        Feature for colour coding. 'auto' = SHAP auto-selection.
    """
    try:
        import shap
    except ImportError:
        print("shap is not installed. Run: pip install shap")
        return

    fig, ax = plt.subplots(figsize=figsize)
    shap.dependence_plot(
        feature,
        shap_values,
        X,
        interaction_index=interaction_feature,
        ax=ax,
        show=False,
    )
    ax.set_title(
        f"SHAP Dependence: {feature.replace('_', ' ').title()}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_feature_importance_bar(
    feature_names: list,
    mean_abs_shap: np.ndarray,
    top_n: int = 15,
    save_path: str | None = None,
    figsize: tuple = (9, 6),
) -> plt.Figure:
    """
    Horizontal bar chart of mean absolute SHAP values (feature importance).

    This is the most interpretable global importance plot — it shows
    which features move model predictions the most on average, in
    the same units as the model output (log-odds for GBM).

    Parameters
    ----------
    feature_names : list
    mean_abs_shap : np.ndarray
        Mean absolute SHAP value per feature.
    top_n : int
        Show only the top N features by importance.
    """
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs_shap,
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        importance_df["feature"][::-1],
        importance_df["importance"][::-1],
        color=PALETTE["blue"],
        edgecolor="none",
    )

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + importance_df["importance"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            fontsize=9,
            color=PALETTE["gray"],
        )

    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title(f"Top {top_n} Features by Mean Absolute SHAP Value", fontsize=13, fontweight="bold")
    ax.set_xlim(0, importance_df["importance"].max() * 1.15)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
