"""
generate_all.py
===============
One-shot script to populate every folder in the project:

    data/raw/loans_raw.csv
    data/processed/loans_processed.csv
    models/logistic_regression.pkl
    models/gradient_boosting.pkl
    models/gradient_boosting_calibrated.pkl
    reports/figures/*.png  (16 plots)

Run from the project root:

    python generate_all.py

Takes roughly 3-5 minutes on a modern laptop (GBM training dominates).
All progress is printed with timestamps.
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# -- Make src/ importable when run from project root -----------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves to file, no window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.data_generation import generate_loan_dataset
from src.feature_engineering import engineer_features
from src.modeling import (
    apply_smote,
    calibrate_model,
    save_model,
    train_gradient_boosting,
    train_logistic_regression,
)
from src.evaluation import (
    build_metrics_summary_table,
    compute_brier_score,
    compute_ks_statistic,
    compute_roc_auc,
    plot_calibration_curve,
    plot_ks_curve,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from src.business_simulation import (
    compute_portfolio_summary,
    find_optimal_threshold,
    plot_threshold_vs_approval_rate,
    plot_threshold_vs_profit,
    threshold_sweep,
)
from src.visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_feature_importance_bar,
    plot_shap_dependence,
    plot_smote_comparison,
    set_portfolio_style,
)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

DATA_RAW_DIR   = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROC_DIR  = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR      = os.path.join(PROJECT_ROOT, "models")
FIG_DIR        = os.path.join(PROJECT_ROOT, "reports", "figures")

for d in [DATA_RAW_DIR, DATA_PROC_DIR, MODEL_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
set_portfolio_style()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _step(msg: str) -> float:
    """Print a timestamped step header and return the start time."""
    t = time.time()
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")
    return t


def _done(t0: float, extra: str = "") -> None:
    print(f"  Done in {time.time() - t0:.1f}s{'.  ' + extra if extra else '.'}")


def _fig(name: str) -> str:
    """Return full save path for a figure filename."""
    return os.path.join(FIG_DIR, name)


# ==========================================================================
# STEP 1 — Data Generation
# ==========================================================================

t0 = _step("Step 1/5 — Generating synthetic loan dataset (n=50,000)")

df = generate_loan_dataset(n=50_000, default_rate=0.20, random_state=RANDOM_STATE)

raw_path = os.path.join(DATA_RAW_DIR, "loans_raw.csv")
df.to_csv(raw_path, index=False)

print(f"  Saved: {raw_path}")
print(f"  Shape: {df.shape}  |  Default rate: {df['default'].mean():.2%}")
_done(t0, f"{os.path.getsize(raw_path) / 1e6:.1f} MB")


# ==========================================================================
# STEP 2 — Feature Engineering
# ==========================================================================

t0 = _step("Step 2/5 — Feature engineering + train/test split")

X_raw = df.drop(columns=["default"])
y     = df["default"].values

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
X_train_raw = X_train_raw.reset_index(drop=True)
X_test_raw  = X_test_raw.reset_index(drop=True)

# Add target back temporarily (engineer_features drops it internally)
train_df = X_train_raw.copy(); train_df["default"] = y_train
test_df  = X_test_raw.copy();  test_df["default"]  = y_test

X_train_fe, enc_state = engineer_features(train_df, fit=True)
X_test_fe,  _         = engineer_features(test_df,  fit=False, encoder_state=enc_state)

feature_names = X_train_fe.columns.tolist()
X_train_np = X_train_fe.values.astype(np.float32)
X_test_np  = X_test_fe.values.astype(np.float32)

proc_path = os.path.join(DATA_PROC_DIR, "loans_processed.csv")
proc_df = X_train_fe.copy(); proc_df["default"] = y_train
proc_df.to_csv(proc_path, index=False)

print(f"  Features after engineering: {len(feature_names)}")
print(f"  Train: {X_train_np.shape}  |  Test: {X_test_np.shape}")
print(f"  Saved: {proc_path}")
_done(t0)


# ==========================================================================
# STEP 3 — Model Training
# ==========================================================================

t0 = _step("Step 3/5 — Training models (LR + GBM + calibration)")

# SMOTE on training data only
print("  Applying SMOTE...")
X_train_smote, y_train_smote = apply_smote(
    X_train_np, y_train, sampling_strategy=0.5, random_state=RANDOM_STATE
)
print(f"  Post-SMOTE: {y_train_smote.mean():.1%} default rate, {len(y_train_smote):,} samples")

# Split SMOTE'd train into model-train (80%) + calibration (20%)
X_tr, X_cal, y_tr, y_cal = train_test_split(
    X_train_smote, y_train_smote,
    test_size=0.20, stratify=y_train_smote, random_state=RANDOM_STATE
)

# Logistic Regression
print("  Training Logistic Regression...")
lr_model = train_logistic_regression(X_tr, y_tr, C=0.1, random_state=RANDOM_STATE)
save_model(lr_model, os.path.join(MODEL_DIR, "logistic_regression.pkl"))

# Gradient Boosting
print("  Training Gradient Boosting (may take ~2-4 min)...")
gb_model = train_gradient_boosting(
    X_tr, y_tr,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=50,
    random_state=RANDOM_STATE,
)
save_model(gb_model, os.path.join(MODEL_DIR, "gradient_boosting.pkl"))

# Platt calibration
print("  Calibrating GBM (Platt scaling)...")
gb_cal = calibrate_model(gb_model, X_cal, y_cal, method="sigmoid")
save_model(gb_cal, os.path.join(MODEL_DIR, "gradient_boosting_calibrated.pkl"))

# Predictions on test set
lr_probs   = lr_model.predict_proba(X_test_np)[:, 1]
gb_probs   = gb_model.predict_proba(X_test_np)[:, 1]
gb_c_probs = gb_cal.predict_proba(X_test_np)[:, 1]

_done(t0, "3 models saved")

# Print metrics table
models_results = {
    "Logistic Regression":      {"y_true": y_test, "y_prob": lr_probs},
    "Gradient Boosting (raw)":  {"y_true": y_test, "y_prob": gb_probs},
    "GBM (calibrated)":         {"y_true": y_test, "y_prob": gb_c_probs},
}
metrics_df = build_metrics_summary_table(models_results)
print("\n  Model Performance:")
print(metrics_df.to_string(index=False))


# ==========================================================================
# STEP 4 — All Figures
# ==========================================================================

t0 = _step("Step 4/5 — Generating all figures (16 plots)")

roc_dict  = {
    "Logistic Regression":      (y_test, lr_probs),
    "Gradient Boosting (raw)":  (y_test, gb_probs),
    "GBM (calibrated)":         (y_test, gb_c_probs),
}

# ---- EDA plots -----------------------------------------------------------

print("  [1/16] class_distribution.png")
fig, ax = plt.subplots(figsize=(6, 5))
plot_class_distribution(y, ax=ax, save_path=_fig("class_distribution.png"))
plt.close()

print("  [2/16] feature_distributions.png")
numeric_feats = [
    "credit_score", "annual_income", "credit_utilization",
    "debt_to_income", "loan_amount", "interest_rate",
    "num_delinquencies", "num_credit_lines",
]
fig = plot_feature_distributions(df, features=numeric_feats, hue_col="default",
                                  n_cols=4, save_path=_fig("feature_distributions.png"))
plt.close()

print("  [3/16] correlation_heatmap.png")
numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["loan_id"], errors="ignore")
fig = plot_correlation_heatmap(numeric_df, figsize=(12, 9),
                                save_path=_fig("correlation_heatmap.png"))
plt.close()

print("  [4/16] default_rate_by_segment.png")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["home_ownership", "loan_purpose", "loan_term"]):
    dr = df.groupby(col)["default"].mean().sort_values(ascending=False)
    cts = df[col].value_counts()
    bars = ax.bar([str(x) for x in dr.index], dr.values * 100,
                  color="#2196F3", edgecolor="none")
    ax.axhline(y=df["default"].mean() * 100, color="red", linestyle="--", lw=1.5,
               label=f"Overall: {df['default'].mean():.1%}")
    for bar, idx in zip(bars, dr.index):
        n = cts.get(idx, 0)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"n={n:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Default Rate by {col.replace('_', ' ').title()}", fontweight="bold")
    ax.set_ylabel("Default Rate (%)")
    ax.legend(fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(_fig("default_rate_by_segment.png"), dpi=150, bbox_inches="tight")
plt.close()

print("  [5/16] smote_comparison.png")
X_smote_for_vis, y_smote_for_vis = apply_smote(
    X_train_np, y_train, sampling_strategy=0.5, random_state=RANDOM_STATE
)
fig = plot_smote_comparison(y_train, y_smote_for_vis,
                             save_path=_fig("smote_comparison.png"))
plt.close()

# ---- Evaluation plots ----------------------------------------------------

print("  [6/16] roc_curves.png")
fig, ax = plt.subplots(figsize=(8, 6))
plot_roc_curves(roc_dict, ax=ax, save_path=_fig("roc_curves.png"))
plt.close()

print("  [7/16] precision_recall_curves.png")
fig, ax = plt.subplots(figsize=(8, 6))
plot_precision_recall_curves(roc_dict, baseline_positive_rate=y_test.mean(),
                              ax=ax, save_path=_fig("precision_recall_curves.png"))
plt.close()

print("  [8/16] ks_chart.png")
fig, ax = plt.subplots(figsize=(9, 6))
plot_ks_curve(y_test, gb_c_probs, model_name="GBM (Calibrated)",
              ax=ax, save_path=_fig("ks_chart.png"))
plt.close()

print("  [9/16] calibration_curve.png")
cal_dict = {
    "GBM (uncalibrated)": (y_test, gb_probs),
    "GBM (Platt scaled)": (y_test, gb_c_probs),
    "Logistic Regression": (y_test, lr_probs),
}
fig, ax = plt.subplots(figsize=(8, 6))
plot_calibration_curve(cal_dict, n_bins=10, ax=ax,
                       save_path=_fig("calibration_curve.png"))
plt.close()

# ---- Business simulation plots -------------------------------------------

print("  [10/16] threshold_approval_rate.png")
test_loan_amounts   = X_test_raw["loan_amount"].values
test_interest_rates = X_test_raw["interest_rate"].values
test_loan_terms     = X_test_raw["loan_term"].values

sweep_df = threshold_sweep(
    y_true=y_test,
    y_prob=gb_c_probs,
    loan_amounts=test_loan_amounts,
    interest_rates=test_interest_rates,
    loan_terms=test_loan_terms,
)
opt_threshold, _ = find_optimal_threshold(sweep_df, objective="total_profit")

fig, ax = plt.subplots(figsize=(10, 6))
plot_threshold_vs_approval_rate(sweep_df, ax=ax,
                                 save_path=_fig("threshold_approval_rate.png"))
plt.close()

print("  [11/16] threshold_profit.png")
fig, ax = plt.subplots(figsize=(10, 6))
plot_threshold_vs_profit(sweep_df, optimal_threshold=opt_threshold,
                          ax=ax, save_path=_fig("threshold_profit.png"))
plt.close()

# ---- SHAP plots ----------------------------------------------------------

print("  [12/16] shap_summary.png  (computing SHAP values...)")
base_gbm = gb_cal.estimator
shap_idx = np.random.default_rng(RANDOM_STATE).choice(len(X_test_np), size=2000, replace=False)
X_shap    = X_test_fe.iloc[shap_idx].reset_index(drop=True)
X_shap_np = X_shap.values

explainer   = shap.TreeExplainer(base_gbm)
shap_values = explainer.shap_values(X_shap_np)

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_shap, max_display=15, show=False)
plt.title("SHAP Feature Impact — Gradient Boosting (Top 15)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(_fig("shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()

print("  [13/16] shap_importance_bar.png")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
fig = plot_feature_importance_bar(feature_names, mean_abs_shap, top_n=15,
                                   save_path=_fig("shap_importance_bar.png"))
plt.close()

print("  [14/16] shap_waterfall.png")
explanation = explainer(X_shap_np)
shap_probs  = gb_c_probs[shap_idx]
high_risk_i = int(np.argmax(shap_probs))

shap.waterfall_plot(explanation[high_risk_i], show=False, max_display=12)
plt.title(f"SHAP Waterfall — High-Risk Loan (PD={shap_probs[high_risk_i]:.1%})",
          fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(_fig("shap_waterfall.png"), dpi=150, bbox_inches="tight")
plt.close()

print("  [15/16] shap_dependence_credit_score.png")
fig = plot_shap_dependence(shap_values, X_shap, feature="credit_score",
                            interaction_feature="credit_utilization",
                            save_path=_fig("shap_dependence_credit_score.png"))
plt.close()

print("  [16/16] shap_dependence_utilization.png")
fig = plot_shap_dependence(shap_values, X_shap, feature="credit_utilization",
                            interaction_feature="num_delinquencies",
                            save_path=_fig("shap_dependence_utilization.png"))
plt.close()

_done(t0, "16 figures saved")


# ==========================================================================
# STEP 5 — Final Summary
# ==========================================================================

_step("Step 5/5 — Summary")

auc_gb, _, _, _ = compute_roc_auc(y_test, gb_c_probs)
ks_gb, _        = compute_ks_statistic(y_test, gb_c_probs)
brier_gb        = compute_brier_score(y_test, gb_c_probs)

portfolio = compute_portfolio_summary(
    y_test, gb_c_probs,
    test_loan_amounts, test_interest_rates, test_loan_terms,
    threshold=opt_threshold,
)

pngs = [f for f in os.listdir(FIG_DIR) if f.endswith(".png")]
total_mb = sum(os.path.getsize(os.path.join(FIG_DIR, f)) for f in pngs) / 1e6

print(f"""
  Best model (GBM calibrated):
    ROC-AUC:    {auc_gb:.4f}
    KS:         {ks_gb:.4f}
    Brier:      {brier_gb:.4f}

  Optimal threshold: {opt_threshold:.2f}
    Approval rate:   {portfolio['approval_rate']:.1%}
    Default rate:    {portfolio['default_rate_in_approved']:.1%}
    Portfolio profit: ${portfolio['expected_total_profit']:,.0f}

  Files generated:
    data/raw/loans_raw.csv                  ({os.path.getsize(raw_path)/1e6:.1f} MB)
    data/processed/loans_processed.csv      ({os.path.getsize(proc_path)/1e6:.1f} MB)
    models/logistic_regression.pkl
    models/gradient_boosting.pkl
    models/gradient_boosting_calibrated.pkl
    reports/figures/  ({len(pngs)} PNG files, {total_mb:.1f} MB total)

  All done. You can now git add . && git push
""")
