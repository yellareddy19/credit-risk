"""
generate_figures_quick.py
=========================
Fast version of generate_all.py — produces all key figures in ~60-90 seconds.

Differences from generate_all.py:
  - n = 8,000 loans instead of 50,000  (10x faster data gen)
  - GBM: n_estimators=80, max_depth=3   (4x faster training)
  - SHAP: 500 samples instead of 2,000  (4x faster SHAP)

Output is identical in format; metrics are slightly lower due to smaller
training set, but figures look the same.

Run from the project root:
    python generate_figures_quick.py
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.data_generation import generate_loan_dataset
from src.feature_engineering import engineer_features
from src.modeling import (
    apply_smote, calibrate_model, save_model,
    train_gradient_boosting, train_logistic_regression,
)
from src.evaluation import (
    build_metrics_summary_table, compute_brier_score,
    compute_ks_statistic, compute_roc_auc,
    plot_calibration_curve, plot_ks_curve,
    plot_precision_recall_curves, plot_roc_curves,
)
from src.business_simulation import (
    find_optimal_threshold, plot_threshold_vs_approval_rate,
    plot_threshold_vs_profit, threshold_sweep,
)
from src.visualization import (
    plot_class_distribution, plot_correlation_heatmap,
    plot_feature_distributions, plot_feature_importance_bar,
    plot_shap_dependence, plot_smote_comparison, set_portfolio_style,
)

# ── paths ──────────────────────────────────────────────────────────────────
FIG_DIR   = os.path.join(PROJECT_ROOT, "reports", "figures")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_RAW  = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROC = os.path.join(PROJECT_ROOT, "data", "processed")
for d in [FIG_DIR, MODEL_DIR, DATA_RAW, DATA_PROC]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
set_portfolio_style()

def fig(name): return os.path.join(FIG_DIR, name)
def stamp(msg): print(f"  [{time.strftime('%H:%M:%S')}] {msg}")

T0 = time.time()
print("=" * 55)
print("  Generating figures  (quick mode — ~90 sec)")
print("=" * 55)

# ── 1. DATA ────────────────────────────────────────────────────────────────
stamp("Generating 8,000 synthetic loans...")
df = generate_loan_dataset(n=8_000, default_rate=0.20, random_state=RANDOM_STATE)
df.to_csv(os.path.join(DATA_RAW, "loans_raw.csv"), index=False)

X_raw = df.drop(columns=["default"])
y     = df["default"].values

X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_raw, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
X_tr_raw = X_tr_raw.reset_index(drop=True)
X_te_raw = X_te_raw.reset_index(drop=True)

train_df = X_tr_raw.copy(); train_df["default"] = y_tr
test_df  = X_te_raw.copy(); test_df["default"]  = y_te

X_tr_fe, state  = engineer_features(train_df, fit=True)
X_te_fe, _      = engineer_features(test_df,  fit=False, encoder_state=state)
feature_names   = X_tr_fe.columns.tolist()
X_tr_np = X_tr_fe.values.astype(np.float32)
X_te_np = X_te_fe.values.astype(np.float32)

proc_df = X_tr_fe.copy(); proc_df["default"] = y_tr
proc_df.to_csv(os.path.join(DATA_PROC, "loans_processed.csv"), index=False)
stamp("Data ready.")

# ── 2. MODELS ──────────────────────────────────────────────────────────────
stamp("Applying SMOTE...")
X_sm, y_sm = apply_smote(X_tr_np, y_tr, sampling_strategy=0.5, random_state=RANDOM_STATE)

X_m, X_cal, y_m, y_cal = train_test_split(
    X_sm, y_sm, test_size=0.20, stratify=y_sm, random_state=RANDOM_STATE
)

stamp("Training Logistic Regression...")
lr  = train_logistic_regression(X_m, y_m, C=0.1, random_state=RANDOM_STATE)
save_model(lr, os.path.join(MODEL_DIR, "logistic_regression.pkl"))

stamp("Training Gradient Boosting (80 trees)...")
gb  = train_gradient_boosting(
    X_m, y_m, n_estimators=80, learning_rate=0.08,
    max_depth=3, subsample=0.8, min_samples_leaf=20, random_state=RANDOM_STATE
)
save_model(gb, os.path.join(MODEL_DIR, "gradient_boosting.pkl"))

stamp("Calibrating GBM...")
gb_cal = calibrate_model(gb, X_cal, y_cal, method="sigmoid")
save_model(gb_cal, os.path.join(MODEL_DIR, "gradient_boosting_calibrated.pkl"))

lr_p  = lr.predict_proba(X_te_np)[:, 1]
gb_p  = gb.predict_proba(X_te_np)[:, 1]
gbc_p = gb_cal.predict_proba(X_te_np)[:, 1]

metrics = build_metrics_summary_table({
    "Logistic Regression":  {"y_true": y_te, "y_prob": lr_p},
    "Gradient Boosting":    {"y_true": y_te, "y_prob": gb_p},
    "GBM (calibrated)":     {"y_true": y_te, "y_prob": gbc_p},
})
print("\n" + metrics.to_string(index=False) + "\n")

roc_dict = {
    "Logistic Regression":  (y_te, lr_p),
    "Gradient Boosting":    (y_te, gb_p),
    "GBM (calibrated)":     (y_te, gbc_p),
}

# ── 3. EDA FIGURES ─────────────────────────────────────────────────────────
stamp("[1/16] class_distribution.png")
_, ax = plt.subplots(figsize=(6,5))
plot_class_distribution(y, ax=ax, save_path=fig("class_distribution.png"))
plt.close()

stamp("[2/16] feature_distributions.png")
feats = ["credit_score","annual_income","credit_utilization",
         "debt_to_income","loan_amount","interest_rate",
         "num_delinquencies","num_credit_lines"]
plot_feature_distributions(df, features=feats, hue_col="default",
                           n_cols=4, save_path=fig("feature_distributions.png"))
plt.close()

stamp("[3/16] correlation_heatmap.png")
num_df = df.select_dtypes(include=[np.number])
plot_correlation_heatmap(num_df, figsize=(12,9),
                         save_path=fig("correlation_heatmap.png"))
plt.close()

stamp("[4/16] default_rate_by_segment.png")
f, axes = plt.subplots(1, 3, figsize=(14,4))
for ax, col in zip(axes, ["home_ownership","loan_purpose","loan_term"]):
    dr  = df.groupby(col)["default"].mean().sort_values(ascending=False)
    cts = df[col].value_counts()
    bars = ax.bar([str(x) for x in dr.index], dr.values*100,
                  color="#2196F3", edgecolor="none")
    ax.axhline(df["default"].mean()*100, color="red", ls="--", lw=1.5,
               label=f"Overall: {df['default'].mean():.1%}")
    for bar, idx in zip(bars, dr.index):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"n={cts.get(idx,0):,}", ha="center", va="bottom", fontsize=8)
    ax.set_title(col.replace("_"," ").title(), fontweight="bold")
    ax.set_ylabel("Default Rate (%)")
    ax.legend(fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(fig("default_rate_by_segment.png"), dpi=150, bbox_inches="tight")
plt.close()

stamp("[5/16] smote_comparison.png")
_, y_sm2 = apply_smote(X_tr_np, y_tr, sampling_strategy=0.5, random_state=RANDOM_STATE)
plot_smote_comparison(y_tr, y_sm2, save_path=fig("smote_comparison.png"))
plt.close()

# ── 4. EVALUATION FIGURES ──────────────────────────────────────────────────
stamp("[6/16] roc_curves.png")
_, ax = plt.subplots(figsize=(8,6))
plot_roc_curves(roc_dict, ax=ax, save_path=fig("roc_curves.png"))
plt.close()

stamp("[7/16] precision_recall_curves.png")
_, ax = plt.subplots(figsize=(8,6))
plot_precision_recall_curves(roc_dict, baseline_positive_rate=y_te.mean(),
                              ax=ax, save_path=fig("precision_recall_curves.png"))
plt.close()

stamp("[8/16] ks_chart.png")
_, ax = plt.subplots(figsize=(9,6))
plot_ks_curve(y_te, gbc_p, model_name="GBM (Calibrated)",
              ax=ax, save_path=fig("ks_chart.png"))
plt.close()

stamp("[9/16] calibration_curve.png")
cal_d = {"GBM (uncalibrated)":(y_te,gb_p),
         "GBM (Platt scaled)":(y_te,gbc_p),
         "Logistic Regression":(y_te,lr_p)}
_, ax = plt.subplots(figsize=(8,6))
plot_calibration_curve(cal_d, n_bins=10, ax=ax,
                       save_path=fig("calibration_curve.png"))
plt.close()

# ── 5. BUSINESS SIMULATION FIGURES ─────────────────────────────────────────
la  = X_te_raw["loan_amount"].values
ir  = X_te_raw["interest_rate"].values
lt  = X_te_raw["loan_term"].values

stamp("[10/16] threshold_approval_rate.png")
sweep = threshold_sweep(y_te, gbc_p, la, ir, lt)
opt_t, _ = find_optimal_threshold(sweep, objective="total_profit")
_, ax = plt.subplots(figsize=(10,6))
plot_threshold_vs_approval_rate(sweep, ax=ax,
                                 save_path=fig("threshold_approval_rate.png"))
plt.close()

stamp("[11/16] threshold_profit.png")
_, ax = plt.subplots(figsize=(10,6))
plot_threshold_vs_profit(sweep, optimal_threshold=opt_t,
                          ax=ax, save_path=fig("threshold_profit.png"))
plt.close()

# ── 6. SHAP FIGURES ────────────────────────────────────────────────────────
stamp("[12/16] shap_summary.png  (computing SHAP values on 500 samples...)")
base_gbm = gb_cal.base_model
rng      = np.random.default_rng(RANDOM_STATE)
idx      = rng.choice(len(X_te_np), size=min(500, len(X_te_np)), replace=False)
X_sh     = X_te_fe.iloc[idx].reset_index(drop=True)

explainer   = shap.TreeExplainer(base_gbm)
shap_vals   = explainer.shap_values(X_sh.values)
explanation = explainer(X_sh.values)

plt.figure(figsize=(10,7))
shap.summary_plot(shap_vals, X_sh, max_display=15, show=False)
plt.title("SHAP Feature Impact — Gradient Boosting (Top 15)",
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(fig("shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()

stamp("[13/16] shap_importance_bar.png")
mean_shap = np.abs(shap_vals).mean(axis=0)
plot_feature_importance_bar(feature_names, mean_shap, top_n=15,
                             save_path=fig("shap_importance_bar.png"))
plt.close()

stamp("[14/16] shap_waterfall.png")
hi_risk = int(np.argmax(gbc_p[idx]))
shap.waterfall_plot(explanation[hi_risk], show=False, max_display=12)
plt.title(f"SHAP Waterfall — High-Risk Loan (PD={gbc_p[idx[hi_risk]]:.1%})",
          fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(fig("shap_waterfall.png"), dpi=150, bbox_inches="tight")
plt.close()

stamp("[15/16] shap_dependence_credit_score.png")
plot_shap_dependence(shap_vals, X_sh, "credit_score", "credit_utilization",
                     save_path=fig("shap_dependence_credit_score.png"))
plt.close()

stamp("[16/16] shap_dependence_utilization.png")
plot_shap_dependence(shap_vals, X_sh, "credit_utilization", "num_delinquencies",
                     save_path=fig("shap_dependence_utilization.png"))
plt.close()

# ── DONE ───────────────────────────────────────────────────────────────────
pngs  = [f for f in os.listdir(FIG_DIR) if f.endswith(".png")]
total = sum(os.path.getsize(os.path.join(FIG_DIR,f)) for f in pngs) / 1e6
elapsed = time.time() - T0

print(f"""
{'='*55}
  Done in {elapsed:.0f}s
  {len(pngs)} figures saved to reports/figures/  ({total:.1f} MB total)
  Optimal approval threshold: {opt_t:.2f}
{'='*55}
""")
