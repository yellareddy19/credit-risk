"""
modeling.py
===========
Model training, imbalance handling, calibration, and serialisation.

Contains four core capabilities:

1. Class Imbalance — SMOTE oversampling (training set only)
2. Logistic Regression — L2-penalised, features scaled inside Pipeline
3. Gradient Boosting — sklearn GradientBoostingClassifier
4. Probability Calibration — Platt scaling via CalibratedClassifierCV

Design decisions
----------------
- Logistic Regression uses a scikit-learn Pipeline with StandardScaler to
  ensure feature scaling happens inside cross-validation / train-test splits,
  preventing any leakage of test-set statistics.

- Gradient Boosting does NOT need scaling (tree splits are scale-invariant),
  so it is trained directly without a preprocessing pipeline.

- SMOTE is applied to the training set ONLY and AFTER the train/test split.
  Applying SMOTE before splitting would create synthetic minority-class
  samples that "leak" information about the test set distribution into the
  training data — a subtle but common mistake.

- Calibration uses cv='prefit' which applies to an already-fitted model.
  A held-out calibration set (or the training set with cross-validation)
  can be used. Here we use a portion of the training set passed in by
  the caller so the user has full control over the data split.

Example
-------
>>> from src.modeling import train_logistic_regression, train_gradient_boosting
>>> from src.modeling import calibrate_model, apply_smote
>>> # After feature engineering and train/test split:
>>> X_res, y_res = apply_smote(X_train, y_train)
>>> lr_model = train_logistic_regression(X_res, y_res)
>>> gb_model = train_gradient_boosting(X_res, y_res)
>>> gb_cal   = calibrate_model(gb_model, X_val, y_val)
"""

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Imbalance handling
# ---------------------------------------------------------------------------

def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampling_strategy: float = 0.5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample the minority class (defaults) using SMOTE.

    SMOTE generates synthetic minority-class samples by interpolating
    between existing minority-class observations in feature space.
    This is more informative than simple random duplication (ROS).

    sampling_strategy=0.5 means we bring the minority class up to 50%
    of the majority class count — a 1:2 ratio. We intentionally do NOT
    fully balance (1:1) because:
    - Full balancing introduces many synthetic points, which can cause
      the model to "memorise" synthetic artefacts
    - In practice, a 1:2 ratio gives most of the benefit with less risk
      of over-fitting to synthetic data

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
    y_train : np.ndarray, shape (n_train,), binary labels
    sampling_strategy : float
        Ratio of minority to majority class after resampling.
        0.5 = bring minority to 50% of majority count.
    random_state : int

    Returns
    -------
    X_resampled : np.ndarray
    y_resampled : np.ndarray
    """
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 0.1,
    class_weight: str = "balanced",
    max_iter: int = 2_000,
    random_state: int = 42,
) -> Pipeline:
    """
    Train an L2-penalised Logistic Regression inside a scaling Pipeline.

    The Pipeline(StandardScaler → LogisticRegression) ensures that:
    - Feature scaling is fit on training data only (no leakage)
    - The model can be called with a single .predict_proba() on raw features
    - The entire pipeline can be saved and loaded as one object

    C=0.1 applies moderate L2 regularisation, which reduces overfitting
    on high-dimensional one-hot encoded feature spaces. The optimal value
    can be tuned with GridSearchCV on the full pipeline.

    class_weight='balanced': sklearn automatically adjusts class weights
    inverse to class frequencies. This is an alternative to SMOTE —
    both approaches address the imbalanced label distribution, and results
    can be compared in the notebook.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    C : float
        Inverse regularisation strength. Smaller = stronger regularisation.
    class_weight : str or dict
        'balanced' uses inverse class frequencies. Pass None to disable.
    max_iter : int
        Max solver iterations; 2000 ensures convergence on large datasets.
    random_state : int

    Returns
    -------
    Pipeline
        Fitted sklearn Pipeline with steps ['scaler', 'classifier'].
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ---------------------------------------------------------------------------
# Gradient Boosting
# ---------------------------------------------------------------------------

def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.8,
    min_samples_leaf: int = 50,
    random_state: int = 42,
) -> GradientBoostingClassifier:
    """
    Train a Gradient Boosting classifier (sklearn implementation).

    Hyperparameter rationale:
    - n_estimators=300, learning_rate=0.05: shallow learning rate with
      many trees prevents overfitting better than fewer, faster trees
    - max_depth=4: moderate tree depth; depth >5 often leads to overfitting
      on tabular credit data
    - subsample=0.8: stochastic gradient boosting (uses 80% of data per tree),
      which acts as implicit regularisation and speeds training
    - min_samples_leaf=50: requires at least 50 observations per leaf node,
      reducing splits on very small subgroups and improving generalisation

    No scaling is needed for tree-based models since splits are determined
    by rank order of feature values, not their magnitudes.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    n_estimators : int
    learning_rate : float
    max_depth : int
    subsample : float
    min_samples_leaf : int
    random_state : int

    Returns
    -------
    GradientBoostingClassifier
        Fitted model.
    """
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        verbose=0,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Probability calibration
# ---------------------------------------------------------------------------

class _CalibratedWrapper:
    """Lightweight wrapper that applies a fitted calibrator to a base model."""

    def __init__(self, base_model, calibrator, method: str):
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method
        self.classes_ = base_model.classes_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base_model.predict_proba(X)[:, 1]
        if self.method == "sigmoid":
            cal = self.calibrator.predict_proba(raw.reshape(-1, 1))
            return cal  # shape (n, 2): [P(0), P(1)]
        else:
            p1 = self.calibrator.predict(raw)
            return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def calibrate_model(
    model,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    method: str = "sigmoid",
) -> "_CalibratedWrapper":
    """
    Apply probability calibration to a pre-fitted model.

    Tree models (and ensemble methods) tend to produce poorly calibrated
    probabilities — they push predictions toward 0 and 1 more than the
    true probabilities warrant. This matters enormously for credit risk
    because we use predicted PD (probability of default) directly in
    profit calculations and threshold decisions.

    Platt Scaling (method='sigmoid'):
        Fits a logistic regression on the model's raw output scores.
        Works well when the calibration set is moderately sized (> 1000
        samples). Assumes that the miscalibration is monotone and sigmoid-
        shaped, which holds in practice for gradient boosting.

    Isotonic Regression (method='isotonic'):
        A non-parametric monotone step function. More flexible but prone
        to overfitting on calibration sets smaller than ~5,000 samples.
        Better choice when you have lots of calibration data.

    Implementation:
        Rather than relying on CalibratedClassifierCV (whose cv='prefit'
        option was removed in sklearn 1.4+), we implement calibration
        directly:
          - Sigmoid/Platt: fit a LogisticRegression on the model's raw
            output probabilities from the calibration set.
          - Isotonic: fit an IsotonicRegression on the same raw probs.
        The result is wrapped in a lightweight object that exposes the
        same predict_proba / predict interface as a sklearn estimator.

    Parameters
    ----------
    model : fitted estimator
        A fitted sklearn estimator with predict_proba.
    X_calib : np.ndarray
        Features for calibration (should NOT overlap with test set).
    y_calib : np.ndarray
        True labels for calibration set.
    method : str
        'sigmoid' (Platt) or 'isotonic'.

    Returns
    -------
    _CalibratedWrapper
        Fitted wrapper with .predict_proba() returning well-calibrated
        probabilities.
    """
    raw = model.predict_proba(X_calib)[:, 1]

    if method == "sigmoid":
        calibrator = LogisticRegression(C=1.0, solver="lbfgs")
        calibrator.fit(raw.reshape(-1, 1), y_calib)
    else:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw, y_calib)

    return _CalibratedWrapper(model, calibrator, method)


# ---------------------------------------------------------------------------
# Cross-validation utility
# ---------------------------------------------------------------------------

def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: list | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run stratified k-fold cross-validation and return per-fold results.

    Uses StratifiedKFold to maintain the class ratio in each fold —
    important for imbalanced datasets where a random split might produce
    folds with very different default rates.

    Parameters
    ----------
    model : sklearn estimator (unfitted)
        Will be cloned for each fold.
    X : np.ndarray
    y : np.ndarray
    cv : int
        Number of folds.
    scoring : list of str
        Sklearn scoring strings. Default: ROC-AUC and Average Precision.
    random_state : int

    Returns
    -------
    pd.DataFrame
        Columns: fold, roc_auc (or other metrics), fit_time, score_time.
        Last row shows mean ± std across folds.
    """
    if scoring is None:
        scoring = ["roc_auc", "average_precision"]

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    rows = []
    for fold_idx in range(cv):
        row = {"fold": fold_idx + 1}
        for metric in scoring:
            key = f"test_{metric}"
            row[metric] = cv_results[key][fold_idx]
        rows.append(row)

    df_cv = pd.DataFrame(rows)

    # Append summary row
    summary = {"fold": "mean ± std"}
    for metric in scoring:
        vals = df_cv[metric]
        summary[metric] = f"{vals.mean():.4f} ± {vals.std():.4f}"
    df_cv = pd.concat([df_cv, pd.DataFrame([summary])], ignore_index=True)

    return df_cv


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_model(model, filepath: str) -> None:
    """
    Serialise a fitted model to disk using joblib.

    joblib is preferred over pickle for scikit-learn objects because it
    handles large numpy arrays more efficiently via memory mapping.

    Parameters
    ----------
    model : fitted sklearn estimator or Pipeline
    filepath : str
        Path ending in .pkl or .joblib.
    """
    joblib.dump(model, filepath, compress=3)


def load_model(filepath: str):
    """
    Load a serialised model from disk.

    Parameters
    ----------
    filepath : str
        Path to the serialised model file.

    Returns
    -------
    Fitted sklearn estimator
    """
    return joblib.load(filepath)
