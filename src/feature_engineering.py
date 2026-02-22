"""
feature_engineering.py
=======================
All feature transformations for the loan default risk model.

Design pattern: stateless transforms that separate *fitting* from *applying*.
The master function `engineer_features` accepts a `fit` flag:
  - fit=True  → compute and store transform parameters (bin edges, fill values,
                 encoder maps) from the training set, return them as `encoder_state`
  - fit=False → apply pre-fitted parameters from a prior `encoder_state` to the
                test set, ensuring no data leakage across the train/test boundary

This pattern mirrors scikit-learn's fit/transform paradigm but operates
directly on DataFrames, making the code easy to audit and debug.

Engineered features summary
---------------------------
Log transforms:
  log_annual_income, log_loan_amount, log_dti

Interaction terms:
  loan_to_income_ratio          — loan_amount / annual_income
  monthly_payment_est           — estimated PMT($)
  payment_to_income             — monthly_payment_est / monthly_income
  utilization_x_delinq          — credit_utilization * num_delinquencies
  credit_score_x_utilization    — credit_score * credit_utilization (captures
                                   synergy: high utilization matters more for
                                   lower-score borrowers)

Missing value handling:
  has_delinquency_history       — 1 if months_since_last_delinq was not NaN
  months_since_last_delinq      — NaN filled to 0 after flag is created

Risk tier bins (ordinal encoded 0, 1, 2, 3, 4):
  credit_score_bin              — Poor / Fair / Good / Very Good / Exceptional
  dti_bin                       — Low / Moderate / Elevated / High
  age_bin                       — Young / Mid / Established / Senior

One-hot encoded (drop_first=True to avoid multicollinearity for linear models):
  home_ownership                — MORTGAGE / OWN (RENT is the dropped baseline)
  loan_purpose                  — home_improvement / medical / car /
                                   small_business / other
                                   (debt_consolidation is the dropped baseline)

Example
-------
>>> from src.data_generation import generate_loan_dataset
>>> from src.feature_engineering import engineer_features, get_feature_columns
>>> df = generate_loan_dataset(n=1000, random_state=0)
>>> df_train, df_test = df.iloc[:800], df.iloc[800:]
>>> df_fe_train, state = engineer_features(df_train, fit=True)
>>> df_fe_test, _      = engineer_features(df_test,  fit=False, encoder_state=state)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    fit: bool = True,
    encoder_state: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Master feature engineering orchestrator.

    Applies all transformations in the correct order and returns the
    model-ready DataFrame plus the encoder_state needed to reproduce
    identical transforms on new data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw loan DataFrame (output of generate_loan_dataset).
    fit : bool
        True → fit all transformers on df (use for training set).
        False → apply pre-fitted state to df (use for test/scoring).
    encoder_state : dict or None
        Required when fit=False. Pass the dict returned by a prior
        fit=True call.

    Returns
    -------
    df_out : pd.DataFrame
        Fully engineered, model-ready DataFrame. Includes all engineered
        columns; 'loan_id' and 'default' are dropped.
    encoder_state : dict
        Fitted state (bin edges, fill values, encoder maps, feature lists).
        Serialisable with joblib or pickle alongside the model.
    """
    if not fit and encoder_state is None:
        raise ValueError(
            "encoder_state must be provided when fit=False. "
            "Run engineer_features(..., fit=True) on the training set first."
        )

    state = encoder_state.copy() if encoder_state else {}
    df = df.copy()

    # 1. Log transforms (no fitting required — deterministic)
    df = add_log_transforms(df)

    # 2. Interaction terms (also deterministic)
    df = add_interaction_terms(df)

    # 3. Missing value handling
    df, _fill_vals = impute_missing_values(df, fill_values=state.get("fill_values"), fit=fit)
    if fit:
        state["fill_values"] = _fill_vals

    # 4. Risk-tier binning
    df, bin_edges = add_binned_features(
        df, bin_edges=state.get("bin_edges"), fit=fit
    )
    if fit:
        state["bin_edges"] = bin_edges

    # 5. Categorical encoding
    df, encoder_map = encode_categoricals(
        df, encoder_map=state.get("encoder_map"), fit=fit
    )
    if fit:
        state["encoder_map"] = encoder_map

    # 6. Drop columns not used in modeling
    drop_cols = ["loan_id", "default"] + [
        c for c in ["home_ownership", "loan_purpose"] if c in df.columns
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 7. Record final feature list
    state["feature_columns"] = df.columns.tolist()

    return df, state


def get_feature_columns() -> dict:
    """
    Return a catalogue of feature column groups for reference.

    Note: This returns the *intended* column sets. The actual column list
    after engineer_features() may vary slightly depending on which
    categories are present in the data. Use encoder_state['feature_columns']
    for the definitive list after fitting.
    """
    return {
        "log_features": [
            "log_annual_income",
            "log_loan_amount",
            "log_dti",
        ],
        "interaction_features": [
            "loan_to_income_ratio",
            "monthly_payment_est",
            "payment_to_income",
            "utilization_x_delinq",
            "credit_score_x_utilization",
        ],
        "imputed_features": [
            "has_delinquency_history",
            "months_since_last_delinq",
        ],
        "bin_features": [
            "credit_score_bin",
            "dti_bin",
            "age_bin",
        ],
        "one_hot_features": [
            "home_ownership_MORTGAGE",
            "home_ownership_OWN",
            "loan_purpose_car",
            "loan_purpose_home_improvement",
            "loan_purpose_medical",
            "loan_purpose_other",
            "loan_purpose_small_business",
        ],
        "passthrough_numeric": [
            "age",
            "annual_income",
            "employment_length",
            "credit_score",
            "credit_utilization",
            "num_delinquencies",
            "num_credit_lines",
            "debt_to_income",
            "loan_amount",
            "interest_rate",
            "loan_term",
        ],
    }


# ---------------------------------------------------------------------------
# Individual transform functions
# ---------------------------------------------------------------------------

def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p to right-skewed financial variables.

    log1p (= log(1 + x)) is used instead of log(x) because it handles
    zero values gracefully and produces the same monotonic transformation
    for positive values.

    New columns added: log_annual_income, log_loan_amount, log_dti
    """
    df = df.copy()
    df["log_annual_income"] = np.log1p(df["annual_income"])
    df["log_loan_amount"] = np.log1p(df["loan_amount"])
    df["log_dti"] = np.log1p(df["debt_to_income"])
    return df


def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features that capture combined risk signals.

    loan_to_income_ratio:
        How large is the loan relative to borrower income? High values
        indicate the borrower is stretching to take a large loan.

    monthly_payment_est:
        Estimated fixed monthly payment using the PMT formula. This is
        a more useful signal than loan_amount alone because it accounts
        for both the size of the loan and the interest rate.

    payment_to_income:
        The estimated monthly payment as a fraction of monthly gross
        income. This is the loan-specific debt service ratio —
        complementary to the aggregate DTI ratio.

    utilization_x_delinq:
        Product of credit utilization and number of delinquencies.
        A borrower with both high utilization AND recent delinquencies
        is substantially riskier than either factor alone.

    credit_score_x_utilization:
        Product term. High utilization is worse for borrowers who also
        have low credit scores — this feature captures that interaction
        which a linear model cannot represent without it.
    """
    df = df.copy()

    # Loan-to-income: guard against zero income (shouldn't happen but defensive)
    df["loan_to_income_ratio"] = df["loan_amount"] / df["annual_income"].clip(lower=1)

    # Monthly payment estimate using PMT formula
    monthly_rate = df["interest_rate"] / 12
    term = df["loan_term"]
    principal = df["loan_amount"]

    # PMT = P * (r * (1+r)^n) / ((1+r)^n - 1)
    # Edge case: if interest_rate is 0, payment = principal / term
    factor = (1 + monthly_rate) ** term
    pmt = np.where(
        monthly_rate > 0,
        principal * (monthly_rate * factor) / (factor - 1),
        principal / term,
    )
    df["monthly_payment_est"] = pmt.round(2)

    # Payment-to-income ratio
    monthly_income = df["annual_income"] / 12
    df["payment_to_income"] = df["monthly_payment_est"] / monthly_income.clip(lower=1)

    # Interaction terms
    df["utilization_x_delinq"] = df["credit_utilization"] * df["num_delinquencies"]
    df["credit_score_x_utilization"] = df["credit_score"] * df["credit_utilization"]

    return df


def impute_missing_values(
    df: pd.DataFrame,
    fill_values: dict | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Handle missing values in months_since_last_delinq.

    Strategy:
    1. Create a binary flag: has_delinquency_history = 1 if the field
       was non-null (borrower has at least one past delinquency on record).
    2. Fill NaN → 0 (meaning "no recorded delinquency, treating recency as 0").

    This preserves the missingness signal — a borrower with NaN is different
    from one with months_since_last_delinq = 0 (very recent delinquency).

    Parameters
    ----------
    fill_values : dict or None
        Pre-fitted fill values. Only used for consistency (here the fill
        is always 0, but the pattern is maintained for extensibility).
    fit : bool
        If True, compute and return fill_values. If False, use provided ones.
    """
    df = df.copy()

    if "months_since_last_delinq" in df.columns:
        # Flag must be created BEFORE filling NaN
        df["has_delinquency_history"] = df["months_since_last_delinq"].notna().astype(int)
        df["months_since_last_delinq"] = df["months_since_last_delinq"].fillna(0)

    if fit:
        fill_values = {"months_since_last_delinq": 0}

    return df, fill_values or {}


def add_binned_features(
    df: pd.DataFrame,
    bin_edges: dict | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Bin continuous features into ordinal risk tiers.

    Binning is useful for:
    - Capturing non-linear relationships in Logistic Regression
    - Making the model less sensitive to outliers at bin boundaries
    - Producing human-readable risk categories for reporting

    Credit score tiers follow FICO's standard classification scheme.
    DTI and age tiers are based on lending industry conventions.

    The ordinal encoding assigns integers (0, 1, 2, ...) so that the
    ordering is preserved for models that can use it.

    Parameters
    ----------
    bin_edges : dict or None
        Pre-fitted bin edges from a prior fit=True call.
    fit : bool
        If True, define bin edges from df (they're hardcoded, not data-driven,
        so this is mainly for API consistency). If False, use provided edges.
    """
    df = df.copy()

    if fit:
        bin_edges = {
            "credit_score": [300, 580, 670, 740, 800, 851],
            "dti": [0.0, 0.15, 0.30, 0.45, 0.66],
            "age": [20, 30, 41, 56, 76],
        }

    # Credit score bins: Poor=0, Fair=1, Good=2, Very Good=3, Exceptional=4
    cs_edges = bin_edges["credit_score"]
    cs_labels = [0, 1, 2, 3, 4]
    df["credit_score_bin"] = pd.cut(
        df["credit_score"],
        bins=cs_edges,
        labels=cs_labels,
        include_lowest=True,
        right=True,
    ).astype(float).fillna(0).astype(int)

    # DTI bins: Low=0, Moderate=1, Elevated=2, High=3
    dti_edges = bin_edges["dti"]
    dti_labels = [0, 1, 2, 3]
    df["dti_bin"] = pd.cut(
        df["debt_to_income"],
        bins=dti_edges,
        labels=dti_labels,
        include_lowest=True,
        right=True,
    ).astype(float).fillna(0).astype(int)

    # Age bins: Young=0, Mid=1, Established=2, Senior=3
    age_edges = bin_edges["age"]
    age_labels = [0, 1, 2, 3]
    df["age_bin"] = pd.cut(
        df["age"],
        bins=age_edges,
        labels=age_labels,
        include_lowest=True,
        right=True,
    ).astype(float).fillna(0).astype(int)

    return df, bin_edges


def encode_categoricals(
    df: pd.DataFrame,
    encoder_map: dict | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    One-hot encode nominal categorical columns.

    drop_first=True drops one dummy per variable to avoid perfect
    multicollinearity in Logistic Regression (the "dummy variable trap").
    The dropped category serves as the reference/baseline level.

      home_ownership baseline: RENT
      loan_purpose baseline: debt_consolidation

    Tree models are not affected by multicollinearity, but the same
    encoding is used for both models to keep the feature space consistent.

    Parameters
    ----------
    encoder_map : dict or None
        Pre-fitted mapping {column: [dummy_column_names_in_order]}
        Used during test-set transform to ensure identical column sets.
    fit : bool
        If True, fit encoder on df and return encoder_map.
        If False, apply encoder_map from prior fit.
    """
    df = df.copy()
    categorical_cols = ["home_ownership", "loan_purpose"]

    if fit:
        encoder_map = {}
        for col in categorical_cols:
            if col not in df.columns:
                continue
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
            encoder_map[col] = dummies.columns.tolist()
            df = pd.concat([df, dummies], axis=1)
    else:
        # Apply pre-fitted encoding: ensure exact same columns appear in test set
        for col, dummy_cols in encoder_map.items():
            if col not in df.columns:
                for dc in dummy_cols:
                    df[dc] = 0
                continue
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
            for dc in dummy_cols:
                if dc not in dummies.columns:
                    df[dc] = 0
                else:
                    df[dc] = dummies[dc]

    return df, encoder_map or {}
