"""
data_generation.py
==================
Synthetic consumer loan dataset factory.

Uses a latent risk score approach to generate realistic, learnable loan data.
Each borrower's default probability is determined by a weighted combination
of their risk characteristics plus Gaussian noise, passed through a sigmoid.
The intercept is solved numerically so the resulting default rate matches
the requested target exactly.

Distributions are calibrated to US consumer lending patterns:
  - FICO score distribution: mean ~680, sd ~80
  - Income: log-normal with median ~$54k
  - DTI: typical consumer range 0.10 – 0.55
  - Default rate: 20% (sub-prime/near-prime portfolio mix)

Example
-------
>>> from src.data_generation import generate_loan_dataset
>>> df = generate_loan_dataset(n=50_000, random_state=42)
>>> df.shape
(50000, 16)
>>> df['default'].mean()
0.2001  # approximately 20%
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit  # numerically stable sigmoid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_loan_dataset(
    n: int = 50_000,
    default_rate: float = 0.20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic consumer loan dataset.

    Parameters
    ----------
    n : int
        Number of loan records to generate.
    default_rate : float
        Target proportion of defaults. The intercept in the latent risk
        score is solved numerically to hit this rate.
    random_state : int
        Seed for NumPy's default_rng — ensures full reproducibility.

    Returns
    -------
    pd.DataFrame
        Shape (n, 16). Columns: loan_id, borrower demographics, loan
        characteristics, credit profile, and 'default' target (0/1).
    """
    rng = np.random.default_rng(random_state)

    demo_df = _generate_demographics(n, rng)
    credit_df = _generate_credit_features(n, rng)
    base_df = pd.concat([demo_df, credit_df], axis=1)
    loan_df = _generate_loan_features(base_df, rng)

    df = pd.concat([base_df, loan_df], axis=1)
    df['default'] = _compute_default_labels(df, default_rate, rng)

    # Assign loan IDs last so the column order is clean
    df.insert(0, 'loan_id', [f"LN-{i:06d}" for i in range(1, n + 1)])

    return df.reset_index(drop=True)


def get_feature_metadata() -> dict:
    """
    Return metadata for each feature: dtype, description, and modeling role.

    The 'role' key takes one of three values:
      'identifier'  — not used in modeling (e.g., loan_id)
      'feature'     — used as a model input
      'target'      — the label we're predicting

    Returns
    -------
    dict
        Mapping feature_name -> {dtype, description, role}
    """
    return {
        "loan_id": {
            "dtype": "str",
            "description": "Unique loan identifier (LN-XXXXXX)",
            "role": "identifier",
        },
        "age": {
            "dtype": "int",
            "description": "Borrower age in years (21–75)",
            "role": "feature",
        },
        "annual_income": {
            "dtype": "float",
            "description": "Gross annual income in USD (log-normal, median ~$54k)",
            "role": "feature",
        },
        "employment_length": {
            "dtype": "int",
            "description": "Years at current employer (0–10+)",
            "role": "feature",
        },
        "home_ownership": {
            "dtype": "str",
            "description": "Housing status: RENT (45%), MORTGAGE (40%), OWN (15%)",
            "role": "feature",
        },
        "credit_score": {
            "dtype": "int",
            "description": "FICO-style credit score (300–850, mean ~680)",
            "role": "feature",
        },
        "credit_utilization": {
            "dtype": "float",
            "description": "Revolving credit utilization ratio (0–1, Beta distributed)",
            "role": "feature",
        },
        "num_delinquencies": {
            "dtype": "int",
            "description": "Number of 30+ day delinquencies in past 2 years (0–10)",
            "role": "feature",
        },
        "num_credit_lines": {
            "dtype": "int",
            "description": "Open trade lines (credit cards, auto loans, etc.)",
            "role": "feature",
        },
        "debt_to_income": {
            "dtype": "float",
            "description": "Monthly debt obligations / monthly gross income (0–0.65)",
            "role": "feature",
        },
        "months_since_last_delinq": {
            "dtype": "float",
            "description": "Months since most recent delinquency; NaN = no history (40%)",
            "role": "feature",
        },
        "loan_amount": {
            "dtype": "float",
            "description": "Requested loan amount in USD ($2,000–$35,000)",
            "role": "feature",
        },
        "loan_purpose": {
            "dtype": "str",
            "description": "Loan purpose: debt_consolidation, home_improvement, medical, car, small_business, other",
            "role": "feature",
        },
        "interest_rate": {
            "dtype": "float",
            "description": "Annual interest rate (5.5%–28%, negatively correlated with credit score)",
            "role": "feature",
        },
        "loan_term": {
            "dtype": "int",
            "description": "Loan term in months: 36 (65%) or 60 (35%)",
            "role": "feature",
        },
        "default": {
            "dtype": "int",
            "description": "Target: 1 = loan defaulted within 2 years, 0 = current/paid off",
            "role": "target",
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_demographics(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate borrower demographic features."""

    # Age: truncated normal, working-age adults
    age = rng.normal(loc=42, scale=12, size=n)
    age = np.clip(age, 21, 75).astype(int)

    # Income: log-normal so the distribution is right-skewed (realistic)
    # mu=10.85 gives median exp(10.85) ≈ $51,800
    log_income = rng.normal(loc=10.85, scale=0.60, size=n)
    annual_income = np.exp(log_income).round(2)

    # Employment length: discrete, weighted toward shorter tenures
    emp_choices = [0, 1, 2, 3, 5, 7, 10]
    emp_weights = [0.12, 0.15, 0.14, 0.13, 0.16, 0.14, 0.16]
    employment_length = rng.choice(emp_choices, size=n, p=emp_weights)

    # Home ownership
    ownership_choices = ["RENT", "MORTGAGE", "OWN"]
    ownership_weights = [0.45, 0.40, 0.15]
    home_ownership = rng.choice(ownership_choices, size=n, p=ownership_weights)

    return pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "employment_length": employment_length,
        "home_ownership": home_ownership,
    })


def _generate_credit_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate credit bureau / credit profile features."""

    # FICO-style credit score: normal, clipped to valid range
    credit_score = rng.normal(loc=680, scale=80, size=n)
    credit_score = np.clip(credit_score, 300, 850).astype(int)

    # Utilization: Beta(2, 5) gives a right-skewed distribution peaking ~0.25
    credit_utilization = rng.beta(2, 5, size=n).round(4)

    # Delinquencies: right-skewed, most borrowers have 0
    num_delinquencies = rng.negative_binomial(1, 0.70, size=n)
    num_delinquencies = np.clip(num_delinquencies, 0, 10).astype(int)

    # Open credit lines: Poisson
    num_credit_lines = rng.poisson(lam=8, size=n)
    num_credit_lines = np.clip(num_credit_lines, 1, 30).astype(int)

    # Debt-to-income: log-normal, realistic consumer range
    log_dti = rng.normal(loc=-1.30, scale=0.50, size=n)  # median ~0.27
    debt_to_income = np.clip(np.exp(log_dti), 0.02, 0.65).round(4)

    # Months since last delinquency: 40% borrowers have no delinquency history
    has_delinq_history = rng.random(size=n) > 0.40
    months_since_last_delinq = np.where(
        has_delinq_history,
        rng.uniform(1, 84, size=n).round(1),
        np.nan,
    )

    return pd.DataFrame({
        "credit_score": credit_score,
        "credit_utilization": credit_utilization,
        "num_delinquencies": num_delinquencies,
        "num_credit_lines": num_credit_lines,
        "debt_to_income": debt_to_income,
        "months_since_last_delinq": months_since_last_delinq,
    })


def _generate_loan_features(
    df_base: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate loan-specific features. Interest rate is negatively correlated
    with credit score to reflect real underwriting pricing logic.
    """
    n = len(df_base)

    # Loan amount: uniform across typical personal loan range, rounded
    loan_amount = rng.uniform(2_000, 35_000, size=n)
    loan_amount = (np.round(loan_amount / 100) * 100).round(2)

    # Loan purpose: weighted toward debt consolidation (most common)
    purpose_choices = [
        "debt_consolidation",
        "home_improvement",
        "medical",
        "car",
        "small_business",
        "other",
    ]
    purpose_weights = [0.42, 0.15, 0.10, 0.08, 0.12, 0.13]
    loan_purpose = rng.choice(purpose_choices, size=n, p=purpose_weights)

    # Interest rate: base rate + risk premium inversely linked to credit score
    # Higher credit score → lower rate. Clipped to real-world range.
    credit_score_normalized = (df_base["credit_score"] - 680) / 80
    base_rate = 0.13
    risk_premium = -0.045 * credit_score_normalized
    noise = rng.normal(0, 0.02, size=n)
    interest_rate = np.clip(base_rate + risk_premium + noise, 0.055, 0.28).round(4)

    # Loan term: 36 or 60 months (longer term → higher risk)
    loan_term = rng.choice([36, 60], size=n, p=[0.65, 0.35]).astype(int)

    return pd.DataFrame({
        "loan_amount": loan_amount,
        "loan_purpose": loan_purpose,
        "interest_rate": interest_rate,
        "loan_term": loan_term,
    })


def _compute_default_labels(
    df: pd.DataFrame,
    default_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Assign default labels using a latent risk score model.

    The latent score is a linear combination of normalized risk features.
    A sigmoid maps the score to P(default). An intercept is solved
    numerically (via brentq) so that mean(P(default)) ≈ target default_rate.
    Final labels are Bernoulli draws from each loan's P(default).

    The coefficient signs and magnitudes reflect real credit risk factor
    importance documented in academic and industry literature.
    """
    # Normalize key features to put coefficients on comparable scales
    credit_score_std = (df["credit_score"] - 680) / 80
    income_log = np.log1p(df["annual_income"])
    income_std = (income_log - income_log.mean()) / income_log.std()
    dti_std = (df["debt_to_income"] - 0.27) / 0.12
    utilization = df["credit_utilization"]
    delinq = df["num_delinquencies"]
    emp_std = (df["employment_length"] - 4) / 3
    loan_amount_std = (df["loan_amount"] - 18_500) / 9_000

    # Latent risk score: positive = higher default risk
    latent = (
        -1.60 * credit_score_std        # lower score → higher risk
        +  1.80 * utilization            # higher utilization → higher risk
        +  0.70 * delinq                 # more past delinquencies → higher risk
        +  1.40 * dti_std                # higher DTI → higher risk
        -  0.80 * income_std             # higher income → lower risk
        -  0.40 * emp_std                # longer employment → lower risk
        +  0.30 * loan_amount_std        # larger loan → slightly higher risk
        +  0.25 * (df["loan_term"] == 60).astype(float)  # 60-month → higher risk
        +  0.20 * (df["loan_purpose"] == "small_business").astype(float)
        +  0.10 * (df["loan_purpose"] == "medical").astype(float)
        +  rng.normal(0, 0.8, size=len(df))  # idiosyncratic noise
    )

    # Find intercept c such that mean(sigmoid(latent + c)) == default_rate
    def objective(c):
        return expit(latent + c).mean() - default_rate

    intercept = brentq(objective, a=-10.0, b=10.0, xtol=1e-6)
    p_default = expit(latent + intercept)

    # Bernoulli draw for each loan
    default_labels = (rng.random(size=len(df)) < p_default).astype(int)
    return default_labels
