"""
business_simulation.py
======================
Loan approval threshold analysis and expected profit/loss calculations.

This module answers the most important question in applied credit risk modeling:
"At what predicted default probability threshold should we stop approving loans
in order to maximise portfolio profitability?"

The standard ML approach (maximise F1, use threshold=0.5) ignores the asymmetric
economic costs of credit decisions. This module replaces that with a proper
economic optimisation framework.

Profit model
------------
For each approved loan, expected profit is:

    monthly_payment = PMT(annual_rate/12, term_months, loan_amount)
    total_interest   = monthly_payment × term_months − loan_amount
    avg_balance      = loan_amount / 2                    (linear amortisation)
    funding_cost     = avg_balance × funding_cost_rate × (term_months / 12)
    origination_cost = fixed cost per loan (~underwriting, servicing setup)

    revenue_if_good  = total_interest − origination_cost − funding_cost
    loss_if_default  = loan_amount × (1 − recovery_rate)
                       + origination_cost + funding_cost

    expected_profit  = (1 − PD) × revenue_if_good − PD × loss_if_default

where PD is the model's calibrated predicted default probability.

Key economic parameters (defaults reflect US consumer lending)
---------------------------------------------------------------
RECOVERY_RATE    = 0.10   # 10% of principal recovered post-charge-off (unsecured)
ORIGINATION_COST = 200    # USD per loan (fixed: underwriting, KYC, setup)
FUNDING_COST_RATE = 0.04  # Annual cost of capital: 4%

Break-even PD
-------------
Setting expected_profit = 0 and solving for PD gives the break-even probability:

    PD_breakeven = revenue_if_good / (revenue_if_good + loss_if_default)

Loans with predicted PD above PD_breakeven are expected to lose money.
The optimal approval threshold must be ≤ PD_breakeven for any loan to be worth approving.

Threshold direction
-------------------
We APPROVE a loan when: predicted_PD ≤ threshold
We DECLINE a loan when: predicted_PD > threshold

This is the opposite of a standard classifier (where you classify as positive
when score > threshold) because here 1 = default (bad), and we approve the
ones predicted to be low-risk.

Example
-------
>>> from src.business_simulation import threshold_sweep, find_optimal_threshold
>>> sweep_df = threshold_sweep(y_test, y_prob_cal, loan_amounts, interest_rates, loan_terms)
>>> opt_thresh, opt_row = find_optimal_threshold(sweep_df)
>>> print(f"Optimal threshold: {opt_thresh:.2f}")
>>> print(f"Expected portfolio profit: ${opt_row['total_expected_profit']:,.0f}")
"""

import os

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Module-level economic constants
# ---------------------------------------------------------------------------

RECOVERY_RATE = 0.10        # Fraction of principal recovered after default
ORIGINATION_COST = 200.0    # USD — fixed cost per approved loan
FUNDING_COST_RATE = 0.04    # Annual cost of capital as a fraction of loan amount


# ---------------------------------------------------------------------------
# Per-loan profit calculation
# ---------------------------------------------------------------------------

def compute_expected_profit_per_loan(
    loan_amount: float,
    interest_rate: float,
    loan_term_months: int,
    default_probability: float,
    recovery_rate: float = RECOVERY_RATE,
    origination_cost: float = ORIGINATION_COST,
    funding_cost_rate: float = FUNDING_COST_RATE,
) -> float:
    """
    Compute the expected profit (or loss) for a single loan.

    This function implements the full profit model described in the module
    docstring. It uses the model's calibrated predicted default probability
    (PD) rather than the actual label, making it suitable for prospective
    approval decisions on new loan applications.

    Parameters
    ----------
    loan_amount : float
        Loan principal in USD.
    interest_rate : float
        Annual interest rate as a decimal (e.g., 0.13 for 13%).
    loan_term_months : int
        Loan term in months (36 or 60).
    default_probability : float
        Predicted probability of default [0, 1] from calibrated model.
    recovery_rate : float
        Fraction of principal recovered after charge-off.
    origination_cost : float
        Fixed cost per loan in USD.
    funding_cost_rate : float
        Annual cost of capital as a fraction.

    Returns
    -------
    float
        Expected profit in USD. Negative values indicate expected loss.
    """
    monthly_rate = interest_rate / 12.0

    # Monthly payment via PMT formula
    if monthly_rate > 0:
        factor = (1 + monthly_rate) ** loan_term_months
        monthly_payment = loan_amount * (monthly_rate * factor) / (factor - 1)
    else:
        monthly_payment = loan_amount / loan_term_months

    # Total interest revenue over the life of the loan
    total_interest = monthly_payment * loan_term_months - loan_amount

    # Funding cost: applied to average outstanding balance
    avg_outstanding_balance = loan_amount / 2.0
    funding_cost = avg_outstanding_balance * funding_cost_rate * (loan_term_months / 12.0)

    # Net revenue if the loan performs (no default)
    revenue_if_good = total_interest - origination_cost - funding_cost

    # Net loss if the loan defaults (assume default at midpoint)
    loss_if_default = (
        loan_amount * (1 - recovery_rate)   # principal not recovered
        + origination_cost                   # sunk cost
        + funding_cost                       # sunk funding cost
    )

    # Expected value
    expected_profit = (
        (1 - default_probability) * revenue_if_good
        - default_probability * loss_if_default
    )

    return float(expected_profit)


def compute_breakeven_pd(
    loan_amount: float,
    interest_rate: float,
    loan_term_months: int,
    recovery_rate: float = RECOVERY_RATE,
    origination_cost: float = ORIGINATION_COST,
    funding_cost_rate: float = FUNDING_COST_RATE,
) -> float:
    """
    Compute the break-even default probability for a loan.

    At PD = PD_breakeven, expected_profit = 0.
    Above this PD, the loan is expected to lose money regardless of threshold.

    PD_breakeven = revenue_if_good / (revenue_if_good + loss_if_default)
    """
    monthly_rate = interest_rate / 12.0
    if monthly_rate > 0:
        factor = (1 + monthly_rate) ** loan_term_months
        monthly_payment = loan_amount * (monthly_rate * factor) / (factor - 1)
    else:
        monthly_payment = loan_amount / loan_term_months

    total_interest = monthly_payment * loan_term_months - loan_amount
    avg_outstanding_balance = loan_amount / 2.0
    funding_cost = avg_outstanding_balance * funding_cost_rate * (loan_term_months / 12.0)

    revenue_if_good = total_interest - origination_cost - funding_cost
    loss_if_default = loan_amount * (1 - recovery_rate) + origination_cost + funding_cost

    if (revenue_if_good + loss_if_default) == 0:
        return 0.5

    return max(0.0, min(1.0, revenue_if_good / (revenue_if_good + loss_if_default)))


# ---------------------------------------------------------------------------
# Threshold sweep engine
# ---------------------------------------------------------------------------

def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    loan_amounts: np.ndarray,
    interest_rates: np.ndarray,
    loan_terms: np.ndarray,
    thresholds: np.ndarray | None = None,
    recovery_rate: float = RECOVERY_RATE,
    origination_cost: float = ORIGINATION_COST,
    funding_cost_rate: float = FUNDING_COST_RATE,
) -> pd.DataFrame:
    """
    Evaluate lending portfolio economics across a range of approval thresholds.

    For each threshold t, we approve all loans where predicted_PD ≤ t.
    At each threshold we compute:
    - Volume metrics: n_approved, approval_rate
    - Credit quality: actual default rate in the approved pool
    - Economic metrics: total_expected_profit, avg_profit_per_loan
    - ML metrics: precision, recall, f1 (treating default prediction as positive class)

    This sweep lets a risk manager see the full trade-off curve between
    loan volume (revenue potential) and credit quality (default risk).

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        True binary labels (1 = default, 0 = performing).
    y_prob : np.ndarray, shape (n,)
        Calibrated predicted default probabilities.
    loan_amounts : np.ndarray, shape (n,)
        Loan principal amounts in USD.
    interest_rates : np.ndarray, shape (n,)
        Annual interest rates as decimals.
    loan_terms : np.ndarray, shape (n,)
        Loan terms in months.
    thresholds : np.ndarray, optional
        Thresholds to evaluate. Default: np.arange(0.05, 0.80, 0.01).
    recovery_rate, origination_cost, funding_cost_rate : float
        Economic parameters passed to compute_expected_profit_per_loan.

    Returns
    -------
    pd.DataFrame
        One row per threshold. Columns:
        threshold, n_approved, approval_rate, default_rate_approved,
        total_expected_profit, avg_profit_per_loan, total_loan_book,
        return_on_portfolio, precision, recall, f1
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.80, 0.01).round(2)

    n_total = len(y_true)
    rows = []

    for t in thresholds:
        approved_mask = y_prob <= t
        n_approved = int(approved_mask.sum())

        if n_approved == 0:
            continue

        # Credit quality in approved pool
        defaults_in_approved = int(y_true[approved_mask].sum())
        default_rate_approved = defaults_in_approved / n_approved

        # Expected profit using model PD (not actual labels)
        profits = np.array([
            compute_expected_profit_per_loan(
                loan_amount=float(loan_amounts[i]),
                interest_rate=float(interest_rates[i]),
                loan_term_months=int(loan_terms[i]),
                default_probability=float(y_prob[i]),
                recovery_rate=recovery_rate,
                origination_cost=origination_cost,
                funding_cost_rate=funding_cost_rate,
            )
            for i in np.where(approved_mask)[0]
        ])

        total_profit = float(profits.sum())
        avg_profit = float(profits.mean())
        total_loan_book = float(loan_amounts[approved_mask].sum())
        rop = total_profit / total_loan_book if total_loan_book > 0 else 0.0

        # ML classification metrics (treating default=1 as positive class)
        # Approved = predicted non-default (predicted class 0)
        # At threshold t: classify as default (1) if y_prob > t
        y_pred = (y_prob > t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        rows.append({
            "threshold": round(float(t), 2),
            "n_approved": n_approved,
            "approval_rate": round(n_approved / n_total, 4),
            "default_rate_approved": round(default_rate_approved, 4),
            "total_expected_profit": round(total_profit, 2),
            "avg_profit_per_loan": round(avg_profit, 2),
            "total_loan_book": round(total_loan_book, 2),
            "return_on_portfolio": round(rop, 6),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Optimal threshold selection
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    sweep_df: pd.DataFrame,
    objective: str = "total_profit",
) -> tuple[float, pd.Series]:
    """
    Find the decision threshold that optimises a given business objective.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Output of threshold_sweep().
    objective : str
        One of:
        - 'total_profit': maximise total_expected_profit
        - 'avg_profit': maximise avg_profit_per_loan
        - 'f1': maximise F1 score
        - 'return_on_portfolio': maximise return_on_portfolio

    Returns
    -------
    optimal_threshold : float
    optimal_row : pd.Series
        The full row from sweep_df at the optimal threshold.
    """
    objective_map = {
        "total_profit": "total_expected_profit",
        "avg_profit": "avg_profit_per_loan",
        "f1": "f1",
        "return_on_portfolio": "return_on_portfolio",
    }

    if objective not in objective_map:
        raise ValueError(f"objective must be one of {list(objective_map.keys())}")

    col = objective_map[objective]
    best_idx = sweep_df[col].idxmax()
    best_row = sweep_df.loc[best_idx]

    return float(best_row["threshold"]), best_row


# ---------------------------------------------------------------------------
# Portfolio summary
# ---------------------------------------------------------------------------

def compute_portfolio_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    loan_amounts: np.ndarray,
    interest_rates: np.ndarray,
    loan_terms: np.ndarray,
    threshold: float,
    recovery_rate: float = RECOVERY_RATE,
    origination_cost: float = ORIGINATION_COST,
    funding_cost_rate: float = FUNDING_COST_RATE,
) -> dict:
    """
    Compute a full business summary for a given approval threshold.

    Suitable for presenting to a credit committee or risk officer:
    breaks down the economics of the approved portfolio into revenue,
    losses, and net profit in dollar terms.

    Parameters
    ----------
    threshold : float
        The maximum predicted PD at which loans are approved.

    Returns
    -------
    dict with keys:
        n_evaluated, n_approved, n_declined,
        approval_rate, decline_rate,
        n_true_defaults_approved, n_true_defaults_declined,
        false_approval_rate (FPR for defaults),
        total_loan_book_value, expected_total_profit,
        expected_revenue_from_good_loans, expected_loss_from_defaults,
        return_on_portfolio
    """
    n_total = len(y_true)
    approved_mask = y_prob <= threshold
    declined_mask = ~approved_mask

    n_approved = int(approved_mask.sum())
    n_declined = int(declined_mask.sum())

    defaults_in_approved = int(y_true[approved_mask].sum())
    defaults_in_declined = int(y_true[declined_mask].sum())

    # Per-loan P&L split
    good_profits = []
    default_losses = []

    for i in np.where(approved_mask)[0]:
        ep = compute_expected_profit_per_loan(
            loan_amount=float(loan_amounts[i]),
            interest_rate=float(interest_rates[i]),
            loan_term_months=int(loan_terms[i]),
            default_probability=float(y_prob[i]),
            recovery_rate=recovery_rate,
            origination_cost=origination_cost,
            funding_cost_rate=funding_cost_rate,
        )
        pd_i = float(y_prob[i])
        # Decompose into expected components
        monthly_rate = float(interest_rates[i]) / 12
        term = int(loan_terms[i])
        principal = float(loan_amounts[i])

        if monthly_rate > 0:
            factor = (1 + monthly_rate) ** term
            mp = principal * (monthly_rate * factor) / (factor - 1)
        else:
            mp = principal / term

        total_int = mp * term - principal
        avg_bal = principal / 2
        fcost = avg_bal * funding_cost_rate * (term / 12)
        rev_good = total_int - origination_cost - fcost
        loss_def = principal * (1 - recovery_rate) + origination_cost + fcost

        good_profits.append((1 - pd_i) * rev_good)
        default_losses.append(pd_i * loss_def)

    total_loan_book = float(loan_amounts[approved_mask].sum())
    expected_revenue = float(sum(good_profits))
    expected_losses = float(sum(default_losses))
    expected_profit = expected_revenue - expected_losses
    rop = expected_profit / total_loan_book if total_loan_book > 0 else 0.0

    n_defaults_total = int(y_true.sum())
    false_approval_rate = defaults_in_approved / max(n_defaults_total, 1)

    return {
        "n_evaluated": n_total,
        "n_approved": n_approved,
        "n_declined": n_declined,
        "approval_rate": round(n_approved / n_total, 4),
        "decline_rate": round(n_declined / n_total, 4),
        "n_true_defaults_approved": defaults_in_approved,
        "n_true_defaults_declined": defaults_in_declined,
        "default_rate_in_approved": round(defaults_in_approved / max(n_approved, 1), 4),
        "false_approval_rate": round(false_approval_rate, 4),
        "total_loan_book_value": round(total_loan_book, 2),
        "expected_total_profit": round(expected_profit, 2),
        "expected_revenue_from_good_loans": round(expected_revenue, 2),
        "expected_loss_from_defaults": round(expected_losses, 2),
        "return_on_portfolio": round(rop, 6),
        "approval_threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Business simulation plots
# ---------------------------------------------------------------------------

def plot_threshold_vs_approval_rate(
    sweep_df: pd.DataFrame,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
    figsize: tuple = (10, 6),
) -> plt.Axes:
    """
    Dual-axis plot: approval rate (left) and default rate in approved pool (right).

    This chart is used by risk management to understand the volume-risk
    trade-off at each possible threshold. A lender might have a regulatory
    constraint on minimum approval rate, or an internal constraint on
    maximum portfolio default rate.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax2 = ax.twinx()

    color_approval = "#2196F3"
    color_default = "#FF5722"

    ax.plot(
        sweep_df["threshold"],
        sweep_df["approval_rate"] * 100,
        color=color_approval,
        lw=2.5,
        label="Approval Rate (%)",
    )
    ax2.plot(
        sweep_df["threshold"],
        sweep_df["default_rate_approved"] * 100,
        color=color_default,
        lw=2.5,
        linestyle="--",
        label="Default Rate in Approved Pool (%)",
    )

    ax.set_xlabel("Approval Threshold (Max Tolerable PD)", fontsize=12)
    ax.set_ylabel("Approval Rate (%)", color=color_approval, fontsize=12)
    ax.tick_params(axis="y", labelcolor=color_approval)
    ax2.set_ylabel("Default Rate in Approved Pool (%)", color=color_default, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_default)

    ax.set_title("Volume vs. Credit Quality Trade-off by Threshold", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_threshold_vs_profit(
    sweep_df: pd.DataFrame,
    optimal_threshold: float | None = None,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
    figsize: tuple = (10, 6),
) -> plt.Axes:
    """
    Plot expected portfolio profit as a function of the approval threshold.

    The profit curve is typically concave — too restrictive (high threshold)
    and we reject profitable good loans; too permissive (low threshold) and
    defaults erode the portfolio. The peak of this curve is the optimal threshold.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    profit_millions = sweep_df["total_expected_profit"] / 1e6

    ax.plot(
        sweep_df["threshold"],
        profit_millions,
        color="#4CAF50",
        lw=2.5,
        label="Expected Portfolio Profit",
    )
    ax.fill_between(
        sweep_df["threshold"],
        profit_millions,
        0,
        where=(profit_millions > 0),
        alpha=0.15,
        color="#4CAF50",
    )
    ax.fill_between(
        sweep_df["threshold"],
        profit_millions,
        0,
        where=(profit_millions <= 0),
        alpha=0.15,
        color="#F44336",
    )
    ax.axhline(y=0, color="black", lw=1.0, linestyle="-", alpha=0.5)

    if optimal_threshold is not None:
        opt_row = sweep_df[sweep_df["threshold"] == optimal_threshold]
        if not opt_row.empty:
            opt_profit = float(opt_row["total_expected_profit"].values[0]) / 1e6
            ax.axvline(x=optimal_threshold, color="#FF9800", lw=2.0, linestyle="--", alpha=0.9)
            ax.annotate(
                f"Optimal threshold\nPD ≤ {optimal_threshold:.2f}\nProfit: ${opt_profit:.2f}M",
                xy=(optimal_threshold, opt_profit),
                xytext=(optimal_threshold + 0.04, opt_profit * 0.85),
                fontsize=9,
                color="#E65100",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#E65100"),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0", edgecolor="#FF9800"),
            )

    ax.set_xlabel("Approval Threshold (Max Tolerable PD)", fontsize=12)
    ax.set_ylabel("Expected Portfolio Profit ($M)", fontsize=12)
    ax.set_title("Expected Portfolio Profit by Approval Threshold", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax
