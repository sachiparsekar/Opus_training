"""
data_processor.py
Takes the standardised DataFrame produced by file_parser.py and:
  1. Enriches it with derived columns used across the app
  2. Computes summary metrics, monthly aggregates, category summary
  3. Returns a dict that app.py unpacks directly
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def process_data(df: pd.DataFrame) -> dict:
    """
    Enrich the canonical DataFrame and return a dict that app.py unpacks:
        {
            "df":               enriched DataFrame,
            "metrics":          summary metrics dict,
            "monthly":          monthly income/expense DataFrame,
            "category_summary": per-category expense DataFrame,
        }
    """
    df = df.copy()

    # ── Derived time columns ─────────────────────────────────────────────────
    df["abs_amount"]  = df["amount"].abs()
    df["month"]       = df["date"].dt.to_period("M").astype(str)
    df["month_str"]   = df["date"].dt.strftime("%b %Y")
    df["year"]        = df["date"].dt.year
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_num"]    = df["date"].dt.isocalendar().week.astype(int)

    # ── Anomaly placeholder columns (filled later by anomaly_detector) ───────
    if "is_anomaly" not in df.columns:
        df["is_anomaly"]     = False
    if "anomaly_score" not in df.columns:
        df["anomaly_score"]  = 0.0
    if "anomaly_reason" not in df.columns:
        df["anomaly_reason"] = ""

    metrics = compute_metrics(df)
    monthly = _build_monthly(df)
    cat_sum = get_category_summary(df)

    return {
        "df":               df,
        "metrics":          metrics,
        "monthly":          monthly,
        "category_summary": cat_sum,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute high-level financial summary metrics.

    Returns keys expected by app.py:
        total_income, total_expenses, net_savings, savings_rate,
        num_transactions, num_income_txns, num_expense_txns,
        avg_expense, top_expense_category, date_range
    """
    income_df  = df[df["transaction_type"] == "Income"]
    expense_df = df[df["transaction_type"] == "Expense"]

    total_income   = income_df["abs_amount"].sum()
    total_expenses = expense_df["abs_amount"].sum()
    net_savings    = total_income - total_expenses
    savings_rate   = (net_savings / total_income * 100) if total_income > 0 else 0.0
    avg_expense    = expense_df["abs_amount"].mean() if not expense_df.empty else 0.0

    top_expense_category = (
        expense_df.groupby("category")["abs_amount"].sum().idxmax()
        if not expense_df.empty else "N/A"
    )

    date_range = (
        df["date"].min().strftime("%Y-%m-%d") if not df.empty else "N/A",
        df["date"].max().strftime("%Y-%m-%d") if not df.empty else "N/A",
    )

    return {
        "total_income":           round(total_income, 2),
        "total_expenses":         round(total_expenses, 2),
        "net_savings":            round(net_savings, 2),
        "savings_rate":           round(savings_rate, 2),
        "num_transactions":       len(df),
        "num_income_txns":        len(income_df),
        "num_expense_txns":       len(expense_df),
        "avg_expense":            round(avg_expense, 2),
        "top_expense_category":   top_expense_category,
        "date_range":             date_range,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Monthly aggregates
# ─────────────────────────────────────────────────────────────────────────────

def _build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a wide DataFrame indexed by month_str with columns:
        Income | Expense | Net
    Used by the income_vs_expense_chart and trend_line_chart visualizations.
    """
    if df.empty:
        return pd.DataFrame(columns=["month_str", "Income", "Expense", "Net"])

    monthly = (
        df.groupby(["month_str", "transaction_type"])["abs_amount"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Ensure both columns exist even if data has only one type
    for col in ("Income", "Expense"):
        if col not in monthly.columns:
            monthly[col] = 0.0

    monthly["Net"] = monthly["Income"] - monthly["Expense"]

    # Sort chronologically using the first date per month_str
    month_order = (
        df.groupby("month_str")["date"].min()
        .sort_values()
        .index.tolist()
    )
    monthly["month_str"] = pd.Categorical(monthly["month_str"], categories=month_order, ordered=True)
    monthly = monthly.sort_values("month_str").reset_index(drop=True)

    return monthly


# ─────────────────────────────────────────────────────────────────────────────
# Category summary
# ─────────────────────────────────────────────────────────────────────────────

def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-category expense totals, counts, averages, and percentage share.
    Columns: category | total | count | avg | pct
    """
    expense_df = df[df["transaction_type"] == "Expense"]

    if expense_df.empty:
        return pd.DataFrame(columns=["category", "total", "count", "avg", "pct"])

    summary = (
        expense_df.groupby("category")["abs_amount"]
        .agg(total="sum", count="count", avg="mean")
        .reset_index()
        .sort_values("total", ascending=False)
    )

    grand_total  = summary["total"].sum()
    summary["pct"]   = (summary["total"] / grand_total * 100).round(2) if grand_total > 0 else 0.0
    summary["total"] = summary["total"].round(2)
    summary["avg"]   = summary["avg"].round(2)

    return summary.reset_index(drop=True)