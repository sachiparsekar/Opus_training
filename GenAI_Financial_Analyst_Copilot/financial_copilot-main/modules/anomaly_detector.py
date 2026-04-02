"""
anomaly_detector.py
Detects unusual financial transactions using two complementary ML methods:
  1. Z-score  – statistical outlier detection per category (fast, interpretable)
  2. Isolation Forest (scikit-learn) – unsupervised ML on amount + time features

A transaction is flagged if EITHER method raises an alarm, and the
combined confidence score is stored for ranking.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds three columns to the DataFrame:
        is_anomaly     – bool
        anomaly_score  – float (higher = more anomalous)
        anomaly_reason – human-readable explanation string
    """
    df = df.copy()
    df["is_anomaly"]     = False
    df["anomaly_score"]  = 0.0
    df["anomaly_reason"] = ""

    expense_mask = df["transaction_type"] == "Expense"
    expense_df   = df[expense_mask].copy()

    if len(expense_df) < 5:
        return df          # Not enough data for meaningful detection

    # ── Z-score per-category ─────────────────────────────────────────────────
    z_flags, z_scores, z_reasons = _zscore_flags(expense_df)

    # ── Isolation Forest ─────────────────────────────────────────────────────
    iso_flags, iso_scores = _isolation_forest_flags(expense_df)

    # ── Combine ──────────────────────────────────────────────────────────────
    for loc_pos, idx in enumerate(expense_df.index):
        z_flag   = z_flags[loc_pos]
        iso_flag = iso_flags[loc_pos]

        if z_flag or iso_flag:
            df.at[idx, "is_anomaly"] = True

            # Combined score: weighted average
            combined = (z_scores[loc_pos] * 0.5 + iso_scores[loc_pos] * 0.5)
            df.at[idx, "anomaly_score"] = round(combined, 3)

            reasons = []
            if z_flag:
                reasons.append(z_reasons[loc_pos])
            if iso_flag:
                reasons.append("Isolation Forest")
            df.at[idx, "anomaly_reason"] = " | ".join(reasons)

    return df


def get_anomaly_summary(df: pd.DataFrame) -> dict:
    """Return summary stats for anomalous transactions."""
    anomalies = df[df["is_anomaly"] == True]

    if anomalies.empty:
        return {
            "count":       0,
            "total_value": 0.0,
            "categories":  {},
            "anomaly_df":  anomalies,
        }

    anomaly_df_out = (
        anomalies[["date", "category", "abs_amount", "anomaly_score", "anomaly_reason"]]
        .sort_values("anomaly_score", ascending=False)
    )
    # Embed the original df row index as a plain column so app.py
    # can reliably map checkboxes back to the right rows — never use
    # .index for this because any pandas op can silently change it
    anomaly_df_out = anomaly_df_out.copy()
    anomaly_df_out["df_row_idx"] = anomaly_df_out.index

    return {
        "count":       len(anomalies),
        "total_value": round(anomalies["abs_amount"].sum(), 2),
        "categories":  anomalies["category"].value_counts().to_dict(),
        "anomaly_df":  anomaly_df_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _zscore_flags(expense_df: pd.DataFrame):
    """Per-category Z-score: flag if |z| > 2.5."""
    n        = len(expense_df)
    flags    = np.zeros(n, dtype=bool)
    scores   = np.zeros(n, dtype=float)
    reasons  = [""] * n

    amounts   = expense_df["abs_amount"].values
    categories = expense_df["category"].values

    # Global Z-score
    global_mean = amounts.mean()
    global_std  = amounts.std(ddof=1) if amounts.std(ddof=1) > 0 else 1.0
    global_z    = np.abs((amounts - global_mean) / global_std)

    # Per-category Z-score
    cat_z = np.zeros(n)
    for cat in np.unique(categories):
        mask    = categories == cat
        sub_amt = amounts[mask]
        if len(sub_amt) < 3:
            continue
        mu  = sub_amt.mean()
        sig = sub_amt.std(ddof=1) if sub_amt.std(ddof=1) > 0 else 1.0
        cat_z[mask] = np.abs((sub_amt - mu) / sig)

    THRESHOLD = 2.5
    for i in range(n):
        z = max(global_z[i], cat_z[i])
        if z > THRESHOLD:
            flags[i]   = True
            scores[i]  = round(float(z), 3)
            tag = "global" if global_z[i] >= cat_z[i] else f"in {categories[i]}"
            reasons[i] = f"Z-score {z:.1f}σ ({tag})"

    return flags, scores, reasons


def _isolation_forest_flags(expense_df: pd.DataFrame):
    """Isolation Forest on [abs_amount, day-of-month] features."""
    n = len(expense_df)

    # Feature matrix
    day_of_month = expense_df["date"].dt.day.values.reshape(-1, 1)
    amounts      = expense_df["abs_amount"].values.reshape(-1, 1)
    X            = np.hstack([amounts, day_of_month])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Contamination: assume ≤10 % of transactions are anomalies
    contamination = min(0.10, max(0.01, 8 / n))

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    predictions = iso.fit_predict(X_scaled)        # -1 = anomaly
    raw_scores  = iso.decision_function(X_scaled)  # lower = more anomalous

    flags  = predictions == -1
    # Normalise to [0, 1] – higher = more anomalous
    min_s, max_s = raw_scores.min(), raw_scores.max()
    if max_s > min_s:
        scores = 1.0 - (raw_scores - min_s) / (max_s - min_s)
    else:
        scores = np.zeros(n)

    return flags, scores