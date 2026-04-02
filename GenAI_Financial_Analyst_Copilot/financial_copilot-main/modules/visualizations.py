"""
visualizations.py
All Plotly chart builders for the Financial Analyst Copilot dashboard.
Every chart is transparent-background so it looks great on Streamlit's dark theme.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "income":    "#2ecc71",
    "expense":   "#e74c3c",
    "savings":   "#3498db",
    "anomaly":   "#f39c12",
    "warning":   "#f39c12",
    "neutral":   "#95a5a6",
    "grid":      "rgba(255,255,255,0.08)",
    "bg":        "rgba(0,0,0,0)",
    "text":      "#e8e8e8",
    "subtext":   "#aaaaaa",
}

PALETTE = px.colors.qualitative.Set3    # 12 colours for categories

_layout_defaults = dict(
    plot_bgcolor  = C["bg"],
    paper_bgcolor = C["bg"],
    font          = dict(color=C["text"], family="Inter, sans-serif"),
    legend        = dict(
        bgcolor     = "rgba(0,0,0,0)",
        bordercolor = "rgba(255,255,255,0.15)",
        borderwidth = 1,
    ),
    margin = dict(t=55, b=40, l=55, r=20),
)

def _axis(title="", **kw):
    return dict(title=title, gridcolor=C["grid"], zerolinecolor=C["grid"],
                tickfont=dict(color=C["subtext"]), title_font=dict(color=C["subtext"]),
                **kw)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Income vs Expenses – grouped bar
# ─────────────────────────────────────────────────────────────────────────────

def income_vs_expense_chart(monthly: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart showing monthly income vs. expenses.
    Purpose: Instantly spot months where spending exceeded earnings.
    """
    fig = go.Figure()

    if "Income" in monthly.columns:
        fig.add_trace(go.Bar(
            name        = "Income",
            x           = monthly["month_str"],
            y           = monthly["Income"],
            marker_color= C["income"],
            text        = [f"₹{v:,.0f}" for v in monthly["Income"]],
            textposition= "outside",
            textfont    = dict(size=10, color=C["income"]),
        ))

    if "Expense" in monthly.columns:
        fig.add_trace(go.Bar(
            name        = "Expenses",
            x           = monthly["month_str"],
            y           = monthly["Expense"],
            marker_color= C["expense"],
            text        = [f"₹{v:,.0f}" for v in monthly["Expense"]],
            textposition= "outside",
            textfont    = dict(size=10, color=C["expense"]),
        ))

    fig.update_layout(
        **_layout_defaults,
        title      = dict(text="Monthly Income vs Expenses", font=dict(size=16)),
        barmode    = "group",
        xaxis      = _axis("Month"),
        yaxis      = _axis("Amount (₹)"),
        height     = 400,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Category-wise Spending – donut
# ─────────────────────────────────────────────────────────────────────────────

def category_donut_chart(category_df: pd.DataFrame) -> go.Figure:
    """
    Donut chart for category-wise expense distribution.
    Purpose: Identify the biggest cost drivers at a glance.
    """
    top = category_df.head(9)

    fig = go.Figure(go.Pie(
        labels      = top["category"],
        values      = top["total"],
        hole        = 0.52,
        textinfo    = "percent+label",
        textposition= "outside",
        marker      = dict(colors=PALETTE[:len(top)],
                           line=dict(color="#1a1a2e", width=2)),
        pull        = [0.04 if i == 0 else 0 for i in range(len(top))],
        hovertemplate = "<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>",
    ))

    fig.update_layout(
        **_layout_defaults,
        title       = dict(text="Category-wise Expense Distribution", font=dict(size=16)),
        annotations = [dict(
            text="Expenses", x=0.5, y=0.5,
            font=dict(size=14, color=C["text"]),
            showarrow=False,
        )],
        height      = 420,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Monthly Trend – multi-line
# ─────────────────────────────────────────────────────────────────────────────

def trend_line_chart(monthly: pd.DataFrame) -> go.Figure:
    """
    Line chart tracking income, expenses, and net savings over time.
    Purpose: Reveal seasonal patterns and long-term trajectory.
    """
    fig = go.Figure()

    def add_line(name, col, color, dash="solid", width=3):
        if col in monthly.columns:
            fig.add_trace(go.Scatter(
                name             = name,
                x                = monthly["month_str"],
                y                = monthly[col],
                mode             = "lines+markers",
                line             = dict(color=color, width=width, dash=dash),
                marker           = dict(size=8, color=color,
                                        line=dict(width=2, color="#1a1a2e")),
                hovertemplate    = f"<b>{name}</b><br>₹%{{y:,.0f}}<extra></extra>",
            ))

    add_line("Income",   "Income",  C["income"])
    add_line("Expenses", "Expense", C["expense"])

    # Derived savings line
    if "Income" in monthly.columns and "Expense" in monthly.columns:
        savings = monthly["Income"] - monthly["Expense"]
        fig.add_trace(go.Scatter(
            name          = "Net Savings",
            x             = monthly["month_str"],
            y             = savings,
            mode          = "lines+markers",
            line          = dict(color=C["savings"], width=2, dash="dot"),
            marker        = dict(size=6),
            fill          = "tozeroy",
            fillcolor     = "rgba(52,152,219,0.08)",
            hovertemplate = "<b>Savings</b><br>₹%{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        **_layout_defaults,
        title  = dict(text="Monthly Financial Trend", font=dict(size=16)),
        xaxis  = _axis("Month"),
        yaxis  = _axis("Amount (₹)"),
        height = 400,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Anomaly Scatter
# ─────────────────────────────────────────────────────────────────────────────

def anomaly_scatter_chart(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of all transactions; ML-flagged anomalies are highlighted.
    Purpose: Visually locate suspicious spikes in the transaction timeline.
    """
    normal    = df[df["is_anomaly"] == False]
    anomalies = df[df["is_anomaly"] == True]

    fig = go.Figure()

    # Normal transactions
    if not normal.empty:
        fig.add_trace(go.Scatter(
            name          = "Normal",
            x             = normal["date"],
            y             = normal["abs_amount"],
            mode          = "markers",
            marker        = dict(color=C["income"], size=6, opacity=0.55),
            customdata    = normal[["category", "transaction_type"]],
            hovertemplate = (
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "₹%{y:,.0f} on %{x|%d %b %Y}<extra></extra>"
            ),
        ))

    # Anomalous transactions
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            name          = "⚠️ Anomaly",
            x             = anomalies["date"],
            y             = anomalies["abs_amount"],
            mode          = "markers+text",
            text          = ["⚠️"] * len(anomalies),
            textposition  = "top center",
            marker        = dict(
                color  = C["anomaly"],
                size   = 16,
                symbol = "x",
                line   = dict(width=2.5, color="white"),
            ),
            customdata    = anomalies[["category", "anomaly_reason"]],
            hovertemplate = (
                "<b>ANOMALY: %{customdata[0]}</b><br>"
                "₹%{y:,.0f} on %{x|%d %b %Y}<br>"
                "Reason: %{customdata[1]}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_layout_defaults,
        title  = dict(text="Transaction Anomaly Detection (ML)", font=dict(size=16)),
        xaxis  = _axis("Date"),
        yaxis  = _axis("Amount (₹)"),
        height = 400,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Savings Rate Gauge
# ─────────────────────────────────────────────────────────────────────────────

def savings_rate_gauge(rate: float) -> go.Figure:
    """
    Gauge showing savings rate vs the 20 % recommended benchmark.
    Purpose: Provide an immediate health signal for savings discipline.
    """
    color = C["income"] if rate >= 20 else C["warning"] if rate >= 10 else C["expense"]

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = rate,
        title = dict(text="Savings Rate", font=dict(color=C["text"], size=13)),
        delta = dict(
            reference  = 20,
            suffix     = "%",
            increasing = dict(color=C["income"]),
            decreasing = dict(color=C["expense"]),
        ),
        number = dict(suffix="%", font=dict(color=C["text"], size=32)),
        gauge  = dict(
            axis      = dict(range=[0, 100],
                             tickfont=dict(color=C["subtext"]),
                             ticksuffix="%"),
            bar       = dict(color=color, thickness=0.28),
            bgcolor   = "rgba(0,0,0,0)",
            steps     = [
                dict(range=[0, 10],  color="rgba(231,76,60,0.18)"),
                dict(range=[10, 20], color="rgba(243,156,18,0.18)"),
                dict(range=[20, 100],color="rgba(46,204,113,0.18)"),
            ],
            threshold = dict(
                line      = dict(color="white", width=2),
                thickness = 0.75,
                value     = 20,
            ),
        ),
    ))

    gauge_layout = {k: v for k, v in _layout_defaults.items() if k != "margin"}
    fig.update_layout(
        **gauge_layout,
        height = 280,
        margin = dict(t=50, b=10, l=30, r=30),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Spending Heatmap – day-of-week × week
# ─────────────────────────────────────────────────────────────────────────────

def spending_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Weekly spending heatmap (expenses only).
    Purpose: Identify which days / weeks have the highest spending patterns.
    """
    expense_df = df[df["transaction_type"] == "Expense"].copy()
    if expense_df.empty:
        return go.Figure()

    expense_df["dow"]   = expense_df["date"].dt.day_name()
    expense_df["week"]  = expense_df["date"].dt.isocalendar().week.astype(int)

    heat = (
        expense_df.groupby(["dow", "week"])["abs_amount"]
        .sum()
        .unstack(fill_value=0)
    )

    # Sort days
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    heat = heat.reindex([d for d in day_order if d in heat.index])

    fig = go.Figure(go.Heatmap(
        z            = heat.values,
        x            = [f"W{w}" for w in heat.columns],
        y            = heat.index.tolist(),
        colorscale   = "RdYlGn_r",
        hovertemplate= "Week %{x} – %{y}<br>₹%{z:,.0f}<extra></extra>",
        colorbar     = dict(tickfont=dict(color=C["subtext"]),
                            title=dict(text="₹", font=dict(color=C["subtext"]))),
    ))

    fig.update_layout(
        **_layout_defaults,
        title  = dict(text="Spending Heatmap (Day × Week)", font=dict(size=16)),
        xaxis  = _axis("Week"),
        yaxis  = _axis("Day of Week"),
        height = 320,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Top-N Expense Categories – horizontal bar
# ─────────────────────────────────────────────────────────────────────────────

def top_categories_bar(category_df: pd.DataFrame, n: int = 8) -> go.Figure:
    """
    Horizontal bar chart of top expense categories.
    Purpose: Ranked view to quickly see where most money goes.
    """
    top = category_df.head(n).sort_values("total")

    fig = go.Figure(go.Bar(
        orientation = "h",
        x           = top["total"],
        y           = top["category"],
        text        = [f"₹{v:,.0f}  ({p:.1f}%)" for v, p in zip(top["total"], top["pct"])],
        textposition= "outside",
        textfont    = dict(size=11, color=C["text"]),
        marker      = dict(
            color      = top["total"],
            colorscale = "RdYlGn_r",
            showscale  = False,
        ),
        hovertemplate = "<b>%{y}</b><br>₹%{x:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        **_layout_defaults,
        title  = dict(text=f"Top {n} Expense Categories", font=dict(size=16)),
        xaxis  = _axis("Amount (₹)"),
        yaxis  = _axis(""),
        height = max(300, n * 42),
    )
    return fig