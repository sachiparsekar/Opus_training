"""
insight_generator.py
Communicates with a locally running Ollama instance to:
  1. Generate a full financial insight report (generate_insight)
  2. Answer ad-hoc user questions about the dataset (chat_with_data)
"""

import requests
import json
import pandas as pd
from typing import Optional

OLLAMA_BASE  = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
TIMEOUT_SECS  = 120


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def get_available_models() -> list[str]:
    """Return list of model names available in the local Ollama instance."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def ollama_is_running() -> bool:
    """Quick health-check for Ollama."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _call_ollama(prompt: str, model: str, max_tokens: int = 700) -> str:
    """Low-level call to Ollama /api/generate endpoint."""
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model":   model,
                "prompt":  prompt,
                "stream":  False,
                "options": {
                    "temperature": 0.25,
                    "num_predict": max_tokens,
                    "top_p":       0.9,
                },
            },
            timeout=TIMEOUT_SECS,
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        return f"⚠️ Ollama returned HTTP {r.status_code}: {r.text[:200]}"
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ **Cannot connect to Ollama.**\n\n"
            "Please start it with: `ollama serve`\n"
            "Then pull the model: `ollama pull llama3.2`"
        )
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. The model may still be loading — try again in a moment."
    except Exception as e:
        return f"⚠️ Unexpected error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Full financial insight report
# ─────────────────────────────────────────────────────────────────────────────

def generate_insight(
    metrics:          dict,
    anomaly_summary:  dict,
    category_summary: pd.DataFrame,
    model:            str = DEFAULT_MODEL,
) -> str:
    """
    Build a rich prompt from processed financial data and ask Ollama
    to generate a professional financial analysis report.
    """

    # ── Category breakdown (top 6) ────────────────────────────────────────
    cat_lines = ""
    if isinstance(category_summary, pd.DataFrame) and not category_summary.empty:
        for _, row in category_summary.head(6).iterrows():
            cat_lines += f"  • {row['category']}: ₹{row['total']:,.0f} ({row.get('pct', 0):.1f}%)\n"



    # ── Anomaly context ──────────────────────────────────────────────────
    anomaly_count = anomaly_summary.get("count", 0)
    anomaly_value = anomaly_summary.get("total_value", 0.0)
    anomaly_cats  = ", ".join(list(anomaly_summary.get("categories", {}).keys())[:3])

    prompt = f"""You are a senior financial analyst AI assistant. Analyze the following financial data and produce a professional report.

═══════════════════════════════════════
FINANCIAL DATA SUMMARY
═══════════════════════════════════════
Period: {metrics.get('date_range', ('N/A','N/A'))[0]} to {metrics.get('date_range', ('N/A','N/A'))[1]}
Total Transactions: {metrics.get('num_transactions', 0):,}

  💰 Total Income:    ₹{metrics.get('total_income', 0):,.0f}
  💸 Total Expenses:  ₹{metrics.get('total_expenses', 0):,.0f}
  🏦 Net Savings:     ₹{metrics.get('net_savings', 0):,.0f}
  📊 Savings Rate:    {metrics.get('savings_rate', 0):.1f}%
  📈 Avg Expense:     ₹{metrics.get('avg_expense', 0):,.0f}
  🏷  Top Category:   {metrics.get('top_expense_category', 'N/A')}

TOP EXPENSE CATEGORIES:
{cat_lines or '  (no expense data)'}

ANOMALY DETECTION (ML):
  • Flagged transactions: {anomaly_count}
  • Combined flagged value: ₹{anomaly_value:,.0f}
  • Affected categories: {anomaly_cats or 'None'}

═══════════════════════════════════════
Please provide a structured report with:

## 📋 Financial Health Summary
(2–3 sentences on overall financial health)

## 🔍 Key Observations
(4 bullet points — be specific with numbers)

## ⚠️ Risk Alerts
(mention anomalies, overspending categories, or concerning trends; "None" if all clear)

## 💡 Actionable Recommendations
(3–4 specific, practical recommendations with estimated impact)

## 🎯 Financial Health Score
(Rate overall financial health: Excellent / Good / Fair / Poor — with one sentence justification)

Use ₹ for currency. Be concise, data-driven, and professional.
"""

    return _call_ollama(prompt, model, max_tokens=800)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Conversational Q&A chatbot
# ─────────────────────────────────────────────────────────────────────────────

def chat_with_data(
    question:         str,
    df:               pd.DataFrame,
    metrics:          dict,
    chat_history:     Optional[list] = None,
    model:            str = DEFAULT_MODEL,
    extra_context:    str = "",          # ← override / correction notes from app.py
) -> str:
    """
    Answer a user's natural-language question about their financial data.
    Includes the last few chat turns for conversational context.
    """

    # ── Build compact data context ────────────────────────────────────────
    cat_summary = (
        df[df["transaction_type"] == "Expense"]
        .groupby("category")["abs_amount"]
        .agg(["sum", "count", "mean"])
        .round(2)
        .to_string()
    )

    # Most recent 15 transactions
    recent_txns = (
        df[["date", "category", "abs_amount", "transaction_type", "is_anomaly"]]
        .tail(15)
        .to_string(index=False)
    )

    # Monthly totals
    monthly_str = (
        df.groupby(["month_str", "transaction_type"])["abs_amount"]
        .sum()
        .round(2)
        .to_string()
    )

    # ── Previous turns ────────────────────────────────────────────────────
    history_text = ""
    if chat_history:
        for turn in chat_history[-4:]:    # last 4 turns for context
            role    = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n"

    prompt = f"""You are a financial analyst AI assistant helping a user understand their personal/business finances. 
Answer questions accurately using ONLY the data provided below. 
If the data does not contain enough information to answer, say so clearly.
Keep answers concise (under 150 words) unless a detailed breakdown is explicitly requested.
Always cite specific numbers from the data.
{f"IMPORTANT USER CORRECTION: {extra_context}" if extra_context else ""}

════════════════════════════════
FINANCIAL DATA SNAPSHOT
════════════════════════════════
Total Income:    ₹{metrics.get('total_income', 0):,.0f}
Total Expenses:  ₹{metrics.get('total_expenses', 0):,.0f}
Net Savings:     ₹{metrics.get('net_savings', 0):,.0f}
Savings Rate:    {metrics.get('savings_rate', 0):.1f}%
Transactions:    {metrics.get('num_transactions', 0)}
Top Expense Cat: {metrics.get('top_expense_category', 'N/A')}
Anomalies Detected: {metrics.get('anomaly_count', 'N/A')}

CATEGORY BREAKDOWN (Expenses):
{cat_summary}

MONTHLY SUMMARY:
{monthly_str}

RECENT TRANSACTIONS (last 15):
{recent_txns}

════════════════════════════════
CONVERSATION HISTORY:
{history_text or '(No previous messages)'}

User: {question}
Assistant:"""

    return _call_ollama(prompt, model, max_tokens=400)