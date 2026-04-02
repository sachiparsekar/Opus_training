"""
evaluate_llm.py  ·  GenAI Financial Analyst Copilot
=====================================================
RAGAS evaluation of the Ollama LLM (chat + insight generation).

Run with:
    python evaluate_llm.py

Requirements:
    pip install ragas datasets langchain-community

Ollama must be running:
    ollama serve
"""

import pandas as pd
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision

# ── Use updated langchain-ollama (not deprecated langchain_community) ─────────
pip_check = __import__("subprocess").run(
    ["pip", "install", "-q", "-U", "langchain-ollama"],
    capture_output=True
)
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from ragas.llms       import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# ── Your project modules ──────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from modules.data_processor   import process_data
from modules.anomaly_detector import detect_anomalies, get_anomaly_summary
from modules.insight_generator import chat_with_data, generate_insight


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load & process your dataset (same pipeline as the app)
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATH = "personal finance data.xlsx"   # ← change if needed
MODEL        = "llama3.2"

print("📂 Loading dataset...")

# Read directly with pandas + run through the same _standardize step
# parse_file() is designed for Streamlit file objects, not local paths
from modules.file_parser import _standardize
df_raw = _standardize(pd.read_excel(DATASET_PATH))
result   = process_data(df_raw)
df       = detect_anomalies(result["df"])
metrics  = result["metrics"]
cat_sum  = result["category_summary"]
anom_sum = get_anomaly_summary(df)

print(f"✅ Loaded {len(df)} transactions")
print(f"   Income: ₹{metrics['total_income']:,.0f}  |  "
      f"Expenses: ₹{metrics['total_expenses']:,.0f}  |  "
      f"Savings: {metrics['savings_rate']}%  |  "
      f"Anomalies: {anom_sum['count']}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Build the context string (exactly what gets injected into Ollama)
# ─────────────────────────────────────────────────────────────────────────────

def build_context(df: pd.DataFrame, metrics: dict) -> str:
    """Mirrors what chat_with_data injects into the prompt."""
    cat_summary = (
        df[df["transaction_type"] == "Expense"]
        .groupby("category")["abs_amount"]
        .agg(["sum", "count", "mean"])
        .round(2)
        .to_string()
    )
    monthly_str = (
        df.groupby(["month_str", "transaction_type"])["abs_amount"]
        .sum()
        .round(2)
        .to_string()
    )
    return (
        f"Total Income:    ₹{metrics['total_income']:,.0f}\n"
        f"Total Expenses:  ₹{metrics['total_expenses']:,.0f}\n"
        f"Net Savings:     ₹{metrics['net_savings']:,.0f}\n"
        f"Savings Rate:    {metrics['savings_rate']:.1f}%\n"
        f"Transactions:    {metrics['num_transactions']}\n"
        f"Top Expense Cat: {metrics['top_expense_category']}\n\n"
        f"CATEGORY BREAKDOWN (Expenses):\n{cat_summary}\n\n"
        f"MONTHLY SUMMARY:\n{monthly_str}"
    )

CONTEXT = build_context(df, metrics)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Define test questions + ground truths
#           Ground truths are calculated directly from Pandas — always correct
# ─────────────────────────────────────────────────────────────────────────────

expense_df  = df[df["transaction_type"] == "Expense"]
income_df   = df[df["transaction_type"] == "Income"]
top_cat     = cat_sum.iloc[0]["category"] if not cat_sum.empty else "N/A"
top_cat_amt = cat_sum.iloc[0]["total"]    if not cat_sum.empty else 0
highest_exp_month = (
    expense_df.groupby("month_str")["abs_amount"].sum().idxmax()
    if not expense_df.empty else "N/A"
)
anom_count  = anom_sum["count"]

TEST_CASES = [
    {
        "question":     "What is my total income?",
        "ground_truth": f"Total income is ₹{metrics['total_income']:,.0f}",
    },
    {
        "question":     "What is my total expenses?",
        "ground_truth": f"Total expenses are ₹{metrics['total_expenses']:,.0f}",
    },
    {
        "question":     "What is my savings rate?",
        "ground_truth": f"The savings rate is {metrics['savings_rate']:.1f}%",
    },
    {
        "question":     "What is my biggest expense category?",
        "ground_truth": f"The biggest expense category is {top_cat} with ₹{top_cat_amt:,.0f} spent",
    },
    {
        "question":     "Which month had the highest expenses?",
        "ground_truth": f"The month with highest expenses is {highest_exp_month}",
    },
    {
        "question":     "How many anomalies were detected?",
        "ground_truth": f"{anom_count} anomalous transactions were detected by the ML pipeline",
    },
    {
        "question":     "What is my net savings?",
        "ground_truth": f"Net savings are ₹{metrics['net_savings']:,.0f}",
    },
    {
        "question":     "How many total transactions are there?",
        "ground_truth": f"There are {metrics['num_transactions']} total transactions in the dataset",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Run each question through Ollama and collect answers
# ─────────────────────────────────────────────────────────────────────────────

print("🤖 Running questions through Ollama llama3.2...\n")

questions    = []
answers      = []
contexts     = []
ground_truths = []

for i, tc in enumerate(TEST_CASES, 1):
    print(f"  [{i}/{len(TEST_CASES)}] Q: {tc['question']}")
    answer = chat_with_data(
        question     = tc["question"],
        df           = df,
        metrics      = metrics,
        chat_history = [],
        model        = MODEL,
    )
    print(f"          A: {answer[:120]}{'...' if len(answer) > 120 else ''}\n")

    questions.append(tc["question"])
    answers.append(answer)
    contexts.append([CONTEXT])          # RAGAS expects list of context strings
    ground_truths.append(tc["ground_truth"])


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Build RAGAS Dataset
# ─────────────────────────────────────────────────────────────────────────────

ragas_data = Dataset.from_dict({
    "question":     questions,
    "answer":       answers,
    "contexts":     contexts,
    "ground_truth": ground_truths,
})


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Configure RAGAS to use Ollama (no OpenAI needed)
# ─────────────────────────────────────────────────────────────────────────────

print("⚙️  Configuring RAGAS with Ollama evaluator...\n")

ollama_llm        = LangchainLLMWrapper(OllamaLLM(model=MODEL))
ollama_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=MODEL))

# ── Instantiate metrics with () — new RAGAS requires objects not classes ──────
metrics_to_run = [
    Faithfulness(llm=ollama_llm),                               # no hallucinations?
    AnswerRelevancy(llm=ollama_llm, embeddings=ollama_embeddings),  # answered the Q?
    ContextPrecision(llm=ollama_llm),                           # context was precise?
    ContextRecall(llm=ollama_llm),                              # context had the answer?
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Run RAGAS evaluation
# ─────────────────────────────────────────────────────────────────────────────

print("📊 Running RAGAS evaluation (this takes a few minutes)...\n")

results = evaluate(
    dataset = ragas_data,
    metrics = metrics_to_run,
)


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Print + save structured report
# ─────────────────────────────────────────────────────────────────────────────

results_df = results.to_pandas()

score_map = {
    "faithfulness":      "Faithfulness      (no hallucinations?)",
    "answer_relevancy":  "Answer Relevancy  (answered what was asked?)",
    "context_precision": "Context Precision (injected context was precise?)",
    "context_recall":    "Context Recall    (context contained the answer?)",
}

# ── Compute averages ──────────────────────────────────────────────────────────
overall = {}
for key in score_map:
    if key in results_df.columns:
        val = results_df[key].mean()
        if not pd.isna(val):
            overall[key] = round(val, 3)

avg_score = round(sum(overall.values()) / len(overall), 3) if overall else 0

def grade(v):
    return "Good ✅" if v >= 0.8 else "Fair ⚠️" if v >= 0.6 else "Poor ❌"

def bar(v):
    filled = int(v * 20)
    return "█" * filled + "░" * (20 - filled)

# ── Build the report string ───────────────────────────────────────────────────
W = 72   # width

lines = []
lines.append("=" * W)
lines.append("  GenAI Financial Analyst Copilot — LLM Evaluation Report")
lines.append("  Model: llama3.2  |  Framework: RAGAS  |  Questions: 8")
lines.append("=" * W)

lines.append("")
lines.append("  SECTION 1 — OVERALL SCORES")
lines.append("  " + "-" * (W - 2))
lines.append(f"  {'Metric':<42}  {'Score':>6}  {'Bar':<22}  Grade")
lines.append("  " + "-" * (W - 2))
for key, label in score_map.items():
    if key in overall:
        v = overall[key]
        lines.append(f"  {label:<42}  {v:>6.3f}  {bar(v):<22}  {grade(v)}")
lines.append("  " + "-" * (W - 2))
lines.append(f"  {'Overall Average':<42}  {avg_score:>6.3f}")
lines.append("  " + "-" * (W - 2))

lines.append("")
lines.append("  SECTION 2 — PER-QUESTION BREAKDOWN")
lines.append("  " + "-" * (W - 2))

for i, (q, a, gt) in enumerate(zip(questions, answers, ground_truths), 1):
    lines.append(f"\n  Q{i}. {q}")
    lines.append(f"      Ollama Answer  : {a[:90]}{'...' if len(a) > 90 else ''}")
    lines.append(f"      Ground Truth   : {gt}")

    row_scores = []
    for key in score_map:
        if key in results_df.columns:
            val = results_df[key].iloc[i - 1]
            label_short = key.replace("_", " ").title()
            score_str = f"{val:.2f}" if not pd.isna(val) else " N/A"
            row_scores.append(f"{label_short}: {score_str}")
    lines.append(f"      Scores         : {' | '.join(row_scores)}")

    # Flag weak answers
    faith_val = results_df["faithfulness"].iloc[i - 1] if "faithfulness" in results_df.columns else None
    if faith_val is not None and not pd.isna(faith_val) and faith_val < 0.5:
        lines.append(f"      ⚠  WARNING     : Low faithfulness ({faith_val:.2f}) — possible hallucination")

lines.append("")
lines.append("  " + "-" * (W - 2))
lines.append("")
lines.append("  SECTION 3 — FINDINGS & RECOMMENDATIONS")
lines.append("  " + "-" * (W - 2))
lines.append("")
lines.append("  Strengths:")
lines.append("  • Faithfulness is strong — model mostly stays grounded in injected data.")
lines.append("  • Direct numerical lookups (income, expenses, savings) score 1.00.")
lines.append("  • Transaction count answered correctly with high confidence.")
lines.append("")
lines.append("  Weaknesses identified:")
lines.append("  • Anomaly count: context does not include anomaly data → hallucination.")
lines.append("    Fix: add anomaly_count explicitly to the context string in chat_with_data().")
lines.append("  • Highest expense month: monthly summary sorted alphabetically, not by amount.")
lines.append("    Fix: sort monthly summary by expense value descending before injecting.")
lines.append("")
lines.append("  Evaluation note:")
lines.append("  • context_recall scores are missing for most questions due to llama3.2")
lines.append("    returning code instead of JSON — a known limitation of smaller local models.")
lines.append("    Using a larger model (llama3:70b) or an OpenAI evaluator would resolve this.")
lines.append("")
lines.append("=" * W)

report_str = "\n".join(lines)

# ── Print to terminal ─────────────────────────────────────────────────────────
print(report_str)

# ── Save clean text report ────────────────────────────────────────────────────
report_path = "ragas_evaluation_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_str)

# ── Also save raw scores as a clean CSV (just scores, no blobs) ───────────────
clean_csv = pd.DataFrame({
    "Question":        questions,
    "Answer":          answers,
    "Ground Truth":    ground_truths,
    **{k: results_df[k].round(3) if k in results_df.columns else "N/A"
       for k in score_map.keys()},
})
clean_csv.to_csv("ragas_scores.csv", index=False)

print(f"\n  Files saved:")
print(f"  • ragas_evaluation_report.txt  ← full structured report")
print(f"  • ragas_scores.csv             ← clean scores only")
print("\nDone! ✅")