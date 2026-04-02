# 💹 GenAI Financial Analyst Copilot

A fully local, AI-powered financial analysis tool built with **Streamlit**, **Pandas**, **Scikit-learn**, and **Ollama**.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed locally

### 2. Install Ollama model
```bash
ollama pull llama3.2
```
> `llama3.2` is recommended — fast and accurate for financial Q&A.
> You can also use `llama3`, `mistral`, or `gemma2`.

### 3. Install Python dependencies
```bash
cd financial_copilot
pip install -r requirements.txt
```

### 4. Start Ollama (in a separate terminal)
```bash
ollama serve
```

### 5. Run the app
```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**

---

## 📁 Project Structure

```
financial_copilot/
├── app.py                      ← Main Streamlit application
├── requirements.txt
├── sample_data/
│   └── sample_transactions.csv ← Test dataset (6 months, ~96 transactions)
└── modules/
    ├── file_parser.py          ← CSV / Excel 
    ├── data_processor.py       ← Pandas: cleaning, metrics, trends
    ├── anomaly_detector.py     ← Scikit-learn: Isolation Forest + Z-score
    ├── insight_generator.py    ← Ollama: report generation + chatbot
    └── visualizations.py       ← Plotly: 7 interactive charts
```

---

## 📊 Features

| Feature | Details |
|---|---|
| **File Support** | CSV, Excel (.xlsx/.xls) |
| **Auto Column Mapping** | Detects `Date`, `Category`, `Amount`, `Type` from any naming |
| **Key Metrics** | Total income, expenses, net savings, savings rate |
| **4 Charts** | Bar, Donut, Line, Scatter |
| **ML Anomaly Detection** | Isolation Forest (200 estimators) + Z-score (2.5σ threshold) |
| **AI Insight Report** | Full financial health report via local Ollama LLM |
| **AI Chatbot** | Conversational Q&A about your data — sidebar always accessible |
| **100% Local** | No API keys, no cloud calls, no data leaves your machine |

---

## 📋 Supported CSV Format

```
Date,Category,Amount,Transaction_Type
2024-01-01,Salary,80000,Income
2024-01-02,Rent,-20000,Expense
2024-01-05,Groceries,-4800,Expense
```

- `Amount` can be positive (Income) or negative (Expense)  
- `Transaction_Type` is optional — inferred from Amount sign if absent  
- Currency symbols (₹, $, €) are auto-stripped  

---

## 🧠 ML Pipeline

```
Expense transactions
       ↓
┌──────────────────────┐     ┌───────────────────────────────┐
│  Z-score per category│  +  │  Isolation Forest (sklearn)   │
│  flag if |z| > 2.5σ  │     │  contamination = auto-tuned   │
└──────────────────────┘     └───────────────────────────────┘
       ↓                              ↓
       └───────── OR ─────────────────┘
                  ↓
         Combined anomaly flag
         + unified score [0–1]
```

---

## 💬 Chatbot Tips

Ask questions like:
- *"What is my biggest expense category?"*
- *"How did my spending change in March?"*
- *"Which transactions look suspicious?"*
- *"How can I improve my savings rate?"*
- *"What percentage of income goes to rent?"*

---

