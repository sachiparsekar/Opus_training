"""
app.py  ·  GenAI Financial Analyst Copilot
==========================================
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd

# ── Modules ───────────────────────────────────────────────────────────────────
from modules.file_parser       import parse_file
from modules.data_processor    import process_data
from modules.anomaly_detector  import detect_anomalies, get_anomaly_summary
from modules.insight_generator import (
    generate_insight,
    chat_with_data,
    get_available_models,
    ollama_is_running,
)
from modules.visualizations    import (
    income_vs_expense_chart,
    category_donut_chart,
    trend_line_chart,
    anomaly_scatter_chart,
    savings_rate_gauge,
    spending_heatmap,
    top_categories_bar,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "GenAI Financial Analyst Copilot",
    page_icon  = "💹",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ── Metric cards ── */
.fin-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 18px 14px;
    text-align: center;
    margin-bottom: 8px;
    transition: transform .15s;
}
.fin-card:hover { transform: translateY(-2px); }
.fin-card .label { font-size: 12px; color: #9aabb5; letter-spacing: .5px; text-transform: uppercase; }
.fin-card .value { font-size: 26px; font-weight: 700; margin: 6px 0 2px; }
.fin-card .sub   { font-size: 11px; color: #6b7f8a; }
.green  { color: #2ecc71; }
.red    { color: #e74c3c; }
.blue   { color: #3498db; }
.orange { color: #f39c12; }
.white  { color: #ecf0f1; }

/* ── Chart purpose strip ── */
.purpose {
    background: rgba(52,152,219,0.08);
    border-left: 3px solid #3498db;
    padding: 7px 14px;
    border-radius: 0 8px 8px 0;
    font-size: 12.5px;
    color: #9aabb5;
    margin: -4px 0 18px;
    line-height: 1.5;
}

/* ── Anomaly badge ── */
.badge {
    display: inline-block;
    background: #f39c12;
    color: #1a1a2e;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 13px;
    font-weight: 700;
}

/* ── Section header ── */
.sec-header {
    font-size: 20px;
    font-weight: 700;
    margin: 28px 0 14px;
    color: #ecf0f1;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 6px;
}

/* ── Chat bubbles ── */
.bubble-user {
    background: rgba(52,152,219,0.18);
    border-radius: 12px 12px 2px 12px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 14px;
}
.bubble-ai {
    background: rgba(255,255,255,0.05);
    border-radius: 12px 12px 12px 2px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 14px;
    border-left: 3px solid #2ecc71;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: opacity .15s;
}
.stButton > button:hover { opacity: .88; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0e1320; }
[data-testid="stSidebar"] .stMarkdown p { font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "processed_data":    None,
        "chat_history":      [],
        "last_insight":      None,
        "pending_question":  None,
        "selected_model":    "llama3.1",
        # Set of row indices the user has manually marked as NOT anomalies
        "user_overrides":    set(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Chatbot
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💬 Financial AI Chat")
    st.caption("Ask anything about your uploaded data")

    # ── Ollama status ────────────────────────────────────────────────────
    alive = ollama_is_running()
    if alive:
        st.success("🟢 Ollama connected", icon=None)
    else:
        st.error("🔴 Ollama offline — run `ollama serve`")

    # ── Model picker ─────────────────────────────────────────────────────
    models = get_available_models()
    if models:
        st.session_state.selected_model = st.selectbox(
            "🤖 Model",
            models,
            index=0,
            help="All models pulled in your local Ollama instance",
        )
    else:
        st.session_state.selected_model = st.text_input(
            "🤖 Model name", value="llama3.1"
        )

    st.divider()

    # ── Chat history display ─────────────────────────────────────────────
    chat_box = st.container()
    with chat_box:
        if not st.session_state.chat_history:
            st.markdown(
                "<div style='text-align:center;color:#5a6a7a;font-size:13px;"
                "padding:30px 10px'>Upload a dataset, then ask me anything 👇</div>",
                unsafe_allow_html=True,
            )
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='bubble-user'>🧑 {msg['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='bubble-ai'>🤖 {msg['content']}</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Quick question buttons ───────────────────────────────────────────
    st.markdown("**💡 Quick questions**")
    quick_qs = [
        "What is my biggest expense category?",
        "How healthy is my savings rate?",
        "Which month had the highest expenses?",
        "Are there any unusual transactions?",
        "How can I reduce my monthly expenses?",
    ]
    for q in quick_qs:
        if st.button(q, key=f"qbtn_{q[:25]}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

    st.divider()

    # ── Free-text input ──────────────────────────────────────────────────
    user_input = st.text_area(
        "Your question",
        placeholder="e.g. What percentage of my income goes to rent?",
        height=90,
        label_visibility="collapsed",
    )
    col_send, col_clear = st.columns(2)
    with col_send:
        if st.button("Send ✉️", type="primary", use_container_width=True):
            if user_input.strip():
                st.session_state.pending_question = user_input.strip()
                st.rerun()
    with col_clear:
        if st.button("Clear 🗑️", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ── Process pending question ─────────────────────────────────────────
    if st.session_state.pending_question:
        q = st.session_state.pending_question
        st.session_state.pending_question = None

        if not st.session_state.processed_data:
            st.warning("⚠️ Please upload a financial dataset first.")
        else:
            data = st.session_state.processed_data
            st.session_state.chat_history.append({"role": "user", "content": q})

            with st.spinner("Thinking…"):
                # Compute anom_sum here directly — can't use the dashboard variable
                # because the sidebar runs before the dashboard block
                n_overrides = len(st.session_state.user_overrides)
                if n_overrides > 0:
                    _df_for_chat = data["df"].copy()
                    _df_for_chat.loc[list(st.session_state.user_overrides), "is_anomaly"] = False
                    from modules.anomaly_detector import get_anomaly_summary
                    _anom_count = get_anomaly_summary(_df_for_chat)["count"]
                    override_note = (
                        f"Note: The user has manually reviewed the ML-flagged anomalies and marked "
                        f"{n_overrides} transaction(s) as NOT anomalous. "
                        f"The current anomaly count is {_anom_count} (after user corrections). "
                        f"Do not refer to the overridden transactions as anomalies."
                    )
                else:
                    override_note = ""
                answer = chat_with_data(
                    question     = q,
                    df           = data["df"],
                    metrics      = {**data["metrics"], "anomaly_count": anom_sum["count"]},  # ← merge it in
                    chat_history = st.session_state.chat_history[:-1],
                    model        = st.session_state.selected_model,
                    extra_context= override_note,
    )
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main — Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-size:32px;font-weight:800;margin-bottom:4px'>"
    "💹 GenAI Financial Analyst Copilot</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "Upload your financial data (CSV · Excel · PDF) → "
    "get AI-powered metrics, trend charts, anomaly detection, and natural-language insights."
)


# ─────────────────────────────────────────────────────────────────────────────
# Upload section
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">📁 Upload Financial Dataset</div>', unsafe_allow_html=True)

up_col, info_col = st.columns([3, 2])

with up_col:
    uploaded = st.file_uploader(
        "Drag & drop or click to browse",
        # ── CHANGE 1: added "pdf" to accepted types ──
        type=["csv", "xlsx", "xls", "pdf"],
        help="Supported: CSV, Excel (.xlsx / .xls), PDF bank statements",
        label_visibility="collapsed",
    )

with info_col:
    # ── CHANGE 2: removed sample file download button entirely ──
    with st.expander("ℹ️ Expected columns & sample file"):
        st.markdown("""
**Required columns (auto-detected):**
| Column | Examples |
|--------|----------|
| Date | `Date`, `Trans Date`, `Value Date`, `Date / Time` |
| Category | `Category`, `Description`, `Narration`, `Sub category` |
| Amount | `Amount`, `Debit/Credit`, `Value` |
| Type *(optional)* | `Income/Expense`, `Dr/Cr` |

If `Transaction Type` is absent it is inferred from the sign of Amount.
Expense amounts are automatically made negative if stored as positive.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Parse & process uploaded file
# ─────────────────────────────────────────────────────────────────────────────
if uploaded:
    # Only re-process if a new file is uploaded
    if (
        st.session_state.processed_data is None
        or st.session_state.get("_last_uploaded") != uploaded.name
    ):
        with st.spinner("🔄 Parsing and analysing your data…"):
            try:
                df_raw  = parse_file(uploaded)
                result  = process_data(df_raw)
                df_proc = detect_anomalies(result["df"])
                result["df"]              = df_proc
                result["anomaly_summary"] = get_anomaly_summary(df_proc)

                st.session_state.processed_data = result
                st.session_state._last_uploaded = uploaded.name
                st.session_state.last_insight   = None  # clear stale insight
                st.session_state.user_overrides = set() # clear overrides for new file

                st.success(
                    f"✅ Processed **{len(df_proc):,} transactions** "
                    f"from {df_proc['date'].min().strftime('%d %b %Y')} "
                    f"to {df_proc['date'].max().strftime('%d %b %Y')}"
                )
            except Exception as exc:
                st.error(f"❌ {exc}")
                st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard (only shown when data is loaded)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.processed_data:
    D        = st.session_state.processed_data
    df       = D["df"].copy()
    metrics  = D["metrics"]
    monthly  = D["monthly"]
    cat_sum  = D["category_summary"]

    # ── Apply user overrides — uncheck = no longer an anomaly ────────────
    if st.session_state.user_overrides:
        df.loc[list(st.session_state.user_overrides), "is_anomaly"] = False

    # Recompute anomaly summary after overrides
    from modules.anomaly_detector import get_anomaly_summary
    anom_sum = get_anomaly_summary(df)

    # ── KPI Cards ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">📊 Key Financial Metrics</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    def _card(col, label, value, color, sub=""):
        with col:
            st.markdown(
                f"""<div class="fin-card">
                    <div class="label">{label}</div>
                    <div class="value {color}">{value}</div>
                    <div class="sub">{sub}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    _card(k1, "💰 Total Income",    f"₹{metrics['total_income']:,.0f}",   "green",
          f"{metrics['num_income_txns']} transactions")
    exp_sub   = "⚠️ No expense rows found" if metrics["total_expenses"] == 0 else f"{metrics['num_expense_txns']} transactions"
    exp_color = "orange" if metrics["total_expenses"] == 0 else "red"
    _card(k2, "💸 Total Expenses",  f"₹{metrics['total_expenses']:,.0f}", exp_color, exp_sub)

    net_sav   = metrics["net_savings"]
    sav_color = "green" if net_sav >= 0 else "red"
    no_expense = metrics["total_expenses"] == 0
    sav_sub   = "⚠️ No expenses in data" if no_expense else ("Positive cash flow" if net_sav >= 0 else "Expenses exceed income")
    _card(k3, "🏦 Net Savings",     f"₹{net_sav:,.0f}",                  sav_color, sav_sub)

    rate      = metrics["savings_rate"]
    rate_color = "green" if rate >= 20 else "orange" if rate >= 10 else "red"
    _card(k4, "📈 Savings Rate",    f"{rate:.1f}%",                       rate_color, "Target: ≥ 20%")

    _card(k5, "🏷️ Top Expense Cat", metrics["top_expense_category"],      "white",
          f"Avg txn ₹{metrics['avg_expense']:,.0f}")

    acount = anom_sum["count"]
    aval   = f'<span class="badge">⚠️ {acount}</span>' if acount > 0 else "✅ 0"
    with k6:
        st.markdown(
            f"""<div class="fin-card">
                <div class="label">🔍 Anomalies</div>
                <div class="value" style="font-size:22px">{aval}</div>
                <div class="sub">ML flagged</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Charts row 1 ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">📈 Visual Analytics</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(income_vs_expense_chart(monthly), use_container_width=True)
        st.markdown(
            "<div class='purpose'>📌 <b>Purpose:</b> Compare monthly income vs. expenses "
            "to spot months where spending exceeded earnings and track cash-flow health over time.</div>",
            unsafe_allow_html=True,
        )
    with c2:
        if not cat_sum.empty:
            st.plotly_chart(category_donut_chart(cat_sum), use_container_width=True)
            st.markdown(
                "<div class='purpose'>📌 <b>Purpose:</b> Reveals where money is going — "
                "the largest slice is your biggest cost driver and the first target for savings.</div>",
                unsafe_allow_html=True,
            )

    # ── Charts row 2 ──────────────────────────────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        if not monthly.empty:
            st.plotly_chart(trend_line_chart(monthly), use_container_width=True)
            st.markdown(
                "<div class='purpose'>📌 <b>Purpose:</b> Multi-line trend shows income, "
                "expenses, and net savings trajectory — useful for spotting drift or seasonal patterns.</div>",
                unsafe_allow_html=True,
            )
    with c4:
        st.plotly_chart(anomaly_scatter_chart(df), use_container_width=True)
        st.markdown(
            "<div class='purpose'>📌 <b>Purpose:</b> Isolation Forest + Z-score ML model "
            "flags statistically unusual transactions (⚠️ orange ✕). Each point is a transaction.</div>",
            unsafe_allow_html=True,
        )

    # ── Charts row 3 ──────────────────────────────────────────────────────
    c5, c6 = st.columns([1, 2])
    with c5:
        st.plotly_chart(savings_rate_gauge(rate), use_container_width=True)
        st.markdown(
            "<div class='purpose'>📌 <b>Purpose:</b> Savings rate vs the 20 % benchmark. "
            "Green = excellent, Orange = caution, Red = below target.</div>",
            unsafe_allow_html=True,
        )
    with c6:
        if not cat_sum.empty:
            st.plotly_chart(top_categories_bar(cat_sum), use_container_width=True)
            st.markdown(
                "<div class='purpose'>📌 <b>Purpose:</b> Ranked horizontal bars highlight "
                "exact ₹ amounts and share of total spend per category — ideal for budget allocation.</div>",
                unsafe_allow_html=True,
            )

    # ── Heatmap ───────────────────────────────────────────────────────────
    hmap = spending_heatmap(df)
    if hmap.data:
        st.plotly_chart(hmap, use_container_width=True)
        st.markdown(
            "<div class='purpose'>📌 <b>Purpose:</b> Day × week heatmap shows which weekdays "
            "you spend the most — helps with lifestyle budgeting and identifying impulse-spend days.</div>",
            unsafe_allow_html=True,
        )

    # ── Anomaly Detail ────────────────────────────────────────────────────
    if anom_sum["count"] > 0:
        st.markdown(
            f'<div class="sec-header">⚠️ Anomalous Transactions '
            f'({anom_sum["count"]} flagged · ₹{anom_sum["total_value"]:,.0f} total)</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Flagged by a two-stage ML pipeline: "
            "**Z-score** (per-category statistical outliers) + **Isolation Forest** (unsupervised ML). "
            "Higher score = more anomalous. "
            "✅ Check **'Not an Anomaly?'** on any row you disagree with — "
            "it will be removed from all metrics, charts, and the AI chatbot instantly."
        )

        # Build editable anomaly table
        # df_row_idx is the real position in the main df — embedded explicitly
        # in get_anomaly_summary so it can never be lost by pandas index ops
        adf       = anom_sum["anomaly_df"].copy()
        orig_idxs = adf["df_row_idx"].tolist()   # ← 100% reliable real df row positions

        adf_display = pd.DataFrame({
            "Date":             adf["date"].dt.strftime("%d %b %Y"),
            "Category":         adf["category"],
            "Amount (₹)":       adf["abs_amount"].apply(lambda x: f"₹{x:,.0f}"),
            "Anomaly Score":    adf["anomaly_score"].apply(lambda x: f"{x:.3f}"),
            "Detection Method": adf["anomaly_reason"],
            "Not an Anomaly?":  [idx in st.session_state.user_overrides for idx in orig_idxs],
            "_row_idx":         orig_idxs,   # hidden carrier — dropped before display
        })

        edited_anom = st.data_editor(
            adf_display.drop(columns=["_row_idx"]),   # hide carrier col from user
            column_config={
                "Not an Anomaly?": st.column_config.CheckboxColumn(
                    "Not an Anomaly?",
                    help="Check this if you believe this transaction is normal. "
                         "It will be removed from all anomaly counts, charts, and AI reports.",
                    default=False,
                    width="small",
                ),
                "Date":             st.column_config.TextColumn("Date",             width="small"),
                "Category":         st.column_config.TextColumn("Category",         width="medium"),
                "Amount (₹)":       st.column_config.TextColumn("Amount (₹)",       width="small"),
                "Anomaly Score":    st.column_config.TextColumn("Anomaly Score",     width="small"),
                "Detection Method": st.column_config.TextColumn("Detection Method", width="large"),
            },
            disabled=["Date", "Category", "Amount (₹)", "Anomaly Score", "Detection Method"],
            use_container_width=True,
            hide_index=True,
            key="anomaly_editor",
        )

        # ── Sync checkboxes → user_overrides ─────────────────────────────
        # orig_idxs comes from adf_display["_row_idx"] — guaranteed to match
        # the editor rows in order since both were built from the same adf
        new_flags = edited_anom["Not an Anomaly?"].tolist()
        changed   = False
        for orig_idx, is_overridden in zip(orig_idxs, new_flags):
            if is_overridden and orig_idx not in st.session_state.user_overrides:
                st.session_state.user_overrides.add(orig_idx)
                changed = True
            elif not is_overridden and orig_idx in st.session_state.user_overrides:
                st.session_state.user_overrides.discard(orig_idx)
                changed = True

        if changed:
            # rerun → df rebuilt with overrides applied at top of dashboard block
            # → scatter chart, KPI cards, anomaly count all re-render correctly
            st.rerun()

        # ── Override summary banner ───────────────────────────────────────
        n_overridden = len(st.session_state.user_overrides)
        if n_overridden > 0:
            col_msg, col_btn = st.columns([4, 1])
            with col_msg:
                st.info(
                    f"ℹ️ **{n_overridden} transaction(s)** marked as not anomalous by you. "
                    "Excluded from all metrics, charts, and AI reports.",
                )
            with col_btn:
                if st.button("↩️ Reset", key="reset_overrides", use_container_width=True):
                    st.session_state.user_overrides = set()
                    st.rerun()

    # ── AI Insight Generation ─────────────────────────────────────────────
    st.markdown('<div class="sec-header">🤖 AI-Generated Financial Insight Report</div>', unsafe_allow_html=True)
    st.caption(
        f"Powered by **Ollama / {st.session_state.selected_model}** running locally on your machine. "
        "No data leaves your device."
    )

    if st.button("✨ Generate Insight Report", type="primary", use_container_width=False):
        if not alive:
            st.error("Ollama is not running. Start it with `ollama serve` and try again.")
        else:
            with st.spinner(
                f"🧠 Analysing with {st.session_state.selected_model} — this may take 20–60 s…"
            ):
                insight = generate_insight(
                    metrics          = metrics,
                    anomaly_summary  = anom_sum,
                    category_summary = cat_sum,
                    model            = st.session_state.selected_model,
                )
                st.session_state.last_insight = insight

    if st.session_state.last_insight:
        with st.container():
            st.markdown(
                "<div style='background:rgba(255,255,255,0.04);border-radius:12px;"
                "padding:20px 24px;border:1px solid rgba(255,255,255,0.08)'>"
                + st.session_state.last_insight.replace("\n", "<br>")
                + "</div>",
                unsafe_allow_html=True,
            )

    # ── Raw Data Explorer ─────────────────────────────────────────────────
    st.markdown('<div class="sec-header">📋 Raw Transaction Data</div>', unsafe_allow_html=True)

    with st.expander("Show / hide all transactions", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            cats    = ["All"] + sorted(df["category"].unique().tolist())
            sel_cat = st.selectbox("Category filter", cats, key="cat_filter")
        with fc2:
            types    = ["All", "Income", "Expense"]
            sel_type = st.selectbox("Type filter", types, key="type_filter")
        with fc3:
            only_anomaly = st.checkbox("Show anomalies only", key="anom_filter")

        fdf = df.copy()
        if sel_cat   != "All": fdf = fdf[fdf["category"] == sel_cat]
        if sel_type  != "All": fdf = fdf[fdf["transaction_type"] == sel_type]
        if only_anomaly:       fdf = fdf[fdf["is_anomaly"] == True]

        disp = fdf[["date", "category", "abs_amount", "transaction_type", "is_anomaly"]].copy()
        disp.columns = ["Date", "Category", "Amount (₹)", "Type", "Anomaly?"]
        disp["Date"]       = disp["Date"].dt.strftime("%d %b %Y")
        disp["Amount (₹)"] = disp["Amount (₹)"].apply(lambda x: f"₹{x:,.0f}")
        disp["Anomaly?"]   = disp["Anomaly?"].apply(lambda x: "⚠️ Yes" if x else "✅ No")

        st.dataframe(disp, use_container_width=True, height=350, hide_index=True)
        st.caption(f"Showing {len(disp):,} of {len(df):,} transactions")


# ─────────────────────────────────────────────────────────────────────────────
# Landing (no data yet)
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(4)
    features = [
        ("📊", "Instant Metrics",     "Income, expenses, savings rate — calculated the moment you upload."),
        ("🔍", "ML Anomaly Detection", "Isolation Forest + Z-score ML pipeline flags unusual transactions."),
        ("📈", "Trend Analytics",     "Month-over-month charts with purpose labels for every graph."),
        ("🤖", "Local AI Insights",   "Ollama LLM generates a full financial report — 100 % on-device."),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(
                f"""<div class="fin-card" style="padding:24px 16px">
                    <div style="font-size:36px">{icon}</div>
                    <div style="font-weight:700;font-size:15px;margin:12px 0 6px">{title}</div>
                    <div style="font-size:12px;color:#6b7f8a">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "👆 Upload a CSV, Excel, or PDF financial dataset above to get started."
    )