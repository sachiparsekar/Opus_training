"""
file_parser.py
Handles ingestion of CSV, Excel (.xlsx/.xls), and PDF files.
Standardizes all inputs into a uniform DataFrame schema.
"""

import pandas as pd
import pdfplumber
import re
import io


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_file(uploaded_file) -> pd.DataFrame:
    """
    Detect file type and delegate to the appropriate parser.
    Returns a standardised DataFrame with columns:
        date | category | amount | transaction_type
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return _parse_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return _parse_excel(uploaded_file)
    elif name.endswith(".pdf"):
        return _parse_pdf(uploaded_file)
    else:
        raise ValueError(
            f"Unsupported file type: '{uploaded_file.name}'. "
            "Please upload a CSV, Excel, or PDF file."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return _standardize(df)


def _parse_excel(file) -> pd.DataFrame:
    # Try every sheet and use the first one that looks like financial data
    xls = pd.ExcelFile(file)
    best_df = None
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
            df = _standardize(df)
            if len(df) > 0:
                best_df = df
                break
        except Exception:
            continue
    if best_df is None:
        raise ValueError("No valid financial table found in the Excel file.")
    return best_df


def _parse_pdf(file) -> pd.DataFrame:
    """
    Extract financial tables from a PDF using pdfplumber.

    Strategy:
      1. Try extracting structured tables from every page first —
         works great for bank statements exported as PDFs.
      2. If no tables found, fall back to raw text parsing using
         regex to pick out date + amount patterns line-by-line.
    """
    raw_bytes = file.read()

    # ── Pass 1: structured table extraction ──────────────────────────────
    all_rows = []
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                header = [str(c).strip() if c else f"col_{i}"
                          for i, c in enumerate(table[0])]
                for row in table[1:]:
                    if row:
                        all_rows.append(dict(zip(header, row)))

    if all_rows:
        df = pd.DataFrame(all_rows)
        df = df.replace("", pd.NA).dropna(how="all", axis=1)
        return _standardize(df)

    # ── Pass 2: regex text fallback ───────────────────────────────────────
    # Matches lines like: "14 Aug 2023   Rent   -20,000"
    date_pat   = re.compile(
        r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}"
        r"|\d{4}[\/\-]\d{2}[\/\-]\d{2}"
        r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4})",
        re.IGNORECASE,
    )
    amount_pat = re.compile(r"[₹$€£]?\s*-?[\d,]+(?:\.\d{1,2})?")

    rows = []
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                line        = line.strip()
                date_match  = date_pat.search(line)
                amt_match   = amount_pat.search(line)
                if date_match and amt_match:
                    cat_raw  = line[date_match.end():amt_match.start()].strip()
                    category = cat_raw or "General"
                    rows.append({
                        "date":     date_match.group(),
                        "category": category,
                        "amount":   amt_match.group()
                                    .replace(",", "").replace("₹", "").strip(),
                    })

    if not rows:
        raise ValueError(
            "No financial data found in the PDF. "
            "The PDF must contain tables or clearly formatted transaction lines "
            "(e.g. a bank statement)."
        )

    return _standardize(pd.DataFrame(rows))


# ─────────────────────────────────────────────────────────────────────────────
# Standardisation helpers
# ─────────────────────────────────────────────────────────────────────────────

# ── Column-name patterns → canonical name ────────────────────────────────────
# Flat reverse map built once at import time: pattern → canonical_name.
# This reduces the O(targets × columns × patterns) triple loop in the old code
# to a single O(columns × total_patterns) pass — much faster on wide DataFrames.
_COL_MAP = {
    "date":             ["date", "trans_date", "transaction_date", "value_date",
                         "booking_date", "posting_date", "day",
                         "date_time", "date__time", "date___time", "datetime"],
    "category":         ["category", "cat", "description", "desc", "narration",
                         "particulars", "type_of_expense", "expense_type",
                         "merchant", "merchant_name", "sub_category", "subcategory"],
    "amount":           ["amount", "value", "sum", "debit", "credit",
                         "transaction_amount", "trans_amount",
                         "debit_credit", "dr_cr_amount"],
    "transaction_type": ["transaction_type", "trans_type", "txn_type",
                         "type", "transaction_kind", "dr_cr",
                         "income_expense", "income_or_expense"],
}

# Reverse map: pattern substring → canonical target (built once, reused on every parse)
_PATTERN_TO_TARGET: dict[str, str] = {
    pattern: target
    for target, patterns in _COL_MAP.items()
    for pattern in patterns
}

# Pre-compiled per-target regex — matches any recognised pattern for that target.
# Using word-boundary anchors so "category_id" doesn't match "category".
_TARGET_REGEX: dict[str, re.Pattern] = {
    target: re.compile("|".join(re.escape(p) for p in patterns))
    for target, patterns in _COL_MAP.items()
}


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename, type-cast, and clean raw DataFrame into the canonical schema."""

    # 1. Normalize column names — collapse repeated underscores so
    #    "Date / Time" → "date___time" → "date_time"
    import re as _re
    df.columns = [
        _re.sub(r"_+", "_", str(c).strip().lower()
                .replace(" ", "_").replace("-", "_").replace("/", "_"))
        for c in df.columns
    ]

    # 2. Drop completely empty columns / rows
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)

    # 3. Map to canonical names using pre-compiled regex (O(columns × targets))
    #    For each column we check each target's regex — first match wins.
    rename = {}
    claimed_targets: set[str] = set()
    for col in df.columns:
        if col in rename:
            continue
        for target, rx in _TARGET_REGEX.items():
            if target in claimed_targets:
                continue
            if rx.search(col):          # regex match on normalized col name
                rename[col] = target
                claimed_targets.add(target)
                break
    df = df.rename(columns=rename)

    # 4. Validate required columns
    for req in ("date", "amount"):
        if req not in df.columns:
            raise ValueError(
                f"Required column '{req}' not found after auto-mapping. "
                f"Columns detected: {list(df.columns)}"
            )

    # 5. If 'category' is missing, create a fallback
    if "category" not in df.columns:
        df["category"] = "General"

    # 6. Parse dates
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, errors="coerce")
    df = df.dropna(subset=["date"])

    # 7. Parse amounts — strip currency symbols / commas
    if df["amount"].dtype == object:
        df["amount"] = (
            df["amount"]
            .astype(str)
            .str.replace(r"[₹$€£,\s]", "", regex=True)
            .str.replace(r"\((.+)\)", r"-\1", regex=True)   # (1,000) → -1000
        )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

    # 8. Infer / normalise transaction_type
    if "transaction_type" not in df.columns:
        df["transaction_type"] = df["amount"].apply(
            lambda x: "Income" if x >= 0 else "Expense"
        )
    else:
        # Normalize: Dr/Debit → Expense, Cr/Credit → Income
        def _normalise_type(val):
            v = str(val).strip().lower()
            if v in ("income", "cr", "credit", "salary", "receipt", "inflow"):
                return "Income"
            return "Expense"
        df["transaction_type"] = df["transaction_type"].apply(_normalise_type)

    # 9. Sort and reset index
    df = df.sort_values("date").reset_index(drop=True)

    return df[["date", "category", "amount", "transaction_type"]]