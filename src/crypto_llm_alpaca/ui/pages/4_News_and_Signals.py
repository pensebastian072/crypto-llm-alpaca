from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from crypto_llm_alpaca.config import load_config
from crypto_llm_alpaca.env import load_env
from crypto_llm_alpaca.storage import SQLiteStore
from crypto_llm_alpaca.ui._runner import active_config_path, render_result, run_cli

load_env()
st.set_page_config(page_title="News & Signals", layout="wide")
st.title("News & Signals")

config_path = active_config_path()
try:
    cfg = load_config(config_path)
except Exception as exc:
    st.error(f"Failed to load config `{config_path}`: {exc}")
    st.stop()
store = SQLiteStore(cfg.storage.path)
store.init_schema()

cols = st.columns(3)
if cols[0].button("Ingest news", type="primary"):
    with st.spinner("Fetching RSS feeds ..."):
        result = run_cli(["ingest-news"], config=config_path, timeout=120)
    render_result(st, result, label="ingest-news")

limit = cols[1].number_input("Extract limit", min_value=1, value=50, step=1)
if cols[2].button("Extract signals"):
    with st.spinner(f"Extracting signals (provider={cfg.llm.provider}) ..."):
        result = run_cli(["extract-signals", "--limit", str(int(limit))], config=config_path, timeout=600)
    render_result(st, result, label="extract-signals")

st.markdown("---")
st.subheader("Latest news (50)")
with store.connect() as conn:
    news_rows = conn.execute(
        "SELECT dedupe_hash, source, title, published_at, symbols_json FROM news_items "
        "ORDER BY published_at DESC LIMIT 50"
    ).fetchall()
    sig_rows = conn.execute(
        "SELECT s.id, s.news_hash, s.symbol, s.sentiment, s.confidence, s.horizon, "
        "s.event_type, s.provider, s.model, s.created_at, n.title "
        "FROM llm_signals s LEFT JOIN news_items n ON n.dedupe_hash = s.news_hash "
        "ORDER BY s.created_at DESC LIMIT 100"
    ).fetchall()

if news_rows:
    st.dataframe(
        pd.DataFrame([dict(r) for r in news_rows]),
        width="stretch",
        hide_index=True,
    )
else:
    st.info("No news yet. Click **Ingest news**.")

st.subheader("Latest LLM signals (100)")
if sig_rows:
    st.dataframe(
        pd.DataFrame([dict(r) for r in sig_rows]),
        width="stretch",
        hide_index=True,
    )
else:
    st.info("No signals yet. Click **Extract signals** after ingesting.")
