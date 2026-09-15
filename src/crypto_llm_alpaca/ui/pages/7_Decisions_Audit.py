from __future__ import annotations

import json
import sys
from datetime import date as _date, timedelta
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from crypto_llm_alpaca.config import load_config
from crypto_llm_alpaca.env import load_env
from crypto_llm_alpaca.storage import SQLiteStore
from crypto_llm_alpaca.ui._runner import active_config_path

load_env()
st.set_page_config(page_title="Decisions Audit", layout="wide")
st.title("Decisions Audit")

config_path = active_config_path()
try:
    cfg = load_config(config_path)
except Exception as exc:
    st.error(f"Failed to load config `{config_path}`: {exc}")
    st.stop()
store = SQLiteStore(cfg.storage.path)
store.init_schema()

cols = st.columns(2)
picked = cols[0].date_input("Date", value=_date.today())
show_rejected = cols[1].checkbox("Include rejected", value=True)

iso = picked.isoformat() if isinstance(picked, _date) else str(picked)
decisions = store.list_decisions(date=iso)

if not show_rejected:
    decisions = [d for d in decisions if d["accepted"]]

st.write(f"**{len(decisions)} decisions on {iso}**")

if not decisions:
    st.info("No decisions for that date. Try a recent autopilot run date.")
    st.stop()

rows = [
    {
        "run_id": d["run_id"],
        "symbol": d["symbol"],
        "accepted": d["accepted"],
        "score": round(d["score"], 4),
        "confidence": round(d["confidence"], 3),
        "rank": d["intent"].get("rank"),
        "notional": d["intent"].get("notional"),
        "submitted_order_id": d["intent"].get("submitted_order_id"),
        "take_profit_price": d["intent"].get("take_profit_price"),
        "stop_loss_price": d["intent"].get("stop_loss_price"),
        "reason": d["reason"],
        "decided_at": d["decided_at"],
    }
    for d in decisions
]
df = pd.DataFrame(rows)
st.dataframe(df, width="stretch", hide_index=True)

st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"decisions-{iso}.csv",
    mime="text/csv",
)

with st.expander("Raw intents (JSON)"):
    st.code(json.dumps([d["intent"] for d in decisions], indent=2, default=str), language="json")
