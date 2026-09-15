from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from crypto_llm_alpaca._paths import find_repo_root
from crypto_llm_alpaca.env import load_env
from crypto_llm_alpaca.ui._db import get_backtest_runs, get_counts, get_config
from crypto_llm_alpaca.ui._runner import active_config_path

load_env()
st.set_page_config(page_title="crypto-llm-alpaca", layout="wide")
st.title("crypto-llm-alpaca")

config_path = active_config_path()
overrides_active = config_path.endswith("ui_overrides.yaml")
caption = f"Config: `{config_path}`" + ("  (UI overrides active)" if overrides_active else "")
st.caption(caption)

try:
    cfg = get_config(config_path)
except Exception as exc:
    st.error(f"Failed to load config `{config_path}`: {exc}")
    st.stop()

st.subheader("Database")
try:
    counts = get_counts(config_path)
except Exception as exc:
    st.error(f"Failed to read DB: {exc}")
    st.stop()
cols = st.columns(4)
items = list(counts.items())
for i, (k, v) in enumerate(items):
    cols[i % 4].metric(k, f"{v:,}")

st.subheader("Environment")
env_cols = st.columns(4)
checks = [
    (".env file", (find_repo_root() / ".env").exists()),
    ("APCA_API_KEY_ID", bool(os.getenv("APCA_API_KEY_ID"))),
    ("APCA_API_SECRET_KEY", bool(os.getenv("APCA_API_SECRET_KEY"))),
    ("HUGGINGFACE_API_TOKEN", bool(os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN"))),
]
for i, (label, ok) in enumerate(checks):
    env_cols[i].markdown(f"**{label}**\n\n{'✅ set' if ok else '⚠️ missing'}")

st.subheader("Latest backtest runs")
runs = get_backtest_runs(config_path)
if not runs:
    st.info("No backtest runs yet. Run: `crypto-llm backtest --strategy adaptive-breakout --persist`")
else:
    rows = []
    for r in runs[:5]:
        m = r["metrics"]
        rows.append(
            {
                "run_id": r["run_id"],
                "strategy": r["strategy"],
                "return_pct": m.get("return_pct", 0.0),
                "max_drawdown_pct": m.get("max_drawdown_pct", 0.0),
                "profit_factor": m.get("profit_factor", 0.0),
                "entries": m.get("entries", 0),
                "completed_at": r["completed_at"],
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)

st.markdown("---")
st.markdown(
    "**Research**\n"
    "- **Strategy Lab** — view backtest runs, trigger backtests/sweeps\n"
    "- **Parameters** — edit strategy/risk params (writes `ui_overrides.yaml`)\n"
    "- **Doctor** — preflight checks for env, keys, DB\n"
    "- **News & Signals** — ingest news, extract LLM signals\n\n"
    "**Live (paper)**\n"
    "- **Daily Reports** — autopilot runs + markdown summaries; trigger new runs\n"
    "- **Live Positions** — Alpaca paper account, positions, open orders\n"
    "- **Decisions Audit** — every decision the bot made, by date"
)
