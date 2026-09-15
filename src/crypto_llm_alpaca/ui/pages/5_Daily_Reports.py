from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from crypto_llm_alpaca._paths import find_repo_root
from crypto_llm_alpaca.config import load_config
from crypto_llm_alpaca.env import load_env
from crypto_llm_alpaca.storage import SQLiteStore
from crypto_llm_alpaca.ui._runner import active_config_path, render_result, run_cli

load_env()
st.set_page_config(page_title="Daily Reports", layout="wide")
st.title("Daily Reports")

config_path = active_config_path()
try:
    cfg = load_config(config_path)
except Exception as exc:
    st.error(f"Failed to load config `{config_path}`: {exc}")
    st.stop()
store = SQLiteStore(cfg.storage.path)
store.init_schema()

cols = st.columns(4)
dry = cols[0].checkbox("Dry run", value=True)
max_trades = cols[1].number_input("Max trades", min_value=1, value=3, step=1)
max_positions = cols[2].number_input("Max positions", min_value=1, value=5, step=1)
if cols[3].button("Run autopilot now", type="primary"):
    args = ["autopilot", "--max-trades", str(int(max_trades)), "--max-positions", str(int(max_positions))]
    if dry:
        args.append("--dry-run")
    with st.spinner(f"Running {' '.join(args)} ..."):
        result = run_cli(args, config=config_path, timeout=900)
    render_result(st, result, label="autopilot")

st.markdown("---")

# --- Free-form artifacts in outputs/ (backtest analyses, CSVs, etc.) ---
out_dir = find_repo_root() / "outputs"
md_files = sorted([p for p in out_dir.glob("*.md") if not p.name.startswith(".")], reverse=True)
csv_files = sorted([p for p in out_dir.glob("*.csv")], reverse=True)
if md_files or csv_files:
    st.subheader("Files in `outputs/`")
    if md_files:
        chosen = st.selectbox(
            "Markdown reports", [p.name for p in md_files], key="md_artifact",
        )
        chosen_path = out_dir / chosen
        st.caption(f"`{chosen_path}`")
        st.markdown(chosen_path.read_text(encoding="utf-8"))
    if csv_files:
        st.write("**CSV downloads**")
        for c in csv_files:
            st.download_button(
                label=f"⬇ {c.name}",
                data=c.read_bytes(),
                file_name=c.name,
                mime="text/csv",
                key=f"dl_{c.name}",
            )

st.markdown("---")
st.subheader("Daily autopilot runs (DB)")
reports = store.list_daily_reports()
if not reports:
    st.info("No daily reports yet. Click **Run autopilot now** above (start with Dry run).")
    st.stop()

df = pd.DataFrame(
    [
        {
            "date": r["date"],
            "equity_open": r["equity_open"],
            "equity_close": r["equity_close"],
            "n_decisions": r["n_decisions"],
            "n_orders": r["n_orders"],
            "markdown_path": r["markdown_path"],
            "generated_at": r["generated_at"],
        }
        for r in reports
    ]
)
st.dataframe(df, width="stretch", hide_index=True)

selected_date = st.selectbox("View report for date", df["date"].tolist())
chosen = next((r for r in reports if r["date"] == selected_date), None)
if not chosen:
    st.stop()

md_path = Path(chosen["markdown_path"])
if md_path.exists():
    st.markdown(md_path.read_text(encoding="utf-8"))
else:
    st.warning(f"Markdown file not found: {md_path}")

st.subheader(f"Decisions on {selected_date}")
decs = store.list_decisions(date=selected_date)
if decs:
    rows = [
        {
            "symbol": d["symbol"],
            "score": d["score"],
            "confidence": d["confidence"],
            "accepted": d["accepted"],
            "reason": d["reason"],
            "decided_at": d["decided_at"],
        }
        for d in decs
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
else:
    st.info("No decisions recorded for this date.")
