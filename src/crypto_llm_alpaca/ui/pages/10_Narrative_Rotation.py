from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crypto_llm_alpaca.config import load_config
from crypto_llm_alpaca.env import load_env
from crypto_llm_alpaca.research.narratives import aggregate_sector_rotation
from crypto_llm_alpaca.storage import SQLiteStore
from crypto_llm_alpaca.ui._runner import active_config_path, render_result, run_cli

load_env()
st.set_page_config(page_title="Narrative Rotation", layout="wide")
st.title("Narrative Rotation")
st.caption("Which crypto sectors are gaining vs losing news attention.")

config_path = active_config_path()
try:
    cfg = load_config(config_path)
except Exception as exc:
    st.error(f"Failed to load config `{config_path}`: {exc}")
    st.stop()

store = SQLiteStore(cfg.storage.path)
store.init_schema()

c1, c2, c3, c4 = st.columns(4)
current_days = c1.number_input("Current window (days)", min_value=1, value=7, step=1)
previous_days = c2.number_input("Prior window (days)", min_value=1, value=7, step=1)
news_limit = c3.number_input("Max news fetched", min_value=100, value=2000, step=100)
if c4.button("Ingest fresh news", type="primary"):
    result = run_cli(["ingest-news"], config=config_path, timeout=120)
    render_result(st, result, label="ingest-news")


@st.cache_data(ttl=120, show_spinner="Computing rotation…")
def _rotations(config_path: str, current_days: int, previous_days: int, news_limit: int):
    cfg2 = load_config(config_path)
    store2 = SQLiteStore(cfg2.storage.path)
    since = datetime.now(tz=timezone.utc) - timedelta(days=current_days + previous_days + 1)
    items = store2.list_news_items(limit=news_limit, since=since)
    rots = aggregate_sector_rotation(items, current_days=current_days, previous_days=previous_days)
    return [asdict(r) for r in rots], len(items)


rotations, n_items = _rotations(config_path, int(current_days), int(previous_days), int(news_limit))
st.caption(f"News in window: {n_items}")

if not rotations:
    st.info(
        "No narratives detected in the news window. Try **Ingest fresh news** above, "
        "or extend the windows."
    )
    st.stop()

# ----- Bar chart current vs previous -----
labels = [r["label"] for r in rotations]
fig = go.Figure()
fig.add_trace(go.Bar(name="Current", x=labels, y=[r["current_score"] for r in rotations], marker_color="#16a34a"))
fig.add_trace(go.Bar(name="Prior", x=labels, y=[r["previous_score"] for r in rotations], marker_color="#94a3b8"))
fig.update_layout(barmode="group", height=420, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                  yaxis={"title": "narrative score"})
st.plotly_chart(fig, width="stretch")

# ----- Momentum table -----
st.subheader("Sector momentum")
rows = []
for r in rotations:
    rows.append({
        "sector": r["label"],
        "current": round(r["current_score"], 3),
        "prior": round(r["previous_score"], 3),
        "Δ momentum": round(r["momentum"], 3),
        "n_curr": r["current_articles"],
        "n_prev": r["previous_articles"],
        "top_symbols": ", ".join(r["top_symbols"][:5]),
    })
df = pd.DataFrame(rows)
st.dataframe(df, width="stretch", hide_index=True)

# ----- Drill-down -----
st.subheader("Drill-down")
selected = st.selectbox("Sector", [r["label"] for r in rotations])
chosen = next((r for r in rotations if r["label"] == selected), None)
if chosen:
    mc = st.columns(4)
    mc[0].metric("Current score", f"{chosen['current_score']:.3f}")
    mc[1].metric("Prior score", f"{chosen['previous_score']:.3f}")
    mc[2].metric("Momentum", f"{chosen['momentum']:+.3f}")
    mc[3].metric("Articles (curr)", f"{chosen['current_articles']}")
    if chosen["top_symbols"]:
        st.write("**Top symbols (current window):** " + ", ".join(chosen["top_symbols"]))
