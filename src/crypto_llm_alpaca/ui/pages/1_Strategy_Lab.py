from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crypto_llm_alpaca.env import load_env
from crypto_llm_alpaca.ui._db import (
    get_backtest_runs,
    get_bars,
    get_config,
    get_equity_curve,
    get_trades,
)
from crypto_llm_alpaca.ui._runner import active_config_path, render_result, run_cli

load_env()
st.set_page_config(page_title="Strategy Lab", layout="wide")
st.title("Strategy Lab")

config_path = active_config_path()
try:
    cfg = get_config(config_path)
except Exception as exc:
    st.error(f"Failed to load config `{config_path}`: {exc}")
    st.stop()

with st.expander("Run a new backtest", expanded=False):
    with st.form("backtest_form"):
        cols = st.columns(5)
        strategy = cols[0].selectbox(
            "Strategy",
            ("quality-growth", "adaptive-breakout", "breakout-volume", "demo"),
        )
        equity = cols[1].number_input("Starting equity", min_value=100.0, value=10_000.0, step=100.0)
        demo = cols[2].checkbox("Use demo bars", value=False)
        persist = cols[3].checkbox("Persist run", value=True)
        all_bars = cols[4].checkbox("All stored bars", value=False)
        submitted = st.form_submit_button("Run backtest")
    if submitted:
        args = ["backtest", "--strategy", strategy, "--equity", str(equity)]
        if demo:
            args.append("--demo")
        if all_bars:
            args.append("--all-bars")
            args.append("--summary")
        if persist:
            args.append("--persist")
        with st.spinner(f"Running {' '.join(args)} ..."):
            result = run_cli(args, config=config_path, timeout=600)
        render_result(st, result, label="backtest")

with st.expander("Run focused parameter sweep", expanded=False):
    with st.form("sweep_form"):
        cols = st.columns(3)
        sw_equity = cols[0].number_input("Starting equity", min_value=100.0, value=10_000.0, step=100.0, key="swp_eq")
        sw_top = cols[1].number_input("Top N", min_value=1, value=10, step=1)
        sw_demo = cols[2].checkbox("Use demo bars", value=False, key="swp_demo")
        sw_submit = st.form_submit_button("Run sweep")
    if sw_submit:
        args = ["sweep", "--equity", str(sw_equity), "--top", str(sw_top)]
        if sw_demo:
            args.append("--demo")
        with st.spinner(f"Running {' '.join(args)} ..."):
            result = run_cli(args, config=config_path, timeout=900)
        render_result(st, result, label="sweep")

st.subheader("Backtest runs")
runs = get_backtest_runs(config_path)
if not runs:
    st.info("No persisted runs yet. Use the form above with **Persist run** checked.")
    st.stop()

run_rows = []
for run in runs:
    m = run["metrics"]
    run_rows.append(
        {
            "run_id": run["run_id"],
            "strategy": run["strategy"],
            "return_pct": m.get("return_pct", 0.0),
            "max_drawdown_pct": m.get("max_drawdown_pct", 0.0),
            "profit_factor": m.get("profit_factor", 0.0),
            "win_rate": m.get("win_rate", 0.0),
            "entries": m.get("entries", 0),
            "trades": m.get("trades", 0),
            "fees": m.get("total_fees", 0.0),
            "rank_score": m.get("assumptions", {}).get("rank_score", 0.0),
            "completed_at": run["completed_at"],
        }
    )
runs_df = pd.DataFrame(run_rows).sort_values("rank_score", ascending=False)
st.dataframe(runs_df, width="stretch", hide_index=True)

run_labels = [f"{r['run_id']} | {r['strategy']} | {r['completed_at']}" for r in runs]
selected_label = st.sidebar.selectbox("Backtest run", run_labels)
selected_run = runs[run_labels.index(selected_label)]
selected_symbol = st.sidebar.selectbox("Symbol", cfg.universe.symbols)

with st.sidebar.expander("Run config snapshot", expanded=False):
    st.json(selected_run["config"])

metrics = selected_run["metrics"]
mc = st.columns(6)
mc[0].metric("Return", f"{metrics.get('return_pct', 0.0) * 100:.2f}%")
mc[1].metric("Buy & Hold", f"{metrics.get('buy_hold_return_pct', 0.0) * 100:.2f}%")
mc[2].metric("Max DD", f"{metrics.get('max_drawdown_pct', 0.0) * 100:.2f}%")
mc[3].metric("Win Rate", f"{metrics.get('win_rate', 0.0) * 100:.2f}%")
mc[4].metric("Profit Factor", f"{metrics.get('profit_factor', 0.0):.2f}")
mc[5].metric("Fees", f"${metrics.get('total_fees', 0.0):,.2f}")

bars = get_bars(config_path, selected_symbol)
if not bars:
    st.warning(f"No bars stored for {selected_symbol}. Run `crypto-llm backfill --demo` first.")
    st.stop()

trades = get_trades(config_path, selected_run["run_id"], selected_symbol)
bar_df = pd.DataFrame([b.__dict__ for b in bars])

fig = go.Figure(
    data=[
        go.Candlestick(
            x=bar_df["timestamp"],
            open=bar_df["open"],
            high=bar_df["high"],
            low=bar_df["low"],
            close=bar_df["close"],
            name=selected_symbol,
        )
    ]
)
if trades:
    td = pd.DataFrame(trades)
    buys = td[td["side"] == "buy"]
    sells = td[td["side"] == "sell"]
    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(buys["timestamp"]),
                y=buys["price"],
                mode="markers",
                marker={"symbol": "triangle-up", "size": 11, "color": "#16a34a"},
                name="Buy",
                text=buys["reason"],
            )
        )
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(sells["timestamp"]),
                y=sells["price"],
                mode="markers",
                marker={"symbol": "triangle-down", "size": 11, "color": "#dc2626"},
                name="Sell",
                text=sells["reason"],
            )
        )
fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin={"l": 10, "r": 10, "t": 30, "b": 10})
st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})

left, right = st.columns([2, 1])
with left:
    st.subheader("Trades")
    if trades:
        st.dataframe(pd.DataFrame(trades), width="stretch", hide_index=True)
    else:
        st.info("No trades for this symbol/run.")
with right:
    st.subheader("Open positions at end")
    open_pos = [p for p in metrics.get("open_positions", []) if p.get("symbol") == selected_symbol]
    if open_pos:
        st.dataframe(pd.DataFrame(open_pos), width="stretch", hide_index=True)
    else:
        st.info("None.")

per_symbol = metrics.get("per_symbol") or {}
if per_symbol:
    st.subheader("Per-symbol attribution")
    rows = []
    for sym, stats in per_symbol.items():
        rows.append({
            "symbol": sym,
            "trades": stats.get("trades", 0),
            "tp": stats.get("tp", 0),
            "sl": stats.get("sl", 0),
            "expired": stats.get("expired", 0),
            "win_rate": stats.get("win_rate", 0.0),
            "pnl": stats.get("pnl", 0.0),
        })
    rows.sort(key=lambda r: r["pnl"], reverse=True)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

curve = get_equity_curve(config_path, selected_run["run_id"])
if curve:
    cdf = pd.DataFrame(curve)
    cf = go.Figure()
    cf.add_trace(go.Scatter(x=pd.to_datetime(cdf["timestamp"]), y=cdf["equity"], name="Equity"))
    cf.add_trace(
        go.Scatter(
            x=pd.to_datetime(cdf["timestamp"]),
            y=cdf["drawdown"],
            name="Drawdown",
            yaxis="y2",
        )
    )
    cf.update_layout(
        height=300,
        yaxis={"title": "Equity"},
        yaxis2={"title": "Drawdown", "overlaying": "y", "side": "right", "tickformat": ".0%"},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    st.subheader("Equity curve")
    st.plotly_chart(cf, width="stretch", config={"scrollZoom": True})
