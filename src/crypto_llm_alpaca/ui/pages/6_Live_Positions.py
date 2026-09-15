from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from crypto_llm_alpaca.broker import AlpacaPaperBroker
from crypto_llm_alpaca.config import load_config
from crypto_llm_alpaca.env import load_env
from crypto_llm_alpaca.ui._runner import active_config_path

load_env()
st.set_page_config(page_title="Live Positions", layout="wide")
st.title("Live Positions (Alpaca paper)")

config_path = active_config_path()
try:
    cfg = load_config(config_path)
except Exception as exc:
    st.error(f"Failed to load config `{config_path}`: {exc}")
    st.stop()

if st.button("Refresh"):
    st.cache_data.clear()
    st.rerun()

try:
    broker = AlpacaPaperBroker(cfg.broker)
    account = broker.get_account()
    positions = broker.get_positions()
    open_orders = broker.get_orders(status="open")
except Exception as exc:
    st.error(f"Broker error: {exc}")
    st.stop()

mc = st.columns(2)
mc[0].metric("Equity", f"${account.equity:,.2f}")
mc[1].metric("Buying power", f"${account.buying_power:,.2f}")

st.subheader(f"Open positions ({len(positions)})")
if positions:
    st.dataframe(pd.DataFrame([asdict(p) for p in positions]), width="stretch", hide_index=True)
else:
    st.info("No open positions.")

st.subheader(f"Open orders ({len(open_orders)})")
if open_orders:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "broker_order_id": o.broker_order_id,
                    "symbol": o.symbol,
                    "side": o.side,
                    "qty": o.qty,
                    "status": o.status,
                    "type": o.order_type,
                    "submitted_at": o.submitted_at,
                    "filled_avg_price": o.filled_avg_price,
                    "legs": ",".join(o.legs) if o.legs else "",
                }
                for o in open_orders
            ]
        ),
        width="stretch",
        hide_index=True,
    )
else:
    st.info("No open orders.")
