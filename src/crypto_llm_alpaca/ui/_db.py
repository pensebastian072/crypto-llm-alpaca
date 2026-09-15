from __future__ import annotations

from typing import Any

import streamlit as st

from ..config import load_config
from ..storage import SQLiteStore
from ._runner import active_config_path


@st.cache_data(ttl=10, show_spinner=False)
def get_counts(config_path: str) -> dict[str, int]:
    cfg = load_config(config_path)
    store = SQLiteStore(cfg.storage.path)
    store.init_schema()
    with store.connect() as conn:
        return {
            "bars": conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0],
            "news_items": conn.execute("SELECT COUNT(*) FROM news_items").fetchone()[0],
            "llm_signals": conn.execute("SELECT COUNT(*) FROM llm_signals").fetchone()[0],
            "trade_intents": conn.execute("SELECT COUNT(*) FROM trade_intents").fetchone()[0],
            "strategy_features": conn.execute("SELECT COUNT(*) FROM strategy_features").fetchone()[0],
            "backtest_runs": conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0],
            "backtest_trades": conn.execute("SELECT COUNT(*) FROM backtest_trades").fetchone()[0],
            "backtest_equity_curve": conn.execute("SELECT COUNT(*) FROM backtest_equity_curve").fetchone()[0],
        }


@st.cache_data(ttl=10, show_spinner=False)
def get_backtest_runs(config_path: str) -> list[dict[str, Any]]:
    cfg = load_config(config_path)
    store = SQLiteStore(cfg.storage.path)
    return store.list_backtest_runs()


@st.cache_data(ttl=10, show_spinner=False)
def get_bars(config_path: str, symbol: str):
    cfg = load_config(config_path)
    store = SQLiteStore(cfg.storage.path)
    return store.list_bars(symbol, timeframe=cfg.data.timeframe)


@st.cache_data(ttl=10, show_spinner=False)
def get_trades(config_path: str, run_id: str, symbol: str):
    cfg = load_config(config_path)
    store = SQLiteStore(cfg.storage.path)
    return store.list_backtest_trades(run_id, symbol)


@st.cache_data(ttl=10, show_spinner=False)
def get_equity_curve(config_path: str, run_id: str):
    cfg = load_config(config_path)
    store = SQLiteStore(cfg.storage.path)
    return store.list_equity_curve(run_id)


def get_config(config_path: str | None = None):
    return load_config(config_path or active_config_path())
