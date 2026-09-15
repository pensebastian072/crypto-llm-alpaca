from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from crypto_llm_alpaca.config import load_config
from crypto_llm_alpaca.env import load_env
from crypto_llm_alpaca.features import compute_features
from crypto_llm_alpaca.scoring import score_with_vector
from crypto_llm_alpaca.sentiment_aggregator import build_sentiment_vector
from crypto_llm_alpaca.storage import SQLiteStore
from crypto_llm_alpaca.ui._runner import active_config_path

load_env()
st.set_page_config(page_title="Sentiment Inspector", layout="wide")
st.title("Sentiment Inspector")

config_path = active_config_path()
try:
    cfg = load_config(config_path)
except Exception as exc:
    st.error(f"Failed to load config `{config_path}`: {exc}")
    st.stop()
store = SQLiteStore(cfg.storage.path)
store.init_schema()

cols = st.columns(3)
picked_date = cols[0].date_input("Date", value=_date.today())
universe = list(cfg.universe.symbols) + [s for s in cfg.per_symbol.keys() if s not in cfg.universe.symbols]
symbol = cols[1].selectbox("Symbol", universe)
notional = cols[2].number_input("Notional ($)", min_value=100.0, value=1000.0, step=100.0)

asof = datetime.combine(picked_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=23, minutes=59)

signals = store.latest_signals(symbol, limit=200)
signals = [s for s in signals if s.created_at <= asof and s.created_at >= asof - timedelta(days=2)]

st.subheader(f"Signals (last 2 days, asof {asof.isoformat()})")
if signals:
    sig_rows = [
        {
            "created_at": s.created_at,
            "sentiment": s.sentiment, "confidence": s.confidence,
            "fear": s.fear, "hype": s.hype, "topic": s.topic,
            "source_weight": s.source_weight, "provider": s.provider,
            "horizon": s.horizon, "event_type": s.event_type,
        }
        for s in signals
    ]
    st.dataframe(pd.DataFrame(sig_rows), width="stretch", hide_index=True)
else:
    st.info("No signals found in this window. Run **News & Signals → Extract signals** first.")

st.markdown("---")
st.subheader("Sentiment vector")
vec = build_sentiment_vector(signals, asof=asof, min_confidence=cfg.llm.min_confidence)
mc = st.columns(4)
mc[0].metric("Polarity", f"{vec.polarity:+.3f}")
mc[1].metric("Source-weighted", f"{vec.source_weighted_polarity:+.3f}")
mc[2].metric("Fear", f"{vec.fear:.2f}")
mc[3].metric("Hype", f"{vec.hype:.2f}")
mc2 = st.columns(4)
mc2[0].metric("Lag 1h", f"{vec.lag_1h:+.3f}")
mc2[1].metric("Lag 6h", f"{vec.lag_6h:+.3f}")
mc2[2].metric("Lag 24h", f"{vec.lag_24h:+.3f}")
mc2[3].metric("Agreement", f"{vec.agreement:.2f}")
st.write(f"**Sources:** {vec.n_sources}  •  **Topics:** {', '.join(vec.topic_flags.keys()) or '—'}")

st.markdown("---")
st.subheader("Score breakdown")
bars = store.latest_bars(symbol, limit=cfg.data.lookback_bars)
if not bars:
    st.warning(f"No bars stored for {symbol}. Run `crypto-llm backfill` first.")
    st.stop()
features = compute_features(bars)
intent = score_with_vector(features, vec, cfg.scoring, notional=float(notional))

s_cols = st.columns(3)
s_cols[0].metric("Score", f"{intent.score:+.3f}", delta=f"vs threshold {cfg.scoring.buy_threshold:+.2f}")
s_cols[1].metric("Decision", intent.side.value.upper())
s_cols[2].metric("Confidence", f"{intent.confidence:.2f}")
st.code(intent.reason, language="text")

with st.expander("Bars window used"):
    bdf = pd.DataFrame([{
        "ts": b.timestamp, "open": b.open, "high": b.high, "low": b.low, "close": b.close, "vol": b.volume
    } for b in bars[-30:]])
    st.dataframe(bdf, width="stretch", hide_index=True)
