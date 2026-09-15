from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from crypto_llm_alpaca.config import load_config
from crypto_llm_alpaca.env import load_env
from crypto_llm_alpaca.research_candidates import aggregate_by_category, compute_candidates
from crypto_llm_alpaca.storage import SQLiteStore
from crypto_llm_alpaca.ui._runner import active_config_path, render_result, run_cli

load_env()
st.set_page_config(page_title="Research Dashboard", layout="wide")
st.title("Research Dashboard — Quality-Growth Watchlist")

config_path = active_config_path()
try:
    cfg = load_config(config_path)
except Exception as exc:
    st.error(f"Failed to load config `{config_path}`: {exc}")
    st.stop()

if not cfg.project_profiles:
    st.warning(
        "No `project_profiles` configured in this config. "
        "Switch to a quality-growth config (e.g. `config/quality_growth.yaml`) "
        "via the sidebar `CRYPTO_LLM_CONFIG` env var or copy that file to `ui_overrides.yaml`."
    )
    st.stop()

store = SQLiteStore(cfg.storage.path)
store.init_schema()

# ---------------- Controls ----------------

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
limit = c1.number_input("Top N", min_value=1, value=min(10, len(cfg.universe.symbols)), step=1)
signal_limit = c2.number_input("Signals/symbol", min_value=10, value=50, step=10)
news_limit = c3.number_input("Narrative news/symbol", min_value=20, value=100, step=20)
refresh = c4.button("Recompute candidates", type="primary")

with st.expander("Refresh data sources before recomputing", expanded=False):
    cols = st.columns(3)
    if cols[0].button("Ingest news"):
        result = run_cli(["ingest-news"], config=config_path, timeout=120)
        render_result(st, result, label="ingest-news")
    if cols[1].button("Extract signals"):
        result = run_cli(["extract-signals", "--limit", "50"], config=config_path, timeout=600)
        render_result(st, result, label="extract-signals")
    if cols[2].button("Backfill 1y"):
        result = run_cli(["backfill", "--days", "365"], config=config_path, timeout=900)
        render_result(st, result, label="backfill")

if refresh:
    st.cache_data.clear()


@st.cache_data(ttl=300, show_spinner="Computing candidates…")
def _candidates(config_path: str, limit: int, signal_limit: int, news_limit: int):
    cfg2 = load_config(config_path)
    store2 = SQLiteStore(cfg2.storage.path)
    return compute_candidates(
        cfg2, store2,
        limit=int(limit), signal_limit=int(signal_limit), narrative_news_limit=int(news_limit),
    )


rows = _candidates(config_path, int(limit), int(signal_limit), int(news_limit))

if not rows:
    st.info(
        "No candidates produced. Likely causes:\n"
        "- No bars stored for any profiled symbol → run **Backfill 1y**\n"
        "- No project profiles match the active universe\n"
        "- DefiLlama / narrative pipelines disabled in config\n"
    )
    st.stop()

# ---------------- Watchlist table ----------------

st.subheader(f"Top {len(rows)} ranked")
table_rows = []
for i, r in enumerate(rows, start=1):
    fs = r["fund_style"]
    defi = r.get("defi") or {}
    narr = r.get("narrative_profile") or {}
    table_rows.append({
        "#": i,
        "symbol": r["symbol"],
        "category": r.get("category", ""),
        "passes": "✅" if r["passes"] else "❌",
        "research": round(fs.get("research_score", 0.0), 3),
        "tradable": round(fs.get("tradable_score", 0.0), 3),
        "exec_score": round(r["score"], 3),
        "tvl": defi.get("tvl"),
        "tvl_7d": defi.get("tvl_growth"),
        "fees_7d": defi.get("fees_7d"),
        "narrative": narr.get("label") or "",
        "narr_mom": round(narr.get("momentum", 0.0), 3) if narr else None,
        "signals_7d": r.get("signals_7d", 0),
        "sentiment": round(r["sentiment"]["source_weighted_polarity"], 3),
        "rejects": ", ".join((r.get("reject_reasons") or [])[:2]),
    })
st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

# ---------------- Sector heatmap ----------------

st.subheader("Sector roll-up")
cats = aggregate_by_category(rows)
if cats:
    st.dataframe(pd.DataFrame(cats), width="stretch", hide_index=True)

# ---------------- Per-symbol drilldown ----------------

st.subheader("Per-symbol drilldown")
for r in rows:
    title_status = "✅ tradable" if r["passes"] else "🔍 watchlist"
    fs = r["fund_style"]
    header = (
        f"**{r['symbol']}** · {r.get('category', '')} "
        f"· research={fs.get('research_score', 0.0):.3f} "
        f"· tradable={fs.get('tradable_score', 0.0):.3f} "
        f"· {title_status}"
    )
    with st.expander(header, expanded=False):
        explain = r["explanation"]
        st.markdown(f"**Thesis:** {explain['thesis']}")
        if r.get("thesis"):
            st.caption(r["thesis"])

        # Score breakdown
        score_cols = st.columns(5)
        score_cols[0].metric("Profile", f"{r['profile_score']:.3f}")
        score_cols[1].metric("Momentum", f"{r['momentum_score']:.3f}")
        score_cols[2].metric("Volume", f"{r['volume_score']:.3f}")
        score_cols[3].metric("Trend", f"{r['trend_score']:.3f}")
        score_cols[4].metric("Sentiment", f"{r['sentiment_score']:.3f}")

        # Gates
        st.markdown("**Gates**")
        for check in explain["fundamentals"]["checks"]:
            st.write(f"- {check}")
        for check in explain["market_structure"]["checks"]:
            st.write(f"- {check}")

        # Multi-timeframe
        mtf_cols = st.columns(4)
        mtf_cols[0].metric("4H trend", f"{r['trend_4h']:+.2%}")
        mtf_cols[1].metric("1D trend", f"{r['trend_1d']:+.2%}")
        mtf_cols[2].metric("1W trend", f"{r['trend_1w']:+.2%}")
        mtf_cols[3].metric("MTF score", f"{r['mtf_trend_score']:.3f}")

        # DefiLlama
        defi = r.get("defi") or {}
        if defi:
            st.markdown("**DefiLlama metrics**")
            d_cols = st.columns(4)
            tvl = defi.get("tvl") or 0.0
            d_cols[0].metric(
                "TVL",
                f"${tvl/1e9:.2f}B" if tvl >= 1e9 else (f"${tvl/1e6:.1f}M" if tvl else "—"),
            )
            tvl_g = defi.get("tvl_growth")
            d_cols[1].metric("TVL 7d", f"{tvl_g:+.2%}" if tvl_g is not None else "—")
            fees_7d = defi.get("fees_7d") or 0.0
            d_cols[2].metric(
                "Fees 7d",
                f"${fees_7d/1e6:.2f}M" if fees_7d >= 1e6 else (f"${fees_7d:,.0f}" if fees_7d else "—"),
            )
            fee_g = defi.get("fee_growth")
            d_cols[3].metric("Fee 24h Δ", f"{fee_g:+.2%}" if fee_g is not None else "—")

        # Narrative
        narr = r.get("narrative_profile") or {}
        if narr:
            st.markdown(
                f"**Narrative:** {narr.get('label','—')} "
                f"· score {narr.get('score',0.0):.3f} "
                f"· momentum {narr.get('momentum',0.0):+.3f} "
                f"· sources {narr.get('source_count',0)}"
            )

        # Sentiment vector
        sent = r["sentiment"]
        s_cols = st.columns(4)
        s_cols[0].metric("Polarity", f"{sent['source_weighted_polarity']:+.3f}")
        s_cols[1].metric("Confidence", f"{sent['confidence']:.2f}")
        s_cols[2].metric("Fear", f"{sent['fear']:.2f}")
        s_cols[3].metric("Hype", f"{sent['hype']:.2f}")
        st.caption(
            f"signals_total={r['signals']} · signals_7d={r['signals_7d']} "
            f"· agreement={sent['agreement']:.2f} · sources={r['signal_sources']}"
        )

        # Risk + missing data
        if explain["not_tradable"]:
            st.error("**Not tradable:** " + ", ".join(explain["not_tradable"]))
        elif r["passes"]:
            st.success("Passes hard gates")
        if explain["missing_data"]:
            st.caption("Missing data: " + ", ".join(explain["missing_data"]))
