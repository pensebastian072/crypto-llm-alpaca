from __future__ import annotations

from datetime import datetime, timezone

import pytest

from crypto_llm_alpaca.broker import BrokerPosition
from crypto_llm_alpaca.storage import SQLiteStore


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(tmp_path / "x.sqlite3")
    s.init_schema()
    return s


def test_insert_and_list_decisions(store):
    store.insert_decision(
        run_id="run-1", symbol="BTC/USD", score=0.85, confidence=0.9,
        accepted=True, reason="strong sentiment", intent={"side": "buy", "notional": 1000},
    )
    store.insert_decision(
        run_id="run-1", symbol="ETH/USD", score=0.2, confidence=0.4,
        accepted=False, reason="below threshold", intent={"side": "hold"},
    )
    decisions = store.list_decisions(run_id="run-1")
    assert len(decisions) == 2
    accepted = [d for d in decisions if d["accepted"]]
    assert len(accepted) == 1
    assert accepted[0]["symbol"] == "BTC/USD"
    assert accepted[0]["intent"]["notional"] == 1000


def test_list_decisions_filtered_by_date(store):
    store.insert_decision(
        run_id="r", symbol="BTC/USD", score=0.5, confidence=0.5,
        accepted=True, reason="ok", intent={},
        decided_at=datetime(2026, 5, 4, 9, 0, tzinfo=timezone.utc),
    )
    store.insert_decision(
        run_id="r", symbol="BTC/USD", score=0.5, confidence=0.5,
        accepted=True, reason="ok", intent={},
        decided_at=datetime(2026, 5, 5, 9, 0, tzinfo=timezone.utc),
    )
    assert len(store.list_decisions(date="2026-05-04")) == 1
    assert len(store.list_decisions(date="2026-05-05")) == 1


def test_sync_and_list_positions(store):
    pos = [
        BrokerPosition(symbol="BTCUSD", qty=0.5, avg_entry_price=50000, market_value=26000, unrealized_pl=1000, side="long"),
        BrokerPosition(symbol="ETHUSD", qty=2.0, avg_entry_price=3000, market_value=6500, unrealized_pl=500, side="long"),
    ]
    n = store.sync_positions(pos)
    assert n == 2
    rows = store.list_positions()
    assert {r["symbol"] for r in rows} == {"BTCUSD", "ETHUSD"}
    btc = next(r for r in rows if r["symbol"] == "BTCUSD")
    assert btc["state"]["unrealized_pl"] == 1000


def test_sync_positions_replaces_existing(store):
    store.sync_positions([BrokerPosition(symbol="BTCUSD", qty=0.5, avg_entry_price=50000, market_value=26000, unrealized_pl=1000)])
    store.sync_positions([BrokerPosition(symbol="ETHUSD", qty=2.0, avg_entry_price=3000, market_value=6500, unrealized_pl=500)])
    rows = store.list_positions()
    assert {r["symbol"] for r in rows} == {"ETHUSD"}


def test_upsert_daily_report_idempotent(store):
    store.upsert_daily_report(
        date="2026-05-04", equity_open=100000, equity_close=100050,
        n_decisions=5, n_orders=2, markdown_path="outputs/2026-05-04.md",
        summary={"top_symbol": "BTC/USD"},
    )
    store.upsert_daily_report(
        date="2026-05-04", equity_open=100000, equity_close=100150,
        n_decisions=6, n_orders=3, markdown_path="outputs/2026-05-04.md",
        summary={"top_symbol": "ETH/USD"},
    )
    reports = store.list_daily_reports()
    assert len(reports) == 1
    assert reports[0]["equity_close"] == 100150
    assert reports[0]["n_orders"] == 3
    assert reports[0]["summary"]["top_symbol"] == "ETH/USD"


def test_get_daily_report_returns_none_when_missing(store):
    assert store.get_daily_report("1999-01-01") is None
