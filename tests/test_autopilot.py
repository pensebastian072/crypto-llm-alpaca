from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from datetime import datetime, timezone

from crypto_llm_alpaca.autopilot import run_autopilot
from crypto_llm_alpaca.broker import BrokerAccount, BrokerPosition, SubmittedOrder
from crypto_llm_alpaca.config import load_config
from crypto_llm_alpaca.schemas import LlmSignal
from crypto_llm_alpaca.storage import SQLiteStore


def _seed_positive_signals(store: SQLiteStore, symbols: list[str]) -> None:
    sigs = [
        LlmSignal(
            news_hash=f"seed-{i}",
            symbol=sym,
            sentiment=0.9,
            confidence=0.9,
            horizon="short",
            event_type="seed_test",
            rationale="positive seed",
            provider="mock",
            model="mock",
            created_at=datetime.now(tz=timezone.utc),
        )
        for i, sym in enumerate(symbols)
    ]
    store.insert_llm_signals(sigs)


@pytest.fixture
def cfg_and_store(tmp_path):
    db = tmp_path / "ap.sqlite3"
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        f"""
universe:
  symbols: ["BTC/USD", "ETH/USD", "SOL/USD"]
  quote_currency: USD
data:
  timeframe: "1Hour"
  lookback_bars: 200
  rss_feeds: []
llm:
  provider: mock
  model: mock
  min_confidence: 0.0
scoring:
  buy_threshold: 0.0
  sell_threshold: -1.0
  sentiment_weight: 1.0
  momentum_weight: 0.5
  volatility_weight: 0.2
strategy:
  breakout_lookback: 20
  volume_lookback: 20
  relative_volume_min: 1.2
  range_expansion_min: 1.2
  min_close_location: 0.5
  momentum_lookback: 5
  htf_timeframe: "4Hour"
  htf_trend_lookback: 6
  require_htf_trend: false
  atr_window: 14
  atr_short_window: 6
  atr_long_window: 24
  atr_stop_multiple: 2.0
  atr_trail_multiple: 2.5
  min_expected_move_fee_multiple: 1.5
  top_n_per_bar: 3
costs:
  taker_bps: 25
  maker_bps: 15
  default_fill_model: taker
risk:
  max_total_exposure_pct: 0.30
  min_positions: 1
  max_positions: 5
  stop_loss_pct: 0.05
  risk_per_trade_pct: 0.01
  max_symbol_exposure_pct: 0.10
  daily_drawdown_stop_pct: 0.05
  volatility_spike_limit: 0.20
  take_profit_levels:
    - gain_pct: 0.10
      sell_fraction: 1.0
broker:
  paper_base_url: https://paper-api.alpaca.markets
  live_trading_enabled: false
storage:
  path: "{db.as_posix()}"
"""
    )
    cfg = load_config(cfg_yaml)
    store = SQLiteStore(cfg.storage.path)
    store.init_schema()
    return cfg, store


def _fake_broker():
    b = MagicMock()
    b.get_account.return_value = BrokerAccount(equity=100000.0, buying_power=200000.0)
    b.get_positions.return_value = []
    b.submit_bracket_order.return_value = SubmittedOrder(
        broker_order_id="ord-x", status="accepted", symbol="BTC/USD", qty=0.001,
        take_profit_price=110.0, stop_loss_price=95.0,
    )
    return b


def test_dry_run_writes_markdown_and_no_submit(cfg_and_store, tmp_path):
    cfg, store = cfg_and_store
    out = tmp_path / "outputs"
    broker = _fake_broker()
    _seed_positive_signals(store, list(cfg.universe.symbols))
    result = run_autopilot(
        cfg, store, dry_run=True, max_trades=3, max_positions=5,
        broker=broker, outputs_dir=out, demo_only=False,
    )
    assert result.dry_run is True
    assert Path(result.markdown_path).exists()
    assert "DRY RUN" in Path(result.markdown_path).read_text(encoding="utf-8")
    broker.submit_bracket_order.assert_not_called()
    decisions = store.list_decisions(run_id=result.run_id)
    assert len(decisions) == 3
    reports = store.list_daily_reports()
    assert len(reports) == 1


def test_live_submits_bracket_orders_capped_by_max_trades(cfg_and_store, tmp_path):
    cfg, store = cfg_and_store
    out = tmp_path / "outputs"
    broker = _fake_broker()
    _seed_positive_signals(store, list(cfg.universe.symbols))
    # provide bars so price lookup works
    from crypto_llm_alpaca.backtest import demo_bars
    for sym in cfg.universe.symbols:
        store.insert_bars(demo_bars(symbol=sym, count=50))
    result = run_autopilot(
        cfg, store, dry_run=False, max_trades=2, max_positions=5,
        broker=broker, outputs_dir=out, demo_only=False,
    )
    accepted = [d for d in result.decisions if d.accepted]
    assert len(accepted) >= 2
    submitted = [d for d in result.decisions if d.submitted_order_id]
    assert len(submitted) <= 2
    assert broker.submit_bracket_order.call_count == len(submitted)


def test_max_positions_budget_reduces_take_n(cfg_and_store, tmp_path):
    cfg, store = cfg_and_store
    out = tmp_path / "outputs"
    broker = _fake_broker()
    broker.get_positions.return_value = [
        BrokerPosition(symbol="BTCUSD", qty=0.5, avg_entry_price=50000, market_value=26000, unrealized_pl=0),
        BrokerPosition(symbol="ETHUSD", qty=2.0, avg_entry_price=3000, market_value=6500, unrealized_pl=0),
    ]
    _seed_positive_signals(store, list(cfg.universe.symbols))
    from crypto_llm_alpaca.backtest import demo_bars
    for sym in cfg.universe.symbols:
        store.insert_bars(demo_bars(symbol=sym, count=50))
    result = run_autopilot(
        cfg, store, dry_run=False, max_trades=5, max_positions=3,
        broker=broker, outputs_dir=out, demo_only=False,
    )
    submitted = [d for d in result.decisions if d.submitted_order_id]
    assert len(submitted) <= 1


def test_idempotent_same_day_overwrites(cfg_and_store, tmp_path):
    cfg, store = cfg_and_store
    out = tmp_path / "outputs"
    broker = _fake_broker()
    run_autopilot(cfg, store, dry_run=True, broker=broker, outputs_dir=out, demo_only=True)
    run_autopilot(cfg, store, dry_run=True, broker=broker, outputs_dir=out, demo_only=True)
    reports = store.list_daily_reports()
    assert len(reports) == 1
