from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from crypto_llm_alpaca.autopilot import run_autopilot
from crypto_llm_alpaca.autopilot_backtest import run_autopilot_backtest
from crypto_llm_alpaca.broker import BrokerAccount, SubmittedOrder
from crypto_llm_alpaca.config import PerSymbolOverride, load_config
from crypto_llm_alpaca.schemas import Bar, LlmSignal
from crypto_llm_alpaca.storage import SQLiteStore


def _write_cfg(tmp_path: Path, per_symbol_block: str = "") -> Path:
    db = tmp_path / "ps.sqlite3"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""
universe:
  symbols: ["BTC/USD", "ETH/USD"]
  quote_currency: USD
{per_symbol_block}
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
  volatility_weight: 0.0
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
  taker_bps: 0
  maker_bps: 0
  default_fill_model: taker
risk:
  max_total_exposure_pct: 0.30
  min_positions: 1
  max_positions: 5
  stop_loss_pct: 0.05
  risk_per_trade_pct: 0.01
  max_symbol_exposure_pct: 0.50
  daily_drawdown_stop_pct: 1.0
  volatility_spike_limit: 10.0
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
    return cfg


def _seed_signals(store: SQLiteStore, symbols: list[str]) -> None:
    sigs = [
        LlmSignal(
            news_hash=f"sig-{i}",
            symbol=sym,
            sentiment=0.9,
            confidence=0.9,
            horizon="short",
            event_type="seed",
            rationale="positive",
            provider="mock",
            model="mock",
            created_at=datetime.now(tz=timezone.utc),
        )
        for i, sym in enumerate(symbols)
    ]
    store.insert_llm_signals(sigs)


def test_load_per_symbol_overrides(tmp_path):
    cfg_path = _write_cfg(
        tmp_path,
        per_symbol_block="""
per_symbol:
  BTC/USD:
    trade_enabled: false
  ETH/USD:
    take_profit_pct: 0.20
    atr_stop_multiple: 1.8
""",
    )
    cfg = load_config(cfg_path)
    assert "BTC/USD" in cfg.per_symbol
    assert cfg.per_symbol["BTC/USD"].trade_enabled is False
    assert cfg.symbol_override("ETH/USD").take_profit_pct == pytest.approx(0.20)
    assert cfg.symbol_override("ETH/USD").atr_stop_multiple == pytest.approx(1.8)
    assert cfg.symbol_override("SOL/USD").trade_enabled is True  # default


def test_is_trade_enabled_helper(tmp_path):
    cfg_path = _write_cfg(
        tmp_path,
        per_symbol_block="""
per_symbol:
  BTC/USD:
    trade_enabled: false
""",
    )
    cfg = load_config(cfg_path)
    assert cfg.is_trade_enabled("ETH/USD") is True
    assert cfg.is_trade_enabled("BTC/USD") is False


def test_autopilot_skips_disabled_symbol(tmp_path):
    cfg_path = _write_cfg(
        tmp_path,
        per_symbol_block="""
per_symbol:
  BTC/USD:
    trade_enabled: false
""",
    )
    cfg = load_config(cfg_path)
    store = SQLiteStore(cfg.storage.path)
    store.init_schema()
    _seed_signals(store, ["BTC/USD", "ETH/USD"])

    broker = MagicMock()
    broker.get_account.return_value = BrokerAccount(equity=100000.0, buying_power=200000.0)
    broker.get_positions.return_value = []
    broker.submit_bracket_order.return_value = SubmittedOrder(
        broker_order_id="x", status="accepted", symbol="ETH/USD", qty=0.1,
    )
    out = tmp_path / "outputs"
    result = run_autopilot(
        cfg, store, dry_run=True, max_trades=5, max_positions=5,
        broker=broker, outputs_dir=out, demo_only=True,
    )
    decided = {d.symbol for d in result.decisions}
    assert "BTC/USD" not in decided
    assert "ETH/USD" in decided


def _flat_day(price: float = 100.0) -> list[tuple[float, float, float, float]]:
    return [(price, price, price, price)] * 24


def _make_bars(symbol: str, days: list[list[tuple[float, float, float, float]]], start: datetime) -> list[Bar]:
    bars = []
    for d, day in enumerate(days):
        for h, (o, hi, lo, c) in enumerate(day):
            ts = start + timedelta(days=d, hours=h)
            bars.append(Bar(symbol=symbol, timestamp=ts, open=o, high=hi, low=lo, close=c, volume=1.0))
    return bars


def test_autopilot_backtest_skips_disabled_symbol(tmp_path):
    cfg_path = _write_cfg(
        tmp_path,
        per_symbol_block="""
per_symbol:
  BTC/USD:
    trade_enabled: false
""",
    )
    cfg = load_config(cfg_path)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    days = [_flat_day(100.0) for _ in range(35)]
    bars = {
        "BTC/USD": _make_bars("BTC/USD", days, start),
        "ETH/USD": _make_bars("ETH/USD", days, start),
    }
    result = run_autopilot_backtest(
        cfg, bars, run_id="t-skip",
        starting_equity=10_000.0, max_trades_per_day=5, max_positions=5,
        feature_lookback=120,
    )
    syms = {t["symbol"] for t in result.trade_log if t["side"] == "buy"}
    assert "BTC/USD" not in syms
    assert "ETH/USD" in syms or len(syms) == 0  # ETH may or may not enter (technical=0)


def test_per_symbol_take_profit_used_in_backtest(tmp_path):
    cfg_path = _write_cfg(
        tmp_path,
        per_symbol_block="""
per_symbol:
  ETH/USD:
    take_profit_pct: 0.05
    stop_loss_pct: 0.03
""",
    )
    cfg = load_config(cfg_path)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    # 30 flat days then a 6% spike on day 30 (would hit 5% TP, not 10% global)
    days = [_flat_day(100.0) for _ in range(30)]
    spike = [(100, 100, 100, 100)] * 9 + [(100, 100, 100, 100), (100, 106, 100, 106)] + [(106, 106, 106, 106)] * 13
    days.append(spike)
    days.append(_flat_day(106.0))
    bars = {"ETH/USD": _make_bars("ETH/USD", days, start)}
    cfg = cfg.__class__(
        universe=cfg.universe.__class__(symbols=("ETH/USD",), quote_currency=cfg.universe.quote_currency, allow_shorts=cfg.universe.allow_shorts),
        data=cfg.data, llm=cfg.llm, scoring=cfg.scoring, strategy=cfg.strategy,
        costs=cfg.costs, risk=cfg.risk, broker=cfg.broker, storage=cfg.storage,
        per_symbol=cfg.per_symbol,
    )
    result = run_autopilot_backtest(
        cfg, bars, run_id="t-tp",
        starting_equity=10_000.0, max_trades_per_day=1, max_positions=1,
        feature_lookback=120,
    )
    # 5% TP would have hit; 10% global would not
    assert result.tp_events >= 1
