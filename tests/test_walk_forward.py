from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from crypto_llm_alpaca.config import load_config
from crypto_llm_alpaca.schemas import Bar
from crypto_llm_alpaca.walk_forward import (
    render_walk_forward_markdown,
    run_walk_forward,
)


def _cfg(tmp_path: Path) -> Path:
    db = tmp_path / "wf.sqlite3"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""
universe:
  symbols: ["ETH/USD"]
data:
  timeframe: "1Hour"
  lookback_bars: 200
  rss_feeds: []
llm:
  provider: mock
  model: mock
  min_confidence: 0.0
scoring:
  buy_threshold: -1.0
  sell_threshold: -2.0
  risk_off_threshold: -2.0
  sentiment_weight: 1.0
  momentum_weight: 0.5
  volatility_weight: 0.0
strategy:
  breakout_lookback: 10
  volume_lookback: 10
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
  max_positions: 3
  stop_loss_pct: 0.05
  risk_per_trade_pct: 0.01
  max_symbol_exposure_pct: 0.50
  daily_drawdown_stop_pct: 1.0
  volatility_spike_limit: 10.0
  take_profit_levels:
    - gain_pct: 0.10
      sell_fraction: 1.0
regime:
  enabled: false
broker:
  paper_base_url: https://paper-api.alpaca.markets
  live_trading_enabled: false
storage:
  path: "{db.as_posix()}"
"""
    )
    return cfg


def _make_bars(symbol: str, days: int, start: datetime) -> list[Bar]:
    bars = []
    for d in range(days):
        for h in range(24):
            ts = start + timedelta(days=d, hours=h)
            price = 100.0 + (d * 0.05)  # mild uptrend
            bars.append(Bar(symbol=symbol, timestamp=ts, open=price, high=price, low=price, close=price, volume=1.0))
    return bars


def test_walk_forward_produces_windows(tmp_path):
    cfg = load_config(_cfg(tmp_path))
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = {"ETH/USD": _make_bars("ETH/USD", 200, start)}
    result = run_walk_forward(
        cfg, bars,
        start=start, end=start + timedelta(days=200),
        train_days=60, test_days=20, step_days=30,
        grid={"max_trades_per_day": [1, 2], "take_profit_pct": [0.10], "stop_loss_pct": [0.05]},
    )
    assert len(result.windows) >= 2
    for w in result.windows:
        assert "max_trades_per_day" in w.best_params
        assert w.test_result["entries"] >= 0


def test_render_markdown_has_table(tmp_path):
    cfg = load_config(_cfg(tmp_path))
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = {"ETH/USD": _make_bars("ETH/USD", 150, start)}
    result = run_walk_forward(
        cfg, bars,
        start=start, end=start + timedelta(days=150),
        train_days=60, test_days=20, step_days=30,
        grid={"max_trades_per_day": [1], "take_profit_pct": [0.10], "stop_loss_pct": [0.05]},
    )
    md = render_walk_forward_markdown(result, cfg)
    assert "Walk-forward" in md
    assert "test window" in md
    if result.aggregate:
        assert "Aggregate" in md


def test_walk_forward_handles_no_windows(tmp_path):
    cfg = load_config(_cfg(tmp_path))
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = {"ETH/USD": _make_bars("ETH/USD", 30, start)}  # too few days
    result = run_walk_forward(
        cfg, bars,
        start=start, end=start + timedelta(days=30),
        train_days=60, test_days=20, step_days=30,
    )
    assert len(result.windows) == 0
    assert result.aggregate == {}
