from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from crypto_llm_alpaca.regime import compute_btc_regime
from crypto_llm_alpaca.schemas import Bar


def _bars(prices: list[float], symbol: str = "BTC/USD") -> list[Bar]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        Bar(symbol=symbol, timestamp=start + timedelta(hours=i), open=p, high=p, low=p, close=p, volume=1.0)
        for i, p in enumerate(prices)
    ]


def test_uptrend_is_risk_on():
    prices = [100.0 + i * 0.1 for i in range(300)]  # steady uptrend
    state = compute_btc_regime(_bars(prices), ema_window=168, vol_window=168)
    assert state.trend_slope > 0
    assert state.risk_on is True


def test_downtrend_is_risk_off():
    prices = [200.0 - i * 0.1 for i in range(300)]
    state = compute_btc_regime(_bars(prices), ema_window=168, vol_window=168)
    assert state.trend_slope < 0
    assert state.risk_on is False
    assert "trend below EMA" in state.reason or "vol regime extreme" in state.reason


def test_extreme_volatility_is_risk_off_even_in_uptrend():
    import math
    prices = [100.0 + (math.sin(i / 3.0) * 30.0) + i * 0.05 for i in range(300)]
    state = compute_btc_regime(
        _bars(prices), ema_window=168, vol_window=168,
        vol_extreme_threshold=0.05, vol_high_threshold=0.02, vol_low_threshold=0.005,
    )
    assert state.vol_regime in ("high", "extreme")


def test_insufficient_bars_defaults_to_risk_on():
    state = compute_btc_regime(_bars([100.0] * 5))
    assert state.risk_on is True
    assert "insufficient" in state.reason


def test_flat_market_is_risk_on():
    prices = [100.0] * 300
    state = compute_btc_regime(_bars(prices), ema_window=168, vol_window=168)
    assert state.risk_on is True
    assert state.vol_regime == "low"


def test_backtest_regime_off_blocks_entries(tmp_path):
    """Synthetic BTC downtrend should make autopilot_backtest skip entries."""
    from datetime import datetime, timedelta, timezone

    from crypto_llm_alpaca.autopilot_backtest import run_autopilot_backtest
    from crypto_llm_alpaca.config import load_config

    db = tmp_path / "x.sqlite3"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""
universe:
  symbols: ["ETH/USD"]
  quote_currency: USD
per_symbol:
  BTC/USD:
    trade_enabled: false
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
  risk_off_threshold: -0.40
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
regime:
  enabled: true
  btc_ema_hours: 72
  vol_window_hours: 72
  sensor_symbol: "BTC/USD"
broker:
  paper_base_url: https://paper-api.alpaca.markets
  live_trading_enabled: false
storage:
  path: "{db.as_posix()}"
"""
    )
    cfg = load_config(cfg_path)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    # ETH bars: flat with some morning slots
    eth_bars = [
        Bar(symbol="ETH/USD", timestamp=start + timedelta(hours=i),
            open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0)
        for i in range(35 * 24)
    ]
    # BTC bars: monotonic downtrend → trend_slope < 0 → risk_off
    btc_bars = [
        Bar(symbol="BTC/USD", timestamp=start + timedelta(hours=i),
            open=200.0 - i * 0.05, high=200.0 - i * 0.05, low=200.0 - i * 0.05,
            close=200.0 - i * 0.05, volume=1.0)
        for i in range(35 * 24)
    ]
    bars = {"ETH/USD": eth_bars, "BTC/USD": btc_bars}
    result = run_autopilot_backtest(
        cfg, bars, run_id="t-regime-block",
        starting_equity=10_000.0, max_trades_per_day=3, max_positions=3,
        feature_lookback=120,
    )
    assert result.assumptions["regime_off_days"] > 0
    assert result.entries == 0  # all blocked by regime gate
