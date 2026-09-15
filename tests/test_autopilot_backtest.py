from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from crypto_llm_alpaca.autopilot_backtest import run_autopilot_backtest
from crypto_llm_alpaca.config import (
    AppConfig,
    CostsConfig,
    DataConfig,
    LlmConfig,
    RiskConfig,
    ScoringConfig,
    StorageConfig,
    StrategyConfig,
    TakeProfitLevel,
    UniverseConfig,
)
from crypto_llm_alpaca.schemas import Bar


def _cfg(symbols=("BTC/USD",), buy_threshold=-1.0):
    """Liberal scoring threshold so technical-only entries fire in tests."""
    return AppConfig(
        universe=UniverseConfig(symbols=symbols),
        data=DataConfig(timeframe="1Hour", lookback_bars=120, rss_feeds=()),
        llm=LlmConfig(provider="mock", model="mock", min_confidence=0.0),
        scoring=ScoringConfig(
            buy_threshold=buy_threshold, sell_threshold=-2.0,
            sentiment_weight=1.0, momentum_weight=0.5, volatility_weight=0.0,
        ),
        strategy=StrategyConfig(),
        costs=CostsConfig(taker_bps=0.0, maker_bps=0.0, default_fill_model="taker"),
        risk=RiskConfig(
            max_total_exposure_pct=0.30, min_positions=1, max_positions=5,
            stop_loss_pct=0.05, risk_per_trade_pct=0.01, max_symbol_exposure_pct=0.50,
            daily_drawdown_stop_pct=1.0, volatility_spike_limit=10.0,
            take_profit_levels=(TakeProfitLevel(0.10, 1.0),),
        ),
        broker=type("B", (), {"paper_base_url": "https://paper-api.alpaca.markets", "live_trading_enabled": False})(),
        storage=StorageConfig(path=":memory:"),
    )


def _make_bars(symbol: str, prices_per_day: list[list[tuple[float, float, float, float]]], start: datetime) -> list[Bar]:
    """prices_per_day[d][h] = (open, high, low, close) for hour h of day d. 24 hours per day."""
    bars = []
    for d, day in enumerate(prices_per_day):
        for h, (o, hi, lo, c) in enumerate(day):
            ts = start + timedelta(days=d, hours=h)
            bars.append(Bar(symbol=symbol, timestamp=ts, open=o, high=hi, low=lo, close=c, volume=1.0))
    return bars


def _flat_day(price: float = 100.0) -> list[tuple[float, float, float, float]]:
    return [(price, price, price, price)] * 24


def _spike_day_up(open_: float, high: float, close: float) -> list[tuple[float, float, float, float]]:
    """High spike day, used to trigger TP."""
    base = [(open_, open_, open_, open_)] * 9  # bars 0..8 flat
    base.append((open_, open_, open_, open_))  # 09:00 = bar 9
    base.append((open_, high, open_, close))   # 10:00 high spike
    base.extend([(close, close, close, close)] * 13)
    return base


def _spike_day_down(open_: float, low: float, close: float) -> list[tuple[float, float, float, float]]:
    base = [(open_, open_, open_, open_)] * 9
    base.append((open_, open_, open_, open_))
    base.append((open_, open_, low, close))
    base.extend([(close, close, close, close)] * 13)
    return base


def test_take_profit_fills():
    """Day 0 setup -> day 1 entry -> day 1 high reaches TP."""
    cfg = _cfg()
    start = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    # 30 prep days flat, day 30 morning entry, day 30 spike up to TP
    days = [_flat_day(100.0) for _ in range(30)]
    days.append(_spike_day_up(open_=100.0, high=120.0, close=120.0))
    days.append(_flat_day(120.0))
    bars = _make_bars("BTC/USD", days, start)
    result = run_autopilot_backtest(
        cfg, {"BTC/USD": bars}, run_id="t1",
        starting_equity=10_000.0, max_trades_per_day=1, max_positions=1,
        feature_lookback=120,
    )
    assert result.entries >= 1
    assert result.tp_events >= 1
    assert result.return_pct > 0


def test_stop_loss_fills():
    cfg = _cfg()
    start = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    days = [_flat_day(100.0) for _ in range(30)]
    days.append(_spike_day_down(open_=100.0, low=90.0, close=90.0))
    days.append(_flat_day(90.0))
    bars = _make_bars("BTC/USD", days, start)
    result = run_autopilot_backtest(
        cfg, {"BTC/USD": bars}, run_id="t2",
        starting_equity=10_000.0, max_trades_per_day=1, max_positions=1,
        feature_lookback=120,
    )
    assert result.stop_events >= 1
    assert result.return_pct < 0


def test_no_fill_remains_open():
    cfg = _cfg()
    start = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    days = [_flat_day(100.0) for _ in range(35)]  # never moves enough
    bars = _make_bars("BTC/USD", days, start)
    result = run_autopilot_backtest(
        cfg, {"BTC/USD": bars}, run_id="t3",
        starting_equity=10_000.0, max_trades_per_day=1, max_positions=1,
        feature_lookback=120,
    )
    # Some entries fire (flat technical = 0 score, threshold -1.0 accepts), none resolve
    assert result.tp_events == 0
    assert result.stop_events == 0


def test_synthetic_signals_branch_runs():
    cfg = _cfg(buy_threshold=0.5)  # high threshold, only synthetic positives can pass
    start = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    days = [_flat_day(100.0) for _ in range(40)]
    bars = _make_bars("BTC/USD", days, start)
    result_no_sig = run_autopilot_backtest(
        cfg, {"BTC/USD": bars}, run_id="t4a",
        starting_equity=10_000.0, max_trades_per_day=1, max_positions=1,
        feature_lookback=120, synthetic_signals=False,
    )
    result_sig = run_autopilot_backtest(
        cfg, {"BTC/USD": bars}, run_id="t4b",
        starting_equity=10_000.0, max_trades_per_day=1, max_positions=1,
        feature_lookback=120, synthetic_signals=True,
    )
    # synthetic should produce entries; no-sig path with high threshold should not
    assert result_no_sig.entries == 0
    assert result_sig.entries > 0


def test_max_positions_caps_concurrent_trades():
    cfg = _cfg(symbols=("BTC/USD", "ETH/USD", "SOL/USD"))
    start = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    days = [_flat_day(100.0) for _ in range(35)]
    bars = {"BTC/USD": _make_bars("BTC/USD", days, start),
            "ETH/USD": _make_bars("ETH/USD", days, start),
            "SOL/USD": _make_bars("SOL/USD", days, start)}
    result = run_autopilot_backtest(
        cfg, bars, run_id="t5",
        starting_equity=10_000.0, max_trades_per_day=3, max_positions=2,
        feature_lookback=120,
    )
    # Concurrent open positions never exceed cap
    max_open = max((p["open_positions"] for p in result.equity_curve), default=0)
    assert max_open <= 2
