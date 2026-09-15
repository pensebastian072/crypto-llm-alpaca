from dataclasses import replace
from datetime import datetime, timedelta, timezone

from crypto_llm_alpaca.backtest import (
    focused_sweep_configs,
    run_adaptive_breakout_backtest,
    run_breakout_backtest,
    run_quality_growth_backtest,
)
from crypto_llm_alpaca.config import AppConfig, CostsConfig, ProjectProfile, QualityGrowthConfig, StrategyConfig
from crypto_llm_alpaca.features import atr, compute_strategy_features
from crypto_llm_alpaca.schemas import Bar
from crypto_llm_alpaca.storage import SQLiteStore


def _breakout_bars(symbol="BTC/USD", count=80):
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = []
    price = 100.0
    for idx in range(count):
        price *= 1.002
        volume = 1_000.0
        high = price * 1.002
        low = price * 0.998
        close = price
        if idx in (50, 65):
            close = max(bar.high for bar in bars[-20:]) * 1.018
            high = close * 1.004
            low = close * 0.986
            volume = 2_500.0
            price = close
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=ts + timedelta(hours=idx),
                open=price * 0.997,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
    return bars


def test_strategy_features_use_prior_bars_for_breakout_level():
    cfg = StrategyConfig(relative_volume_min=1.2, range_expansion_min=1.1)
    bars = _breakout_bars(count=51)
    features = compute_strategy_features(bars, cfg)
    assert features.breakout_level == max(bar.high for bar in bars[-21:-1])
    assert features.entry_signal
    assert features.relative_volume > 1.2
    assert features.range_expansion > 1.1
    assert features.atr > 0
    assert features.atr_regime > 0


def test_strategy_features_include_multi_timeframe_trend_alignment():
    cfg = StrategyConfig(relative_volume_min=1.2, range_expansion_min=1.1)
    bars = _breakout_bars(count=1500)
    features = compute_strategy_features(bars, cfg)

    assert features.trend_4h != 0.0
    assert features.trend_1d != 0.0
    assert features.trend_1w != 0.0
    assert 0.0 <= features.mtf_trend_score <= 1.0


def test_atr_uses_current_and_prior_ranges_without_future_lookahead():
    bars = _breakout_bars(count=40)
    value_without_future = atr(bars[:30], 14)
    value_with_future_removed = atr(bars[:30], 14)
    assert value_without_future == value_with_future_removed
    assert atr(bars[:31], 14) != atr(bars[:30], 14)


def test_breakout_backtest_runs_with_fee_assumptions():
    cfg = AppConfig(
        strategy=StrategyConfig(relative_volume_min=1.2, range_expansion_min=1.1),
        costs=CostsConfig(taker_bps=25.0, default_fill_model="taker"),
    )
    result = run_breakout_backtest(cfg, {"BTC/USD": _breakout_bars()}, run_id="test-run")
    assert result.entries >= 1
    assert result.total_fees > 0
    assert result.equity_curve
    assert result.assumptions["fee_bps_per_fill"] == 25.0
    assert result.buy_hold_return_pct != 0


def test_adaptive_backtest_uses_atr_trailing_and_risk_sizing():
    cfg = AppConfig(strategy=StrategyConfig(relative_volume_min=1.2, range_expansion_min=1.1))
    result = run_adaptive_breakout_backtest(cfg, {"BTC/USD": _breakout_bars()}, run_id="adaptive-test")
    assert result.strategy == "adaptive_breakout"
    assert result.entries >= 1
    assert result.stop_events == result.exits
    assert result.assumptions["exit_model"] == "atr_trailing_stop"
    assert result.assumptions["sizing_model"] == "risk_per_trade"
    assert "rank_score" in result.assumptions


def test_quality_growth_backtest_requires_profile_and_volume_confirmation():
    cfg = AppConfig(
        strategy=StrategyConfig(relative_volume_min=1.2, range_expansion_min=1.1),
        project_profiles={
            "AAVE/USD": ProjectProfile(
                quality_score=0.8,
                growth_score=0.7,
                backer_score=0.75,
                liquidity_score=0.7,
                risk_score=0.4,
            )
        },
    )
    result = run_quality_growth_backtest(cfg, {"AAVE/USD": _breakout_bars("AAVE/USD")}, run_id="qg-test")
    assert result.strategy == "quality_growth"
    assert result.entries >= 1
    assert result.assumptions["profile_model"] == "quality_growth_backer_liquidity_risk"
    assert result.per_symbol


def test_quality_growth_research_promotions_are_probe_sized(tmp_path):
    store = SQLiteStore(tmp_path / "promotions.sqlite3")
    store.init_schema()
    cfg = AppConfig(
        strategy=StrategyConfig(relative_volume_min=1.2, range_expansion_min=1.1),
        quality_growth=QualityGrowthConfig(
            min_relative_volume=1.2,
            min_volume_acceleration=1.0,
            backtest_use_fund_style=True,
            backtest_allow_research_promotions=True,
            research_promotion_min_score=0.0,
            research_promotion_size_scale=0.25,
        ),
        project_profiles={
            "BAT/USD": ProjectProfile(
                quality_score=0.45,
                growth_score=0.45,
                backer_score=0.45,
                liquidity_score=0.45,
                risk_score=0.40,
            )
        },
    )

    result = run_quality_growth_backtest(
        cfg,
        {"BAT/USD": _breakout_bars("BAT/USD")},
        run_id="qg-promotion",
        store=store,
    )

    assert result.assumptions["research_promotion_entries"] >= 1
    assert result.assumptions["standard_entries"] == 0
    assert any("quality_research_promotion" in str(t["reason"]) for t in result.trade_log)

    blocked = run_quality_growth_backtest(
        replace(
            cfg,
            quality_growth=replace(cfg.quality_growth, research_promotion_symbols=("CRV/USD",)),
        ),
        {"BAT/USD": _breakout_bars("BAT/USD")},
        run_id="qg-promotion-blocked",
        store=store,
    )
    assert blocked.assumptions["research_promotion_entries"] == 0


def test_focused_sweep_shape_is_deterministic():
    configs = focused_sweep_configs(AppConfig())
    assert len(configs) == 162
    assert configs[0].strategy.breakout_lookback == 20
    assert configs[-1].strategy.top_n_per_bar == 5


def test_storage_persists_features_and_backtest(tmp_path):
    store = SQLiteStore(tmp_path / "research.sqlite3")
    store.init_schema()
    cfg = AppConfig(strategy=StrategyConfig(relative_volume_min=1.2, range_expansion_min=1.1))
    bars = _breakout_bars(count=51)
    features = compute_strategy_features(bars, cfg.strategy)
    assert store.upsert_strategy_features([features]) == 1
    with store.connect() as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(strategy_features)").fetchall()}
    assert "mtf_trend_score" in columns
    result = run_breakout_backtest(cfg, {"BTC/USD": _breakout_bars()}, run_id="stored-run")
    changed = store.insert_backtest_run(
        result.run_id,
        result.strategy,
        {"strategy": "test"},
        {"return_pct": result.return_pct},
        result.trade_log,
        result.equity_curve,
    )
    assert changed >= 1
    assert store.list_backtest_runs()[0]["run_id"] == "stored-run"
    assert store.list_backtest_trades("stored-run")
    assert store.list_equity_curve("stored-run")
    assert store.list_bars("BTC/USD") == []
