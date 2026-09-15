from __future__ import annotations

from datetime import datetime, timedelta, timezone

from crypto_llm_alpaca.config import (
    AppConfig,
    ProjectProfile,
    QualityGrowthConfig,
    StrategyConfig,
    UniverseConfig,
)
from crypto_llm_alpaca.research_impact import format_research_impact, run_research_impact
from crypto_llm_alpaca.schemas import Bar, LlmSignal, NewsItem
from crypto_llm_alpaca.storage import SQLiteStore


def _breakout_bars(symbol: str = "AAVE/USD", count: int = 90) -> list[Bar]:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars: list[Bar] = []
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


def _cfg() -> AppConfig:
    return AppConfig(
        universe=UniverseConfig(symbols=("AAVE/USD",)),
        strategy=StrategyConfig(relative_volume_min=1.2, range_expansion_min=1.1),
        quality_growth=QualityGrowthConfig(
            min_relative_volume=1.2,
            min_volume_acceleration=1.0,
            backtest_news_window_days=14,
        ),
        project_profiles={
            "AAVE/USD": ProjectProfile(
                category="defi_lending",
                thesis="lending protocol growth",
                quality_score=0.8,
                growth_score=0.75,
                backer_score=0.75,
                liquidity_score=0.7,
                risk_score=0.35,
            )
        },
    )


def test_research_impact_reports_eligible_candidates_and_snapshot_data(tmp_path):
    store = SQLiteStore(tmp_path / "impact.sqlite3")
    store.init_schema()
    bars = _breakout_bars()
    store.insert_bars(bars)
    store.upsert_defi_snapshot(
        "AAVE/USD",
        bars[0].timestamp - timedelta(hours=1),
        {
            "tvl": 10_000_000_000,
            "tvl_growth": 0.10,
            "fee_growth": 0.05,
            "revenue_growth": 0.04,
            "fees_7d": 1_000_000,
        },
    )
    store.upsert_news(
        [
            NewsItem(
                source="fixture",
                url="https://example.test/aave",
                title="Aave lending demand rises",
                summary="Aave sees stronger lending activity and new institutional demand.",
                published_at=bars[45].timestamp,
                symbols=("AAVE/USD",),
                dedupe_hash="aave-news",
            )
        ]
    )
    store.insert_llm_signals(
        [
            LlmSignal(
                news_hash="aave-news",
                symbol="AAVE/USD",
                sentiment=0.6,
                confidence=0.85,
                horizon="medium",
                event_type="growth",
                rationale="positive attributed news",
                created_at=bars[45].timestamp,
                symbol_confidence=1.0,
            )
        ]
    )

    result = run_research_impact(
        _cfg(),
        store,
        start=bars[0].timestamp,
        end=bars[-1].timestamp,
    )

    assert result["decision_bars"] > 0
    assert result["eligible_candidates"] > 0
    assert result["symbol_impacts"]["AAVE/USD"]["eligible"] > 0
    assert result["symbol_impacts"]["AAVE/USD"]["avg_research_score"] > 0
    assert result["missing_data"].get("protocol_tvl", 0) == 0


def test_format_research_impact_includes_actionable_sections():
    text = format_research_impact(
        {
            "timeframe": "1Hour",
            "start": None,
            "end": None,
            "decision_bars": 10,
            "eligible_bars": 2,
            "eligible_candidates": 3,
            "rank_changed_bars": 1,
            "rank_changed_pct": 0.5,
            "fund_style_vetoes": 1,
            "fund_style_promotions": 0,
            "recommendation": "test_research_ranked_entries_or_min_research_score",
            "top_symbol_changes": {"UNI/USD->AAVE/USD": 1},
            "symbol_impacts": {
                "AAVE/USD": {
                    "evaluated": 10,
                    "eligible": 2,
                    "fund_veto": 0,
                    "fund_promotion": 0,
                    "top_base": 0,
                    "top_fund": 1,
                    "avg_base_score": 0.5,
                    "avg_research_score": 0.7,
                    "avg_tradable_score": 0.4,
                }
            },
            "missing_data": {"developer_activity": 10},
        }
    )

    assert "Research impact audit" in text
    assert "rank changed bars: 1 (50.0%)" in text
    assert "AAVE/USD" in text
    assert "developer_activity" in text
