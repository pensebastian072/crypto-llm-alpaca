from __future__ import annotations

from datetime import datetime, timezone

from crypto_llm_alpaca.config import ProjectProfile, QualityGrowthConfig
from crypto_llm_alpaca.quality_growth import fund_style_quality_score, quality_growth_score
from crypto_llm_alpaca.schemas import LlmSignal, StrategyFeatures
from crypto_llm_alpaca.sentiment_aggregator import build_sentiment_vector


def _features(*, momentum=0.05, relative_volume=1.6, volume_acceleration=1.3, is_breakout=True):
    return StrategyFeatures(
        symbol="AAVE/USD",
        timestamp=datetime(2026, 5, 5, tzinfo=timezone.utc),
        breakout_level=100.0,
        breakout_pct=0.04,
        relative_volume=relative_volume,
        volume_acceleration=volume_acceleration,
        range_expansion=1.2,
        close_location=0.75,
        momentum=momentum,
        htf_trend=0.04,
        atr=4.0,
        atr_pct=0.04,
        atr_short=3.0,
        atr_long=4.0,
        atr_regime=0.75,
        expected_move_pct=0.08,
        signal_strength=0.7,
        is_breakout=is_breakout,
        passes_volume=relative_volume >= 1.25,
        passes_range=True,
        passes_htf=True,
    )


def _profile():
    return ProjectProfile(
        category="DeFi lending",
        quality_score=0.8,
        growth_score=0.7,
        backer_score=0.75,
        liquidity_score=0.7,
        risk_score=0.35,
    )


def _signal(sentiment=0.6):
    return LlmSignal(
        news_hash="n1",
        symbol="AAVE/USD",
        sentiment=sentiment,
        confidence=0.85,
        horizon="short",
        event_type="fixture",
        rationale="positive attributed news",
    )


def test_fund_style_score_blocks_tradable_score_without_momentum_gate():
    result = fund_style_quality_score(
        _profile(),
        _features(momentum=0.0, is_breakout=False),
        QualityGrowthConfig(),
        build_sentiment_vector([_signal()]),
        signal_count_7d=1,
    )

    assert result.research_score > 0
    assert result.tradable_score == 0.0
    assert result.momentum_gate is False


def test_fund_style_score_activates_when_momentum_and_volume_confirm():
    result = fund_style_quality_score(
        _profile(),
        _features(),
        QualityGrowthConfig(),
        build_sentiment_vector([_signal(), _signal(0.4)]),
        signal_count_7d=2,
    )

    assert result.research_score > 0
    assert result.tradable_score == result.research_score
    assert result.momentum_gate is True
    assert result.volume_gate is True


def test_fund_style_sentiment_increases_with_attributed_signal_count():
    no_news = fund_style_quality_score(
        _profile(),
        _features(),
        QualityGrowthConfig(),
        build_sentiment_vector([]),
        signal_count_7d=0,
    )
    with_news = fund_style_quality_score(
        _profile(),
        _features(),
        QualityGrowthConfig(),
        build_sentiment_vector([_signal(), _signal(), _signal()]),
        signal_count_7d=3,
    )

    assert with_news.sentiment > no_news.sentiment
    assert with_news.research_score > no_news.research_score


def test_quality_growth_can_require_multi_timeframe_alignment():
    score = quality_growth_score(
        _profile(),
        _features(),
        QualityGrowthConfig(require_mtf_alignment=True, min_mtf_trend_score=0.90),
        build_sentiment_vector([_signal()]),
    )

    assert score.passes is False
    assert "mtf_trend<0.90" in score.reject_reasons


def test_fund_style_uses_defi_and_narrative_inputs():
    result = fund_style_quality_score(
        _profile(),
        _features(),
        QualityGrowthConfig(),
        build_sentiment_vector([_signal()]),
        signal_count_7d=1,
        defi_metrics={
            "tvl": 10_000_000,
            "tvl_growth": 0.20,
            "fee_growth": 0.10,
            "revenue_growth": 0.10,
            "fees_7d": 5_000,
        },
        narrative={"label": "defi_lending", "score": 0.90, "momentum": 0.25},
    )

    assert result.tvl == 10_000_000
    assert result.narrative_label == "defi_lending"
    assert result.narrative_momentum == 0.25
    assert "protocol_tvl" not in result.missing_data
    assert "fee_generation" not in result.missing_data
