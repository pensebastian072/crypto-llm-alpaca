from __future__ import annotations

from datetime import datetime, timezone

from crypto_llm_alpaca.config import ScoringConfig
from crypto_llm_alpaca.schemas import LlmSignal, Side, TechnicalFeatures
from crypto_llm_alpaca.scoring import aggregate_sentiment, score_symbol, score_with_vector
from crypto_llm_alpaca.sentiment_aggregator import build_sentiment_vector


def _features(symbol="BTC/USD", momentum=0.0, rsi=50.0, vol=0.02) -> TechnicalFeatures:
    return TechnicalFeatures(
        symbol=symbol, timestamp=datetime.now(tz=timezone.utc),
        momentum=momentum, rsi=rsi, volatility=vol, volume_change=0.0, trend_slope=0.0,
    )


def _sig(sent, conf=0.8, fear=0.0, hype=0.0, sw=1.0):
    return LlmSignal(
        news_hash=f"h-{sent}-{fear}-{hype}-{sw}",
        symbol="BTC/USD", sentiment=sent, confidence=conf,
        horizon="short", event_type="t", rationale="r",
        fear=fear, hype=hype, source_weight=sw,
    )


def test_score_symbol_legacy_path_unchanged():
    """Backward compat: existing scoring path produces the same numbers it always did."""
    features = _features(momentum=0.05, rsi=60.0, vol=0.02)
    sigs = [_sig(0.5, conf=0.8)]
    cfg = ScoringConfig()
    intent = score_symbol(features, sigs, cfg, min_confidence=0.0, notional=1000.0)
    sentiment, _ = aggregate_sentiment(sigs, 0.0)
    assert "sentiment=" in intent.reason
    # exact reproduction of the old formula
    technical = 0.65 * (features.momentum / 0.08) + 0.35 * ((features.rsi - 50.0) / 50.0)
    vol_pen = features.volatility / 0.12
    expected = cfg.sentiment_weight * sentiment + cfg.momentum_weight * technical - cfg.volatility_weight * vol_pen
    assert intent.score == expected


def test_score_with_vector_buys_on_strong_positive():
    features = _features(momentum=0.04, rsi=58.0, vol=0.02)
    sigs = [_sig(0.8, conf=0.9), _sig(0.7, conf=0.85)]
    vec = build_sentiment_vector(sigs)
    cfg = ScoringConfig(buy_threshold=0.20)
    intent = score_with_vector(features, vec, cfg, notional=1000.0)
    assert intent.side == Side.BUY
    assert intent.score >= cfg.buy_threshold


def test_score_with_vector_blocked_by_low_agreement():
    features = _features(momentum=0.05, rsi=60.0, vol=0.02)
    sigs = [_sig(0.9, conf=0.9), _sig(-0.9, conf=0.9)]  # huge disagreement
    vec = build_sentiment_vector(sigs)
    cfg = ScoringConfig(buy_threshold=0.0, agreement_min=0.5)
    intent = score_with_vector(features, vec, cfg, notional=1000.0)
    assert intent.side == Side.HOLD
    assert "blocked_low_agreement" in intent.reason


def test_score_with_vector_fear_lowers_score():
    features = _features(momentum=0.05, rsi=60.0, vol=0.02)
    base = [_sig(0.8, conf=0.9)]
    fearful = [_sig(0.8, conf=0.9, fear=0.9)]
    cfg = ScoringConfig(fear_weight=0.3)
    s_base = score_with_vector(_features(), build_sentiment_vector(base), cfg, 1000.0)
    s_fear = score_with_vector(_features(), build_sentiment_vector(fearful), cfg, 1000.0)
    assert s_fear.score < s_base.score


def test_score_with_vector_source_weighting_helps_credible_sources():
    features = _features(momentum=0.0, rsi=50.0, vol=0.02)
    cred = [_sig(1.0, conf=0.9, sw=1.0), _sig(-0.9, conf=0.9, sw=0.1)]
    no_w = [_sig(1.0, conf=0.9, sw=1.0), _sig(-0.9, conf=0.9, sw=1.0)]
    cfg_on = ScoringConfig(use_source_weighted_polarity=True)
    cfg_off = ScoringConfig(use_source_weighted_polarity=False)
    s_on = score_with_vector(features, build_sentiment_vector(cred), cfg_on, 1000.0)
    s_off = score_with_vector(features, build_sentiment_vector(no_w), cfg_off, 1000.0)
    assert s_on.score > s_off.score
