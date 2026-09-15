from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from crypto_llm_alpaca.schemas import LlmSignal
from crypto_llm_alpaca.sentiment_aggregator import build_sentiment_vector


def _sig(
    sym,
    sent,
    conf=0.8,
    fear=0.0,
    hype=0.0,
    topic="",
    sw=1.0,
    symbol_conf=1.0,
    prov="mock",
    t=None,
) -> LlmSignal:
    return LlmSignal(
        news_hash=f"h-{sym}-{sent}-{prov}-{topic}-{conf}",
        symbol=sym, sentiment=sent, confidence=conf,
        horizon="short", event_type="t", rationale="r",
        provider=prov, model="m", fear=fear, hype=hype, topic=topic, source_weight=sw,
        symbol_confidence=symbol_conf,
        created_at=t or datetime.now(tz=timezone.utc),
    )


def test_empty_returns_zero_vector():
    v = build_sentiment_vector([])
    assert v.polarity == 0.0
    assert v.confidence == 0.0
    assert v.n_sources == 0
    assert v.agreement == 1.0


def test_polarity_is_confidence_weighted():
    sigs = [
        _sig("BTC", 1.0, conf=1.0),
        _sig("BTC", -1.0, conf=0.0),  # zero conf should not pull polarity
    ]
    v = build_sentiment_vector(sigs)
    assert v.polarity == pytest.approx(1.0, abs=1e-6)


def test_source_weighted_polarity_uses_source_weight():
    sigs = [
        _sig("BTC", 1.0, conf=1.0, sw=1.0, prov="cryptonative"),
        _sig("BTC", -1.0, conf=1.0, sw=0.1, prov="legacymedia"),
    ]
    v = build_sentiment_vector(sigs)
    assert v.source_weighted_polarity > v.polarity


def test_source_weighted_polarity_uses_symbol_confidence():
    sigs = [
        _sig("BTC", 1.0, conf=1.0, sw=1.0, symbol_conf=1.0, prov="curated"),
        _sig("BTC", -1.0, conf=1.0, sw=1.0, symbol_conf=0.1, prov="weak_attribution"),
    ]
    v = build_sentiment_vector(sigs)
    assert v.source_weighted_polarity > v.polarity


def test_lag_buckets():
    now = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)
    sigs = [
        _sig("BTC", 1.0, t=now - timedelta(minutes=30)),  # in (0,1h]
        _sig("BTC", -1.0, t=now - timedelta(hours=3)),    # in (1h,6h]
        _sig("BTC", 0.5, t=now - timedelta(hours=12)),    # in (6h,24h]
    ]
    v = build_sentiment_vector(sigs, asof=now)
    assert v.lag_1h == pytest.approx(1.0)
    assert v.lag_6h == pytest.approx(-1.0)
    assert v.lag_24h == pytest.approx(0.5)


def test_agreement_low_when_signals_disagree():
    sigs = [_sig("BTC", 1.0), _sig("BTC", -1.0)]
    v = build_sentiment_vector(sigs)
    assert v.agreement < 0.5


def test_agreement_high_when_signals_align():
    sigs = [_sig("BTC", 0.8), _sig("BTC", 0.85), _sig("BTC", 0.82)]
    v = build_sentiment_vector(sigs)
    assert v.agreement > 0.9


def test_topic_flags_collected():
    sigs = [
        _sig("BTC", 1.0, topic="etf"),
        _sig("BTC", 0.5, topic="adoption"),
    ]
    v = build_sentiment_vector(sigs)
    assert v.topic_flags.get("etf") is True
    assert v.topic_flags.get("adoption") is True


def test_fear_and_hype_aggregate():
    sigs = [
        _sig("BTC", -0.5, conf=0.9, fear=0.8, hype=0.0),
        _sig("BTC", -0.3, conf=0.5, fear=0.4, hype=0.1),
    ]
    v = build_sentiment_vector(sigs)
    assert 0.4 < v.fear < 0.8
    assert v.hype < 0.1
