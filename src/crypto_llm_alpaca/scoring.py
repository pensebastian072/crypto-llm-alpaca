from __future__ import annotations

from statistics import mean

from .config import ScoringConfig
from .features import normalize_momentum, normalize_rsi
from .schemas import LlmSignal, Side, TechnicalFeatures, TradeIntent
from .sentiment_aggregator import SentimentVector, build_sentiment_vector


def aggregate_sentiment(signals: list[LlmSignal], min_confidence: float) -> tuple[float, float]:
    usable = [s for s in signals if s.confidence >= min_confidence]
    if not usable:
        return 0.0, 0.0
    total_weight = sum(s.confidence for s in usable)
    sentiment = sum(s.sentiment * s.confidence for s in usable) / total_weight
    return sentiment, mean(s.confidence for s in usable)


def score_symbol(
    features: TechnicalFeatures,
    signals: list[LlmSignal],
    scoring: ScoringConfig,
    min_confidence: float,
    notional: float,
) -> TradeIntent:
    """Backward-compatible scalar scoring path. New callers should use score_with_vector."""
    sentiment, confidence = aggregate_sentiment(signals, min_confidence)
    technical = 0.65 * normalize_momentum(features.momentum) + 0.35 * normalize_rsi(features.rsi)
    volatility_penalty = max(0.0, min(1.0, features.volatility / 0.12))
    score = (
        scoring.sentiment_weight * sentiment
        + scoring.momentum_weight * technical
        - scoring.volatility_weight * volatility_penalty
    )
    if score >= scoring.buy_threshold:
        side = Side.BUY
    elif score <= scoring.sell_threshold:
        side = Side.SELL
    else:
        side = Side.HOLD
    return TradeIntent(
        symbol=features.symbol,
        side=side,
        score=score,
        confidence=confidence,
        notional=notional,
        reason=(
            f"sentiment={sentiment:.3f}; technical={technical:.3f}; "
            f"volatility_penalty={volatility_penalty:.3f}"
        ),
    )


def score_with_vector(
    features: TechnicalFeatures,
    vec: SentimentVector,
    scoring: ScoringConfig,
    notional: float,
) -> TradeIntent:
    """Sentiment v2 path: multi-dim vector with lag decay + fear/hype + agreement gate."""
    polarity = vec.source_weighted_polarity if scoring.use_source_weighted_polarity else vec.polarity
    decayed = (
        polarity
        + scoring.lag_decay * vec.lag_1h
        + (scoring.lag_decay ** 2) * vec.lag_6h
        + (scoring.lag_decay ** 3) * vec.lag_24h
    ) / (1.0 + scoring.lag_decay + scoring.lag_decay ** 2 + scoring.lag_decay ** 3)
    technical = 0.65 * normalize_momentum(features.momentum) + 0.35 * normalize_rsi(features.rsi)
    volatility_penalty = max(0.0, min(1.0, features.volatility / 0.12))
    fear_term = scoring.fear_weight * vec.fear
    hype_term = scoring.hype_weight * vec.hype
    score = (
        scoring.sentiment_weight * decayed
        + scoring.momentum_weight * technical
        - scoring.volatility_weight * volatility_penalty
        - fear_term
        + hype_term
    )
    blocked_by_agreement = scoring.agreement_min > 0.0 and vec.agreement < scoring.agreement_min
    if score >= scoring.buy_threshold and not blocked_by_agreement:
        side = Side.BUY
    elif score <= scoring.sell_threshold:
        side = Side.SELL
    else:
        side = Side.HOLD
    reason = (
        f"polarity={decayed:+.3f}; tech={technical:+.3f}; "
        f"vol_pen={volatility_penalty:.3f}; fear={vec.fear:.2f}; hype={vec.hype:.2f}; "
        f"agreement={vec.agreement:.2f}; n_sources={vec.n_sources}"
    )
    if blocked_by_agreement:
        reason = "blocked_low_agreement; " + reason
    return TradeIntent(
        symbol=features.symbol, side=side,
        score=score, confidence=vec.confidence,
        notional=notional, reason=reason,
    )
