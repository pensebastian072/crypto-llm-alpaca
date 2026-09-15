"""Multi-dim sentiment aggregator with lag windows + source weighting.

Replaces scalar sentiment in scoring. Backward-compatible: feeding bare
LlmSignal lists still works (defaults fill the new dims).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev

from .schemas import LlmSignal


@dataclass(frozen=True)
class SentimentVector:
    polarity: float                    # weighted mean sentiment ∈ [-1, 1]
    confidence: float                  # weighted mean confidence ∈ [0, 1]
    fear: float                        # weighted mean fear ∈ [0, 1]
    hype: float                        # weighted mean hype ∈ [0, 1]
    source_weighted_polarity: float    # polarity weighted by source × confidence
    lag_1h: float                      # mean polarity in (-1h, 0]
    lag_6h: float                      # mean polarity in (-6h, -1h]
    lag_24h: float                     # mean polarity in (-24h, -6h]
    n_sources: int                     # distinct providers + topics
    agreement: float                   # 1 - stdev(polarity); high = consensus
    topic_flags: dict[str, bool] = field(default_factory=dict)


def _weighted(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total


def build_sentiment_vector(
    signals: list[LlmSignal],
    *,
    asof: datetime | None = None,
    min_confidence: float = 0.0,
) -> SentimentVector:
    """Build the multi-dim vector from a flat list of signals.

    `asof` defines "now" for lag bucketing. Defaults to the most recent created_at.
    """
    if not signals:
        return SentimentVector(
            polarity=0.0, confidence=0.0, fear=0.0, hype=0.0,
            source_weighted_polarity=0.0,
            lag_1h=0.0, lag_6h=0.0, lag_24h=0.0,
            n_sources=0, agreement=1.0, topic_flags={},
        )
    usable = [s for s in signals if s.confidence >= min_confidence]
    if not usable:
        usable = list(signals)

    if asof is None:
        asof = max(s.created_at for s in usable)
    if asof.tzinfo is None:
        asof = asof.replace(tzinfo=timezone.utc)

    polarities = [s.sentiment for s in usable]
    confidences = [s.confidence for s in usable]
    fears = [s.fear for s in usable]
    hypes = [s.hype for s in usable]
    source_weights = [s.source_weight * s.symbol_confidence for s in usable]

    polarity = _weighted(polarities, confidences)
    confidence = mean(confidences) if confidences else 0.0
    fear = _weighted(fears, confidences)
    hype = _weighted(hypes, confidences)
    swp_weights = [c * sw for c, sw in zip(confidences, source_weights)]
    source_weighted_polarity = _weighted(polarities, swp_weights)

    def _bucket(window_start_hours: float, window_end_hours: float) -> float:
        items = [
            s.sentiment for s in usable
            if window_start_hours < (asof - (s.created_at if s.created_at.tzinfo else s.created_at.replace(tzinfo=timezone.utc))).total_seconds() / 3600.0 <= window_end_hours
        ]
        return mean(items) if items else 0.0

    lag_1h = _bucket(0.0, 1.0)
    lag_6h = _bucket(1.0, 6.0)
    lag_24h = _bucket(6.0, 24.0)

    providers = {s.provider for s in usable}
    topics = {s.topic for s in usable if s.topic}
    n_sources = len(providers) + len(topics)
    agreement = 1.0 - (pstdev(polarities) if len(polarities) > 1 else 0.0)
    if agreement < 0.0:
        agreement = 0.0

    topic_flags = {t: True for t in topics}

    return SentimentVector(
        polarity=polarity,
        confidence=confidence,
        fear=fear,
        hype=hype,
        source_weighted_polarity=source_weighted_polarity,
        lag_1h=lag_1h, lag_6h=lag_6h, lag_24h=lag_24h,
        n_sources=n_sources,
        agreement=agreement,
        topic_flags=topic_flags,
    )
