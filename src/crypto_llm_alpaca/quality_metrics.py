from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QualityMetrics:
    tvl_growth: float | None = None
    revenue_growth: float | None = None
    fee_growth: float | None = None
    active_addresses_growth: float | None = None
    tx_growth: float | None = None
    dev_activity: float | None = None
    momentum: float = 0.0
    trend: float = 0.0
    relative_strength: float = 0.0
    mtf_trend_score: float | None = None
    trend_4h: float = 0.0
    trend_1d: float = 0.0
    trend_1w: float = 0.0
    sentiment: float = 0.0
    signal_count_7d: int = 0
    volume_usd: float = 0.0
    narrative_score: float = 0.0
    unlock_risk: float = 0.0
    governance_risk: float = 0.0
    momentum_gate: bool = False
    volume_gate: bool = False


@dataclass(frozen=True)
class QualityBreakdown:
    score: float
    tradable_score: float
    fundamentals: float
    adoption: float
    market_structure: float
    narrative: float
    sentiment: float
    liquidity: float
    risk_penalty: float
    multi_timeframe: float
    missing_data: tuple[str, ...]


def safe_norm(x: float | None, min_val: float = -1.0, max_val: float = 1.0) -> float:
    if x is None:
        return 0.0
    if max_val == min_val:
        return 0.0
    return max(min((x - min_val) / (max_val - min_val), 1.0), 0.0)


def log_scale(x: float) -> float:
    return math.log1p(max(x, 0.0))


def compute_fundamentals(m: QualityMetrics) -> float:
    values = [
        safe_norm(m.tvl_growth),
        safe_norm(m.revenue_growth),
        safe_norm(m.fee_growth),
    ]
    return sum(values) / len(values)


def compute_adoption(m: QualityMetrics) -> float:
    values = [
        safe_norm(m.active_addresses_growth),
        safe_norm(m.tx_growth),
        safe_norm(m.dev_activity, 0.0, 100.0),
    ]
    return sum(values) / len(values)


def compute_multi_timeframe(m: QualityMetrics) -> float:
    if m.mtf_trend_score is not None:
        return max(0.0, min(1.0, m.mtf_trend_score))
    values = [
        safe_norm(m.trend_4h, -0.10, 0.10),
        safe_norm(m.trend_1d, -0.20, 0.20),
        safe_norm(m.trend_1w, -0.50, 0.50),
    ]
    return 0.30 * values[0] + 0.30 * values[1] + 0.40 * values[2]


def compute_market_structure(m: QualityMetrics) -> float:
    base = max(m.momentum, 0.0) * max(m.trend, 0.0) * max(m.relative_strength, 0.0)
    return min(1.0, base * (0.50 + 0.50 * compute_multi_timeframe(m)))


def compute_sentiment(m: QualityMetrics) -> float:
    if m.signal_count_7d == 0:
        return 0.0
    polarity = max(0.0, min(1.0, (m.sentiment + 1.0) / 2.0))
    frequency = min(1.0, log_scale(m.signal_count_7d) / log_scale(10))
    return polarity * frequency


def compute_liquidity(m: QualityMetrics, threshold: float = 5_000_000.0) -> float:
    return min(1.0, max(m.volume_usd, 0.0) / threshold)


def compute_risk_penalty(m: QualityMetrics) -> float:
    return -(max(m.unlock_risk, 0.0) + max(m.governance_risk, 0.0))


def build_metrics(asset_data: dict[str, Any]) -> QualityMetrics:
    return QualityMetrics(
        tvl_growth=asset_data.get("tvl_growth"),
        revenue_growth=asset_data.get("revenue_growth"),
        fee_growth=asset_data.get("fee_growth"),
        active_addresses_growth=asset_data.get("addr_growth"),
        tx_growth=asset_data.get("tx_growth"),
        dev_activity=asset_data.get("dev_score"),
        momentum=asset_data.get("momentum", 0.0),
        trend=asset_data.get("trend", 0.0),
        relative_strength=asset_data.get("rel_strength", 0.0),
        mtf_trend_score=asset_data.get("mtf_trend_score"),
        trend_4h=asset_data.get("trend_4h", 0.0),
        trend_1d=asset_data.get("trend_1d", 0.0),
        trend_1w=asset_data.get("trend_1w", 0.0),
        sentiment=asset_data.get("sentiment", 0.0),
        signal_count_7d=int(asset_data.get("signal_count_7d", 0) or 0),
        volume_usd=asset_data.get("volume", 0.0),
        narrative_score=asset_data.get("narrative_score", 0.0),
        unlock_risk=asset_data.get("unlock_risk", 0.0),
        governance_risk=asset_data.get("gov_risk", 0.0),
        momentum_gate=asset_data.get("momentum_gate", False),
        volume_gate=asset_data.get("volume_gate", False),
    )


def compute_quality_breakdown(m: QualityMetrics, *, missing_data: tuple[str, ...] = ()) -> QualityBreakdown:
    fundamentals = compute_fundamentals(m)
    adoption = compute_adoption(m)
    market = compute_market_structure(m)
    sentiment = compute_sentiment(m)
    liquidity = compute_liquidity(m)
    risk = compute_risk_penalty(m)
    narrative = max(0.0, min(1.0, m.narrative_score))
    multi_timeframe = compute_multi_timeframe(m)

    score = max(
        0.0,
        0.20 * fundamentals
        + 0.20 * adoption
        + 0.15 * market
        + 0.15 * narrative
        + 0.10 * sentiment
        + 0.10 * liquidity
        + 0.10 * risk,
    )
    return QualityBreakdown(
        score=score,
        tradable_score=score if (m.momentum_gate and m.volume_gate) else 0.0,
        fundamentals=fundamentals,
        adoption=adoption,
        market_structure=market,
        narrative=narrative,
        sentiment=sentiment,
        liquidity=liquidity,
        risk_penalty=risk,
        multi_timeframe=multi_timeframe,
        missing_data=missing_data,
    )


def compute_quality_score(m: QualityMetrics) -> float:
    return compute_quality_breakdown(m).tradable_score
