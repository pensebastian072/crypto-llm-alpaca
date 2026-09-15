from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import ProjectProfile, QualityGrowthConfig
from .quality_metrics import build_metrics, compute_quality_breakdown
from .research.narratives import NarrativeProfile
from .schemas import StrategyFeatures
from .sentiment_aggregator import SentimentVector


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def project_profile_score(profile: ProjectProfile) -> float:
    """Quality/growth/backer score, with risk treated as a drag."""

    risk_safety = 1.0 - _clamp(profile.risk_score)
    return _clamp(
        0.30 * _clamp(profile.quality_score)
        + 0.25 * _clamp(profile.growth_score)
        + 0.20 * _clamp(profile.backer_score)
        + 0.15 * _clamp(profile.liquidity_score)
        + 0.10 * risk_safety
    )


@dataclass(frozen=True)
class QualityGrowthScore:
    score: float
    profile_score: float
    momentum_score: float
    volume_score: float
    trend_score: float
    sentiment_score: float
    reject_reasons: tuple[str, ...]

    @property
    def passes(self) -> bool:
        return not self.reject_reasons


@dataclass(frozen=True)
class FundStyleQualityScore:
    """Research score: fundamentals first, execution gates separate."""

    research_score: float
    tradable_score: float
    fundamentals: float
    adoption: float
    market_structure: float
    narrative: float
    sentiment: float
    liquidity: float
    risk_penalty: float
    multi_timeframe: float
    tvl: float
    tvl_growth: float | None
    fee_growth: float | None
    revenue_growth: float | None
    narrative_label: str
    narrative_momentum: float
    momentum_gate: bool
    volume_gate: bool
    missing_data: tuple[str, ...]


def quality_growth_score(
    profile: ProjectProfile,
    features: StrategyFeatures,
    cfg: QualityGrowthConfig,
    sentiment: SentimentVector | None = None,
) -> QualityGrowthScore:
    profile_component = project_profile_score(profile)
    momentum_component = _clamp(features.momentum / 0.18)
    relative_volume_component = _clamp((features.relative_volume - 1.0) / 2.5)
    volume_acceleration_component = _clamp((features.volume_acceleration - 1.0) / 2.0)
    volume_component = (relative_volume_component + volume_acceleration_component) / 2.0
    htf_component = _clamp(features.htf_trend / 0.20)
    trend_component = (htf_component + _clamp(features.mtf_trend_score)) / 2.0

    sentiment_component = 0.50
    if sentiment and sentiment.n_sources:
        polarity = sentiment.source_weighted_polarity
        sentiment_component = _clamp(0.50 + polarity * 0.50)
        sentiment_component *= _clamp(0.50 + sentiment.confidence * 0.50)
        sentiment_component *= _clamp(sentiment.agreement)

    weights = (
        cfg.profile_weight
        + cfg.momentum_weight
        + cfg.volume_weight
        + cfg.trend_weight
        + cfg.sentiment_weight
    )
    if weights <= 0:
        weights = 1.0

    score = (
        profile_component * cfg.profile_weight
        + momentum_component * cfg.momentum_weight
        + volume_component * cfg.volume_weight
        + trend_component * cfg.trend_weight
        + sentiment_component * cfg.sentiment_weight
    ) / weights

    reject_reasons: list[str] = []
    if profile_component < cfg.min_profile_score:
        reject_reasons.append(f"profile<{cfg.min_profile_score:.2f}")
    if features.momentum < cfg.min_momentum_pct:
        reject_reasons.append(f"momentum<{cfg.min_momentum_pct:.2%}")
    if features.relative_volume < cfg.min_relative_volume:
        reject_reasons.append(f"rel_volume<{cfg.min_relative_volume:.2f}")
    if features.relative_volume > cfg.max_relative_volume:
        reject_reasons.append(f"rel_volume>{cfg.max_relative_volume:.2f}")
    if features.volume_acceleration < cfg.min_volume_acceleration:
        reject_reasons.append(f"vol_accel<{cfg.min_volume_acceleration:.2f}")
    if features.htf_trend < cfg.min_htf_trend_pct:
        reject_reasons.append(f"htf_trend<{cfg.min_htf_trend_pct:.2%}")
    if cfg.require_mtf_alignment and features.mtf_trend_score < cfg.min_mtf_trend_score:
        reject_reasons.append(f"mtf_trend<{cfg.min_mtf_trend_score:.2f}")
    if features.atr_pct > cfg.max_atr_pct:
        reject_reasons.append(f"atr_pct>{cfg.max_atr_pct:.2%}")
    if not features.is_breakout and features.momentum < cfg.min_momentum_pct * 1.75:
        reject_reasons.append("no_breakout_or_momentum_extension")

    return QualityGrowthScore(
        score=_clamp(score),
        profile_score=profile_component,
        momentum_score=momentum_component,
        volume_score=volume_component,
        trend_score=trend_component,
        sentiment_score=sentiment_component,
        reject_reasons=tuple(reject_reasons),
    )


def fund_style_quality_score(
    profile: ProjectProfile,
    features: StrategyFeatures,
    cfg: QualityGrowthConfig,
    sentiment: SentimentVector | None = None,
    *,
    signal_count_7d: int = 0,
    volume_usd: float = 0.0,
    defi_metrics: Mapping[str, Any] | None = None,
    narrative: NarrativeProfile | Mapping[str, Any] | None = None,
) -> FundStyleQualityScore:
    """Fund-style research model with momentum/volume kept as hard execution gates.

    This uses existing local data only. External protocol metrics can replace the
    profile/news proxies later without changing the candidate CLI shape.
    """

    breakout_strength = 1.0 if features.is_breakout else _clamp(
        features.momentum / max(cfg.min_momentum_pct * 1.75, 1e-9)
    )
    trend_alignment = _clamp((features.htf_trend / 0.20 + features.mtf_trend_score) / 2.0)
    relative_strength = _clamp(features.momentum / 0.18)
    momentum_gate = (
        features.momentum >= cfg.min_momentum_pct
        and (features.is_breakout or features.momentum >= cfg.min_momentum_pct * 1.75)
        and (not cfg.require_mtf_alignment or features.mtf_trend_score >= cfg.min_mtf_trend_score)
    )
    volume_gate = (
        features.relative_volume >= cfg.min_relative_volume
        and features.relative_volume <= cfg.max_relative_volume
        and features.volume_acceleration >= cfg.min_volume_acceleration
    )

    sentiment_polarity = sentiment.source_weighted_polarity if sentiment else 0.0
    defi = dict(defi_metrics or {})
    narrative_data = _narrative_data(narrative)
    narrative_score = _clamp(0.45 * profile.growth_score + 0.35 * profile.backer_score)
    if sentiment and (sentiment.n_sources or signal_count_7d):
        narrative_score = _clamp(
            narrative_score
            + 0.10 * sentiment.hype
            + 0.05 * min(len(sentiment.topic_flags), 3)
            - 0.10 * sentiment.fear
        )
    if narrative_data["score"] > 0:
        narrative_score = _clamp(
            0.55 * narrative_score
            + 0.35 * narrative_data["score"]
            + 0.10 * max(narrative_data["momentum"], 0.0)
        )

    tvl_growth = defi.get("tvl_growth", profile.growth_score * 2.0 - 1.0)
    revenue_growth = defi.get("revenue_growth", profile.quality_score * 2.0 - 1.0)
    fee_growth = defi.get("fee_growth", profile.liquidity_score * 2.0 - 1.0)
    missing_data = _missing_data(defi)
    metrics = build_metrics(
        {
            "tvl_growth": tvl_growth,
            "revenue_growth": revenue_growth,
            "fee_growth": fee_growth,
            "addr_growth": profile.growth_score * 2.0 - 1.0,
            "tx_growth": profile.growth_score * 2.0 - 1.0,
            "dev_score": profile.backer_score * 100.0,
            "momentum": _clamp((_clamp(features.momentum / 0.18) + breakout_strength) / 2.0),
            "trend": trend_alignment,
            "rel_strength": relative_strength,
            "mtf_trend_score": features.mtf_trend_score,
            "trend_4h": features.trend_4h,
            "trend_1d": features.trend_1d,
            "trend_1w": features.trend_1w,
            "sentiment": sentiment_polarity,
            "signal_count_7d": signal_count_7d,
            "volume": volume_usd,
            "narrative_score": narrative_score,
            "unlock_risk": profile.risk_score * 0.5,
            "gov_risk": profile.risk_score * 0.5,
            "momentum_gate": momentum_gate,
            "volume_gate": volume_gate,
        }
    )
    breakdown = compute_quality_breakdown(metrics, missing_data=missing_data)

    return FundStyleQualityScore(
        research_score=_clamp(breakdown.score),
        tradable_score=_clamp(breakdown.tradable_score),
        fundamentals=breakdown.fundamentals,
        adoption=breakdown.adoption,
        market_structure=breakdown.market_structure,
        narrative=breakdown.narrative,
        sentiment=breakdown.sentiment,
        liquidity=breakdown.liquidity,
        risk_penalty=breakdown.risk_penalty,
        multi_timeframe=breakdown.multi_timeframe,
        tvl=float(defi.get("tvl", 0.0) or 0.0),
        tvl_growth=float(tvl_growth) if tvl_growth is not None else None,
        fee_growth=float(fee_growth) if fee_growth is not None else None,
        revenue_growth=float(revenue_growth) if revenue_growth is not None else None,
        narrative_label=str(narrative_data["label"]),
        narrative_momentum=float(narrative_data["momentum"]),
        momentum_gate=momentum_gate,
        volume_gate=volume_gate,
        missing_data=breakdown.missing_data,
    )


def _missing_data(defi: Mapping[str, Any]) -> tuple[str, ...]:
    missing: list[str] = []
    if not defi.get("tvl"):
        missing.append("protocol_tvl")
    if "fee_growth" not in defi and not defi.get("fees_7d"):
        missing.append("fee_generation")
    if "revenue_growth" not in defi:
        missing.append("protocol_revenue")
    missing.extend(("developer_activity", "token_unlock_schedule"))
    return tuple(missing)


def _narrative_data(narrative: NarrativeProfile | Mapping[str, Any] | None) -> dict[str, float | str]:
    if narrative is None:
        return {"label": "", "score": 0.0, "momentum": 0.0}
    if isinstance(narrative, NarrativeProfile):
        return {
            "label": narrative.label,
            "score": _clamp(narrative.score),
            "momentum": narrative.momentum,
        }
    return {
        "label": str(narrative.get("label", "")),
        "score": _clamp(float(narrative.get("score", 0.0) or 0.0)),
        "momentum": float(narrative.get("momentum", 0.0) or 0.0),
    }
