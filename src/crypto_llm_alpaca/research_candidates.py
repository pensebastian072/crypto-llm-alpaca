"""Pure helper that computes the quality-growth candidate ranking.

Factored out of cli.cmd_quality_growth_candidates so both the CLI and the
Streamlit research dashboard can call it without spawning a subprocess.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from .candidate_explain import generate_candidate_explanation
from .config import AppConfig
from .defillama import extract_defi_metrics
from .features import compute_strategy_features
from .quality_growth import fund_style_quality_score, quality_growth_score
from .research.narratives import aggregate_symbol_narrative
from .sentiment_aggregator import build_sentiment_vector
from .storage import SQLiteStore


def compute_candidates(
    cfg: AppConfig,
    store: SQLiteStore,
    *,
    limit: int = 10,
    signal_limit: int = 50,
    narrative_news_limit: int = 100,
    asof: Optional[datetime] = None,
) -> list[dict]:
    """Compute ranked quality-growth candidates. Returns list of explanation rows."""
    rows: list[dict] = []
    now = asof or datetime.now(tz=timezone.utc)
    for symbol in cfg.universe.symbols:
        bars = store.latest_bars(symbol, limit=max(cfg.data.lookback_bars, 160))
        profile = cfg.project_profiles.get(symbol)
        if not bars or not profile:
            continue
        try:
            features = compute_strategy_features(bars, cfg.strategy)
        except ValueError:
            continue
        signals = store.latest_signals(symbol, limit=signal_limit)
        vector = build_sentiment_vector(
            signals,
            asof=features.timestamp,
            min_confidence=cfg.llm.min_confidence,
        )
        score = quality_growth_score(profile, features, cfg.quality_growth, vector)
        signal_count_7d = sum(
            1
            for signal in signals
            if now - (signal.created_at if signal.created_at.tzinfo else signal.created_at.replace(tzinfo=timezone.utc))
            <= timedelta(days=7)
        )
        defi_metrics = (
            extract_defi_metrics(symbol, max_age_sec=cfg.quality_growth.defillama_cache_ttl_sec)
            if cfg.quality_growth.use_defillama
            else {}
        )
        narrative_profile = (
            aggregate_symbol_narrative(
                symbol,
                store.list_news_for_symbol(symbol, limit=narrative_news_limit),
                asof=now,
                current_days=cfg.quality_growth.narrative_current_days,
                previous_days=cfg.quality_growth.narrative_previous_days,
            )
            if cfg.quality_growth.use_narratives
            else None
        )
        fund_style = fund_style_quality_score(
            profile,
            features,
            cfg.quality_growth,
            vector,
            signal_count_7d=signal_count_7d,
            volume_usd=bars[-1].close * bars[-1].volume,
            defi_metrics=defi_metrics,
            narrative=narrative_profile,
        )
        row = {
            "symbol": symbol,
            "trade_enabled": cfg.is_trade_enabled(symbol),
            "score": score.score,
            "passes": score.passes,
            "reject_reasons": score.reject_reasons,
            "profile_score": score.profile_score,
            "momentum_score": score.momentum_score,
            "volume_score": score.volume_score,
            "trend_score": score.trend_score,
            "sentiment_score": score.sentiment_score,
            "momentum_pct": features.momentum,
            "relative_volume": features.relative_volume,
            "volume_acceleration": features.volume_acceleration,
            "htf_trend_pct": features.htf_trend,
            "trend_4h": features.trend_4h,
            "trend_1d": features.trend_1d,
            "trend_1w": features.trend_1w,
            "mtf_trend_score": features.mtf_trend_score,
            "atr_pct": features.atr_pct,
            "is_breakout": features.is_breakout,
            "category": profile.category,
            "thesis": profile.thesis,
            "signals": len(signals),
            "signals_7d": signal_count_7d,
            "signal_sources": vector.n_sources,
            "sentiment": {
                "polarity": vector.polarity,
                "source_weighted_polarity": vector.source_weighted_polarity,
                "confidence": vector.confidence,
                "fear": vector.fear,
                "hype": vector.hype,
                "agreement": vector.agreement,
            },
            "fund_style": asdict(fund_style),
            "defi": defi_metrics,
            "narrative_profile": asdict(narrative_profile) if narrative_profile else {},
        }
        row["explanation"] = generate_candidate_explanation(row)
        rows.append(row)
    rows.sort(
        key=lambda item: (item["passes"], item["fund_style"]["research_score"], item["score"]),
        reverse=True,
    )
    return rows[:limit]


def aggregate_by_category(rows: list[dict]) -> list[dict]:
    """Roll up candidates by `category` for sector heatmap."""
    bucket: dict[str, dict] = {}
    for r in rows:
        cat = r.get("category") or "uncategorized"
        b = bucket.setdefault(cat, {
            "category": cat, "n": 0, "passes": 0,
            "research_score_sum": 0.0, "tradable_score_sum": 0.0,
            "tvl_sum": 0.0, "tvl_growth_avg": 0.0,
            "narrative_momentum_avg": 0.0, "signals_7d_sum": 0,
            "_growth_n": 0, "_narr_n": 0,
        })
        b["n"] += 1
        b["passes"] += int(bool(r.get("passes")))
        b["research_score_sum"] += float(r["fund_style"].get("research_score", 0.0))
        b["tradable_score_sum"] += float(r["fund_style"].get("tradable_score", 0.0))
        b["signals_7d_sum"] += int(r.get("signals_7d", 0))
        defi = r.get("defi") or {}
        b["tvl_sum"] += float(defi.get("tvl") or 0.0)
        if defi.get("tvl_growth") is not None:
            b["tvl_growth_avg"] += float(defi["tvl_growth"])
            b["_growth_n"] += 1
        nprof = r.get("narrative_profile") or {}
        if "momentum" in nprof:
            b["narrative_momentum_avg"] += float(nprof["momentum"])
            b["_narr_n"] += 1
    out: list[dict] = []
    for cat, b in bucket.items():
        n = b["n"] or 1
        out.append({
            "category": cat,
            "symbols": n,
            "passes": b["passes"],
            "avg_research": b["research_score_sum"] / n,
            "avg_tradable": b["tradable_score_sum"] / n,
            "tvl_total": b["tvl_sum"],
            "avg_tvl_growth": (b["tvl_growth_avg"] / b["_growth_n"]) if b["_growth_n"] else None,
            "avg_narrative_momentum": (b["narrative_momentum_avg"] / b["_narr_n"]) if b["_narr_n"] else None,
            "signals_7d_total": b["signals_7d_sum"],
        })
    out.sort(key=lambda c: (c["avg_research"], c["passes"]), reverse=True)
    return out
