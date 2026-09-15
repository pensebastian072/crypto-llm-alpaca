"""Conviction-based position sizing + per-category exposure caps.

Used by autopilot to override default notional per accepted decision.
Pure helper — no I/O, no broker calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .config import AppConfig, PortfolioConfig


@dataclass
class SizedDecision:
    symbol: str
    category: str
    conviction: float        # tradable_score input (0..1)
    base_notional: float     # what the upstream gate proposed
    sized_notional: float    # what portfolio layer set; 0 if dropped
    reason: str              # explanation for the sizing decision


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _category_for(cfg: AppConfig, symbol: str) -> str:
    profile = cfg.project_profiles.get(symbol)
    if profile and profile.category:
        return profile.category
    return "uncategorized"


def size_decisions(
    candidates: Iterable[dict],
    *,
    equity: float,
    cfg: AppConfig,
    existing_exposure_by_category: Optional[dict[str, float]] = None,
) -> list[SizedDecision]:
    """Apply conviction sizing + category caps.

    `candidates` is a list of dicts with at least:
        symbol: str
        conviction: float (0..1) — typically fund_style.tradable_score
        base_notional: float — what the upstream layer suggested
    Order matters: highest-priority candidates first; sizing is greedy by order.
    """
    p: PortfolioConfig = cfg.portfolio
    existing = dict(existing_exposure_by_category or {})
    results: list[SizedDecision] = []

    if equity <= 0:
        for c in candidates:
            results.append(SizedDecision(
                symbol=c["symbol"],
                category=_category_for(cfg, c["symbol"]),
                conviction=float(c.get("conviction", 0.0)),
                base_notional=float(c.get("base_notional", 0.0)),
                sized_notional=0.0,
                reason="zero_equity",
            ))
        return results

    base_notional_dollars = p.base_notional_pct * equity
    max_notional_dollars = p.max_notional_pct * equity
    cat_cap_dollars = p.max_category_exposure_pct * equity

    for c in candidates:
        symbol = c["symbol"]
        conviction = float(c.get("conviction", 0.0))
        base = float(c.get("base_notional", base_notional_dollars))
        category = _category_for(cfg, symbol)

        if not p.enabled:
            results.append(SizedDecision(
                symbol=symbol, category=category, conviction=conviction,
                base_notional=base, sized_notional=base,
                reason="portfolio_disabled",
            ))
            continue

        if conviction < p.conviction_floor:
            results.append(SizedDecision(
                symbol=symbol, category=category, conviction=conviction,
                base_notional=base, sized_notional=0.0,
                reason=f"conviction_below_floor ({conviction:.2f} < {p.conviction_floor})",
            ))
            continue

        # Linear scale from base→max as conviction goes floor→1.0
        denom = max(1e-6, 1.0 - p.conviction_floor)
        scale_t = _clamp((conviction - p.conviction_floor) / denom, 0.0, 1.0)
        target = base_notional_dollars + scale_t * (max_notional_dollars - base_notional_dollars)

        # Per-category cap
        already_in_cat = existing.get(category, 0.0)
        headroom = max(0.0, cat_cap_dollars - already_in_cat)
        if headroom <= 0:
            results.append(SizedDecision(
                symbol=symbol, category=category, conviction=conviction,
                base_notional=base, sized_notional=0.0,
                reason=f"category_cap_full ({category}: {already_in_cat:.0f}/{cat_cap_dollars:.0f})",
            ))
            continue

        sized = min(target, headroom)
        existing[category] = already_in_cat + sized
        results.append(SizedDecision(
            symbol=symbol, category=category, conviction=conviction,
            base_notional=base, sized_notional=sized,
            reason=f"sized scale_t={scale_t:.2f} cat_used={existing[category]:.0f}/{cat_cap_dollars:.0f}",
        ))

    return results
