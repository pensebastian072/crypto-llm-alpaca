from __future__ import annotations


def _status(ok: bool) -> str:
    return "OK" if ok else "BLOCK"


def _fmt_score(value: float) -> str:
    return f"{value:.3f}"


def _fmt_usd(value: float) -> str:
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if value > 0:
        return f"${value:,.0f}"
    return "missing"


def _fmt_pct(value) -> str:
    return "missing" if value is None else f"{float(value):+.2%}"


def generate_candidate_explanation(row: dict) -> dict:
    factors = row["fund_style"]
    sentiment = row["sentiment"]
    defi = row.get("defi", {})
    narrative_profile = row.get("narrative_profile", {})
    signals = int(row.get("signals", 0))
    reject_reasons = list(row.get("reject_reasons", ()))
    missing = list(factors.get("missing_data", ()))
    mtf_score = row.get("mtf_trend_score", factors.get("multi_timeframe", 0.0))

    if sentiment["source_weighted_polarity"] > 0.25 and signals >= 3:
        thesis = "Positive repeated news flow supports the project profile."
    elif factors["narrative"] >= 0.60:
        thesis = f"Project profile and narrative are strongest in {row['category']}."
    elif row["passes"]:
        thesis = "Technical confirmation is present, with research quality still mixed."
    else:
        thesis = "Good watchlist candidate, but execution gates are not confirmed."

    return {
        "thesis": thesis,
        "signals": {
            "count": signals,
            "sentiment": sentiment["source_weighted_polarity"],
            "confidence": sentiment["confidence"],
            "summary": "confidence-weighted sentiment from attributed LLM signals",
        },
        "fundamentals": {
            "score": factors["fundamentals"],
            "checks": [
                f"manual quality profile {_status(row['profile_score'] >= 0.60)}",
                f"DefiLlama TVL {_status(bool(defi.get('tvl')))}",
                f"fee growth data {_status('fee_growth' in defi or bool(defi.get('fees_7d')))}",
                f"growth/backer profile {_status(factors['adoption'] >= 0.50)}",
            ],
            "tvl": defi.get("tvl", factors.get("tvl", 0.0)),
            "tvl_growth": defi.get("tvl_growth"),
            "fee_growth": defi.get("fee_growth"),
            "fees_7d": defi.get("fees_7d", 0.0),
        },
        "market_structure": {
            "score": factors["market_structure"],
            "checks": [
                f"momentum gate {_status(factors['momentum_gate'])}",
                f"volume gate {_status(factors['volume_gate'])}",
                f"multi-timeframe {_status(mtf_score >= 0.45)}",
                f"breakout {_status(bool(row['is_breakout']))}",
            ],
        },
        "multi_timeframe": {
            "score": mtf_score,
            "trend_4h": row.get("trend_4h", 0.0),
            "trend_1d": row.get("trend_1d", 0.0),
            "trend_1w": row.get("trend_1w", 0.0),
        },
        "liquidity": {
            "score": factors["liquidity"],
            "checks": [f"profile plus relative volume {_status(factors['liquidity'] >= 0.50)}"],
        },
        "narrative": {
            "score": factors["narrative"],
            "summary": narrative_profile.get("label") or factors.get("narrative_label") or row["category"] or "No category profile",
            "momentum": narrative_profile.get("momentum", factors.get("narrative_momentum", 0.0)),
            "source_count": narrative_profile.get("source_count", 0),
        },
        "risks": [
            f"risk penalty {_fmt_score(factors['risk_penalty'])}",
            f"atr_pct {row['atr_pct']:.2%}",
        ],
        "not_tradable": reject_reasons if reject_reasons else [],
        "missing_data": missing,
    }


def format_candidate_report(rows: list[dict]) -> str:
    blocks: list[str] = []
    for row in rows:
        explanation = row["explanation"]
        factors = row["fund_style"]
        sentiment = explanation["signals"]
        blocks.extend(
            [
                row["symbol"],
                f"Execution Score: {_fmt_score(row['score'])}",
                f"Research Score: {_fmt_score(factors['research_score'])}",
                f"Tradable Score: {_fmt_score(factors['tradable_score'])}",
                "",
                "Thesis:",
                explanation["thesis"],
                "",
                "Signals:",
                (
                    f"- Sentiment: {sentiment['sentiment']:+.3f} "
                    f"({sentiment['count']} signals, confidence {sentiment['confidence']:.2f})"
                ),
                "",
                "Fundamentals:",
                f"- Score: {_fmt_score(explanation['fundamentals']['score'])}",
                f"- DefiLlama TVL: {_fmt_usd(float(explanation['fundamentals']['tvl'] or 0.0))}",
                f"- TVL 7d growth: {_fmt_pct(explanation['fundamentals']['tvl_growth'])}",
                f"- Fee growth: {_fmt_pct(explanation['fundamentals']['fee_growth'])}",
                f"- Fees 7d: {_fmt_usd(float(explanation['fundamentals']['fees_7d'] or 0.0))}",
                *[f"- {item}" for item in explanation["fundamentals"]["checks"]],
                "",
                "Market Structure:",
                f"- Score: {_fmt_score(explanation['market_structure']['score'])}",
                *[f"- {item}" for item in explanation["market_structure"]["checks"]],
                "",
                "Multi-Timeframe:",
                f"- Score: {_fmt_score(explanation['multi_timeframe']['score'])}",
                f"- 4H trend: {explanation['multi_timeframe']['trend_4h']:+.2%}",
                f"- Daily trend: {explanation['multi_timeframe']['trend_1d']:+.2%}",
                f"- Weekly trend: {explanation['multi_timeframe']['trend_1w']:+.2%}",
                "",
                "Liquidity:",
                f"- Score: {_fmt_score(explanation['liquidity']['score'])}",
                *[f"- {item}" for item in explanation["liquidity"]["checks"]],
                "",
                "Narrative:",
                f"- Score: {_fmt_score(explanation['narrative']['score'])}",
                f"- {explanation['narrative']['summary']}",
                f"- Momentum: {float(explanation['narrative']['momentum']):+.3f}",
                f"- Sources: {int(explanation['narrative']['source_count'])}",
                "",
                "Risks:",
                *[f"- {item}" for item in explanation["risks"]],
                "",
                "Not Tradable:" if explanation["not_tradable"] else "Tradable:",
                *(
                    [f"- {item}" for item in explanation["not_tradable"]]
                    if explanation["not_tradable"]
                    else ["- passes current hard gates"]
                ),
                "",
                "Missing Data:",
                *[f"- {item}" for item in explanation["missing_data"]],
                "",
                "=" * 60,
                "",
            ]
        )
    return "\n".join(blocks).rstrip()
