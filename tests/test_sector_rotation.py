from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

from crypto_llm_alpaca.research.narratives import (
    SectorRotation,
    aggregate_sector_rotation,
)
from crypto_llm_alpaca.schemas import NewsItem


def _item(title: str, summary: str, ts: datetime, symbols=("BTC/USD",), idx: int = 0) -> NewsItem:
    return NewsItem(
        source="test",
        url=f"https://x/{idx}",
        title=title,
        summary=summary,
        published_at=ts,
        symbols=symbols,
        dedupe_hash=f"h-{idx}-{title[:8]}",
    )


def test_empty_returns_empty_list():
    assert aggregate_sector_rotation([]) == []


def test_uptrend_sector_has_positive_momentum():
    asof = datetime(2026, 5, 10, tzinfo=timezone.utc)
    items = [
        # Prior 7d: little defi lending news
        _item("uniswap volume", "dex swap exchange amm fees", asof - timedelta(days=10), idx=1),
        # Current 7d: lots of defi lending news
        _item("aave lending surge", "defi lending borrow collateral aave liquidity yield", asof - timedelta(days=2), idx=2),
        _item("collateral up", "defi lending borrow collateral liquidity yield aave", asof - timedelta(days=3), idx=3),
        _item("aave yield up", "defi lending yield aave borrow collateral", asof - timedelta(days=1), idx=4),
    ]
    rots = aggregate_sector_rotation(items, asof=asof, current_days=7, previous_days=7)
    assert rots, "should produce at least one sector"
    lending = next((r for r in rots if r.label == "defi_lending"), None)
    assert lending is not None
    assert lending.current_articles >= 3
    assert lending.previous_articles == 0
    assert lending.momentum > 0


def test_top_symbols_ordered_by_count():
    asof = datetime(2026, 5, 10, tzinfo=timezone.utc)
    items = [
        _item("aave", "defi lending borrow aave", asof - timedelta(days=1), symbols=("AAVE/USD",), idx=1),
        _item("aave", "defi lending borrow aave", asof - timedelta(days=2), symbols=("AAVE/USD",), idx=2),
        _item("aave", "defi lending borrow aave", asof - timedelta(days=3), symbols=("AAVE/USD",), idx=3),
        _item("compound", "defi lending borrow", asof - timedelta(days=1), symbols=("COMP/USD",), idx=4),
    ]
    rots = aggregate_sector_rotation(items, asof=asof, current_days=7, previous_days=7)
    lending = next((r for r in rots if r.label == "defi_lending"), None)
    assert lending is not None
    assert lending.top_symbols[0] == "AAVE/USD"


def test_results_sorted_by_momentum_desc():
    asof = datetime(2026, 5, 10, tzinfo=timezone.utc)
    # Two sectors: defi lending growing, dex declining
    items = [
        # Prior dex spike
        _item("uniswap", "dex swap exchange amm fees uniswap curve", asof - timedelta(days=10), idx=1),
        _item("uniswap", "dex swap exchange amm fees uniswap curve", asof - timedelta(days=11), idx=2),
        # Current lending spike
        _item("aave", "defi lending borrow aave liquidity yield collateral", asof - timedelta(days=1), idx=3),
        _item("aave", "defi lending borrow aave liquidity yield collateral", asof - timedelta(days=2), idx=4),
        _item("aave", "defi lending borrow aave liquidity yield collateral", asof - timedelta(days=3), idx=5),
    ]
    rots = aggregate_sector_rotation(items, asof=asof, current_days=7, previous_days=7)
    if len(rots) >= 2:
        # First entry must have momentum >= last entry
        assert rots[0].momentum >= rots[-1].momentum


def test_page_10_parses():
    page = Path(__file__).resolve().parents[1].joinpath(
        "src/crypto_llm_alpaca/ui/pages/10_Narrative_Rotation.py"
    )
    assert page.exists()
    ast.parse(page.read_text(encoding="utf-8"))
