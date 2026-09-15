from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from crypto_llm_alpaca.cache import load_cache, save_cache
from crypto_llm_alpaca.defillama import historical_slugs_for_symbol, extract_defi_metrics
from crypto_llm_alpaca.research.narratives import aggregate_symbol_narrative, detect_narrative
from crypto_llm_alpaca.schemas import NewsItem
from crypto_llm_alpaca.storage import SQLiteStore


def _news(hash_id: str, summary: str, published_at: datetime) -> NewsItem:
    return NewsItem(
        source="fixture",
        url=f"https://example.com/{hash_id}",
        title="Aave lending liquidity update",
        summary=summary,
        published_at=published_at,
        symbols=("AAVE/USD",),
        dedupe_hash=hash_id,
    )


def test_cache_round_trips_json_payload(tmp_path):
    save_cache("llama/protocols", {"ok": True}, cache_dir=tmp_path)

    assert load_cache("llama/protocols", 60, cache_dir=tmp_path) == {"ok": True}


def test_extract_defi_metrics_uses_protocol_and_fee_fixtures():
    metrics = extract_defi_metrics(
        "AAVE/USD",
        protocols=[{"slug": "aave-v3", "tvl": 10_000_000, "change_7d": 5.0, "change_1m": 12.0}],
        fees_summary={"total24h": 120.0, "total48hto24h": 100.0, "total7d": 700.0, "total30d": 3_000.0},
    )

    assert metrics["defillama_slug"] == "aave-v3"
    assert metrics["defillama_fee_slug"] == "aave"
    assert metrics["tvl"] == 10_000_000
    assert metrics["tvl_growth"] == 0.05
    assert metrics["fee_growth"] == pytest.approx(0.2)
    assert metrics["fees_7d"] == 700.0


def test_historical_slugs_use_protocol_history_coverage():
    assert historical_slugs_for_symbol("UNI/USD")[0] == "uniswap"
    assert historical_slugs_for_symbol("CRV/USD")[0] == "curve-finance"
    assert historical_slugs_for_symbol("YFI/USD")[0] == "yearn"
    assert historical_slugs_for_symbol("SUSHI/USD")[0] == "sushiswap"


def test_narrative_detection_maps_lending_news_to_defi_lending():
    match = detect_narrative("Aave lending liquidity and yield demand improves")

    assert match.label == "defi_lending"
    assert match.score > 0


def test_aggregate_symbol_narrative_tracks_momentum():
    asof = datetime(2026, 5, 5, tzinfo=timezone.utc)
    profile = aggregate_symbol_narrative(
        "AAVE/USD",
        [
            _news("old", "Small governance note.", asof - timedelta(days=10)),
            _news("new", "DeFi lending liquidity yield demand surges.", asof - timedelta(days=1)),
        ],
        asof=asof,
    )

    assert profile.label == "defi_lending"
    assert profile.current_score > profile.previous_score
    assert profile.momentum > 0


def test_storage_lists_news_for_symbol(tmp_path):
    store = SQLiteStore(tmp_path / "news.sqlite3")
    store.init_schema()
    item = _news("stored", "Aave lending liquidity improves.", datetime(2026, 5, 5, tzinfo=timezone.utc))
    store.upsert_news([item])

    assert store.list_news_for_symbol("AAVE/USD")[0].dedupe_hash == "stored"
    assert store.list_news_for_symbol("UNI/USD") == []
