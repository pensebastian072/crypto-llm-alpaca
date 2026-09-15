from __future__ import annotations

import ast
from pathlib import Path

import pytest

from crypto_llm_alpaca.research_candidates import aggregate_by_category, compute_candidates


def test_research_dashboard_page_parses():
    page = Path(__file__).resolve().parents[1].joinpath(
        "src/crypto_llm_alpaca/ui/pages/9_Research_Dashboard.py"
    )
    assert page.exists(), "page 9 should exist"
    ast.parse(page.read_text(encoding="utf-8"))


def test_aggregate_by_category_empty():
    assert aggregate_by_category([]) == []


def test_aggregate_by_category_rolls_up_metrics():
    rows = [
        {
            "symbol": "AAVE/USD",
            "category": "defi_lending",
            "passes": True,
            "fund_style": {"research_score": 0.8, "tradable_score": 0.5},
            "defi": {"tvl": 14.65e9, "tvl_growth": 0.065, "fees_7d": 11.17e6},
            "narrative_profile": {"momentum": 0.4},
            "signals_7d": 5,
        },
        {
            "symbol": "UNI/USD",
            "category": "dex_infrastructure",
            "passes": False,
            "fund_style": {"research_score": 0.6, "tradable_score": 0.0},
            "defi": {"tvl": 5e9, "tvl_growth": 0.02, "fees_7d": 3e6},
            "narrative_profile": {"momentum": 0.1},
            "signals_7d": 2,
        },
        {
            "symbol": "CRV/USD",
            "category": "dex_infrastructure",
            "passes": False,
            "fund_style": {"research_score": 0.5, "tradable_score": 0.0},
            "defi": {"tvl": 2e9, "tvl_growth": -0.01, "fees_7d": 0.5e6},
            "narrative_profile": {"momentum": -0.05},
            "signals_7d": 1,
        },
    ]
    cats = aggregate_by_category(rows)
    assert {c["category"] for c in cats} == {"defi_lending", "dex_infrastructure"}
    dl = next(c for c in cats if c["category"] == "defi_lending")
    dex = next(c for c in cats if c["category"] == "dex_infrastructure")
    assert dl["symbols"] == 1
    assert dex["symbols"] == 2
    assert dex["signals_7d_total"] == 3
    assert dex["tvl_total"] == pytest.approx(7e9)
    # defi_lending should rank above dex (higher avg research)
    assert cats[0]["category"] == "defi_lending"


def test_compute_candidates_empty_when_no_profiles(tmp_path):
    """No profiles → no rows."""
    from crypto_llm_alpaca.config import (
        AppConfig, BrokerConfig, CostsConfig, DataConfig, LlmConfig, RegimeConfig,
        RiskConfig, ScoringConfig, StorageConfig, StrategyConfig, UniverseConfig,
        QualityGrowthConfig,
    )
    from crypto_llm_alpaca.storage import SQLiteStore

    db = tmp_path / "x.sqlite3"
    cfg = AppConfig(
        universe=UniverseConfig(symbols=("AAVE/USD",)),
        data=DataConfig(),
        llm=LlmConfig(),
        scoring=ScoringConfig(),
        strategy=StrategyConfig(),
        costs=CostsConfig(),
        risk=RiskConfig(),
        broker=BrokerConfig(),
        storage=StorageConfig(path=str(db)),
        regime=RegimeConfig(enabled=False),
        quality_growth=QualityGrowthConfig(use_defillama=False, use_narratives=False),
        per_symbol={},
        project_profiles={},
    )
    store = SQLiteStore(cfg.storage.path)
    store.init_schema()
    rows = compute_candidates(cfg, store, limit=5)
    assert rows == []
