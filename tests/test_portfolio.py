from __future__ import annotations

import pytest

from crypto_llm_alpaca.config import (
    AppConfig,
    PortfolioConfig,
    ProjectProfile,
)
from crypto_llm_alpaca.portfolio import size_decisions


def _cfg(*, enabled=True, **kwargs) -> AppConfig:
    pc = PortfolioConfig(
        enabled=enabled,
        base_notional_pct=kwargs.get("base", 0.06),
        max_notional_pct=kwargs.get("maxn", 0.15),
        conviction_floor=kwargs.get("floor", 0.30),
        max_category_exposure_pct=kwargs.get("cap", 0.25),
    )
    profiles = {
        "AAVE/USD": ProjectProfile(category="defi_lending"),
        "UNI/USD": ProjectProfile(category="dex_infrastructure"),
        "CRV/USD": ProjectProfile(category="dex_infrastructure"),
        "ARB/USD": ProjectProfile(category="layer2_scaling"),
    }
    return AppConfig(portfolio=pc, project_profiles=profiles)


def test_disabled_returns_base_notional_unchanged():
    cfg = _cfg(enabled=False)
    out = size_decisions(
        [{"symbol": "AAVE/USD", "conviction": 0.9, "base_notional": 1000.0}],
        equity=100_000.0, cfg=cfg,
    )
    assert out[0].sized_notional == 1000.0
    assert out[0].reason == "portfolio_disabled"


def test_low_conviction_skipped():
    cfg = _cfg()
    out = size_decisions(
        [{"symbol": "AAVE/USD", "conviction": 0.1, "base_notional": 1000.0}],
        equity=100_000.0, cfg=cfg,
    )
    assert out[0].sized_notional == 0.0
    assert "below_floor" in out[0].reason


def test_high_conviction_scales_up_to_max():
    cfg = _cfg(base=0.06, maxn=0.15, floor=0.30)
    out = size_decisions(
        [{"symbol": "AAVE/USD", "conviction": 1.0, "base_notional": 6000.0}],
        equity=100_000.0, cfg=cfg,
    )
    # max_notional_pct * equity = 15_000 ; conviction=1.0 → full max
    assert out[0].sized_notional == pytest.approx(15_000.0, rel=0.01)


def test_category_cap_caps_total_per_category():
    cfg = _cfg(maxn=0.15, cap=0.20)  # equity=100k, cap = 20k per category
    candidates = [
        {"symbol": "UNI/USD", "conviction": 1.0, "base_notional": 6000.0},
        {"symbol": "CRV/USD", "conviction": 1.0, "base_notional": 6000.0},
    ]
    out = size_decisions(candidates, equity=100_000.0, cfg=cfg)
    total_dex = sum(s.sized_notional for s in out)
    assert total_dex <= 20_000.0 + 1e-3
    # second hit should be the one capped (less or zero)
    assert out[1].sized_notional <= out[0].sized_notional


def test_existing_exposure_reduces_headroom():
    cfg = _cfg(maxn=0.15, cap=0.20)
    # 18k already in dex; cap=20k → only 2k headroom
    out = size_decisions(
        [{"symbol": "UNI/USD", "conviction": 1.0, "base_notional": 6000.0}],
        equity=100_000.0, cfg=cfg,
        existing_exposure_by_category={"dex_infrastructure": 18_000.0},
    )
    assert out[0].sized_notional == pytest.approx(2_000.0, rel=0.01)


def test_zero_equity_drops_all():
    cfg = _cfg()
    out = size_decisions(
        [{"symbol": "AAVE/USD", "conviction": 0.9, "base_notional": 1000.0}],
        equity=0.0, cfg=cfg,
    )
    assert out[0].sized_notional == 0.0
    assert out[0].reason == "zero_equity"
