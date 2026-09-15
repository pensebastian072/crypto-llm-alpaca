from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

from .schemas import TakeProfitLevel


@dataclass(frozen=True)
class UniverseConfig:
    symbols: tuple[str, ...] = ("BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD")
    quote_currency: str = "USD"
    allow_shorts: bool = False


@dataclass(frozen=True)
class DataConfig:
    timeframe: str = "1Hour"
    lookback_bars: int = 120
    rss_feeds: tuple[str, ...] = ()


@dataclass(frozen=True)
class LlmConfig:
    provider: str = "mock"
    model: str = "gpt-4.1-mini"
    min_confidence: float = 0.55


@dataclass(frozen=True)
class ScoringConfig:
    buy_threshold: float = 0.35
    sell_threshold: float = -0.35
    risk_off_threshold: float = -0.40   # aggregate sentiment below this => block entries
    sentiment_weight: float = 0.45
    momentum_weight: float = 0.35
    volatility_weight: float = 0.20
    # Phase 3 v2 weights (only applied via score_with_vector path).
    fear_weight: float = 0.15           # subtract fear * weight
    hype_weight: float = 0.05           # add hype * weight (small — hype is noisy)
    lag_decay: float = 0.5              # lag features decay by this factor each bucket
    agreement_min: float = 0.0          # require agreement >= this (0 disables)
    use_source_weighted_polarity: bool = True


@dataclass(frozen=True)
class RegimeConfig:
    enabled: bool = True
    btc_ema_hours: int = 168
    vol_window_hours: int = 168
    vol_extreme_threshold: float = 0.06
    vol_high_threshold: float = 0.03
    vol_low_threshold: float = 0.01
    sensor_symbol: str = "BTC/USD"


@dataclass(frozen=True)
class StrategyConfig:
    breakout_lookback: int = 20
    volume_lookback: int = 20
    relative_volume_min: float = 1.25
    range_expansion_min: float = 1.10
    min_close_location: float = 0.60
    momentum_lookback: int = 12
    htf_timeframe: str = "4Hour"
    htf_trend_lookback: int = 6
    require_htf_trend: bool = True
    atr_window: int = 14
    atr_short_window: int = 6
    atr_long_window: int = 24
    atr_stop_multiple: float = 2.0
    atr_trail_multiple: float = 2.5
    min_expected_move_fee_multiple: float = 3.0
    top_n_per_bar: int = 3


@dataclass(frozen=True)
class QualityGrowthConfig:
    """Profile-first momentum strategy knobs for smaller crypto projects."""

    min_profile_score: float = 0.60
    min_momentum_pct: float = 0.035
    min_relative_volume: float = 1.35
    max_relative_volume: float = 75.0
    min_volume_acceleration: float = 1.05
    min_htf_trend_pct: float = 0.0
    max_atr_pct: float = 0.20
    min_expected_move_fee_multiple: float = 4.0
    min_mtf_trend_score: float = 0.45
    require_mtf_alignment: bool = False
    decision_interval_hours: int = 1
    use_defillama: bool = False
    defillama_cache_ttl_sec: int = 21_600
    use_narratives: bool = True
    narrative_current_days: int = 7
    narrative_previous_days: int = 7
    profile_weight: float = 0.35
    momentum_weight: float = 0.25
    volume_weight: float = 0.20
    trend_weight: float = 0.10
    sentiment_weight: float = 0.10
    atr_stop_multiple: float = 2.4
    atr_trail_multiple: float = 3.5
    top_n_per_bar: int = 3
    # Phase C: lookahead-free fund-style scoring during backtest.
    backtest_use_fund_style: bool = False
    backtest_news_window_days: int = 14
    # Phase D: opt-in probes for research candidates blocked by the legacy gate.
    backtest_allow_research_promotions: bool = False
    research_promotion_min_score: float = 0.20
    research_promotion_size_scale: float = 0.35
    research_promotion_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class CostsConfig:
    taker_bps: float = 25.0
    maker_bps: float = 15.0
    default_fill_model: str = "taker"

    @property
    def default_bps(self) -> float:
        return self.maker_bps if self.default_fill_model == "maker" else self.taker_bps


@dataclass(frozen=True)
class RiskConfig:
    max_total_exposure_pct: float = 0.30
    min_positions: int = 3
    max_positions: int = 5
    stop_loss_pct: float = 0.01
    risk_per_trade_pct: float = 0.01
    max_symbol_exposure_pct: float = 0.10
    daily_drawdown_stop_pct: float = 0.05
    volatility_spike_limit: float = 0.08
    take_profit_levels: tuple[TakeProfitLevel, ...] = field(
        default_factory=lambda: (
            TakeProfitLevel(0.10, 0.25),
            TakeProfitLevel(0.20, 0.25),
            TakeProfitLevel(0.50, 0.50),
        )
    )


@dataclass(frozen=True)
class BrokerConfig:
    paper_base_url: str = "https://paper-api.alpaca.markets"
    live_trading_enabled: bool = False


@dataclass(frozen=True)
class StorageConfig:
    path: str = "crypto_llm_alpaca.sqlite3"


@dataclass(frozen=True)
class PortfolioConfig:
    """Phase B: conviction-based sizing + per-category exposure caps."""
    enabled: bool = False
    base_notional_pct: float = 0.06            # of equity, neutral conviction
    max_notional_pct: float = 0.15             # of equity, max conviction
    conviction_floor: float = 0.30             # tradable_score below floor => skip
    max_category_exposure_pct: float = 0.25    # cap per-category as fraction of equity


@dataclass(frozen=True)
class PerSymbolOverride:
    """Optional per-symbol overrides. Unspecified fields fall back to global config."""
    trade_enabled: bool = True
    buy_threshold: float | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    max_notional_pct: float | None = None
    atr_stop_multiple: float | None = None


@dataclass(frozen=True)
class ProjectProfile:
    """Editable research priors for a project before technical confirmation."""

    category: str = ""
    thesis: str = ""
    quality_score: float = 0.50
    growth_score: float = 0.50
    backer_score: float = 0.50
    liquidity_score: float = 0.50
    risk_score: float = 0.50
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class AppConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    quality_growth: QualityGrowthConfig = field(default_factory=QualityGrowthConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    per_symbol: dict[str, PerSymbolOverride] = field(default_factory=dict)
    project_profiles: dict[str, ProjectProfile] = field(default_factory=dict)

    def symbol_override(self, symbol: str) -> PerSymbolOverride:
        return self.per_symbol.get(symbol, PerSymbolOverride())

    def is_trade_enabled(self, symbol: str) -> bool:
        return self.symbol_override(symbol).trade_enabled


def _merge_dataclass_defaults(config_cls: type, raw_values: dict) -> dict:
    allowed_keys = {item.name for item in fields(config_cls)}
    defaults = config_cls().__dict__
    return {**defaults, **{k: v for k, v in raw_values.items() if k in allowed_keys}}


def _load_project_profiles(raw: dict) -> dict[str, ProjectProfile]:
    profiles: dict[str, ProjectProfile] = {}
    for sym, values in raw.items():
        if not isinstance(values, dict):
            continue
        kwargs = _merge_dataclass_defaults(ProjectProfile, values)
        kwargs["keywords"] = tuple(kwargs.get("keywords") or ())
        profiles[sym] = ProjectProfile(**kwargs)
    return profiles


def load_config(path: str | Path) -> AppConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML config files") from exc

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    storage_raw = raw.get("storage", {}) or {}
    storage_path = storage_raw.get("path")
    if storage_path and not Path(storage_path).is_absolute():
        from ._paths import find_repo_root

        storage_raw = {**storage_raw, "path": str(find_repo_root() / storage_path)}
        raw["storage"] = storage_raw
    risk_raw = raw.get("risk", {})
    risk_levels = tuple(
        TakeProfitLevel(float(item["gain_pct"]), float(item["sell_fraction"]))
        for item in risk_raw.get("take_profit_levels", [])
    )
    default_risk = RiskConfig()
    risk_kwargs = {k: v for k, v in default_risk.__dict__.items() if k != "take_profit_levels"}
    risk_kwargs.update({k: v for k, v in risk_raw.items() if k != "take_profit_levels"})
    risk_kwargs["take_profit_levels"] = risk_levels or default_risk.take_profit_levels
    per_symbol_raw = raw.get("per_symbol", {}) or {}
    per_symbol: dict[str, PerSymbolOverride] = {}
    for sym, overrides in per_symbol_raw.items():
        if not isinstance(overrides, dict):
            continue
        kwargs = _merge_dataclass_defaults(PerSymbolOverride, overrides)
        per_symbol[sym] = PerSymbolOverride(**kwargs)
    return AppConfig(
        universe=UniverseConfig(
            symbols=tuple(raw.get("universe", {}).get("symbols", UniverseConfig().symbols)),
            quote_currency=raw.get("universe", {}).get("quote_currency", "USD"),
            allow_shorts=bool(raw.get("universe", {}).get("allow_shorts", False)),
        ),
        data=DataConfig(
            timeframe=raw.get("data", {}).get("timeframe", "1Hour"),
            lookback_bars=int(raw.get("data", {}).get("lookback_bars", 120)),
            rss_feeds=tuple(raw.get("data", {}).get("rss_feeds", [])),
        ),
        llm=LlmConfig(**_merge_dataclass_defaults(LlmConfig, raw.get("llm", {}))),
        scoring=ScoringConfig(**_merge_dataclass_defaults(ScoringConfig, raw.get("scoring", {}))),
        strategy=StrategyConfig(**_merge_dataclass_defaults(StrategyConfig, raw.get("strategy", {}))),
        costs=CostsConfig(**_merge_dataclass_defaults(CostsConfig, raw.get("costs", {}))),
        quality_growth=QualityGrowthConfig(
            **_merge_dataclass_defaults(QualityGrowthConfig, raw.get("quality_growth", {}))
        ),
        risk=RiskConfig(**risk_kwargs),
        broker=BrokerConfig(**_merge_dataclass_defaults(BrokerConfig, raw.get("broker", {}))),
        storage=StorageConfig(**_merge_dataclass_defaults(StorageConfig, raw.get("storage", {}))),
        regime=RegimeConfig(**_merge_dataclass_defaults(RegimeConfig, raw.get("regime", {}))),
        portfolio=PortfolioConfig(**_merge_dataclass_defaults(PortfolioConfig, raw.get("portfolio", {}))),
        per_symbol=per_symbol,
        project_profiles=_load_project_profiles(raw.get("project_profiles", {}) or {}),
    )
