from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str = "1Hour"
    source: str = "alpaca"


@dataclass(frozen=True)
class NewsItem:
    source: str
    url: str
    title: str
    summary: str
    published_at: datetime
    symbols: tuple[str, ...]
    dedupe_hash: str


@dataclass(frozen=True)
class LlmSignal:
    news_hash: str
    symbol: str
    sentiment: float
    confidence: float
    horizon: str
    event_type: str
    rationale: str
    provider: str = "mock"
    model: str = "mock"
    created_at: datetime = field(default_factory=utc_now)
    # Phase 3 additions (all optional, backward-compatible).
    fear: float = 0.0          # 0..1 — fear/uncertainty/regulation/hack
    hype: float = 0.0          # 0..1 — euphoria/rally/adoption
    topic: str = ""            # free-form: regulation, etf, hack, partnership, etc.
    source_weight: float = 1.0  # 0..1 — credibility weight for source
    symbol_confidence: float = 1.0  # 0..1 — confidence in article-to-symbol attribution

    def __post_init__(self) -> None:
        if not -1.0 <= self.sentiment <= 1.0:
            raise ValueError("sentiment must be in [-1, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= self.fear <= 1.0:
            raise ValueError("fear must be in [0, 1]")
        if not 0.0 <= self.hype <= 1.0:
            raise ValueError("hype must be in [0, 1]")
        if not 0.0 <= self.source_weight <= 1.0:
            raise ValueError("source_weight must be in [0, 1]")
        if not 0.0 <= self.symbol_confidence <= 1.0:
            raise ValueError("symbol_confidence must be in [0, 1]")


@dataclass(frozen=True)
class TechnicalFeatures:
    symbol: str
    timestamp: datetime
    momentum: float
    rsi: float
    volatility: float
    volume_change: float
    trend_slope: float


@dataclass(frozen=True)
class StrategyFeatures:
    symbol: str
    timestamp: datetime
    breakout_level: float
    breakout_pct: float
    relative_volume: float
    volume_acceleration: float
    range_expansion: float
    close_location: float
    momentum: float
    htf_trend: float
    atr: float
    atr_pct: float
    atr_short: float
    atr_long: float
    atr_regime: float
    expected_move_pct: float
    signal_strength: float
    is_breakout: bool
    passes_volume: bool
    passes_range: bool
    passes_htf: bool
    trend_4h: float = 0.0
    trend_1d: float = 0.0
    trend_1w: float = 0.0
    mtf_trend_score: float = 0.0

    @property
    def entry_signal(self) -> bool:
        return self.is_breakout and self.passes_volume and self.passes_range and self.passes_htf


@dataclass(frozen=True)
class TradeIntent:
    symbol: str
    side: Side
    score: float
    confidence: float
    notional: float
    reason: str
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class TakeProfitLevel:
    gain_pct: float
    sell_fraction: float


@dataclass
class PositionState:
    symbol: str
    entry_price: float
    quantity: float
    remaining_quantity: float
    stop_price: float
    take_profit_levels: list[TakeProfitLevel]
    filled_tp_indices: set[int] = field(default_factory=set)
    stop_moved_to_breakeven: bool = False

    @property
    def is_open(self) -> bool:
        return self.remaining_quantity > 1e-12


@dataclass(frozen=True)
class RiskDecision:
    accepted: bool
    reason: str
    adjusted_notional: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
