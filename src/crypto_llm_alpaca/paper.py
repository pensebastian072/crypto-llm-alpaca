from __future__ import annotations

from dataclasses import dataclass

from .broker import AlpacaPaperBroker
from .config import AppConfig
from .features import compute_features
from .risk import RiskManager
from .schemas import Bar, LlmSignal, PositionState, TradeIntent
from .scoring import score_symbol


@dataclass(frozen=True)
class PaperDecision:
    intent: TradeIntent
    accepted: bool
    reason: str
    submitted_order_id: str | None = None


def build_paper_decision(
    cfg: AppConfig,
    symbol: str,
    bars: list[Bar],
    signals: list[LlmSignal],
    equity: float,
    open_positions: dict[str, PositionState] | None = None,
) -> PaperDecision:
    open_positions = open_positions or {}
    features = compute_features(bars)
    notional = equity * cfg.risk.max_total_exposure_pct / cfg.risk.max_positions
    intent = score_symbol(features, signals, cfg.scoring, cfg.llm.min_confidence, notional)
    decision = RiskManager(cfg.risk).evaluate_entry(
        intent,
        equity=equity,
        open_positions=open_positions,
        marks={symbol: bars[-1].close},
        volatility=features.volatility,
    )
    return PaperDecision(intent=intent, accepted=decision.accepted, reason=decision.reason)


def submit_if_requested(cfg: AppConfig, paper_decision: PaperDecision, price: float, submit: bool) -> PaperDecision:
    if not submit or not paper_decision.accepted:
        return paper_decision
    order = AlpacaPaperBroker(cfg.broker).submit_market_order(paper_decision.intent, price)
    return PaperDecision(
        intent=paper_decision.intent,
        accepted=True,
        reason=paper_decision.reason,
        submitted_order_id=order.broker_order_id,
    )
