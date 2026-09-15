from __future__ import annotations

from dataclasses import replace

from .config import RiskConfig
from .schemas import PositionState, RiskDecision, Side, TakeProfitLevel, TradeIntent


class RiskManager:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def position_notional(self, position: PositionState, mark_price: float) -> float:
        return position.remaining_quantity * mark_price

    def evaluate_entry(
        self,
        intent: TradeIntent,
        equity: float,
        open_positions: dict[str, PositionState],
        marks: dict[str, float],
        realized_pnl_24h: float = 0.0,
        volatility: float = 0.0,
    ) -> RiskDecision:
        if intent.side == Side.HOLD:
            return RiskDecision(False, "hold_signal")
        if intent.side == Side.SELL:
            return RiskDecision(False, "shorts_disabled_in_v1")
        if equity <= 0:
            return RiskDecision(False, "invalid_equity")
        if realized_pnl_24h <= -equity * self.cfg.daily_drawdown_stop_pct:
            return RiskDecision(False, "daily_drawdown_stop")
        if volatility > self.cfg.volatility_spike_limit:
            return RiskDecision(False, "volatility_spike")
        if len(open_positions) >= self.cfg.max_positions and intent.symbol not in open_positions:
            return RiskDecision(False, "max_positions_reached")

        current_exposure = sum(
            self.position_notional(pos, marks.get(symbol, pos.entry_price))
            for symbol, pos in open_positions.items()
        )
        max_exposure = equity * self.cfg.max_total_exposure_pct
        remaining = max(0.0, max_exposure - current_exposure)
        if remaining <= 0:
            return RiskDecision(False, "max_total_exposure_reached")
        adjusted = min(intent.notional, remaining)
        if adjusted <= 0:
            return RiskDecision(False, "zero_adjusted_notional")
        return RiskDecision(True, "accepted", adjusted, {"remaining_exposure": remaining})

    def build_position(self, symbol: str, entry_price: float, notional: float) -> PositionState:
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        quantity = notional / entry_price
        return PositionState(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            remaining_quantity=quantity,
            stop_price=entry_price * (1.0 - self.cfg.stop_loss_pct),
            take_profit_levels=[replace(level) for level in self.cfg.take_profit_levels],
        )

    def update_position_for_price(self, position: PositionState, price: float) -> list[dict[str, float | str]]:
        events: list[dict[str, float | str]] = []
        if not position.is_open:
            return events
        if price <= position.stop_price:
            qty = position.remaining_quantity
            position.remaining_quantity = 0.0
            events.append({"event": "stop_loss", "quantity": qty, "price": price})
            return events
        for idx, level in enumerate(position.take_profit_levels):
            if idx in position.filled_tp_indices:
                continue
            target = position.entry_price * (1.0 + level.gain_pct)
            if price + 1e-9 >= target and position.remaining_quantity > 0:
                sell_qty = min(position.quantity * level.sell_fraction, position.remaining_quantity)
                position.remaining_quantity -= sell_qty
                position.filled_tp_indices.add(idx)
                events.append({"event": f"tp{idx + 1}", "quantity": sell_qty, "price": price})
                if idx == 0 and not position.stop_moved_to_breakeven:
                    position.stop_price = position.entry_price
                    position.stop_moved_to_breakeven = True
                    events.append({"event": "stop_to_breakeven", "quantity": 0.0, "price": position.stop_price})
        return events


def default_take_profit_levels() -> tuple[TakeProfitLevel, ...]:
    return (
        TakeProfitLevel(0.10, 0.25),
        TakeProfitLevel(0.20, 0.25),
        TakeProfitLevel(0.50, 0.50),
    )
