from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .config import BrokerConfig
from .schemas import TradeIntent


@dataclass(frozen=True)
class BrokerAccount:
    equity: float
    buying_power: float


@dataclass(frozen=True)
class SubmittedOrder:
    broker_order_id: str
    status: str
    symbol: str
    qty: float
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    child_order_ids: tuple[str, ...] = field(default_factory=tuple)
    used_fallback: bool = False


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    side: str = "long"


@dataclass(frozen=True)
class BrokerOrder:
    broker_order_id: str
    symbol: str
    side: str
    qty: float
    status: str
    order_type: str
    submitted_at: Optional[datetime] = None
    filled_avg_price: Optional[float] = None
    legs: tuple[str, ...] = field(default_factory=tuple)


def _alpaca_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


class AlpacaPaperBroker:
    def __init__(self, cfg: BrokerConfig) -> None:
        if cfg.live_trading_enabled:
            raise ValueError("live trading is not implemented in v1")
        base_url = os.getenv("APCA_API_BASE_URL", cfg.paper_base_url)
        if "paper-api" not in base_url:
            raise ValueError("Alpaca broker requires the paper API base URL")
        key = os.getenv("APCA_API_KEY_ID")
        secret = os.getenv("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("APCA_API_KEY_ID and APCA_API_SECRET_KEY are required for paper broker use")
        from alpaca.trading.client import TradingClient

        self.client = TradingClient(api_key=key, secret_key=secret, paper=True)

    def get_account(self) -> BrokerAccount:
        account = self.client.get_account()
        return BrokerAccount(equity=float(account.equity), buying_power=float(account.buying_power))

    def get_positions(self) -> list[BrokerPosition]:
        out: list[BrokerPosition] = []
        for p in self.client.get_all_positions():
            out.append(
                BrokerPosition(
                    symbol=str(p.symbol),
                    qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    market_value=float(p.market_value),
                    unrealized_pl=float(getattr(p, "unrealized_pl", 0.0) or 0.0),
                    side=str(getattr(p, "side", "long")),
                )
            )
        return out

    def get_orders(self, status: str = "open") -> list[BrokerOrder]:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        status_map = {
            "open": QueryOrderStatus.OPEN,
            "closed": QueryOrderStatus.CLOSED,
            "all": QueryOrderStatus.ALL,
        }
        query = GetOrdersRequest(status=status_map.get(status, QueryOrderStatus.OPEN))
        out: list[BrokerOrder] = []
        for o in self.client.get_orders(filter=query):
            legs = tuple(str(getattr(leg, "id", "")) for leg in (getattr(o, "legs", None) or []))
            out.append(
                BrokerOrder(
                    broker_order_id=str(o.id),
                    symbol=str(o.symbol),
                    side=str(o.side),
                    qty=float(o.qty or 0.0),
                    status=str(o.status),
                    order_type=str(getattr(o, "order_type", "")),
                    submitted_at=getattr(o, "submitted_at", None),
                    filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
                    legs=legs,
                )
            )
        return out

    def submit_market_order(self, intent: TradeIntent, price: float) -> SubmittedOrder:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        qty = intent.notional / price
        order = self.client.submit_order(
            MarketOrderRequest(
                symbol=_alpaca_symbol(intent.symbol),
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
            )
        )
        return SubmittedOrder(str(order.id), str(order.status), intent.symbol, qty)

    def submit_bracket_order(
        self,
        intent: TradeIntent,
        price: float,
        take_profit_pct: float,
        stop_loss_pct: float,
    ) -> SubmittedOrder:
        """Submit market entry with bracket TP/SL. Falls back to split orders on crypto.

        ``take_profit_pct`` and ``stop_loss_pct`` are positive fractions (e.g. 0.10 = 10%).
        """
        from alpaca.common.exceptions import APIError
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLossRequest,
            StopOrderRequest,
            TakeProfitRequest,
        )

        if take_profit_pct <= 0 or stop_loss_pct <= 0:
            raise ValueError("take_profit_pct and stop_loss_pct must be positive fractions")

        qty = intent.notional / price
        sym = _alpaca_symbol(intent.symbol)
        tp_price = round(price * (1.0 + take_profit_pct), 2)
        sl_price = round(price * (1.0 - stop_loss_pct), 2)

        try:
            req = MarketOrderRequest(
                symbol=sym,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=tp_price),
                stop_loss=StopLossRequest(stop_price=sl_price),
            )
            order = self.client.submit_order(req)
            legs = tuple(str(getattr(leg, "id", "")) for leg in (getattr(order, "legs", None) or []))
            return SubmittedOrder(
                broker_order_id=str(order.id),
                status=str(order.status),
                symbol=intent.symbol,
                qty=qty,
                take_profit_price=tp_price,
                stop_loss_price=sl_price,
                child_order_ids=legs,
                used_fallback=False,
            )
        except APIError as exc:
            msg = str(exc).lower()
            if "asset class" not in msg and "not supported" not in msg and "bracket" not in msg:
                raise
            # Fallback: market entry, poll for fill, then submit child sells separately.
            entry_req = MarketOrderRequest(
                symbol=sym, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC
            )
            entry = self.client.submit_order(entry_req)
            filled_qty = self._wait_for_fill(str(entry.id), default_qty=qty, timeout=3.0)
            tp = self.client.submit_order(
                LimitOrderRequest(
                    symbol=sym,
                    qty=filled_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    limit_price=tp_price,
                )
            )
            sl = self.client.submit_order(
                StopOrderRequest(
                    symbol=sym,
                    qty=filled_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    stop_price=sl_price,
                )
            )
            return SubmittedOrder(
                broker_order_id=str(entry.id),
                status=str(entry.status),
                symbol=intent.symbol,
                qty=filled_qty,
                take_profit_price=tp_price,
                stop_loss_price=sl_price,
                child_order_ids=(str(tp.id), str(sl.id)),
                used_fallback=True,
            )

    def _wait_for_fill(self, order_id: str, *, default_qty: float, timeout: float) -> float:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                order = self.client.get_order_by_id(order_id)
            except Exception:
                time.sleep(0.25)
                continue
            qty = float(getattr(order, "filled_qty", 0.0) or 0.0)
            if qty > 0:
                return qty
            if str(getattr(order, "status", "")).lower() in {"filled", "partially_filled"}:
                return qty or default_qty
            time.sleep(0.25)
        return default_qty
