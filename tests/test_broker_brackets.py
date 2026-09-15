from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from crypto_llm_alpaca.broker import (
    AlpacaPaperBroker,
    BrokerOrder,
    BrokerPosition,
    SubmittedOrder,
)
from crypto_llm_alpaca.config import BrokerConfig
from crypto_llm_alpaca.schemas import Side, TradeIntent


@pytest.fixture
def broker_with_mock(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "fake")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "fake")
    monkeypatch.setenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    class _FakeClient:
        def __init__(self, *_, **__):
            pass

        def get_account(self):
            return SimpleNamespace(equity="100000", buying_power="200000")

    monkeypatch.setattr(
        "alpaca.trading.client.TradingClient",
        _FakeClient,
        raising=False,
    )
    cfg = BrokerConfig(paper_base_url="https://paper-api.alpaca.markets", live_trading_enabled=False)
    broker = AlpacaPaperBroker(cfg)
    broker.client = MagicMock()
    return broker


def _intent(symbol: str = "BTC/USD", notional: float = 100.0) -> TradeIntent:
    return TradeIntent(
        symbol=symbol,
        side=Side.BUY,
        score=0.8,
        confidence=0.9,
        notional=notional,
        reason="test",
    )


def test_submit_bracket_order_happy_path(broker_with_mock):
    broker_with_mock.client.submit_order.return_value = SimpleNamespace(
        id="order-1", status="accepted", legs=[SimpleNamespace(id="leg-tp"), SimpleNamespace(id="leg-sl")]
    )
    order = broker_with_mock.submit_bracket_order(_intent(notional=100), price=10.0, take_profit_pct=0.10, stop_loss_pct=0.05)
    assert order.used_fallback is False
    assert order.broker_order_id == "order-1"
    assert order.qty == pytest.approx(10.0)
    assert order.take_profit_price == pytest.approx(11.0)
    assert order.stop_loss_price == pytest.approx(9.5)
    assert order.child_order_ids == ("leg-tp", "leg-sl")
    args, _ = broker_with_mock.client.submit_order.call_args
    req = args[0]
    assert getattr(req, "order_class").value.lower() == "bracket"


def test_submit_bracket_falls_back_on_asset_class_error(broker_with_mock):
    from alpaca.common.exceptions import APIError

    fill_order = SimpleNamespace(id="entry-1", status="filled", filled_qty="10")
    tp_order = SimpleNamespace(id="tp-1", status="accepted")
    sl_order = SimpleNamespace(id="sl-1", status="accepted")

    broker_with_mock.client.submit_order.side_effect = [
        APIError("bracket order classes are not supported for this asset class"),
        fill_order,
        tp_order,
        sl_order,
    ]
    broker_with_mock.client.get_order_by_id.return_value = fill_order

    order = broker_with_mock.submit_bracket_order(_intent(notional=100), price=10.0, take_profit_pct=0.10, stop_loss_pct=0.05)

    assert order.used_fallback is True
    assert order.broker_order_id == "entry-1"
    assert order.child_order_ids == ("tp-1", "sl-1")
    assert broker_with_mock.client.submit_order.call_count == 4


def test_submit_bracket_re_raises_other_api_errors(broker_with_mock):
    from alpaca.common.exceptions import APIError

    broker_with_mock.client.submit_order.side_effect = APIError("insufficient buying power")
    with pytest.raises(APIError):
        broker_with_mock.submit_bracket_order(_intent(), price=10.0, take_profit_pct=0.10, stop_loss_pct=0.05)


def test_get_positions_maps_alpaca_positions(broker_with_mock):
    broker_with_mock.client.get_all_positions.return_value = [
        SimpleNamespace(symbol="BTCUSD", qty="0.5", avg_entry_price="50000", market_value="26000", unrealized_pl="1000", side="long"),
    ]
    positions = broker_with_mock.get_positions()
    assert len(positions) == 1
    p = positions[0]
    assert isinstance(p, BrokerPosition)
    assert p.symbol == "BTCUSD"
    assert p.qty == pytest.approx(0.5)
    assert p.unrealized_pl == pytest.approx(1000.0)


def test_get_orders_maps_alpaca_orders(broker_with_mock):
    broker_with_mock.client.get_orders.return_value = [
        SimpleNamespace(
            id="ord-1",
            symbol="BTCUSD",
            side="buy",
            qty="0.5",
            status="open",
            order_type="market",
            submitted_at=None,
            filled_avg_price=None,
            legs=[SimpleNamespace(id="leg-1")],
        )
    ]
    orders = broker_with_mock.get_orders()
    assert len(orders) == 1
    assert isinstance(orders[0], BrokerOrder)
    assert orders[0].legs == ("leg-1",)
