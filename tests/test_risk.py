from crypto_llm_alpaca.config import RiskConfig
from crypto_llm_alpaca.risk import RiskManager
from crypto_llm_alpaca.schemas import Side, TradeIntent


def _intent(notional=5000):
    return TradeIntent("BTC/USD", Side.BUY, score=0.8, confidence=0.9, notional=notional, reason="test")


def test_exposure_cap_adjusts_notional_to_30_percent_total():
    risk = RiskManager(RiskConfig(max_total_exposure_pct=0.30, max_positions=5))
    decision = risk.evaluate_entry(_intent(5000), equity=10_000, open_positions={}, marks={})
    assert decision.accepted
    assert decision.adjusted_notional == 3000


def test_stop_loss_and_take_profit_ladder():
    risk = RiskManager(RiskConfig())
    pos = risk.build_position("BTC/USD", entry_price=100.0, notional=1000.0)
    assert pos.stop_price == 99.0
    events = risk.update_position_for_price(pos, 110.0)
    assert [event["event"] for event in events] == ["tp1", "stop_to_breakeven"]
    assert pos.stop_price == 100.0
    assert pos.remaining_quantity == 7.5
    events = risk.update_position_for_price(pos, 120.0)
    assert [event["event"] for event in events] == ["tp2"]
    assert pos.remaining_quantity == 5.0
    events = risk.update_position_for_price(pos, 150.0)
    assert [event["event"] for event in events] == ["tp3"]
    assert pos.remaining_quantity == 0.0


def test_daily_drawdown_blocks_new_entries():
    risk = RiskManager(RiskConfig(daily_drawdown_stop_pct=0.05))
    decision = risk.evaluate_entry(_intent(100), equity=10_000, open_positions={}, marks={}, realized_pnl_24h=-500)
    assert not decision.accepted
    assert decision.reason == "daily_drawdown_stop"
