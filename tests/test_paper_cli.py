from crypto_llm_alpaca.backtest import demo_bars
from crypto_llm_alpaca.config import AppConfig
from crypto_llm_alpaca.paper import build_paper_decision
from crypto_llm_alpaca.schemas import LlmSignal, Side


def test_paper_decision_builds_accepted_buy_without_broker_submit():
    cfg = AppConfig()
    signal = LlmSignal("n1", "BTC/USD", 0.9, 0.9, "short", "inflow", "test")
    decision = build_paper_decision(cfg, "BTC/USD", demo_bars(count=40), [signal], equity=10_000)
    assert decision.accepted
    assert decision.reason == "accepted"
    assert decision.intent.side == Side.BUY
    assert decision.submitted_order_id is None
