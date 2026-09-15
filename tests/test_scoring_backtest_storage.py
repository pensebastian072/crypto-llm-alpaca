from pathlib import Path

from crypto_llm_alpaca.backtest import demo_bars, run_simple_backtest
from crypto_llm_alpaca.config import AppConfig, LlmConfig, ScoringConfig
from crypto_llm_alpaca.features import compute_features
from crypto_llm_alpaca.schemas import LlmSignal, Side
from crypto_llm_alpaca.scoring import score_symbol
from crypto_llm_alpaca.storage import SQLiteStore


def test_scoring_produces_buy_for_positive_news_and_momentum():
    bars = demo_bars(count=40)
    features = compute_features(bars)
    signal = LlmSignal("n1", "BTC/USD", 0.9, 0.9, "short", "inflow", "test")
    intent = score_symbol(features, [signal], ScoringConfig(), 0.55, 1000)
    assert intent.side == Side.BUY
    assert intent.score > 0


def test_demo_backtest_runs_without_external_services():
    cfg = AppConfig(llm=LlmConfig(provider="mock"))
    result = run_simple_backtest(cfg, {"BTC/USD": demo_bars()})
    assert result.trades >= 1
    assert result.tp_events >= 1


def test_storage_schema_and_insert_intent(tmp_path: Path):
    store = SQLiteStore(tmp_path / "test.sqlite3")
    store.init_schema()
    bars = demo_bars(count=1)
    assert store.insert_bars(bars) == 1
    features = compute_features(demo_bars(count=40))
    signal = LlmSignal("n1", "BTC/USD", 0.9, 0.9, "short", "inflow", "test")
    intent = score_symbol(features, [signal], ScoringConfig(), 0.55, 1000)
    row_id = store.insert_trade_intent(intent)
    assert row_id == 1
