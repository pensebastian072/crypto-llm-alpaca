from datetime import datetime, timezone
import sqlite3

import pytest

from crypto_llm_alpaca.backtest import demo_bars
from crypto_llm_alpaca import cli as cli_module
from crypto_llm_alpaca.llm import MockLLMProvider
from crypto_llm_alpaca.news import parse_rss
from crypto_llm_alpaca.schemas import LlmSignal, NewsItem
from crypto_llm_alpaca.storage import SQLiteStore


def _write_config(path, db_path):
    path.write_text(
        f"""
storage:
  path: "{db_path.as_posix()}"
llm:
  provider: "mock"
universe:
  symbols: ["BTC/USD", "AAVE/USD", "ARB/USD", "UNI/USD"]
""",
        encoding="utf-8",
    )


class DriftingProvider:
    def __init__(self, drift_symbol: str = "BTC/USD") -> None:
        self.drift_symbol = drift_symbol

    def extract(self, item: NewsItem, symbol: str) -> LlmSignal:
        return LlmSignal(
            news_hash=item.dedupe_hash,
            symbol=self.drift_symbol,
            sentiment=0.55,
            confidence=0.85,
            horizon="short",
            event_type="test_drift",
            rationale=f"Provider drifted away from requested {symbol}.",
            provider="drift",
            model="test",
        )


def _news(hash_id: str, symbols: tuple[str, ...]) -> NewsItem:
    return NewsItem(
        source="fixture",
        url=f"https://example.com/{hash_id}",
        title="Aave files protocol update",
        summary="AAVE protocol news.",
        published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        symbols=symbols,
        dedupe_hash=hash_id,
    )


def test_extract_signals_from_stored_news(tmp_path):
    db_path = tmp_path / "signals.sqlite3"
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, db_path)
    store = SQLiteStore(db_path)
    store.init_schema()
    xml = """<rss><channel><item><title>Bitcoin rally accelerates</title><link>https://example.com/a</link><description>BTC inflow improves.</description><pubDate>Mon, 01 Jan 2026 00:00:00 GMT</pubDate></item></channel></rss>"""
    store.upsert_news(parse_rss(xml, "fixture", ("BTC/USD",)))
    assert cli_module.main(["--config", str(cfg_path), "extract-signals"]) == 0
    assert len(store.latest_signals("BTC/USD")) == 1


def test_paper_once_uses_stored_bars_and_signals(tmp_path):
    db_path = tmp_path / "paper.sqlite3"
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, db_path)
    store = SQLiteStore(db_path)
    store.init_schema()
    store.insert_bars(demo_bars(symbol="BTC/USD", count=40))
    xml = """<rss><channel><item><title>Bitcoin rally accelerates</title><link>https://example.com/a</link><description>BTC inflow improves.</description><pubDate>Mon, 01 Jan 2026 00:00:00 GMT</pubDate></item></channel></rss>"""
    item = parse_rss(xml, "fixture", ("BTC/USD",))[0]
    store.insert_llm_signals([MockLLMProvider().extract(item, "BTC/USD")])
    assert cli_module.main(["--config", str(cfg_path), "paper-once", "--symbol", "BTC/USD"]) == 0


def test_extract_reconciles_drifting_provider_to_article_symbol(tmp_path, monkeypatch):
    db_path = tmp_path / "drift.sqlite3"
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, db_path)
    store = SQLiteStore(db_path)
    store.init_schema()
    store.upsert_news([_news("aave-drift", ("AAVE/USD",))])
    monkeypatch.setattr(cli_module, "build_provider", lambda *_: DriftingProvider())

    assert cli_module.main(["--config", str(cfg_path), "extract-signals"]) == 0

    assert len(store.latest_signals("BTC/USD")) == 0
    signals = store.latest_signals("AAVE/USD")
    assert len(signals) == 1
    assert signals[0].symbol_confidence == pytest.approx(1.0)


def test_extract_multi_symbol_news_uses_primary_symbol_only(tmp_path, monkeypatch):
    db_path = tmp_path / "primary.sqlite3"
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, db_path)
    store = SQLiteStore(db_path)
    store.init_schema()
    store.upsert_news([_news("aave-arb", ("AAVE/USD", "ARB/USD"))])
    monkeypatch.setattr(cli_module, "build_provider", lambda *_: DriftingProvider("ARB/USD"))

    assert cli_module.main(["--config", str(cfg_path), "extract-signals"]) == 0

    assert len(store.latest_signals("AAVE/USD")) == 1
    assert len(store.latest_signals("ARB/USD")) == 0


def test_extract_rerun_does_not_duplicate_completed_primary_symbol(tmp_path, monkeypatch):
    db_path = tmp_path / "rerun.sqlite3"
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, db_path)
    store = SQLiteStore(db_path)
    store.init_schema()
    store.upsert_news([_news("aave-rerun", ("AAVE/USD",))])
    monkeypatch.setattr(cli_module, "build_provider", lambda *_: DriftingProvider())

    assert cli_module.main(["--config", str(cfg_path), "extract-signals"]) == 0
    assert cli_module.main(["--config", str(cfg_path), "extract-signals"]) == 0

    assert len(store.latest_signals("AAVE/USD")) == 1


def test_existing_bad_symbol_row_does_not_mark_article_complete(tmp_path, monkeypatch):
    db_path = tmp_path / "bad-existing.sqlite3"
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, db_path)
    store = SQLiteStore(db_path)
    store.init_schema()
    item = _news("aave-with-bad-btc", ("AAVE/USD",))
    store.upsert_news([item])
    store.insert_llm_signals([DriftingProvider().extract(item, "AAVE/USD")])
    monkeypatch.setattr(cli_module, "build_provider", lambda *_: DriftingProvider())

    assert cli_module.main(["--config", str(cfg_path), "extract-signals"]) == 0

    assert len(store.latest_signals("BTC/USD")) == 1
    assert len(store.latest_signals("AAVE/USD")) == 1


def test_old_llm_signal_schema_migrates_symbol_confidence(tmp_path):
    db_path = tmp_path / "old.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE llm_signals (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              news_hash TEXT NOT NULL,
              symbol TEXT NOT NULL,
              sentiment REAL NOT NULL,
              confidence REAL NOT NULL,
              horizon TEXT NOT NULL,
              event_type TEXT NOT NULL,
              rationale TEXT NOT NULL,
              provider TEXT NOT NULL,
              model TEXT NOT NULL,
              created_at TEXT NOT NULL,
              fear REAL DEFAULT 0,
              hype REAL DEFAULT 0,
              topic TEXT DEFAULT '',
              source_weight REAL DEFAULT 1.0
            )
            """
        )

    store = SQLiteStore(db_path)
    store.init_schema()

    with store.connect() as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(llm_signals)").fetchall()}
    assert "symbol_confidence" in columns


def test_llm_signal_symbol_confidence_defaults_and_round_trips(tmp_path):
    signal = LlmSignal(
        news_hash="n1",
        symbol="AAVE/USD",
        sentiment=0.25,
        confidence=0.8,
        horizon="short",
        event_type="fixture",
        rationale="default attribution confidence",
    )
    assert signal.symbol_confidence == 1.0

    db_path = tmp_path / "confidence.sqlite3"
    store = SQLiteStore(db_path)
    store.init_schema()
    store.insert_llm_signals([
        LlmSignal(
            news_hash="n2",
            symbol="AAVE/USD",
            sentiment=0.4,
            confidence=0.9,
            horizon="short",
            event_type="fixture",
            rationale="explicit attribution confidence",
            symbol_confidence=0.7,
        )
    ])

    assert store.latest_signals("AAVE/USD")[0].symbol_confidence == pytest.approx(0.7)
