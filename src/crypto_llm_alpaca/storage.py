from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable
from datetime import datetime, timezone

from .schemas import Bar, LlmSignal, NewsItem, StrategyFeatures, TradeIntent

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  open REAL NOT NULL,
  high REAL NOT NULL,
  low REAL NOT NULL,
  close REAL NOT NULL,
  volume REAL NOT NULL,
  source TEXT NOT NULL,
  PRIMARY KEY(symbol, timeframe, timestamp, source)
);
CREATE TABLE IF NOT EXISTS news_items (
  dedupe_hash TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  url TEXT NOT NULL,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  published_at TEXT NOT NULL,
  symbols_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_signals (
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
  source_weight REAL DEFAULT 1.0,
  symbol_confidence REAL DEFAULT 1.0
);
CREATE TABLE IF NOT EXISTS trade_intents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  score REAL NOT NULL,
  confidence REAL NOT NULL,
  notional REAL NOT NULL,
  reason TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  broker_order_id TEXT,
  status TEXT NOT NULL,
  symbol TEXT NOT NULL,
  qty REAL NOT NULL,
  submitted_at TEXT NOT NULL,
  filled_at TEXT
);
CREATE TABLE IF NOT EXISTS positions (
  symbol TEXT PRIMARY KEY,
  entry_price REAL NOT NULL,
  quantity REAL NOT NULL,
  remaining_quantity REAL NOT NULL,
  stop_price REAL NOT NULL,
  state_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  mode TEXT NOT NULL,
  config_json TEXT NOT NULL,
  started_at TEXT NOT NULL,
  completed_at TEXT
);
CREATE TABLE IF NOT EXISTS strategy_features (
  symbol TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  breakout_level REAL NOT NULL,
  breakout_pct REAL NOT NULL,
  relative_volume REAL NOT NULL,
  volume_acceleration REAL NOT NULL,
  range_expansion REAL NOT NULL,
  close_location REAL NOT NULL,
  momentum REAL NOT NULL,
  htf_trend REAL NOT NULL,
  trend_4h REAL DEFAULT 0,
  trend_1d REAL DEFAULT 0,
  trend_1w REAL DEFAULT 0,
  mtf_trend_score REAL DEFAULT 0,
  signal_strength REAL NOT NULL,
  is_breakout INTEGER NOT NULL,
  passes_volume INTEGER NOT NULL,
  passes_range INTEGER NOT NULL,
  passes_htf INTEGER NOT NULL,
  PRIMARY KEY(symbol, timestamp)
);
CREATE TABLE IF NOT EXISTS backtest_runs (
  run_id TEXT PRIMARY KEY,
  strategy TEXT NOT NULL,
  started_at TEXT NOT NULL,
  completed_at TEXT NOT NULL,
  config_json TEXT NOT NULL,
  metrics_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS backtest_trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  price REAL NOT NULL,
  quantity REAL NOT NULL,
  fee REAL NOT NULL,
  pnl REAL NOT NULL,
  reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS backtest_equity_curve (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  equity REAL NOT NULL,
  drawdown REAL NOT NULL,
  open_positions REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  score REAL NOT NULL,
  confidence REAL NOT NULL,
  accepted INTEGER NOT NULL,
  reason TEXT NOT NULL,
  intent_json TEXT NOT NULL,
  decided_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS daily_reports (
  date TEXT PRIMARY KEY,
  equity_open REAL NOT NULL,
  equity_close REAL NOT NULL,
  n_decisions INTEGER NOT NULL,
  n_orders INTEGER NOT NULL,
  markdown_path TEXT NOT NULL,
  summary_json TEXT NOT NULL,
  generated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS defillama_snapshots (
  symbol TEXT NOT NULL,
  asof TEXT NOT NULL,
  tvl REAL,
  tvl_growth REAL,
  tvl_growth_30d REAL,
  fees_24h REAL,
  fees_7d REAL,
  fees_30d REAL,
  fee_growth REAL,
  revenue_growth REAL,
  source TEXT DEFAULT 'defillama',
  PRIMARY KEY(symbol, asof)
);
"""


class SQLiteStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        if self.path.parent != Path("."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            # Additive migrations for older DBs.
            existing = {row[1] for row in conn.execute("PRAGMA table_info(llm_signals)").fetchall()}
            for col, ddl in (
                ("fear", "ALTER TABLE llm_signals ADD COLUMN fear REAL DEFAULT 0"),
                ("hype", "ALTER TABLE llm_signals ADD COLUMN hype REAL DEFAULT 0"),
                ("topic", "ALTER TABLE llm_signals ADD COLUMN topic TEXT DEFAULT ''"),
                ("source_weight", "ALTER TABLE llm_signals ADD COLUMN source_weight REAL DEFAULT 1.0"),
                ("symbol_confidence", "ALTER TABLE llm_signals ADD COLUMN symbol_confidence REAL DEFAULT 1.0"),
            ):
                if col not in existing:
                    conn.execute(ddl)
            feature_existing = {
                row[1] for row in conn.execute("PRAGMA table_info(strategy_features)").fetchall()
            }
            for col, ddl in (
                ("trend_4h", "ALTER TABLE strategy_features ADD COLUMN trend_4h REAL DEFAULT 0"),
                ("trend_1d", "ALTER TABLE strategy_features ADD COLUMN trend_1d REAL DEFAULT 0"),
                ("trend_1w", "ALTER TABLE strategy_features ADD COLUMN trend_1w REAL DEFAULT 0"),
                ("mtf_trend_score", "ALTER TABLE strategy_features ADD COLUMN mtf_trend_score REAL DEFAULT 0"),
            ):
                if col not in feature_existing:
                    conn.execute(ddl)

    def upsert_news(self, items: Iterable[NewsItem]) -> int:
        rows = [
            (item.dedupe_hash, item.source, item.url, item.title, item.summary, item.published_at.isoformat(), json.dumps(item.symbols))
            for item in items
        ]
        with self.connect() as conn:
            conn.executemany(
                """INSERT OR IGNORE INTO news_items
                (dedupe_hash, source, url, title, summary, published_at, symbols_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            return conn.total_changes

    def insert_bars(self, bars: Iterable[Bar]) -> int:
        rows = [(b.symbol, b.timeframe, b.timestamp.isoformat(), b.open, b.high, b.low, b.close, b.volume, b.source) for b in bars]
        with self.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO bars
                (symbol, timeframe, timestamp, open, high, low, close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            return conn.total_changes

    def insert_llm_signals(self, signals: Iterable[LlmSignal]) -> int:
        rows = [
            (
                s.news_hash, s.symbol, s.sentiment, s.confidence, s.horizon, s.event_type, s.rationale,
                s.provider, s.model, s.created_at.isoformat(),
                float(getattr(s, "fear", 0.0) or 0.0),
                float(getattr(s, "hype", 0.0) or 0.0),
                str(getattr(s, "topic", "") or ""),
                float(getattr(s, "source_weight", 1.0) or 1.0),
                float(getattr(s, "symbol_confidence", 1.0) or 1.0),
            )
            for s in signals
        ]
        with self.connect() as conn:
            conn.executemany(
                """INSERT INTO llm_signals
                (news_hash, symbol, sentiment, confidence, horizon, event_type, rationale,
                 provider, model, created_at, fear, hype, topic, source_weight, symbol_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            return conn.total_changes

    def insert_trade_intent(self, intent: TradeIntent) -> int:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO trade_intents
                (symbol, side, score, confidence, notional, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (intent.symbol, intent.side.value, intent.score, intent.confidence, intent.notional, intent.reason, intent.created_at.isoformat()),
            )
            return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    def upsert_strategy_features(self, features: Iterable[StrategyFeatures]) -> int:
        rows = [
            (
                item.symbol,
                item.timestamp.isoformat(),
                item.breakout_level,
                item.breakout_pct,
                item.relative_volume,
                item.volume_acceleration,
                item.range_expansion,
                item.close_location,
                item.momentum,
                item.htf_trend,
                item.trend_4h,
                item.trend_1d,
                item.trend_1w,
                item.mtf_trend_score,
                item.signal_strength,
                int(item.is_breakout),
                int(item.passes_volume),
                int(item.passes_range),
                int(item.passes_htf),
            )
            for item in features
        ]
        with self.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO strategy_features
                (symbol, timestamp, breakout_level, breakout_pct, relative_volume,
                 volume_acceleration, range_expansion, close_location, momentum,
                 htf_trend, trend_4h, trend_1d, trend_1w, mtf_trend_score,
                 signal_strength, is_breakout, passes_volume, passes_range, passes_htf)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            return conn.total_changes

    def insert_backtest_run(
        self,
        run_id: str,
        strategy: str,
        config: dict,
        metrics: dict,
        trades: Iterable[dict],
        equity_curve: Iterable[dict] = (),
    ) -> int:
        now = datetime.now(tz=timezone.utc).isoformat()
        trade_rows = [
            (
                run_id,
                trade["symbol"],
                trade["side"],
                trade["timestamp"],
                float(trade["price"]),
                float(trade["quantity"]),
                float(trade["fee"]),
                float(trade["pnl"]),
                trade["reason"],
            )
            for trade in trades
        ]
        curve_rows = [
            (
                run_id,
                point["timestamp"],
                float(point["equity"]),
                float(point["drawdown"]),
                float(point["open_positions"]),
            )
            for point in equity_curve
        ]
        with self.connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO backtest_runs
                (run_id, strategy, started_at, completed_at, config_json, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, strategy, now, now, json.dumps(config, default=str), json.dumps(metrics, default=str)),
            )
            conn.executemany(
                """INSERT INTO backtest_trades
                (run_id, symbol, side, timestamp, price, quantity, fee, pnl, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                trade_rows,
            )
            conn.executemany(
                """INSERT INTO backtest_equity_curve
                (run_id, timestamp, equity, drawdown, open_positions)
                VALUES (?, ?, ?, ?, ?)""",
                curve_rows,
            )
            return conn.total_changes

    def list_bars(self, symbol: str, timeframe: str | None = None, limit: int | None = None) -> list[Bar]:
        params: list[object] = [symbol]
        query = "SELECT * FROM bars WHERE symbol = ?"
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)
        query += " ORDER BY timestamp ASC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            Bar(
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                source=row["source"],
            )
            for row in rows
        ]

    def list_backtest_runs(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, strategy, started_at, completed_at, config_json, metrics_json
                FROM backtest_runs
                ORDER BY completed_at DESC
                """
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "strategy": row["strategy"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "config": json.loads(row["config_json"]),
                "metrics": json.loads(row["metrics_json"]),
            }
            for row in rows
        ]

    def list_backtest_trades(self, run_id: str, symbol: str | None = None) -> list[dict]:
        params: list[object] = [run_id]
        query = "SELECT * FROM backtest_trades WHERE run_id = ?"
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        query += " ORDER BY timestamp ASC, id ASC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def list_equity_curve(self, run_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, equity, drawdown, open_positions
                FROM backtest_equity_curve
                WHERE run_id = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_news_without_signals(self, limit: int = 50) -> list[NewsItem]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT n.*
                FROM news_items n
                ORDER BY n.published_at ASC
                """,
            ).fetchall()
            existing_rows = conn.execute("SELECT news_hash, symbol FROM llm_signals").fetchall()
        existing: dict[str, set[str]] = {}
        for row in existing_rows:
            existing.setdefault(row["news_hash"], set()).add(row["symbol"])
        items: list[NewsItem] = []
        for row in rows:
            symbols = tuple(json.loads(row["symbols_json"]))
            required_symbols = symbols[:1]
            if required_symbols and set(required_symbols).issubset(existing.get(row["dedupe_hash"], set())):
                continue
            items.append(
                NewsItem(
                    source=row["source"],
                    url=row["url"],
                    title=row["title"],
                    summary=row["summary"],
                    published_at=datetime.fromisoformat(row["published_at"]),
                    symbols=symbols,
                    dedupe_hash=row["dedupe_hash"],
                )
            )
            if len(items) >= limit:
                break
        return items

    def existing_signal_symbols(self, news_hash: str) -> set[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT symbol FROM llm_signals WHERE news_hash = ?",
                (news_hash,),
            ).fetchall()
        return {row["symbol"] for row in rows}

    def upsert_defi_snapshot(self, symbol: str, asof: datetime, metrics: dict) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO defillama_snapshots
                (symbol, asof, tvl, tvl_growth, tvl_growth_30d, fees_24h, fees_7d, fees_30d,
                 fee_growth, revenue_growth, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    symbol, asof.isoformat(),
                    metrics.get("tvl"),
                    metrics.get("tvl_growth"),
                    metrics.get("tvl_growth_30d"),
                    metrics.get("fees_24h"),
                    metrics.get("fees_7d"),
                    metrics.get("fees_30d"),
                    metrics.get("fee_growth"),
                    metrics.get("revenue_growth"),
                    metrics.get("source", "defillama"),
                ),
            )

    def latest_defi_snapshot_before(self, symbol: str, asof: datetime) -> dict | None:
        """Return the most recent snapshot row at or before asof; None if none."""
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM defillama_snapshots
                WHERE symbol = ? AND asof <= ?
                ORDER BY asof DESC
                LIMIT 1
                """,
                (symbol, asof.isoformat()),
            ).fetchone()
        if not row:
            return None
        return {k: row[k] for k in row.keys()}

    def list_defi_snapshots(self, symbol: str | None = None, *, limit: int = 1000) -> list[dict]:
        params: list[object] = []
        query = "SELECT * FROM defillama_snapshots"
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
        query += " ORDER BY symbol ASC, asof ASC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [{k: row[k] for k in row.keys()} for row in rows]

    def list_news_items(self, *, limit: int = 500, since: datetime | None = None) -> list[NewsItem]:
        """List recent news items optionally filtered by published_at >= since."""
        params: list[object] = []
        query = "SELECT * FROM news_items"
        if since is not None:
            query += " WHERE published_at >= ?"
            params.append(since.isoformat())
        query += " ORDER BY published_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            NewsItem(
                source=row["source"],
                url=row["url"],
                title=row["title"],
                summary=row["summary"],
                published_at=datetime.fromisoformat(row["published_at"]),
                symbols=tuple(json.loads(row["symbols_json"])),
                dedupe_hash=row["dedupe_hash"],
            )
            for row in rows
        ]

    def list_news_for_symbol(self, symbol: str, limit: int = 100) -> list[NewsItem]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM news_items
                ORDER BY published_at DESC
                LIMIT ?
                """,
                (max(limit * 5, limit),),
            ).fetchall()
        items: list[NewsItem] = []
        for row in rows:
            symbols = tuple(json.loads(row["symbols_json"]))
            if symbol not in symbols:
                continue
            items.append(
                NewsItem(
                    source=row["source"],
                    url=row["url"],
                    title=row["title"],
                    summary=row["summary"],
                    published_at=datetime.fromisoformat(row["published_at"]),
                    symbols=symbols,
                    dedupe_hash=row["dedupe_hash"],
                )
            )
            if len(items) >= limit:
                break
        return items

    def latest_bars(self, symbol: str, limit: int = 120) -> list[Bar]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM bars
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (symbol, limit),
            ).fetchall()
        bars = [
            Bar(
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                source=row["source"],
            )
            for row in rows
        ]
        return list(reversed(bars))

    def latest_signals(self, symbol: str, limit: int = 20) -> list[LlmSignal]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM llm_signals
                WHERE symbol = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (symbol, limit),
            ).fetchall()
        return [
            LlmSignal(
                news_hash=row["news_hash"],
                symbol=row["symbol"],
                sentiment=float(row["sentiment"]),
                confidence=float(row["confidence"]),
                horizon=row["horizon"],
                event_type=row["event_type"],
                rationale=row["rationale"],
                provider=row["provider"],
                model=row["model"],
                created_at=datetime.fromisoformat(row["created_at"]),
                fear=float(row["fear"]) if "fear" in row.keys() and row["fear"] is not None else 0.0,
                hype=float(row["hype"]) if "hype" in row.keys() and row["hype"] is not None else 0.0,
                topic=row["topic"] if "topic" in row.keys() and row["topic"] is not None else "",
                source_weight=float(row["source_weight"]) if "source_weight" in row.keys() and row["source_weight"] is not None else 1.0,
                symbol_confidence=float(row["symbol_confidence"]) if "symbol_confidence" in row.keys() and row["symbol_confidence"] is not None else 1.0,
            )
            for row in rows
        ]

    # ---------- Autopilot helpers ----------

    def insert_decision(
        self,
        *,
        run_id: str,
        symbol: str,
        score: float,
        confidence: float,
        accepted: bool,
        reason: str,
        intent: dict,
        decided_at: datetime | None = None,
    ) -> int:
        ts = (decided_at or datetime.now(tz=timezone.utc)).isoformat()
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO decisions
                (run_id, symbol, score, confidence, accepted, reason, intent_json, decided_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, symbol, float(score), float(confidence), int(bool(accepted)), reason, json.dumps(intent, default=str), ts),
            )
            return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    def list_decisions(self, date: str | None = None, run_id: str | None = None) -> list[dict]:
        params: list[object] = []
        query = "SELECT * FROM decisions WHERE 1=1"
        if date:
            query += " AND substr(decided_at, 1, 10) = ?"
            params.append(date)
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        query += " ORDER BY decided_at DESC, id DESC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        out: list[dict] = []
        for row in rows:
            d = dict(row)
            d["intent"] = json.loads(d.pop("intent_json"))
            d["accepted"] = bool(d["accepted"])
            out.append(d)
        return out

    def sync_positions(self, broker_positions) -> int:
        """Replace positions table with current broker view. Returns row count."""
        with self.connect() as conn:
            conn.execute("DELETE FROM positions")
            for p in broker_positions:
                conn.execute(
                    """INSERT INTO positions
                    (symbol, entry_price, quantity, remaining_quantity, stop_price, state_json)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        p.symbol,
                        float(p.avg_entry_price),
                        float(p.qty),
                        float(p.qty),
                        0.0,
                        json.dumps(
                            {
                                "market_value": float(p.market_value),
                                "unrealized_pl": float(p.unrealized_pl),
                                "side": p.side,
                            },
                            default=str,
                        ),
                    ),
                )
            return int(conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0])

    def list_positions(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM positions").fetchall()
        out: list[dict] = []
        for row in rows:
            d = dict(row)
            d["state"] = json.loads(d.pop("state_json")) if d.get("state_json") else {}
            out.append(d)
        return out

    def upsert_daily_report(
        self,
        *,
        date: str,
        equity_open: float,
        equity_close: float,
        n_decisions: int,
        n_orders: int,
        markdown_path: str,
        summary: dict,
    ) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO daily_reports
                (date, equity_open, equity_close, n_decisions, n_orders, markdown_path, summary_json, generated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                  equity_open=excluded.equity_open,
                  equity_close=excluded.equity_close,
                  n_decisions=excluded.n_decisions,
                  n_orders=excluded.n_orders,
                  markdown_path=excluded.markdown_path,
                  summary_json=excluded.summary_json,
                  generated_at=excluded.generated_at""",
                (date, float(equity_open), float(equity_close), int(n_decisions), int(n_orders), markdown_path, json.dumps(summary, default=str), now),
            )

    def list_daily_reports(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM daily_reports ORDER BY date DESC").fetchall()
        out: list[dict] = []
        for row in rows:
            d = dict(row)
            d["summary"] = json.loads(d.pop("summary_json"))
            out.append(d)
        return out

    def get_daily_report(self, date: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM daily_reports WHERE date = ?", (date,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["summary"] = json.loads(d.pop("summary_json"))
        return d
