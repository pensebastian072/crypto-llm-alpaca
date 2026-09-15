from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from .data import fetch_alpaca_crypto_bars
from .backtest import (
    demo_bars,
    focused_sweep_configs,
    run_adaptive_breakout_backtest,
    run_breakout_backtest,
    run_quality_growth_backtest,
    run_simple_backtest,
)
from ._paths import find_repo_root
from .autopilot import run_autopilot
from .autopilot_backtest import run_autopilot_backtest
from .attribution import reconcile_symbols
from .candidate_explain import format_candidate_report
from .config import load_config
from .env import load_env
from .walk_forward import render_walk_forward_markdown, run_walk_forward
from .features import compute_strategy_features
from .llm import build_provider
from .news import fetch_rss
from .paper import build_paper_decision, submit_if_requested
from .research.narratives import aggregate_sector_rotation
from .schemas import LlmSignal
from .storage import SQLiteStore


def _load(args: argparse.Namespace):
    load_env()
    cfg = load_config(args.config)
    store = SQLiteStore(cfg.storage.path)
    return cfg, store


def cmd_init_db(args: argparse.Namespace) -> int:
    _, store = _load(args)
    store.init_schema()
    print(f"initialized {store.path}")
    return 0


def cmd_ingest_news(args: argparse.Namespace) -> int:
    cfg, store = _load(args)
    store.init_schema()
    items = []
    errors = []
    for feed in cfg.data.rss_feeds:
        try:
            items.extend(fetch_rss(feed, cfg.universe.symbols))
        except Exception as exc:
            errors.append({"feed": feed, "error": str(exc)})
    inserted = store.upsert_news(items)
    print(json.dumps({"fetched": len(items), "inserted_or_changed": inserted, "errors": errors}, indent=2))
    return 0


def cmd_extract_signals(args: argparse.Namespace) -> int:
    cfg, store = _load(args)
    store.init_schema()
    provider = build_provider(cfg.llm.provider, cfg.llm.model)
    news_items = store.list_news_without_signals(limit=args.limit)
    signals = []
    by_symbol: Counter[str] = Counter()
    fallback_symbols = 0
    corrected_symbols = 0
    skipped_existing = 0
    for item in news_items:
        article_symbols = tuple(item.symbols)
        target_symbols = article_symbols[:1]
        allowed_symbols = tuple(sorted(set(cfg.universe.symbols) | set(article_symbols)))
        existing_symbols = store.existing_signal_symbols(item.dedupe_hash)
        for symbol in target_symbols:
            if symbol in existing_symbols:
                skipped_existing += 1
                continue
            raw_signal = provider.extract(item, symbol)
            reconciliation = reconcile_symbols(
                (raw_signal.symbol,),
                article_symbols,
                allowed_symbols,
                primary_symbol_bias=True,
            )
            if reconciliation.corrected:
                corrected_symbols += 1
            if reconciliation.fallback:
                fallback_symbols += 1
            for final_symbol in reconciliation.symbols:
                if final_symbol in existing_symbols:
                    skipped_existing += 1
                    continue
                signals.append(
                    replace(
                        raw_signal,
                        symbol=final_symbol,
                        symbol_confidence=reconciliation.symbol_confidence,
                    )
                )
                by_symbol[final_symbol] += 1
                existing_symbols.add(final_symbol)
    inserted = store.insert_llm_signals(signals)
    print(
        json.dumps(
            {
                "provider": provider.__class__.__name__,
                "news_items": len(news_items),
                "signals": len(signals),
                "inserted": inserted,
                "by_symbol": dict(sorted(by_symbol.items())),
                "fallback_symbols": fallback_symbols,
                "corrected_symbols": corrected_symbols,
                "skipped_existing": skipped_existing,
            },
            indent=2,
        )
    )
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg, store = _load(args)
    store.init_schema()
    if args.strategy == "quality-growth" and (
        args.research_promotions
        or args.research_promotion_min_score is not None
        or args.research_promotion_size_scale is not None
        or args.research_promotion_symbols is not None
    ):
        qcfg = cfg.quality_growth
        cfg = replace(
            cfg,
            quality_growth=replace(
                qcfg,
                backtest_use_fund_style=True,
                backtest_allow_research_promotions=(
                    True if args.research_promotions else qcfg.backtest_allow_research_promotions
                ),
                research_promotion_min_score=(
                    args.research_promotion_min_score
                    if args.research_promotion_min_score is not None
                    else qcfg.research_promotion_min_score
                ),
                research_promotion_size_scale=(
                    args.research_promotion_size_scale
                    if args.research_promotion_size_scale is not None
                    else qcfg.research_promotion_size_scale
                ),
                research_promotion_symbols=(
                    tuple(s.strip() for s in args.research_promotion_symbols.split(",") if s.strip())
                    if args.research_promotion_symbols
                    else qcfg.research_promotion_symbols
                ),
            ),
        )
    if args.strategy == "demo":
        if not args.demo:
            raise SystemExit("demo strategy requires --demo")
        bars = {cfg.universe.symbols[0]: demo_bars(cfg.universe.symbols[0])}
        result = run_simple_backtest(cfg, bars)
        print(json.dumps(asdict(result), indent=2))
        return 0

    bars_by_symbol = _load_backtest_bars(
        cfg,
        store,
        demo=args.demo,
        bars_limit=args.bars_limit,
        all_bars=args.all_bars,
    )
    if args.strategy == "adaptive-breakout":
        result = run_adaptive_breakout_backtest(
            cfg,
            bars_by_symbol,
            starting_equity=args.equity,
            run_id=f"abt-{uuid4().hex[:12]}",
        )
    elif args.strategy == "quality-growth":
        result = run_quality_growth_backtest(
            cfg,
            bars_by_symbol,
            starting_equity=args.equity,
            run_id=f"qg-{uuid4().hex[:12]}",
            store=store,
        )
    else:
        result = run_breakout_backtest(
            cfg,
            bars_by_symbol,
            starting_equity=args.equity,
            run_id=f"bt-{uuid4().hex[:12]}",
        )
    if args.persist:
        _persist_backtest_result(store, cfg, result)
    payload = asdict(result)
    if args.summary:
        payload = {k: v for k, v in payload.items() if k not in {"trade_log", "equity_curve"}}
    print(json.dumps(payload, indent=2))
    return 0


def _persist_backtest_result(store: SQLiteStore, cfg, result) -> int:
    payload = asdict(result)
    return store.insert_backtest_run(
        result.run_id,
        result.strategy,
        {
            "strategy": asdict(cfg.strategy),
            "quality_growth": asdict(cfg.quality_growth),
            "project_profiles": {sym: asdict(profile) for sym, profile in cfg.project_profiles.items()},
            "costs": asdict(cfg.costs),
            "risk": asdict(cfg.risk),
        },
        {k: v for k, v in payload.items() if k not in {"trade_log", "equity_curve"}},
        result.trade_log,
        result.equity_curve,
    )


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def cmd_backfill(args: argparse.Namespace) -> int:
    cfg, store = _load(args)
    store.init_schema()
    if args.demo:
        bars = []
        count = int(args.days * 24) if args.days else cfg.data.lookback_bars
        for symbol in cfg.universe.symbols:
            bars.extend(demo_bars(symbol=symbol, count=count))
    else:
        end = _parse_datetime(args.end) if args.end else datetime.now(tz=timezone.utc)
        days = args.days or 365
        if args.start:
            start = _parse_datetime(args.start)
        else:
            start = end - timedelta(days=days)
        timeframe = args.timeframe or cfg.data.timeframe
        bars = fetch_alpaca_crypto_bars(cfg.universe.symbols, start, end, timeframe)
    changed = store.insert_bars(bars)
    print(
        json.dumps(
            {
                "bars": len(bars),
                "inserted_or_changed": changed,
                "start": bars[0].timestamp.isoformat() if bars else None,
                "end": bars[-1].timestamp.isoformat() if bars else None,
            },
            indent=2,
        )
    )
    return 0


def _paper_iteration(args: argparse.Namespace, cfg, store: SQLiteStore) -> dict:
    symbol = args.symbol or cfg.universe.symbols[0]
    bars = [] if args.demo_only else store.latest_bars(symbol, limit=cfg.data.lookback_bars)
    if len(bars) < 25:
        bars = demo_bars(symbol=symbol, count=cfg.data.lookback_bars)
        bars_source = "demo"
    else:
        bars_source = "sqlite"
    signals = [] if args.demo_only else store.latest_signals(symbol, limit=args.signal_limit)
    if not signals:
        signals = [
            LlmSignal(
                news_hash="manual-demo",
                symbol=symbol,
                sentiment=args.sentiment,
                confidence=args.confidence,
                horizon="short",
                event_type="manual_demo",
                rationale="manual paper-once demo signal",
            )
        ]
        signals_source = "manual"
    else:
        signals_source = "sqlite"
    decision = build_paper_decision(cfg, symbol, bars, signals, equity=args.equity)
    final = submit_if_requested(cfg, decision, bars[-1].close, submit=args.submit)
    store.insert_trade_intent(final.intent)
    payload = asdict(final)
    payload["inputs"] = {"symbol": symbol, "bars_source": bars_source, "signals_source": signals_source}
    return payload


def cmd_paper_once(args: argparse.Namespace) -> int:
    cfg, store = _load(args)
    store.init_schema()
    payload = _paper_iteration(args, cfg, store)
    print(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_paper_loop(args: argparse.Namespace) -> int:
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    cfg, store = _load(args)
    store.init_schema()
    payloads = [_paper_iteration(args, cfg, store) for _ in range(args.iterations)]
    print(json.dumps({"iterations": args.iterations, "results": payloads}, indent=2, default=str))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    cfg, store = _load(args)
    store.init_schema()
    with store.connect() as conn:
        counts = {
            "bars": conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0],
            "news_items": conn.execute("SELECT COUNT(*) FROM news_items").fetchone()[0],
            "llm_signals": conn.execute("SELECT COUNT(*) FROM llm_signals").fetchone()[0],
            "trade_intents": conn.execute("SELECT COUNT(*) FROM trade_intents").fetchone()[0],
            "strategy_features": conn.execute("SELECT COUNT(*) FROM strategy_features").fetchone()[0],
            "backtest_runs": conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0],
            "backtest_trades": conn.execute("SELECT COUNT(*) FROM backtest_trades").fetchone()[0],
            "backtest_equity_curve": conn.execute("SELECT COUNT(*) FROM backtest_equity_curve").fetchone()[0],
        }
    print(json.dumps({"storage": cfg.storage.path, "counts": counts}, indent=2))
    return 0


def _doctor_checks(args: argparse.Namespace) -> list[dict]:
    import os
    from datetime import datetime as _dt, timezone as _tz

    from .schemas import NewsItem

    checks: list[dict] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    load_env()
    env_path = find_repo_root() / ".env"
    if env_path.exists():
        add(".env loaded", "ok", str(env_path))
    else:
        add(".env loaded", "warn", f"no .env at {env_path} (process env still used)")

    apca_key = os.getenv("APCA_API_KEY_ID")
    apca_secret = os.getenv("APCA_API_SECRET_KEY")
    if apca_key and apca_secret:
        add("Alpaca keys present", "ok", "APCA_API_KEY_ID and APCA_API_SECRET_KEY set")
    else:
        missing = [n for n, v in (("APCA_API_KEY_ID", apca_key), ("APCA_API_SECRET_KEY", apca_secret)) if not v]
        add("Alpaca keys present", "fail", f"missing: {', '.join(missing)}")

    if apca_key and apca_secret:
        try:
            from alpaca.trading.client import TradingClient

            client = TradingClient(api_key=apca_key, secret_key=apca_secret, paper=True)
            account = client.get_account()
            add("Alpaca paper reachable", "ok", f"equity={account.equity}")
        except Exception as exc:
            msg = str(exc)
            status = "fail" if "401" in msg or "unauthorized" in msg.lower() else "warn"
            add("Alpaca paper reachable", status, msg[:200])
    else:
        add("Alpaca paper reachable", "warn", "skipped (no keys)")

    hf_token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
    if hf_token:
        add("HF token present", "ok", "HUGGINGFACE_API_TOKEN set")
    else:
        add("HF token present", "warn", "no HUGGINGFACE_API_TOKEN/HF_TOKEN (mock provider still works)")

    cfg = None
    try:
        cfg = load_config(args.config)
        add("Config loadable", "ok", args.config)
    except Exception as exc:
        add("Config loadable", "fail", f"{args.config}: {exc}")

    if cfg is not None and hf_token and cfg.llm.provider == "huggingface":
        try:
            provider = build_provider("huggingface", os.getenv("HF_MODEL", cfg.llm.model))
            stub = NewsItem(
                source="doctor",
                url="https://example.com/doctor",
                title="BTC rally test",
                summary="Doctor preflight check.",
                published_at=_dt.now(tz=_tz.utc),
                symbols=("BTC/USD",),
                dedupe_hash="doctor-preflight",
            )
            provider.extract(stub, "BTC/USD")
            add("HF preflight", "ok", f"model={provider.model}")
        except Exception as exc:
            add("HF preflight", "warn", str(exc)[:200])
    else:
        add("HF preflight", "warn", "skipped (provider != huggingface or no token)")

    if cfg is not None:
        try:
            store = SQLiteStore(cfg.storage.path)
            store.init_schema()
            with store.connect() as conn:
                tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "backtest_runs" in tables:
                    add("DB schema present", "ok", f"{len(tables)} tables")
                else:
                    add("DB schema present", "fail", "backtest_runs missing")
                bar_count = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
            if bar_count > 0:
                add("DB has bars", "ok", f"{bar_count} rows")
            else:
                add("DB has bars", "warn", "0 bars (run: crypto-llm backfill --demo)")
        except Exception as exc:
            add("DB schema present", "fail", str(exc)[:200])
            add("DB has bars", "warn", "skipped (DB error)")
    else:
        add("DB schema present", "warn", "skipped (config not loadable)")
        add("DB has bars", "warn", "skipped (config not loadable)")

    return checks


def cmd_doctor(args: argparse.Namespace) -> int:
    results = _doctor_checks(args)
    if args.json:
        print(json.dumps({"checks": results}, indent=2))
    else:
        width = max(len(r["name"]) for r in results)
        for r in results:
            badge = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}[r["status"]]
            print(f"[{badge}] {r['name'].ljust(width)}  {r['detail']}")
    return 1 if any(r["status"] == "fail" for r in results) else 0


def _load_backtest_bars(
    cfg,
    store: SQLiteStore,
    *,
    demo: bool,
    bars_limit: int | None = None,
    all_bars: bool = False,
) -> dict[str, list]:
    limit = bars_limit or max(cfg.data.lookback_bars, 160)
    if demo:
        return {
            symbol: demo_bars(symbol=symbol, count=limit)
            for symbol in cfg.universe.symbols
        }
    if all_bars:
        bars_by_symbol = {
            symbol: store.list_bars(symbol, timeframe=cfg.data.timeframe)
            for symbol in cfg.universe.symbols
        }
        return {symbol: bars for symbol, bars in bars_by_symbol.items() if bars}
    bars_by_symbol = {
        symbol: store.latest_bars(symbol, limit=limit)
        for symbol in cfg.universe.symbols
    }
    return {symbol: bars for symbol, bars in bars_by_symbol.items() if bars}


def cmd_build_features(args: argparse.Namespace) -> int:
    cfg, store = _load(args)
    store.init_schema()
    generated = []
    for symbol in cfg.universe.symbols:
        bars = store.latest_bars(symbol, limit=max(cfg.data.lookback_bars, 160))
        start = max(
            cfg.strategy.breakout_lookback,
            cfg.strategy.volume_lookback,
            cfg.strategy.momentum_lookback,
            cfg.strategy.htf_trend_lookback * 4,
        ) + 1
        for idx in range(start, len(bars)):
            generated.append(compute_strategy_features(bars[: idx + 1], cfg.strategy))
    changed = store.upsert_strategy_features(generated)
    print(json.dumps({"features": len(generated), "inserted_or_changed": changed}, indent=2))
    return 0


def cmd_rank_setups(args: argparse.Namespace) -> int:
    cfg, store = _load(args)
    store.init_schema()
    setups = []
    for symbol in cfg.universe.symbols:
        bars = store.latest_bars(symbol, limit=max(cfg.data.lookback_bars, 160))
        if not bars:
            continue
        try:
            features = compute_strategy_features(bars, cfg.strategy)
        except ValueError:
            continue
        item = asdict(features)
        item["entry_signal"] = features.entry_signal
        setups.append(item)
    setups.sort(key=lambda item: (item["entry_signal"], item["signal_strength"]), reverse=True)
    print(json.dumps({"setups": setups[: args.limit]}, indent=2, default=str))
    return 0


def cmd_quality_growth_candidates(args: argparse.Namespace) -> int:
    cfg, store = _load(args)
    store.init_schema()
    from .research_candidates import compute_candidates

    rows = compute_candidates(
        cfg, store,
        limit=args.limit,
        signal_limit=args.signal_limit,
        narrative_news_limit=args.narrative_news_limit,
    )
    if args.explain:
        print(format_candidate_report(rows))
    else:
        print(json.dumps({"candidates": rows}, indent=2, default=str))
    return 0


def cmd_backfill_defi_snapshots(args: argparse.Namespace) -> int:
    cfg, store = _load(args)
    store.init_schema()
    from .defillama import fetch_historical_tvl, historical_slugs_for_symbol

    start = _parse_datetime(args.start) if args.start else None
    end = _parse_datetime(args.end) if args.end else datetime.now(tz=timezone.utc)
    step_sec = max(int(args.step_days), 1) * 86_400

    universe = list(cfg.universe.symbols) + [s for s in cfg.per_symbol.keys() if s not in cfg.universe.symbols]
    summary = {"symbols": {}, "errors": []}
    for sym in universe:
        slugs = historical_slugs_for_symbol(sym)
        if not slugs:
            continue
        series = []
        try:
            for candidate_slug in slugs:
                candidate_series = fetch_historical_tvl(candidate_slug, max_age_sec=86_400 * 30)
                if candidate_series:
                    series = candidate_series
                    break
        except Exception as exc:
            summary["errors"].append(f"{sym}: {exc}")
            continue
        if not series:
            summary["symbols"][sym] = 0
            continue
        first_ts = series[0]["date"]
        last_ts = series[-1]["date"]
        if start is None:
            start_ts = first_ts
        else:
            start_ts = max(start.timestamp(), first_ts)
        end_ts = min(end.timestamp(), last_ts)
        if start_ts >= end_ts:
            summary["symbols"][sym] = 0
            continue

        # Walk forward at step_sec increments, snapshot at each.
        from .defillama import _tvl_at  # type: ignore

        rows = 0
        ts = start_ts
        while ts <= end_ts:
            tvl = _tvl_at(series, ts)
            tvl_7d_ago = _tvl_at(series, ts - 86_400 * 7)
            tvl_30d_ago = _tvl_at(series, ts - 86_400 * 30)
            metrics = {"tvl": tvl}
            if tvl and tvl_7d_ago and tvl_7d_ago > 0:
                metrics["tvl_growth"] = max(-1.0, min(5.0, tvl / tvl_7d_ago - 1.0))
            if tvl and tvl_30d_ago and tvl_30d_ago > 0:
                metrics["tvl_growth_30d"] = max(-1.0, min(10.0, tvl / tvl_30d_ago - 1.0))
            asof = datetime.fromtimestamp(ts, tz=timezone.utc)
            store.upsert_defi_snapshot(sym, asof, metrics)
            rows += 1
            ts += step_sec
        summary["symbols"][sym] = rows
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_narrative_rotation(args: argparse.Namespace) -> int:
    cfg, store = _load(args)
    store.init_schema()
    since = datetime.now(tz=timezone.utc) - timedelta(days=args.current_days + args.previous_days + 1)
    items = store.list_news_items(limit=args.news_limit, since=since)
    rotations = aggregate_sector_rotation(
        items,
        current_days=args.current_days,
        previous_days=args.previous_days,
        top_symbols_per_sector=args.top_symbols,
    )
    payload = {
        "current_days": args.current_days,
        "previous_days": args.previous_days,
        "news_window_total": len(items),
        "rotations": [asdict(r) for r in rotations],
    }
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"# Narrative rotation — current {args.current_days}d vs prior {args.previous_days}d")
        print(f"# news in window: {len(items)}")
        print()
        if not rotations:
            print("No narratives detected.")
            return 0
        width = max(len(r.label) for r in rotations)
        print(f"  {'sector'.ljust(width)}  curr   prev   diff    n_curr  n_prev  top symbols")
        for r in rotations:
            print(
                f"  {r.label.ljust(width)}  "
                f"{r.current_score:+.3f}  {r.previous_score:+.3f}  {r.momentum:+.3f}  "
                f"{r.current_articles:>5}  {r.previous_articles:>5}   {', '.join(r.top_symbols[:5])}"
            )
    return 0


def cmd_research_impact(args: argparse.Namespace) -> int:
    cfg, store = _load(args)
    store.init_schema()
    from .research_impact import format_research_impact, run_research_impact

    start = _parse_datetime(args.start) if args.start else None
    end = _parse_datetime(args.end) if args.end else None
    result = run_research_impact(cfg, store, start=start, end=end)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(format_research_impact(result))
    return 0


def cmd_paper_candidates(args: argparse.Namespace) -> int:
    cfg, store = _load(args)
    store.init_schema()
    candidates = []
    for symbol in cfg.universe.symbols:
        bars = store.latest_bars(symbol, limit=max(cfg.data.lookback_bars, 160))
        if not bars:
            continue
        try:
            features = compute_strategy_features(bars, cfg.strategy)
        except ValueError:
            continue
        if features.entry_signal:
            item = asdict(features)
            item["entry_signal"] = True
            candidates.append(item)
    candidates.sort(key=lambda item: item["signal_strength"], reverse=True)
    print(json.dumps({"candidates": candidates[: args.limit]}, indent=2, default=str))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    cfg, store = _load(args)
    store.init_schema()
    if args.strategy != "adaptive-breakout":
        raise SystemExit("Only adaptive-breakout sweep is implemented")
    if args.preset != "focused":
        raise SystemExit("Only focused sweep preset is implemented")
    bars_by_symbol = _load_backtest_bars(cfg, store, demo=args.demo)
    results = []
    for idx, sweep_cfg in enumerate(focused_sweep_configs(cfg), start=1):
        result = run_adaptive_breakout_backtest(
            sweep_cfg,
            bars_by_symbol,
            starting_equity=args.equity,
            run_id=f"swp-{idx:03d}-{uuid4().hex[:8]}",
        )
        _persist_backtest_result(store, sweep_cfg, result)
        results.append(
            {
                "run_id": result.run_id,
                "return_pct": result.return_pct,
                "max_drawdown_pct": result.max_drawdown_pct,
                "profit_factor": result.profit_factor,
                "entries": result.entries,
                "rank_score": result.assumptions.get("rank_score", 0.0),
                "strategy": asdict(sweep_cfg.strategy),
            }
        )
    results.sort(key=lambda item: item["rank_score"], reverse=True)
    print(json.dumps({"runs": len(results), "top": results[: args.top]}, indent=2))
    return 0


def cmd_walk_forward(args: argparse.Namespace) -> int:
    cfg, store = _load(args)
    store.init_schema()
    bars_by_symbol: dict[str, list] = {}
    for sym in list(cfg.universe.symbols) + list(cfg.per_symbol.keys()):
        if sym in bars_by_symbol:
            continue
        bars_by_symbol[sym] = store.list_bars(sym, timeframe=cfg.data.timeframe)
    start = _parse_datetime(args.start)
    end = _parse_datetime(args.end)
    result = run_walk_forward(
        cfg, bars_by_symbol,
        start=start, end=end,
        train_days=args.train_days, test_days=args.test_days, step_days=args.step_days,
        starting_equity=args.equity, max_positions=args.max_positions,
    )
    md_path = find_repo_root() / "outputs" / f"walk-forward-{datetime.now(tz=timezone.utc).date().isoformat()}-{result.run_id[:8]}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_walk_forward_markdown(result, cfg), encoding="utf-8")

    if args.persist:
        from dataclasses import asdict as _asdict
        for w in result.windows:
            tr = w.test_result
            store.insert_backtest_run(
                tr["run_id"], "autopilot-walkforward",
                {"strategy": _asdict(cfg.strategy), "costs": _asdict(cfg.costs),
                 "risk": _asdict(cfg.risk), "best_params": w.best_params,
                 "test_window": [w.test_start.isoformat(), w.test_end.isoformat()]},
                tr,
                trades=[], equity_curve=[],
            )

    summary = {
        "run_id": result.run_id, "n_windows": len(result.windows),
        "aggregate": result.aggregate, "markdown": str(md_path),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_autopilot_backtest(args: argparse.Namespace) -> int:
    cfg, store = _load(args)
    store.init_schema()
    bars_by_symbol: dict[str, list] = {}
    for sym in cfg.universe.symbols:
        bars = store.list_bars(sym, timeframe=cfg.data.timeframe)
        if args.start:
            cutoff_start = _parse_datetime(args.start)
            bars = [b for b in bars if b.timestamp >= cutoff_start]
        if args.end:
            cutoff_end = _parse_datetime(args.end)
            bars = [b for b in bars if b.timestamp <= cutoff_end]
        bars_by_symbol[sym] = bars
    result = run_autopilot_backtest(
        cfg, bars_by_symbol,
        run_id=f"apbt-{uuid4().hex[:12]}",
        starting_equity=args.equity,
        max_trades_per_day=args.max_trades,
        max_positions=args.max_positions,
        take_profit_pct=args.take_profit,
        stop_loss_pct=args.stop_loss,
        synthetic_signals=args.synthetic_signals,
        seed=args.seed,
    )
    if args.persist:
        from dataclasses import asdict as _asdict
        payload = _asdict(result)
        store.insert_backtest_run(
            result.run_id, result.strategy,
            {"strategy": _asdict(cfg.strategy), "costs": _asdict(cfg.costs), "risk": _asdict(cfg.risk),
             "assumptions": result.assumptions},
            {k: v for k, v in payload.items() if k not in {"trade_log", "equity_curve"}},
            result.trade_log, result.equity_curve,
        )
    summary = {
        "run_id": result.run_id,
        "trades": result.trades,
        "entries": result.entries,
        "exits": result.exits,
        "final_equity": result.final_equity,
        "return_pct": result.return_pct,
        "buy_hold_return_pct": result.buy_hold_return_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "total_fees": result.total_fees,
        "tp_events": result.tp_events,
        "stop_events": result.stop_events,
        "expired_events": result.expired_events,
        "assumptions": result.assumptions,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_autopilot(args: argparse.Namespace) -> int:
    cfg, store = _load(args)
    store.init_schema()
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    result = run_autopilot(
        cfg, store,
        dry_run=args.dry_run,
        max_trades=args.max_trades,
        max_positions=args.max_positions,
        symbols=symbols,
        equity_override=args.equity,
        demo_only=args.demo_only,
    )
    payload = {
        "run_id": result.run_id,
        "date": result.date,
        "dry_run": result.dry_run,
        "equity_open": result.equity_open,
        "equity_close": result.equity_close,
        "n_decisions": len(result.decisions),
        "n_accepted": sum(1 for d in result.decisions if d.accepted),
        "n_orders": result.n_orders,
        "markdown_path": result.markdown_path,
        "errors": result.errors,
        "decisions": [asdict(d) for d in result.decisions],
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0 if not result.errors else 0  # errors logged, not fatal


def cmd_dashboard(args: argparse.Namespace) -> int:
    from ._paths import find_repo_root

    root = find_repo_root()
    home_path = Path(__file__).parent / "ui" / "Home.py"
    command = [sys.executable, "-m", "streamlit", "run", str(home_path)]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    env = os.environ.copy()
    env["CRYPTO_LLM_CONFIG"] = str(config_path)
    return subprocess.run(command, cwd=str(root), env=env, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crypto-llm")
    parser.add_argument("--config", default=str(find_repo_root() / "config" / "default.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db").set_defaults(func=cmd_init_db)
    sub.add_parser("ingest-news").set_defaults(func=cmd_ingest_news)
    extract = sub.add_parser("extract-signals")
    extract.add_argument("--limit", type=int, default=50)
    extract.set_defaults(func=cmd_extract_signals)
    backfill = sub.add_parser("backfill")
    backfill.add_argument("--demo", action="store_true")
    backfill.add_argument("--days", type=int)
    backfill.add_argument("--start")
    backfill.add_argument("--end")
    backfill.add_argument("--timeframe")
    backfill.set_defaults(func=cmd_backfill)
    backtest = sub.add_parser("backtest")
    backtest.add_argument("--demo", action="store_true")
    backtest.add_argument(
        "--strategy",
        choices=("demo", "breakout-volume", "adaptive-breakout", "quality-growth"),
        default="demo",
    )
    backtest.add_argument("--equity", type=float, default=10_000.0)
    backtest.add_argument("--bars-limit", type=int, default=None)
    backtest.add_argument("--all-bars", action="store_true")
    backtest.add_argument("--summary", action="store_true")
    backtest.add_argument("--persist", action="store_true")
    backtest.add_argument("--research-promotions", action="store_true")
    backtest.add_argument("--research-promotion-min-score", type=float, default=None)
    backtest.add_argument("--research-promotion-size-scale", type=float, default=None)
    backtest.add_argument("--research-promotion-symbols", type=str, default=None)
    backtest.set_defaults(func=cmd_backtest)
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--demo", action="store_true")
    sweep.add_argument("--strategy", choices=("adaptive-breakout",), default="adaptive-breakout")
    sweep.add_argument("--preset", choices=("focused",), default="focused")
    sweep.add_argument("--equity", type=float, default=10_000.0)
    sweep.add_argument("--top", type=int, default=10)
    sweep.set_defaults(func=cmd_sweep)
    sub.add_parser("build-features").set_defaults(func=cmd_build_features)
    rank = sub.add_parser("rank-setups")
    rank.add_argument("--limit", type=int, default=10)
    rank.set_defaults(func=cmd_rank_setups)
    qg_rank = sub.add_parser("quality-growth-candidates")
    qg_rank.add_argument("--limit", type=int, default=10)
    qg_rank.add_argument("--signal-limit", type=int, default=30)
    qg_rank.add_argument("--narrative-news-limit", type=int, default=80)
    qg_rank.add_argument("--explain", action="store_true")
    qg_rank.set_defaults(func=cmd_quality_growth_candidates)
    bfd = sub.add_parser("backfill-defi-snapshots")
    bfd.add_argument("--start", type=str, default=None, help="ISO date inclusive lower bound")
    bfd.add_argument("--end", type=str, default=None, help="ISO date inclusive upper bound (default: now)")
    bfd.add_argument("--step-days", type=int, default=1)
    bfd.set_defaults(func=cmd_backfill_defi_snapshots)
    nrot = sub.add_parser("narrative-rotation")
    nrot.add_argument("--current-days", type=int, default=7)
    nrot.add_argument("--previous-days", type=int, default=7)
    nrot.add_argument("--news-limit", type=int, default=2000)
    nrot.add_argument("--top-symbols", type=int, default=5)
    nrot.add_argument("--json", action="store_true")
    nrot.set_defaults(func=cmd_narrative_rotation)
    impact = sub.add_parser("research-impact")
    impact.add_argument("--start", type=str, default=None, help="ISO date/time inclusive lower bound")
    impact.add_argument("--end", type=str, default=None, help="ISO date/time inclusive upper bound")
    impact.add_argument("--json", action="store_true")
    impact.set_defaults(func=cmd_research_impact)
    candidates = sub.add_parser("paper-candidates")
    candidates.add_argument("--limit", type=int, default=5)
    candidates.set_defaults(func=cmd_paper_candidates)
    paper_once = sub.add_parser("paper-once")
    paper_once.add_argument("--equity", type=float, default=10_000.0)
    paper_once.add_argument("--symbol")
    paper_once.add_argument("--sentiment", type=float, default=0.8)
    paper_once.add_argument("--confidence", type=float, default=0.9)
    paper_once.add_argument("--signal-limit", type=int, default=20)
    paper_once.add_argument("--demo-only", action="store_true")
    paper_once.add_argument("--submit", action="store_true")
    paper_once.set_defaults(func=cmd_paper_once)
    paper_loop = sub.add_parser("paper-loop")
    paper_loop.add_argument("--iterations", type=int, default=1)
    paper_loop.add_argument("--equity", type=float, default=10_000.0)
    paper_loop.add_argument("--symbol")
    paper_loop.add_argument("--sentiment", type=float, default=0.8)
    paper_loop.add_argument("--confidence", type=float, default=0.9)
    paper_loop.add_argument("--signal-limit", type=int, default=20)
    paper_loop.add_argument("--demo-only", action="store_true")
    paper_loop.add_argument("--submit", action="store_true")
    paper_loop.set_defaults(func=cmd_paper_loop)
    sub.add_parser("dashboard").set_defaults(func=cmd_dashboard)
    sub.add_parser("report").set_defaults(func=cmd_report)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)
    autopilot = sub.add_parser("autopilot")
    autopilot.add_argument("--dry-run", action="store_true")
    autopilot.add_argument("--max-trades", type=int, default=3)
    autopilot.add_argument("--max-positions", type=int, default=5)
    autopilot.add_argument("--symbols", type=str, default=None, help="comma-separated override (e.g. BTC/USD,ETH/USD)")
    autopilot.add_argument("--equity", type=float, default=None, help="override broker equity")
    autopilot.add_argument("--demo-only", action="store_true")
    autopilot.set_defaults(func=cmd_autopilot)
    apbt = sub.add_parser("autopilot-backtest")
    apbt.add_argument("--start", type=str, default=None, help="ISO date/time inclusive lower bound")
    apbt.add_argument("--end", type=str, default=None, help="ISO date/time inclusive upper bound")
    apbt.add_argument("--equity", type=float, default=10_000.0)
    apbt.add_argument("--max-trades", type=int, default=3)
    apbt.add_argument("--max-positions", type=int, default=5)
    apbt.add_argument("--take-profit", type=float, default=None, help="override TP fraction (e.g. 0.10)")
    apbt.add_argument("--stop-loss", type=float, default=None, help="override SL fraction (e.g. 0.05)")
    apbt.add_argument("--synthetic-signals", action="store_true", help="sample sentiment from a distribution for stress testing")
    apbt.add_argument("--seed", type=int, default=42)
    apbt.add_argument("--persist", action="store_true")
    apbt.set_defaults(func=cmd_autopilot_backtest)
    wf = sub.add_parser("walk-forward")
    wf.add_argument("--start", type=str, required=True)
    wf.add_argument("--end", type=str, required=True)
    wf.add_argument("--train-days", type=int, default=90)
    wf.add_argument("--test-days", type=int, default=30)
    wf.add_argument("--step-days", type=int, default=30)
    wf.add_argument("--equity", type=float, default=10_000.0)
    wf.add_argument("--max-positions", type=int, default=5)
    wf.add_argument("--persist", action="store_true")
    wf.set_defaults(func=cmd_walk_forward)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    argv = _normalize_global_config_arg(list(argv))
    args = parser.parse_args(argv)
    return int(args.func(args))


def _normalize_global_config_arg(argv: list[str]) -> list[str]:
    if "--config" in argv:
        idx = argv.index("--config")
        if idx > 0 and idx + 1 < len(argv):
            value = argv[idx + 1]
            del argv[idx : idx + 2]
            return ["--config", value, *argv]
    for idx, item in enumerate(list(argv)):
        if item.startswith("--config=") and idx > 0:
            del argv[idx]
            return [item, *argv]
    return argv


if __name__ == "__main__":
    raise SystemExit(main())
