"""Morning autopilot: ingest news -> signals -> features -> decisions -> bracket orders."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from ._paths import find_repo_root
from .backtest import demo_bars
from .broker import AlpacaPaperBroker, SubmittedOrder
from .config import AppConfig
from .paper import build_paper_decision
from .regime import RegimeState, compute_btc_regime
from .schemas import LlmSignal
from .scoring import aggregate_sentiment
from .storage import SQLiteStore


@dataclass
class DecisionRow:
    symbol: str
    score: float
    confidence: float
    accepted: bool
    reason: str
    notional: float
    breakdown: str = ""
    rank: Optional[int] = None
    submitted_order_id: Optional[str] = None
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    used_fallback: bool = False
    submit_error: Optional[str] = None


@dataclass
class AutopilotResult:
    run_id: str
    date: str
    dry_run: bool
    equity_open: float
    equity_close: float
    decisions: list[DecisionRow] = field(default_factory=list)
    n_orders: int = 0
    open_positions: list[dict] = field(default_factory=list)
    markdown_path: str = ""
    errors: list[str] = field(default_factory=list)
    regime: Optional[dict] = None


def _today_iso() -> str:
    return datetime.now(tz=timezone.utc).date().isoformat()


def _summary_for_storage(result: AutopilotResult) -> dict:
    return {
        "run_id": result.run_id,
        "dry_run": result.dry_run,
        "n_decisions": len(result.decisions),
        "n_accepted": sum(1 for d in result.decisions if d.accepted),
        "n_orders": result.n_orders,
        "errors": list(result.errors),
        "top_symbols": [d.symbol for d in result.decisions if d.accepted][:5],
    }


def _render_markdown(result: AutopilotResult, cfg: AppConfig) -> str:
    lines: list[str] = []
    lines.append(f"# Autopilot — {result.date}")
    lines.append("")
    mode = "DRY RUN (no orders submitted)" if result.dry_run else "LIVE (Alpaca paper)"
    lines.append(f"**Mode:** {mode}  ")
    lines.append(f"**Run id:** `{result.run_id}`  ")
    lines.append(f"**Equity open:** ${result.equity_open:,.2f}  ")
    lines.append(f"**Equity close:** ${result.equity_close:,.2f}  ")
    lines.append(f"**Orders submitted:** {result.n_orders}  ")
    lines.append(f"**Universe:** {', '.join(cfg.universe.symbols)}  ")
    if result.regime:
        r = result.regime
        flag = "🟢 RISK ON" if r.get("risk_on") else "🔴 RISK OFF"
        lines.append(f"**Regime:** {flag} — {r.get('reason','')}  ")
    lines.append("")

    if result.open_positions:
        lines.append("## Open positions (Alpaca paper)")
        lines.append("")
        lines.append("| symbol | qty | entry | mkt value | unrealized P/L |")
        lines.append("|---|---|---|---|---|")
        for p in result.open_positions:
            lines.append(
                f"| {p.get('symbol','')} | {p.get('qty','')} | {p.get('avg_entry_price','')} "
                f"| {p.get('market_value','')} | {p.get('unrealized_pl','')} |"
            )
        lines.append("")

    accepted = [d for d in result.decisions if d.accepted]
    rejected = [d for d in result.decisions if not d.accepted]

    lines.append(f"## Accepted decisions ({len(accepted)})")
    lines.append("")
    if not accepted:
        lines.append("_None._")
    else:
        lines.append("| rank | symbol | score | conf | notional | TP | SL | order id | reason |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for d in accepted:
            tp = f"{d.take_profit_price:.2f}" if d.take_profit_price else ""
            sl = f"{d.stop_loss_price:.2f}" if d.stop_loss_price else ""
            oid = d.submitted_order_id or ("dry-run" if result.dry_run else (d.submit_error or ""))
            lines.append(
                f"| {d.rank or ''} | {d.symbol} | {d.score:.3f} | {d.confidence:.2f} "
                f"| ${d.notional:,.2f} | {tp} | {sl} | {oid} | {d.reason} |"
            )
    lines.append("")

    lines.append(f"## Rejected ({len(rejected)})")
    lines.append("")
    if not rejected:
        lines.append("_None._")
    else:
        lines.append("| symbol | score | conf | reason | breakdown |")
        lines.append("|---|---|---|---|---|")
        for d in rejected:
            lines.append(f"| {d.symbol} | {d.score:.3f} | {d.confidence:.2f} | {d.reason} | {d.breakdown} |")
    lines.append("")

    if result.errors:
        lines.append("## Errors")
        lines.append("")
        for err in result.errors:
            lines.append(f"- {err}")
        lines.append("")

    return "\n".join(lines)


def _build_decision_for_symbol(
    cfg: AppConfig,
    store: SQLiteStore,
    symbol: str,
    *,
    equity: float,
    signal_limit: int,
    demo_only: bool,
) -> DecisionRow:
    bars = [] if demo_only else store.latest_bars(symbol, limit=cfg.data.lookback_bars)
    if len(bars) < 25:
        bars = demo_bars(symbol=symbol, count=cfg.data.lookback_bars)
    signals: list[LlmSignal] = [] if demo_only else store.latest_signals(symbol, limit=signal_limit)
    paper_decision = build_paper_decision(cfg, symbol, bars, signals, equity=equity)
    intent = paper_decision.intent
    return DecisionRow(
        symbol=symbol,
        score=intent.score,
        confidence=intent.confidence,
        accepted=paper_decision.accepted and intent.side.value == "buy",
        reason=paper_decision.reason,
        notional=intent.notional,
        breakdown=intent.reason,
    )


def run_autopilot(
    cfg: AppConfig,
    store: SQLiteStore,
    *,
    dry_run: bool = False,
    max_trades: int = 3,
    max_positions: int = 5,
    symbols: Optional[list[str]] = None,
    equity_override: Optional[float] = None,
    signal_limit: int = 20,
    demo_only: bool = False,
    broker: Optional[AlpacaPaperBroker] = None,
    outputs_dir: Optional[Path] = None,
) -> AutopilotResult:
    store.init_schema()
    requested = symbols or list(cfg.universe.symbols)
    universe = [s for s in requested if cfg.is_trade_enabled(s)]
    run_id = f"ap-{_today_iso()}-{uuid4().hex[:8]}"
    today = _today_iso()
    result = AutopilotResult(run_id=run_id, date=today, dry_run=dry_run, equity_open=0.0, equity_close=0.0)

    if broker is None and not (dry_run and demo_only):
        try:
            broker = AlpacaPaperBroker(cfg.broker)
        except Exception as exc:
            result.errors.append(f"broker init failed: {exc}")
            broker = None

    if equity_override is not None:
        result.equity_open = equity_override
    elif broker is not None:
        try:
            account = broker.get_account()
            result.equity_open = float(account.equity)
        except Exception as exc:
            result.errors.append(f"get_account failed: {exc}")
            result.equity_open = 10_000.0
    else:
        result.equity_open = 10_000.0
    result.equity_close = result.equity_open

    if broker is not None:
        try:
            positions = broker.get_positions()
            store.sync_positions(positions)
            result.open_positions = [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "avg_entry_price": p.avg_entry_price,
                    "market_value": p.market_value,
                    "unrealized_pl": p.unrealized_pl,
                }
                for p in positions
            ]
        except Exception as exc:
            result.errors.append(f"sync_positions failed: {exc}")

    # Regime sensor (Phase 2): use BTC bars even though BTC may not be in trade universe.
    regime_state: Optional[RegimeState] = None
    if cfg.regime.enabled:
        sensor = cfg.regime.sensor_symbol
        btc_bars = [] if demo_only else store.latest_bars(sensor, limit=max(cfg.regime.btc_ema_hours, cfg.regime.vol_window_hours) + 50)
        if btc_bars:
            regime_state = compute_btc_regime(
                btc_bars,
                ema_window=cfg.regime.btc_ema_hours,
                vol_window=cfg.regime.vol_window_hours,
                vol_extreme_threshold=cfg.regime.vol_extreme_threshold,
                vol_high_threshold=cfg.regime.vol_high_threshold,
                vol_low_threshold=cfg.regime.vol_low_threshold,
            )
            result.regime = {
                "risk_on": regime_state.risk_on,
                "trend_slope": regime_state.trend_slope,
                "vol_regime": regime_state.vol_regime,
                "realized_vol": regime_state.realized_vol,
                "reason": regime_state.reason,
            }

    open_count = len(result.open_positions)
    budget = max(0, max_positions - open_count)
    take_n = min(max_trades, budget)
    if regime_state is not None and not regime_state.risk_on:
        take_n = 0  # block all new entries when risk-off

    decisions: list[DecisionRow] = []
    for symbol in universe:
        try:
            d = _build_decision_for_symbol(
                cfg, store, symbol,
                equity=result.equity_open, signal_limit=signal_limit, demo_only=demo_only,
            )
            # Asymmetric risk-off: aggregate sentiment of recent signals for this symbol
            recent = [] if demo_only else store.latest_signals(symbol, limit=signal_limit)
            agg_sent, _conf = aggregate_sentiment(recent, cfg.llm.min_confidence) if recent else (0.0, 0.0)
            if agg_sent < cfg.scoring.risk_off_threshold:
                d.accepted = False
                d.reason = f"risk_off_sentiment ({agg_sent:.2f} < {cfg.scoring.risk_off_threshold})"
            elif regime_state is not None and not regime_state.risk_on and d.accepted:
                d.accepted = False
                d.reason = f"regime_risk_off: {regime_state.reason}"
        except Exception as exc:
            d = DecisionRow(
                symbol=symbol, score=0.0, confidence=0.0,
                accepted=False, reason=f"decision_error: {exc}", notional=0.0,
            )
            result.errors.append(f"{symbol}: {exc}")
        decisions.append(d)

    accepted = sorted([d for d in decisions if d.accepted], key=lambda d: d.score, reverse=True)
    selected = accepted[:take_n]
    for rank, d in enumerate(selected, start=1):
        d.rank = rank

    # Phase B: portfolio sizing (conviction + per-category caps).
    if cfg.portfolio.enabled and selected:
        from .portfolio import size_decisions

        # Existing exposure-by-category from open positions on the broker.
        existing_by_cat: dict[str, float] = {}
        for p in result.open_positions:
            sym = (p.get("symbol") or "").replace("USD", "/USD") if "/" not in (p.get("symbol") or "") else p.get("symbol", "")
            cat = (cfg.project_profiles.get(sym).category if cfg.project_profiles.get(sym) else "uncategorized")
            existing_by_cat[cat] = existing_by_cat.get(cat, 0.0) + float(p.get("market_value") or 0.0)

        # Use score in [-1,1] mapped to [0,1] as conviction proxy when no explicit tradable_score.
        candidates = [
            {
                "symbol": d.symbol,
                "conviction": max(0.0, min(1.0, (d.score + 1.0) / 2.0)),
                "base_notional": d.notional,
            }
            for d in selected
        ]
        sized = size_decisions(candidates, equity=result.equity_open, cfg=cfg, existing_exposure_by_category=existing_by_cat)
        sized_by_sym = {s.symbol: s for s in sized}
        kept: list[DecisionRow] = []
        for d in selected:
            s = sized_by_sym.get(d.symbol)
            if s is None or s.sized_notional <= 0:
                d.accepted = False
                d.reason = (s.reason if s else "portfolio_drop")
                continue
            d.notional = s.sized_notional
            d.reason = f"{d.reason} | portfolio: {s.reason}"
            kept.append(d)
        selected = kept

    global_tp = cfg.risk.take_profit_levels[0].gain_pct if cfg.risk.take_profit_levels else 0.10
    global_sl = cfg.risk.stop_loss_pct

    for d in selected:
        override = cfg.symbol_override(d.symbol)
        sym_tp = override.take_profit_pct if override.take_profit_pct is not None else global_tp
        sym_sl = override.stop_loss_pct if override.stop_loss_pct is not None else global_sl

        if dry_run or broker is None:
            d.take_profit_price = None
            d.stop_loss_price = None
            continue
        bars = [] if demo_only else store.latest_bars(d.symbol, limit=cfg.data.lookback_bars)
        price = bars[-1].close if bars else 0.0
        if price <= 0:
            d.submit_error = "no price available"
            result.errors.append(f"{d.symbol}: no price for entry")
            continue
        # ATR-scaled SL if requested
        if override.atr_stop_multiple and len(bars) >= cfg.strategy.atr_window + 1:
            from .features import atr as _atr
            atr_value = _atr(bars, cfg.strategy.atr_window)
            if atr_value > 0:
                sym_sl = max(0.005, (atr_value / price) * float(override.atr_stop_multiple))
        try:
            from .schemas import Side, TradeIntent

            entry_intent = TradeIntent(
                symbol=d.symbol, side=Side.BUY, score=d.score, confidence=d.confidence,
                notional=d.notional, reason=d.reason,
            )
            order: SubmittedOrder = broker.submit_bracket_order(
                entry_intent, price=price,
                take_profit_pct=sym_tp, stop_loss_pct=sym_sl,
            )
            d.submitted_order_id = order.broker_order_id
            d.take_profit_price = order.take_profit_price
            d.stop_loss_price = order.stop_loss_price
            d.used_fallback = order.used_fallback
            result.n_orders += 1
        except Exception as exc:
            d.submit_error = str(exc)
            result.errors.append(f"{d.symbol} submit failed: {exc}")

    result.decisions = decisions

    for d in decisions:
        store.insert_decision(
            run_id=run_id, symbol=d.symbol, score=d.score, confidence=d.confidence,
            accepted=d.accepted, reason=d.reason, intent=asdict(d),
        )

    if not dry_run and broker is not None and result.n_orders > 0:
        try:
            account = broker.get_account()
            result.equity_close = float(account.equity)
        except Exception:
            pass

    out_dir = outputs_dir or (find_repo_root() / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{today}.md"
    md_path.write_text(_render_markdown(result, cfg), encoding="utf-8")
    result.markdown_path = str(md_path)

    store.upsert_daily_report(
        date=today,
        equity_open=result.equity_open,
        equity_close=result.equity_close,
        n_decisions=len(result.decisions),
        n_orders=result.n_orders,
        markdown_path=str(md_path),
        summary=_summary_for_storage(result),
    )
    return result
