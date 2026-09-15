"""Historical backtest of the autopilot decision logic.

Replays the morning-decision loop daily on stored bars. Two modes:
- default: technical + risk gate only (no LLM signals)
- --synthetic-signals: sample sentiment from a normal distribution for stress testing

Persists to the existing `backtest_runs` table with strategy="autopilot" so the
Strategy Lab UI shows it for free.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from typing import Iterable, Optional

from .config import AppConfig
from .features import compute_features
from .regime import compute_btc_regime
from .risk import RiskManager
from .schemas import Bar, LlmSignal, Side
from .scoring import aggregate_sentiment, score_symbol


@dataclass
class _OpenTrade:
    symbol: str
    entry_index: int  # global index into bars list for that symbol
    entry_price: float
    qty: float
    notional: float
    tp_price: float
    sl_price: float
    entry_ts: datetime


@dataclass
class AutopilotBacktestResult:
    run_id: str
    strategy: str
    trades: int
    entries: int
    exits: int
    final_equity: float
    return_pct: float
    buy_hold_return_pct: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_fees: float
    tp_events: int
    stop_events: int
    expired_events: int
    assumptions: dict
    trade_log: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    open_positions: list[dict] = field(default_factory=list)
    per_symbol: dict[str, dict] = field(default_factory=dict)


def _slice_window(bars: list[Bar], end_ts: datetime, lookback: int) -> list[Bar]:
    cutoff = [b for b in bars if b.timestamp <= end_ts]
    return cutoff[-lookback:] if cutoff else []


def _synthetic_signal(symbol: str, ts: datetime, rng: random.Random) -> list[LlmSignal]:
    sentiment = max(-1.0, min(1.0, rng.gauss(0.3, 0.45)))
    confidence = max(0.0, min(1.0, rng.gauss(0.6, 0.2)))
    if confidence < 0.1:
        return []
    return [
        LlmSignal(
            news_hash=f"synthetic-{symbol}-{ts.isoformat()}",
            symbol=symbol,
            sentiment=sentiment,
            confidence=confidence,
            horizon="short",
            event_type="synthetic_test",
            rationale="synthetic signal for backtest",
            provider="synthetic",
            model="rng",
            created_at=ts,
        )
    ]


def _morning_index(bars: list[Bar], hour_utc: int = 9) -> list[int]:
    """Return indices of bars whose timestamp is the daily 09:00 UTC slot."""
    seen: dict[datetime.date, int] = {}
    for i, b in enumerate(bars):
        if b.timestamp.hour == hour_utc:
            d = b.timestamp.date()
            if d not in seen:
                seen[d] = i
    return sorted(seen.values())


def run_autopilot_backtest(
    cfg: AppConfig,
    bars_by_symbol: dict[str, list[Bar]],
    *,
    run_id: str,
    starting_equity: float = 10_000.0,
    max_trades_per_day: int = 3,
    max_positions: int = 5,
    take_profit_pct: Optional[float] = None,
    stop_loss_pct: Optional[float] = None,
    taker_bps: Optional[float] = None,
    synthetic_signals: bool = False,
    seed: int = 42,
    feature_lookback: int = 120,
    morning_hour_utc: int = 9,
) -> AutopilotBacktestResult:
    """Replay the morning-decision loop bar-by-bar over historical data.

    For each daily morning slot:
      1. For every symbol, score with empty (or synthetic) signals + technicals.
      2. Apply risk gate.
      3. Pick top-N accepted by score, capped by max_trades_per_day and budget.
      4. Open a bracket trade at next bar's open price; simulate fill on subsequent bars.
    """
    rng = random.Random(seed)
    global_tp = take_profit_pct or (cfg.risk.take_profit_levels[0].gain_pct if cfg.risk.take_profit_levels else 0.10)
    global_sl = stop_loss_pct or cfg.risk.stop_loss_pct
    taker_bps = taker_bps if taker_bps is not None else cfg.costs.taker_bps
    fee_rate = float(taker_bps) / 10_000.0

    def _resolve_tp_sl(sym: str, window: list[Bar]) -> tuple[float, float]:
        ov = cfg.symbol_override(sym)
        tp = ov.take_profit_pct if ov.take_profit_pct is not None else global_tp
        sl = ov.stop_loss_pct if ov.stop_loss_pct is not None else global_sl
        if ov.atr_stop_multiple and len(window) >= cfg.strategy.atr_window + 1:
            from .features import atr as _atr
            atr_value = _atr(window, cfg.strategy.atr_window)
            price = window[-1].close if window else 0.0
            if atr_value > 0 and price > 0:
                sl = max(0.005, (atr_value / price) * float(ov.atr_stop_multiple))
        return tp, sl

    symbols = [
        s for s in cfg.universe.symbols
        if s in bars_by_symbol and len(bars_by_symbol[s]) > feature_lookback and cfg.is_trade_enabled(s)
    ]
    if not symbols:
        raise ValueError("no symbols have enough bars for backtest")

    # Build a unified daily morning timeline from the union of symbol morning indices
    timeline: dict[datetime, dict[str, int]] = {}
    for sym in symbols:
        for idx in _morning_index(bars_by_symbol[sym], morning_hour_utc):
            ts = bars_by_symbol[sym][idx].timestamp
            timeline.setdefault(ts, {})[sym] = idx
    sorted_ts = sorted(timeline.keys())
    if not sorted_ts:
        raise ValueError("no morning slots found in bar data")

    equity = float(starting_equity)
    cash = equity
    open_trades: list[_OpenTrade] = []
    closed_trades: list[dict] = []
    equity_curve: list[dict] = []
    tp_events = stop_events = expired_events = 0
    total_fees = 0.0
    entries = exits = 0
    peak_equity = equity
    max_dd = 0.0
    risk_mgr = RiskManager(cfg.risk)

    def _mark_to_market(ts: datetime) -> float:
        mtm = cash
        for tr in open_trades:
            sym_bars = bars_by_symbol[tr.symbol]
            last = next((b for b in reversed(sym_bars) if b.timestamp <= ts), None)
            if last:
                mtm += tr.qty * last.close
        return mtm

    def _close_trade(tr: _OpenTrade, exit_price: float, exit_ts: datetime, reason: str) -> None:
        nonlocal cash, total_fees, exits
        gross = tr.qty * exit_price
        fee = gross * fee_rate
        cash += gross - fee
        total_fees += fee
        pnl = (exit_price - tr.entry_price) * tr.qty - fee
        closed_trades.append(
            {
                "symbol": tr.symbol,
                "side": "sell",
                "timestamp": exit_ts.isoformat(),
                "price": exit_price,
                "quantity": tr.qty,
                "fee": fee,
                "pnl": pnl,
                "reason": reason,
            }
        )
        exits += 1

    btc_bars_full = bars_by_symbol.get(cfg.regime.sensor_symbol, [])
    regime_off_days = 0

    for ts in sorted_ts:
        # 1) Resolve any open trades whose TP/SL was hit since previous step
        still_open: list[_OpenTrade] = []
        for tr in open_trades:
            sym_bars = bars_by_symbol[tr.symbol]
            # Walk bars strictly after entry up to ts (inclusive)
            resolved = False
            for j in range(tr.entry_index + 1, len(sym_bars)):
                bar = sym_bars[j]
                if bar.timestamp > ts:
                    break
                # SL takes precedence if both touched in same bar (conservative)
                if bar.low <= tr.sl_price:
                    _close_trade(tr, tr.sl_price, bar.timestamp, "stop_loss")
                    stop_events += 1
                    resolved = True
                    break
                if bar.high >= tr.tp_price:
                    _close_trade(tr, tr.tp_price, bar.timestamp, "take_profit")
                    tp_events += 1
                    resolved = True
                    break
            if not resolved:
                still_open.append(tr)
        open_trades = still_open

        # 2) Make decisions for this morning
        budget_positions = max(0, max_positions - len(open_trades))
        take_n = min(max_trades_per_day, budget_positions)
        # Regime gate (Phase 2): if BTC risk-off, skip entries this morning.
        if cfg.regime.enabled and btc_bars_full:
            window_btc = [b for b in btc_bars_full if b.timestamp <= ts]
            window_btc = window_btc[-max(cfg.regime.btc_ema_hours, cfg.regime.vol_window_hours):]
            if len(window_btc) >= 25:
                regime = compute_btc_regime(
                    window_btc,
                    ema_window=cfg.regime.btc_ema_hours,
                    vol_window=cfg.regime.vol_window_hours,
                    vol_extreme_threshold=cfg.regime.vol_extreme_threshold,
                    vol_high_threshold=cfg.regime.vol_high_threshold,
                    vol_low_threshold=cfg.regime.vol_low_threshold,
                )
                if not regime.risk_on:
                    take_n = 0
                    regime_off_days += 1
        if take_n <= 0:
            equity = _mark_to_market(ts)
            peak_equity = max(peak_equity, equity)
            if peak_equity > 0:
                max_dd = max(max_dd, (peak_equity - equity) / peak_equity)
            equity_curve.append({"timestamp": ts.isoformat(), "equity": equity, "drawdown": max_dd, "open_positions": float(len(open_trades))})
            continue

        candidates: list[tuple[str, int, float, float, float]] = []  # (sym, idx, score, conf, notional)
        for sym, idx in timeline[ts].items():
            window = _slice_window(bars_by_symbol[sym], ts, feature_lookback)
            if len(window) < 25:
                continue
            try:
                feats = compute_features(window)
            except Exception:
                continue
            sigs = _synthetic_signal(sym, ts, rng) if synthetic_signals else []
            notional_per = max(50.0, equity * cfg.risk.max_total_exposure_pct / max_positions)
            intent = score_symbol(feats, sigs, cfg.scoring, cfg.llm.min_confidence, notional_per)
            # Note: open_positions is enforced via max_positions budget in the outer loop;
            # passing {} keeps RiskManager from indexing missing PositionState fields.
            decision = risk_mgr.evaluate_entry(
                intent,
                equity=equity,
                open_positions={},
                marks={sym: window[-1].close},
                volatility=feats.volatility,
            )
            if decision.accepted and intent.side == Side.BUY:
                candidates.append((sym, idx, intent.score, intent.confidence, notional_per))

        candidates.sort(key=lambda c: c[2], reverse=True)
        for sym, idx, score, conf, notional in candidates[:take_n]:
            sym_bars = bars_by_symbol[sym]
            if idx + 1 >= len(sym_bars):
                continue
            entry_bar = sym_bars[idx + 1]
            price = entry_bar.open or entry_bar.close
            if price <= 0:
                continue
            window_for_tpsl = _slice_window(sym_bars, ts, feature_lookback)
            sym_tp, sym_sl = _resolve_tp_sl(sym, window_for_tpsl)
            qty = notional / price
            fee = qty * price * fee_rate
            if cash < (qty * price + fee):
                continue
            cash -= qty * price + fee
            total_fees += fee
            tp_price = price * (1.0 + sym_tp)
            sl_price = price * (1.0 - sym_sl)
            open_trades.append(_OpenTrade(
                symbol=sym, entry_index=idx + 1, entry_price=price, qty=qty,
                notional=notional, tp_price=tp_price, sl_price=sl_price, entry_ts=entry_bar.timestamp,
            ))
            closed_trades.append({
                "symbol": sym, "side": "buy", "timestamp": entry_bar.timestamp.isoformat(),
                "price": price, "quantity": qty, "fee": fee, "pnl": 0.0,
                "reason": f"autopilot_entry score={score:.3f} conf={conf:.2f}",
            })
            entries += 1

        equity = _mark_to_market(ts)
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            max_dd = max(max_dd, (peak_equity - equity) / peak_equity)
        equity_curve.append({"timestamp": ts.isoformat(), "equity": equity, "drawdown": max_dd, "open_positions": float(len(open_trades))})

    # End: leave remaining open trades unrealized but record them
    open_pos = []
    for tr in open_trades:
        sym_bars = bars_by_symbol[tr.symbol]
        last = sym_bars[-1]
        unreal = (last.close - tr.entry_price) * tr.qty
        open_pos.append({
            "symbol": tr.symbol, "entry_price": tr.entry_price, "qty": tr.qty,
            "tp_price": tr.tp_price, "sl_price": tr.sl_price,
            "current_price": last.close, "unrealized_pnl": unreal,
            "entry_ts": tr.entry_ts.isoformat(),
        })
        expired_events += 1

    final_equity = _mark_to_market(sorted_ts[-1])
    return_pct = (final_equity / starting_equity) - 1.0 if starting_equity else 0.0

    # Buy-and-hold reference: equal-weight basket from first morning to last
    bh_returns = []
    for sym in symbols:
        sb = bars_by_symbol[sym]
        first = next((b for b in sb if b.timestamp >= sorted_ts[0]), None)
        last = next((b for b in reversed(sb) if b.timestamp <= sorted_ts[-1]), None)
        if first and last and first.close:
            bh_returns.append(last.close / first.close - 1.0)
    buy_hold_return = sum(bh_returns) / len(bh_returns) if bh_returns else 0.0

    sells = [t for t in closed_trades if t["side"] == "sell"]
    wins = [t for t in sells if t["pnl"] > 0]
    losses = [t for t in sells if t["pnl"] <= 0]
    win_rate = len(wins) / len(sells) if sells else 0.0
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0)
    if profit_factor == math.inf:
        profit_factor = 999.0  # serialize-friendly

    # Per-symbol attribution: pair buys to sells in order.
    per_symbol_stats: dict[str, dict] = {}
    pending_entries: dict[str, list[dict]] = {}
    for t in closed_trades:
        sym = t["symbol"]
        if t["side"] == "buy":
            pending_entries.setdefault(sym, []).append(t)
        else:
            stats = per_symbol_stats.setdefault(sym, {"trades": 0, "tp": 0, "sl": 0, "expired": 0, "pnl": 0.0, "wins": 0})
            stats["trades"] += 1
            stats["pnl"] += float(t["pnl"])
            if "take_profit" in t["reason"]:
                stats["tp"] += 1
                stats["wins"] += 1
            elif "stop_loss" in t["reason"]:
                stats["sl"] += 1
            if pending_entries.get(sym):
                pending_entries[sym].pop(0)
    for sym, entries_left in pending_entries.items():
        if entries_left:
            stats = per_symbol_stats.setdefault(sym, {"trades": 0, "tp": 0, "sl": 0, "expired": 0, "pnl": 0.0, "wins": 0})
            stats["expired"] += len(entries_left)
    for sym, stats in per_symbol_stats.items():
        n = stats["trades"]
        stats["win_rate"] = stats["wins"] / n if n else 0.0

    return AutopilotBacktestResult(
        run_id=run_id,
        strategy="autopilot",
        trades=len(closed_trades),
        entries=entries,
        exits=exits,
        final_equity=final_equity,
        return_pct=return_pct,
        buy_hold_return_pct=buy_hold_return,
        max_drawdown_pct=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_fees=total_fees,
        tp_events=tp_events,
        stop_events=stop_events,
        expired_events=expired_events,
        assumptions={
            "synthetic_signals": synthetic_signals,
            "max_trades_per_day": max_trades_per_day,
            "max_positions": max_positions,
            "take_profit_pct": global_tp,
            "stop_loss_pct": global_sl,
            "taker_bps": taker_bps,
            "morning_hour_utc": morning_hour_utc,
            "regime_enabled": cfg.regime.enabled,
            "regime_off_days": regime_off_days,
            "rank_score": return_pct - max_dd,
        },
        trade_log=closed_trades,
        equity_curve=equity_curve,
        open_positions=open_pos,
        per_symbol=per_symbol_stats,
    )
