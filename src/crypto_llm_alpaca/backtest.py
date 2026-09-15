from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import product

from .config import AppConfig, StrategyConfig
from .features import compute_features, compute_strategy_features
from .quality_growth import quality_growth_score
from .risk import RiskManager
from .schemas import Bar, LlmSignal, PositionState
from .scoring import score_symbol


MTF_LOOKBACK_HOURS = 24 * 7 * 8


@dataclass(frozen=True)
class BacktestResult:
    trades: int
    final_equity: float
    return_pct: float
    max_drawdown_pct: float
    tp_events: int
    stop_events: int


@dataclass(frozen=True)
class BreakoutBacktestResult:
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
    assumptions: dict[str, float | str | bool]
    trade_log: list[dict[str, float | str]]
    equity_curve: list[dict[str, float | str]]
    open_positions: list[dict[str, float | str]]
    per_symbol: dict[str, dict[str, float | int]] = field(default_factory=dict)


@dataclass
class AdaptivePosition:
    symbol: str
    entry_price: float
    quantity: float
    stop_price: float
    atr: float

    @property
    def is_open(self) -> bool:
        return self.quantity > 1e-12


@dataclass
class QualityPosition:
    symbol: str
    entry_price: float
    initial_quantity: float
    remaining_quantity: float
    stop_price: float
    peak_price: float
    profile_score: float
    filled_tp_indices: set[int] = field(default_factory=set)

    @property
    def is_open(self) -> bool:
        return self.remaining_quantity > 1e-12


def demo_bars(symbol: str = "BTC/USD", count: int = 80, start: float = 100.0) -> list[Bar]:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars: list[Bar] = []
    price = start
    for idx in range(count):
        price *= 1.006 if idx < 50 else 0.998
        close = price
        high = price * 1.005
        low = price * 0.995
        volume = 1000.0 + idx
        if idx in (35, 48) and len(bars) >= 20:
            close = max(bar.high for bar in bars[-20:]) * 1.025
            high = close * 1.006
            low = close * 0.982
            volume = 2600.0 + idx
            price = close
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=ts + timedelta(hours=idx),
                open=price * 0.998,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
    return bars


def run_simple_backtest(cfg: AppConfig, bars_by_symbol: dict[str, list[Bar]]) -> BacktestResult:
    equity = 10_000.0
    peak = equity
    max_drawdown = 0.0
    positions: dict[str, PositionState] = {}
    risk = RiskManager(cfg.risk)
    tp_events = 0
    stop_events = 0
    trades = 0

    for symbol, bars in bars_by_symbol.items():
        for idx in range(25, len(bars)):
            window = bars[: idx + 1]
            features = compute_features(window)
            signal = LlmSignal(
                news_hash="demo",
                symbol=symbol,
                sentiment=0.7,
                confidence=0.8,
                horizon="short",
                event_type="demo",
                rationale="demo bullish fixture",
            )
            intent = score_symbol(
                features,
                [signal],
                cfg.scoring,
                cfg.llm.min_confidence,
                notional=equity * cfg.risk.max_total_exposure_pct / cfg.risk.max_positions,
            )
            mark = bars[idx].close
            if symbol not in positions:
                decision = risk.evaluate_entry(
                    intent,
                    equity,
                    positions,
                    {s: mark for s in positions},
                    volatility=features.volatility,
                )
                if decision.accepted:
                    positions[symbol] = risk.build_position(symbol, mark, decision.adjusted_notional)
                    trades += 1
            if symbol in positions:
                events = risk.update_position_for_price(positions[symbol], mark)
                for event in events:
                    if str(event["event"]).startswith("tp"):
                        tp_events += 1
                        equity += float(event["quantity"]) * (mark - positions[symbol].entry_price)
                    elif event["event"] == "stop_loss":
                        stop_events += 1
                        equity += float(event["quantity"]) * (mark - positions[symbol].entry_price)
                if not positions[symbol].is_open:
                    del positions[symbol]
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, (peak - equity) / peak if peak else 0.0)
    return BacktestResult(trades, equity, (equity / 10_000.0) - 1.0, max_drawdown, tp_events, stop_events)


def _fee(notional: float, bps: float) -> float:
    return abs(notional) * bps / 10_000.0


def _strategy_feature_window(
    bars: list[Bar],
    idx: int,
    cfg: StrategyConfig,
    *,
    extra_lookback_hours: int = 0,
) -> list[Bar]:
    lookback = max(
        cfg.breakout_lookback,
        cfg.volume_lookback,
        cfg.momentum_lookback,
        cfg.htf_trend_lookback * 4,
        cfg.atr_long_window,
        extra_lookback_hours,
    ) + 8
    start = max(0, idx + 1 - lookback)
    return bars[start : idx + 1]


def _buy_hold_return(bars_by_symbol: dict[str, list[Bar]], fee_bps: float) -> float:
    returns = []
    for bars in bars_by_symbol.values():
        if len(bars) < 2 or bars[0].close <= 0:
            continue
        gross = bars[-1].close / bars[0].close - 1.0
        round_trip_fee = 2.0 * fee_bps / 10_000.0
        returns.append(gross - round_trip_fee)
    return sum(returns) / len(returns) if returns else 0.0


def _open_equity(equity: float, positions: dict[str, PositionState], marks: dict[str, float]) -> float:
    unrealized = 0.0
    for symbol, position in positions.items():
        mark = marks.get(symbol, position.entry_price)
        unrealized += position.remaining_quantity * (mark - position.entry_price)
    return equity + unrealized


def _open_adaptive_equity(equity: float, positions: dict[str, AdaptivePosition], marks: dict[str, float]) -> float:
    unrealized = 0.0
    for symbol, position in positions.items():
        mark = marks.get(symbol, position.entry_price)
        unrealized += position.quantity * (mark - position.entry_price)
    return equity + unrealized


def _open_quality_equity(equity: float, positions: dict[str, QualityPosition], marks: dict[str, float]) -> float:
    unrealized = 0.0
    for symbol, position in positions.items():
        mark = marks.get(symbol, position.entry_price)
        unrealized += position.remaining_quantity * (mark - position.entry_price)
    return equity + unrealized


def _per_symbol_attribution(trade_log: list[dict[str, float | str]]) -> dict[str, dict[str, float | int]]:
    rows: dict[str, dict[str, float | int]] = {}
    for trade in trade_log:
        if trade.get("side") != "sell":
            continue
        symbol = str(trade["symbol"])
        reason = str(trade.get("reason", ""))
        pnl = float(trade.get("pnl", 0.0))
        row = rows.setdefault(
            symbol,
            {"trades": 0, "tp": 0, "sl": 0, "expired": 0, "wins": 0, "win_rate": 0.0, "pnl": 0.0},
        )
        row["trades"] = int(row["trades"]) + 1
        row["pnl"] = float(row["pnl"]) + pnl
        if pnl > 0:
            row["wins"] = int(row["wins"]) + 1
        if "tp" in reason:
            row["tp"] = int(row["tp"]) + 1
        elif "stop" in reason:
            row["sl"] = int(row["sl"]) + 1
        elif reason == "end_of_backtest":
            row["expired"] = int(row["expired"]) + 1

    for row in rows.values():
        trades = int(row["trades"])
        row["win_rate"] = (int(row["wins"]) / trades) if trades else 0.0
    return rows


def _rank_score(metrics: dict) -> float:
    drawdown = float(metrics.get("max_drawdown_pct", 0.0))
    trades = int(metrics.get("entries", 0))
    if trades < 3:
        return -999.0 + trades
    return (
        float(metrics.get("return_pct", 0.0)) / max(drawdown, 0.01)
        + min(float(metrics.get("profit_factor", 0.0)), 10.0) * 0.10
        - float(metrics.get("total_fees", 0.0)) / max(float(metrics.get("final_equity", 1.0)), 1.0)
    )


def run_breakout_backtest(
    cfg: AppConfig,
    bars_by_symbol: dict[str, list[Bar]],
    *,
    starting_equity: float = 10_000.0,
    run_id: str = "breakout-volume",
) -> BreakoutBacktestResult:
    equity = starting_equity
    peak = starting_equity
    max_drawdown = 0.0
    positions: dict[str, PositionState] = {}
    risk = RiskManager(cfg.risk)
    fee_bps = cfg.costs.default_bps
    trade_log: list[dict[str, float | str]] = []
    equity_curve: list[dict[str, float | str]] = []
    realized_trade_pnls: list[float] = []
    total_fees = 0.0
    tp_events = 0
    stop_events = 0
    entries = 0
    exits = 0
    max_len = max((len(bars) for bars in bars_by_symbol.values()), default=0)
    min_window = max(
        cfg.strategy.breakout_lookback,
        cfg.strategy.volume_lookback,
        cfg.strategy.momentum_lookback,
        cfg.strategy.htf_trend_lookback * 4,
    ) + 1

    for idx in range(min_window, max_len):
        marks = {
            symbol: bars[idx].close
            for symbol, bars in bars_by_symbol.items()
            if idx < len(bars)
        }
        for symbol, bars in bars_by_symbol.items():
            if idx >= len(bars):
                continue
            mark = bars[idx].close
            if symbol in positions:
                position = positions[symbol]
                events = risk.update_position_for_price(position, mark)
                for event in events:
                    if event["event"] == "stop_to_breakeven":
                        continue
                    qty = float(event["quantity"])
                    if qty <= 0:
                        continue
                    notional = qty * mark
                    fee = _fee(notional, fee_bps)
                    pnl = qty * (mark - position.entry_price) - fee
                    equity += pnl
                    total_fees += fee
                    realized_trade_pnls.append(pnl)
                    exits += 1
                    reason = str(event["event"])
                    if reason.startswith("tp"):
                        tp_events += 1
                    elif reason == "stop_loss":
                        stop_events += 1
                    trade_log.append(
                        {
                            "symbol": symbol,
                            "side": "sell",
                            "timestamp": bars[idx].timestamp.isoformat(),
                            "price": mark,
                            "quantity": qty,
                            "fee": fee,
                            "pnl": pnl,
                            "reason": reason,
                        }
                    )
                if not position.is_open:
                    del positions[symbol]

            if symbol in positions:
                continue

            window = bars[: idx + 1]
            features = compute_strategy_features(window, cfg.strategy)
            if not features.entry_signal:
                continue
            intent_notional = equity * cfg.risk.max_total_exposure_pct / cfg.risk.max_positions
            pseudo_intent = score_symbol(
                compute_features(window),
                [],
                cfg.scoring,
                cfg.llm.min_confidence,
                notional=intent_notional,
            )
            pseudo_intent = dataclass_replace_intent(pseudo_intent, side_value="buy", score=features.signal_strength)
            decision = risk.evaluate_entry(
                pseudo_intent,
                equity,
                positions,
                marks,
                volatility=compute_features(window).volatility,
            )
            if not decision.accepted:
                continue
            entry_fee = _fee(decision.adjusted_notional, fee_bps)
            equity -= entry_fee
            total_fees += entry_fee
            positions[symbol] = risk.build_position(symbol, mark, decision.adjusted_notional)
            entries += 1
            trade_log.append(
                {
                    "symbol": symbol,
                    "side": "buy",
                    "timestamp": bars[idx].timestamp.isoformat(),
                    "price": mark,
                    "quantity": decision.adjusted_notional / mark,
                    "fee": entry_fee,
                    "pnl": -entry_fee,
                    "reason": "breakout_volume_entry",
                }
            )

        current_equity = _open_equity(equity, positions, marks)
        peak = max(peak, current_equity)
        drawdown = (peak - current_equity) / peak if peak else 0.0
        max_drawdown = max(max_drawdown, drawdown)
        if marks:
            timestamp = max(bars[idx].timestamp for bars in bars_by_symbol.values() if idx < len(bars))
            equity_curve.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "equity": current_equity,
                    "drawdown": drawdown,
                    "open_positions": float(len(positions)),
                }
            )

    end_positions: list[dict[str, float | str]] = []
    for symbol, position in list(positions.items()):
        bars = bars_by_symbol[symbol]
        if not bars:
            continue
        mark = bars[-1].close
        end_positions.append(
            {
                "symbol": symbol,
                "entry_price": position.entry_price,
                "mark_price": mark,
                "quantity": position.remaining_quantity,
                "unrealized_pnl": position.remaining_quantity * (mark - position.entry_price),
                "stop_price": position.stop_price,
            }
        )
        qty = position.remaining_quantity
        fee = _fee(qty * mark, fee_bps)
        pnl = qty * (mark - position.entry_price) - fee
        equity += pnl
        total_fees += fee
        realized_trade_pnls.append(pnl)
        exits += 1
        trade_log.append(
            {
                "symbol": symbol,
                "side": "sell",
                "timestamp": bars[-1].timestamp.isoformat(),
                "price": mark,
                "quantity": qty,
                "fee": fee,
                "pnl": pnl,
                "reason": "end_of_backtest",
            }
        )

    wins = [pnl for pnl in realized_trade_pnls if pnl > 0]
    losses = [abs(pnl) for pnl in realized_trade_pnls if pnl < 0]
    profit_factor = sum(wins) / sum(losses) if losses else (999.0 if wins else 0.0)
    return BreakoutBacktestResult(
        run_id=run_id,
        strategy="breakout_volume",
        trades=len(trade_log),
        entries=entries,
        exits=exits,
        final_equity=equity,
        return_pct=equity / starting_equity - 1.0,
        buy_hold_return_pct=_buy_hold_return(bars_by_symbol, fee_bps),
        max_drawdown_pct=max_drawdown,
        win_rate=len(wins) / len(realized_trade_pnls) if realized_trade_pnls else 0.0,
        profit_factor=profit_factor,
        total_fees=total_fees,
        tp_events=tp_events,
        stop_events=stop_events,
        assumptions={
            "fill_model": cfg.costs.default_fill_model,
            "fee_bps_per_fill": fee_bps,
            "long_only": True,
            "timeframe": cfg.data.timeframe,
            "htf_timeframe": cfg.strategy.htf_timeframe,
        },
        trade_log=trade_log,
        equity_curve=equity_curve,
        open_positions=end_positions,
    )


def run_adaptive_breakout_backtest(
    cfg: AppConfig,
    bars_by_symbol: dict[str, list[Bar]],
    *,
    starting_equity: float = 10_000.0,
    run_id: str = "adaptive-breakout",
) -> BreakoutBacktestResult:
    equity = starting_equity
    peak = starting_equity
    max_drawdown = 0.0
    positions: dict[str, AdaptivePosition] = {}
    fee_bps = cfg.costs.default_bps
    trade_log: list[dict[str, float | str]] = []
    equity_curve: list[dict[str, float | str]] = []
    realized_trade_pnls: list[float] = []
    total_fees = 0.0
    entries = 0
    exits = 0
    max_len = max((len(bars) for bars in bars_by_symbol.values()), default=0)
    min_window = max(
        cfg.strategy.breakout_lookback,
        cfg.strategy.volume_lookback,
        cfg.strategy.momentum_lookback,
        cfg.strategy.htf_trend_lookback * 4,
        cfg.strategy.atr_long_window,
    ) + 1

    for idx in range(min_window, max_len):
        marks = {
            symbol: bars[idx].close
            for symbol, bars in bars_by_symbol.items()
            if idx < len(bars)
        }
        for symbol, position in list(positions.items()):
            bars = bars_by_symbol[symbol]
            if idx >= len(bars):
                continue
            mark = bars[idx].close
            features = compute_strategy_features(_strategy_feature_window(bars, idx, cfg.strategy), cfg.strategy)
            trail_stop = mark - cfg.strategy.atr_trail_multiple * features.atr
            position.stop_price = max(position.stop_price, trail_stop)
            if mark <= position.stop_price:
                fee = _fee(position.quantity * mark, fee_bps)
                pnl = position.quantity * (mark - position.entry_price) - fee
                equity += pnl
                total_fees += fee
                realized_trade_pnls.append(pnl)
                exits += 1
                trade_log.append(
                    {
                        "symbol": symbol,
                        "side": "sell",
                        "timestamp": bars[idx].timestamp.isoformat(),
                        "price": mark,
                        "quantity": position.quantity,
                        "fee": fee,
                        "pnl": pnl,
                        "reason": "atr_trailing_stop",
                    }
                )
                del positions[symbol]

        candidates = []
        for symbol, bars in bars_by_symbol.items():
            if idx >= len(bars) or symbol in positions:
                continue
            features = compute_strategy_features(_strategy_feature_window(bars, idx, cfg.strategy), cfg.strategy)
            round_trip_fee_pct = 2.0 * fee_bps / 10_000.0
            if not features.entry_signal:
                continue
            if features.atr_regime <= 1.0:
                continue
            if features.expected_move_pct < round_trip_fee_pct * cfg.strategy.min_expected_move_fee_multiple:
                continue
            candidates.append((features.signal_strength, symbol, features))
        candidates.sort(reverse=True, key=lambda item: item[0])

        for _, symbol, features in candidates[: cfg.strategy.top_n_per_bar]:
            if len(positions) >= cfg.risk.max_positions:
                break
            mark = bars_by_symbol[symbol][idx].close
            stop_distance = max(cfg.strategy.atr_stop_multiple * features.atr, mark * 0.001)
            stop_distance_pct = stop_distance / mark if mark else 0.0
            if stop_distance_pct <= 0:
                continue
            risk_dollars = equity * cfg.risk.risk_per_trade_pct
            notional = min(
                risk_dollars / stop_distance_pct,
                equity * cfg.risk.max_symbol_exposure_pct,
                equity * cfg.risk.max_total_exposure_pct,
            )
            current_exposure = sum(pos.quantity * marks.get(sym, pos.entry_price) for sym, pos in positions.items())
            remaining_exposure = equity * cfg.risk.max_total_exposure_pct - current_exposure
            notional = min(notional, remaining_exposure)
            if notional <= 0:
                continue
            entry_fee = _fee(notional, fee_bps)
            equity -= entry_fee
            total_fees += entry_fee
            quantity = notional / mark
            positions[symbol] = AdaptivePosition(
                symbol=symbol,
                entry_price=mark,
                quantity=quantity,
                stop_price=mark - stop_distance,
                atr=features.atr,
            )
            entries += 1
            trade_log.append(
                {
                    "symbol": symbol,
                    "side": "buy",
                    "timestamp": bars_by_symbol[symbol][idx].timestamp.isoformat(),
                    "price": mark,
                    "quantity": quantity,
                    "fee": entry_fee,
                    "pnl": -entry_fee,
                    "reason": "adaptive_breakout_entry",
                }
            )

        current_equity = _open_adaptive_equity(equity, positions, marks)
        peak = max(peak, current_equity)
        drawdown = (peak - current_equity) / peak if peak else 0.0
        max_drawdown = max(max_drawdown, drawdown)
        if marks:
            timestamp = max(bars[idx].timestamp for bars in bars_by_symbol.values() if idx < len(bars))
            equity_curve.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "equity": current_equity,
                    "drawdown": drawdown,
                    "open_positions": float(len(positions)),
                }
            )

    end_positions: list[dict[str, float | str]] = []
    for symbol, position in list(positions.items()):
        bars = bars_by_symbol[symbol]
        if not bars:
            continue
        mark = bars[-1].close
        end_positions.append(
            {
                "symbol": symbol,
                "entry_price": position.entry_price,
                "mark_price": mark,
                "quantity": position.quantity,
                "unrealized_pnl": position.quantity * (mark - position.entry_price),
                "stop_price": position.stop_price,
            }
        )
        fee = _fee(position.quantity * mark, fee_bps)
        pnl = position.quantity * (mark - position.entry_price) - fee
        equity += pnl
        total_fees += fee
        realized_trade_pnls.append(pnl)
        exits += 1
        trade_log.append(
            {
                "symbol": symbol,
                "side": "sell",
                "timestamp": bars[-1].timestamp.isoformat(),
                "price": mark,
                "quantity": position.quantity,
                "fee": fee,
                "pnl": pnl,
                "reason": "end_of_backtest",
            }
        )

    wins = [pnl for pnl in realized_trade_pnls if pnl > 0]
    losses = [abs(pnl) for pnl in realized_trade_pnls if pnl < 0]
    profit_factor = sum(wins) / sum(losses) if losses else (999.0 if wins else 0.0)
    metrics = {
        "entries": entries,
        "final_equity": equity,
        "return_pct": equity / starting_equity - 1.0,
        "buy_hold_return_pct": _buy_hold_return(bars_by_symbol, fee_bps),
        "max_drawdown_pct": max_drawdown,
        "win_rate": len(wins) / len(realized_trade_pnls) if realized_trade_pnls else 0.0,
        "profit_factor": profit_factor,
        "total_fees": total_fees,
    }
    rank_score = _rank_score(metrics)
    return BreakoutBacktestResult(
        run_id=run_id,
        strategy="adaptive_breakout",
        trades=len(trade_log),
        entries=entries,
        exits=exits,
        final_equity=equity,
        return_pct=metrics["return_pct"],
        buy_hold_return_pct=metrics["buy_hold_return_pct"],
        max_drawdown_pct=max_drawdown,
        win_rate=metrics["win_rate"],
        profit_factor=profit_factor,
        total_fees=total_fees,
        tp_events=0,
        stop_events=exits,
        assumptions={
            "fill_model": cfg.costs.default_fill_model,
            "fee_bps_per_fill": fee_bps,
            "long_only": True,
            "timeframe": cfg.data.timeframe,
            "exit_model": "atr_trailing_stop",
            "sizing_model": "risk_per_trade",
            "rank_score": rank_score,
        },
        trade_log=trade_log,
        equity_curve=equity_curve,
        open_positions=end_positions,
    )


def run_quality_growth_backtest(
    cfg: AppConfig,
    bars_by_symbol: dict[str, list[Bar]],
    *,
    starting_equity: float = 10_000.0,
    run_id: str = "quality-growth",
    store=None,
) -> BreakoutBacktestResult:
    equity = starting_equity
    peak = starting_equity
    max_drawdown = 0.0
    positions: dict[str, QualityPosition] = {}
    fee_bps = cfg.costs.default_bps
    qcfg = cfg.quality_growth
    trade_log: list[dict[str, float | str]] = []
    equity_curve: list[dict[str, float | str]] = []
    realized_trade_pnls: list[float] = []
    total_fees = 0.0
    entries = 0
    exits = 0
    tp_events = 0
    stop_events = 0
    standard_entries = 0
    research_promotion_entries = 0
    max_len = max((len(bars) for bars in bars_by_symbol.values()), default=0)
    min_window = max(
        cfg.strategy.breakout_lookback,
        cfg.strategy.volume_lookback,
        cfg.strategy.momentum_lookback,
        cfg.strategy.htf_trend_lookback * 4,
        cfg.strategy.atr_long_window,
    ) + 1
    if qcfg.require_mtf_alignment:
        min_window = max(min_window, MTF_LOOKBACK_HOURS + 1)
    decision_interval = max(1, int(qcfg.decision_interval_hours or 1))
    promotion_symbols = set(qcfg.research_promotion_symbols or ())

    for idx in range(min_window, max_len):
        marks = {
            symbol: bars[idx].close
            for symbol, bars in bars_by_symbol.items()
            if idx < len(bars)
        }

        for symbol, position in list(positions.items()):
            bars = bars_by_symbol[symbol]
            if idx >= len(bars):
                continue
            mark = bars[idx].close
            features = compute_strategy_features(_strategy_feature_window(bars, idx, cfg.strategy), cfg.strategy)
            position.peak_price = max(position.peak_price, mark)
            trail_stop = position.peak_price - qcfg.atr_trail_multiple * features.atr
            position.stop_price = max(position.stop_price, trail_stop)

            for tp_idx, level in enumerate(cfg.risk.take_profit_levels):
                if tp_idx in position.filled_tp_indices:
                    continue
                if mark < position.entry_price * (1.0 + level.gain_pct):
                    continue
                qty = min(position.remaining_quantity, position.initial_quantity * level.sell_fraction)
                if qty <= 0:
                    continue
                fee = _fee(qty * mark, fee_bps)
                pnl = qty * (mark - position.entry_price) - fee
                equity += pnl
                total_fees += fee
                realized_trade_pnls.append(pnl)
                position.remaining_quantity -= qty
                position.filled_tp_indices.add(tp_idx)
                exits += 1
                tp_events += 1
                trade_log.append(
                    {
                        "symbol": symbol,
                        "side": "sell",
                        "timestamp": bars[idx].timestamp.isoformat(),
                        "price": mark,
                        "quantity": qty,
                        "fee": fee,
                        "pnl": pnl,
                        "reason": f"quality_tp_{level.gain_pct:.0%}",
                    }
                )

            if position.is_open and mark <= position.stop_price:
                qty = position.remaining_quantity
                fee = _fee(qty * mark, fee_bps)
                pnl = qty * (mark - position.entry_price) - fee
                equity += pnl
                total_fees += fee
                realized_trade_pnls.append(pnl)
                exits += 1
                stop_events += 1
                trade_log.append(
                    {
                        "symbol": symbol,
                        "side": "sell",
                        "timestamp": bars[idx].timestamp.isoformat(),
                        "price": mark,
                        "quantity": qty,
                        "fee": fee,
                        "pnl": pnl,
                        "reason": "quality_atr_trailing_stop",
                    }
                )
                del positions[symbol]
            elif not position.is_open:
                del positions[symbol]

        candidates = []
        for symbol, bars in bars_by_symbol.items():
            if idx >= len(bars) or symbol in positions or not cfg.is_trade_enabled(symbol):
                continue
            if decision_interval > 1:
                bar_hour = int(bars[idx].timestamp.timestamp() // 3600)
                if bar_hour % decision_interval != 0:
                    continue
            profile = cfg.project_profiles.get(symbol)
            if not profile:
                continue
            features = compute_strategy_features(
                _strategy_feature_window(
                    bars,
                    idx,
                    cfg.strategy,
                    extra_lookback_hours=MTF_LOOKBACK_HOURS if qcfg.require_mtf_alignment else 0,
                ),
                cfg.strategy,
            )
            score = quality_growth_score(profile, features, qcfg)
            round_trip_fee_pct = 2.0 * fee_bps / 10_000.0
            expected_move_ok = (
                features.expected_move_pct
                >= round_trip_fee_pct * qcfg.min_expected_move_fee_multiple
            )
            # Phase C: optional lookahead-free fund-style scoring overrides score.passes
            # when backtest_use_fund_style is enabled and a store is supplied.
            if qcfg.backtest_use_fund_style and store is not None and expected_move_ok:
                from datetime import timedelta as _td

                from .defillama import extract_defi_metrics_at
                from .research.narratives import aggregate_symbol_narrative
                from .quality_growth import fund_style_quality_score
                from .sentiment_aggregator import build_sentiment_vector

                asof = bars[idx].timestamp
                # Point-in-time DefiLlama
                try:
                    defi_metrics_at = extract_defi_metrics_at(symbol, asof, store=store)
                except Exception:
                    defi_metrics_at = {}
                # Point-in-time narrative — only news published <= asof
                try:
                    all_news = store.list_news_for_symbol(symbol, limit=400)
                    news_window_start = asof - _td(days=qcfg.backtest_news_window_days)
                    news_at = [n for n in all_news if news_window_start <= (n.published_at if n.published_at.tzinfo else n.published_at.replace(tzinfo=timezone.utc)) <= asof]
                    narrative = aggregate_symbol_narrative(symbol, news_at, asof=asof)
                except Exception:
                    narrative = None
                # Point-in-time signals
                try:
                    raw_signals = store.latest_signals(symbol, limit=200)
                    signals_at = [s for s in raw_signals if (s.created_at if s.created_at.tzinfo else s.created_at.replace(tzinfo=timezone.utc)) <= asof]
                    vector = build_sentiment_vector(signals_at, asof=asof, min_confidence=cfg.llm.min_confidence)
                    signals_7d = sum(1 for s in signals_at if (asof - (s.created_at if s.created_at.tzinfo else s.created_at.replace(tzinfo=timezone.utc))).total_seconds() <= 86_400 * 7)
                except Exception:
                    vector = build_sentiment_vector([], asof=asof)
                    signals_7d = 0
                fs = fund_style_quality_score(
                    profile, features, qcfg, vector,
                    signal_count_7d=signals_7d,
                    volume_usd=bars[idx].close * bars[idx].volume,
                    defi_metrics=defi_metrics_at,
                    narrative=narrative,
                )
                if score.passes:
                    # Replace score with fund-style tradable when above floor.
                    if fs.tradable_score <= 0.0:
                        continue  # blocked by fund-style gate
                    # Use tradable_score as the entry priority instead of raw quality_growth_score.score
                    candidates.append(
                        (fs.tradable_score, symbol, features, score, 1.0, "quality_growth_entry")
                    )
                elif (
                    qcfg.backtest_allow_research_promotions
                    and (not promotion_symbols or symbol in promotion_symbols)
                    and fs.tradable_score >= qcfg.research_promotion_min_score
                ):
                    priority = fs.tradable_score * qcfg.research_promotion_size_scale
                    candidates.append(
                        (
                            priority,
                            symbol,
                            features,
                            score,
                            qcfg.research_promotion_size_scale,
                            "quality_research_promotion",
                        )
                    )
            elif score.passes and expected_move_ok:
                candidates.append((score.score, symbol, features, score, 1.0, "quality_growth_entry"))
        candidates.sort(reverse=True, key=lambda item: item[0])

        for priority, symbol, features, score, size_scale, entry_reason in candidates[: qcfg.top_n_per_bar]:
            if len(positions) >= cfg.risk.max_positions:
                break
            mark = bars_by_symbol[symbol][idx].close
            stop_distance = max(qcfg.atr_stop_multiple * features.atr, mark * 0.002)
            stop_distance_pct = stop_distance / mark if mark else 0.0
            if stop_distance_pct <= 0:
                continue
            risk_dollars = (
                equity
                * cfg.risk.risk_per_trade_pct
                * (0.75 + score.profile_score * 0.50)
                * size_scale
            )
            notional = min(
                risk_dollars / stop_distance_pct,
                equity * cfg.risk.max_symbol_exposure_pct,
                equity * cfg.risk.max_total_exposure_pct,
            )
            current_exposure = sum(
                pos.remaining_quantity * marks.get(sym, pos.entry_price)
                for sym, pos in positions.items()
            )
            remaining_exposure = equity * cfg.risk.max_total_exposure_pct - current_exposure
            notional = min(notional, remaining_exposure)
            if notional <= 0:
                continue
            entry_fee = _fee(notional, fee_bps)
            equity -= entry_fee
            total_fees += entry_fee
            quantity = notional / mark
            positions[symbol] = QualityPosition(
                symbol=symbol,
                entry_price=mark,
                initial_quantity=quantity,
                remaining_quantity=quantity,
                stop_price=mark - stop_distance,
                peak_price=mark,
                profile_score=score.profile_score,
            )
            entries += 1
            if entry_reason == "quality_research_promotion":
                research_promotion_entries += 1
            else:
                standard_entries += 1
            trade_log.append(
                {
                    "symbol": symbol,
                    "side": "buy",
                    "timestamp": bars_by_symbol[symbol][idx].timestamp.isoformat(),
                    "price": mark,
                    "quantity": quantity,
                    "fee": entry_fee,
                    "pnl": -entry_fee,
                    "reason": (
                        f"{entry_reason} "
                        f"priority={priority:.3f} score={score.score:.3f} "
                        f"profile={score.profile_score:.3f} size_scale={size_scale:.2f} "
                        f"mom={features.momentum:.2%} rvol={features.relative_volume:.2f}"
                    ),
                }
            )

        current_equity = _open_quality_equity(equity, positions, marks)
        peak = max(peak, current_equity)
        drawdown = (peak - current_equity) / peak if peak else 0.0
        max_drawdown = max(max_drawdown, drawdown)
        if marks:
            timestamp = max(bars[idx].timestamp for bars in bars_by_symbol.values() if idx < len(bars))
            equity_curve.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "equity": current_equity,
                    "drawdown": drawdown,
                    "open_positions": float(len(positions)),
                }
            )

    end_positions: list[dict[str, float | str]] = []
    for symbol, position in list(positions.items()):
        bars = bars_by_symbol[symbol]
        if not bars:
            continue
        mark = bars[-1].close
        end_positions.append(
            {
                "symbol": symbol,
                "entry_price": position.entry_price,
                "mark_price": mark,
                "quantity": position.remaining_quantity,
                "unrealized_pnl": position.remaining_quantity * (mark - position.entry_price),
                "stop_price": position.stop_price,
                "profile_score": position.profile_score,
            }
        )
        qty = position.remaining_quantity
        fee = _fee(qty * mark, fee_bps)
        pnl = qty * (mark - position.entry_price) - fee
        equity += pnl
        total_fees += fee
        realized_trade_pnls.append(pnl)
        exits += 1
        trade_log.append(
            {
                "symbol": symbol,
                "side": "sell",
                "timestamp": bars[-1].timestamp.isoformat(),
                "price": mark,
                "quantity": qty,
                "fee": fee,
                "pnl": pnl,
                "reason": "end_of_backtest",
            }
        )

    wins = [pnl for pnl in realized_trade_pnls if pnl > 0]
    losses = [abs(pnl) for pnl in realized_trade_pnls if pnl < 0]
    profit_factor = sum(wins) / sum(losses) if losses else (999.0 if wins else 0.0)
    metrics = {
        "entries": entries,
        "final_equity": equity,
        "return_pct": equity / starting_equity - 1.0,
        "buy_hold_return_pct": _buy_hold_return(bars_by_symbol, fee_bps),
        "max_drawdown_pct": max_drawdown,
        "win_rate": len(wins) / len(realized_trade_pnls) if realized_trade_pnls else 0.0,
        "profit_factor": profit_factor,
        "total_fees": total_fees,
    }
    rank_score = _rank_score(metrics)
    return BreakoutBacktestResult(
        run_id=run_id,
        strategy="quality_growth",
        trades=len(trade_log),
        entries=entries,
        exits=exits,
        final_equity=equity,
        return_pct=metrics["return_pct"],
        buy_hold_return_pct=metrics["buy_hold_return_pct"],
        max_drawdown_pct=max_drawdown,
        win_rate=metrics["win_rate"],
        profit_factor=profit_factor,
        total_fees=total_fees,
        tp_events=tp_events,
        stop_events=stop_events,
        assumptions={
            "fill_model": cfg.costs.default_fill_model,
            "fee_bps_per_fill": fee_bps,
            "long_only": True,
            "timeframe": cfg.data.timeframe,
            "exit_model": "wide_atr_trail_plus_moonbag_tps",
            "sizing_model": "risk_per_trade_scaled_by_profile",
            "profile_model": "quality_growth_backer_liquidity_risk",
            "decision_interval_hours": decision_interval,
            "multi_timeframe_confirmation": qcfg.require_mtf_alignment,
            "min_mtf_trend_score": qcfg.min_mtf_trend_score,
            "backtest_use_fund_style": qcfg.backtest_use_fund_style,
            "backtest_news_window_days": qcfg.backtest_news_window_days,
            "backtest_allow_research_promotions": qcfg.backtest_allow_research_promotions,
            "research_promotion_min_score": qcfg.research_promotion_min_score,
            "research_promotion_size_scale": qcfg.research_promotion_size_scale,
            "research_promotion_symbols": tuple(qcfg.research_promotion_symbols),
            "standard_entries": standard_entries,
            "research_promotion_entries": research_promotion_entries,
            "rank_score": rank_score,
        },
        trade_log=trade_log,
        equity_curve=equity_curve,
        open_positions=end_positions,
        per_symbol=_per_symbol_attribution(trade_log),
    )


def focused_sweep_configs(cfg: AppConfig) -> list[AppConfig]:
    from dataclasses import replace

    configs: list[AppConfig] = []
    for breakout, rel_vol, stop_mult, trail_mult, top_n in product(
        (20, 40),
        (1.1, 1.25, 1.5),
        (1.5, 2.0, 2.5),
        (2.0, 2.5, 3.0),
        (1, 3, 5),
    ):
        configs.append(
            replace(
                cfg,
                strategy=replace(
                    cfg.strategy,
                    breakout_lookback=breakout,
                    relative_volume_min=rel_vol,
                    atr_stop_multiple=stop_mult,
                    atr_trail_multiple=trail_mult,
                    top_n_per_bar=top_n,
                ),
            )
        )
    return configs


def dataclass_replace_intent(intent, *, side_value: str, score: float):
    from dataclasses import replace

    from .schemas import Side

    return replace(
        intent,
        side=Side(side_value),
        score=score,
        confidence=max(0.0, min(1.0, score)),
        reason="breakout_volume_signal",
    )
