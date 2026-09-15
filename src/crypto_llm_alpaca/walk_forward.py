"""Walk-forward harness for autopilot_backtest.

Splits the bar timeline into rolling (train, test) windows. For each window:
  - Run a small param sweep on the train slice.
  - Pick the best params by rank_score.
  - Apply those params to the test slice.
Aggregates per-window results so you can see out-of-sample stability.

Usage:
    crypto-llm walk-forward --start 2024-05-01 --end 2026-05-01 \
        --train-days 90 --test-days 30 --step-days 30 --persist
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import product
from typing import Optional
from uuid import uuid4

from .autopilot_backtest import AutopilotBacktestResult, run_autopilot_backtest
from .config import AppConfig
from .schemas import Bar


@dataclass
class WalkForwardWindow:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: dict
    train_result_summary: dict
    test_result: dict


@dataclass
class WalkForwardResult:
    run_id: str
    train_days: int
    test_days: int
    step_days: int
    windows: list[WalkForwardWindow] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)


_DEFAULT_GRID: dict[str, list] = {
    "max_trades_per_day": [2, 3],
    "take_profit_pct": [0.10, 0.15],
    "stop_loss_pct": [0.03, 0.05],
}


def _slice_bars(bars_by_symbol: dict[str, list[Bar]], start: datetime, end: datetime) -> dict[str, list[Bar]]:
    return {
        sym: [b for b in bars if start <= b.timestamp <= end]
        for sym, bars in bars_by_symbol.items()
    }


def _summary(r: AutopilotBacktestResult) -> dict:
    return {
        "run_id": r.run_id, "trades": r.trades, "entries": r.entries,
        "return_pct": r.return_pct, "buy_hold_return_pct": r.buy_hold_return_pct,
        "max_drawdown_pct": r.max_drawdown_pct, "win_rate": r.win_rate,
        "profit_factor": r.profit_factor, "tp_events": r.tp_events, "stop_events": r.stop_events,
        "rank_score": r.assumptions.get("rank_score", 0.0),
        "per_symbol": r.per_symbol,
    }


def _aggregate(windows: list[WalkForwardWindow]) -> dict:
    if not windows:
        return {}
    rets = [w.test_result["return_pct"] for w in windows]
    dds = [w.test_result["max_drawdown_pct"] for w in windows]
    wrs = [w.test_result["win_rate"] for w in windows]
    pfs = [w.test_result["profit_factor"] for w in windows if w.test_result["profit_factor"] != 999.0]
    return {
        "n_windows": len(windows),
        "mean_return": sum(rets) / len(rets),
        "best_return": max(rets),
        "worst_return": min(rets),
        "mean_max_dd": sum(dds) / len(dds),
        "worst_max_dd": max(dds),
        "mean_win_rate": sum(wrs) / len(wrs),
        "mean_pf": (sum(pfs) / len(pfs)) if pfs else 0.0,
        "n_positive": sum(1 for r in rets if r > 0),
    }


def run_walk_forward(
    cfg: AppConfig,
    bars_by_symbol: dict[str, list[Bar]],
    *,
    start: datetime,
    end: datetime,
    train_days: int = 90,
    test_days: int = 30,
    step_days: int = 30,
    grid: Optional[dict[str, list]] = None,
    starting_equity: float = 10_000.0,
    max_positions: int = 5,
) -> WalkForwardResult:
    grid = grid or _DEFAULT_GRID
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    run_id = f"wf-{uuid4().hex[:12]}"
    windows: list[WalkForwardWindow] = []

    cursor = start
    while True:
        train_start = cursor
        train_end = cursor + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)
        if test_end > end:
            break

        train_bars = _slice_bars(bars_by_symbol, train_start, train_end)
        test_bars = _slice_bars(bars_by_symbol, test_start, test_end)
        if any(len(v) < 25 for v in train_bars.values()) or any(len(v) < 5 for v in test_bars.values()):
            cursor += timedelta(days=step_days)
            continue

        # Train sweep
        best_params: dict = {}
        best_rank = float("-inf")
        best_train_summary: dict = {}
        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                tr = run_autopilot_backtest(
                    cfg, train_bars,
                    run_id=f"{run_id}-train-{train_start.date()}-{combo}",
                    starting_equity=starting_equity,
                    max_positions=max_positions, **params,
                )
            except Exception:
                continue
            rank = tr.return_pct - tr.max_drawdown_pct
            if rank > best_rank:
                best_rank = rank
                best_params = params
                best_train_summary = _summary(tr)

        if not best_params:
            cursor += timedelta(days=step_days)
            continue

        # Test apply
        try:
            te = run_autopilot_backtest(
                cfg, test_bars,
                run_id=f"{run_id}-test-{test_start.date()}",
                starting_equity=starting_equity,
                max_positions=max_positions, **best_params,
            )
        except Exception as exc:
            cursor += timedelta(days=step_days)
            continue

        windows.append(WalkForwardWindow(
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            best_params=best_params,
            train_result_summary=best_train_summary,
            test_result=_summary(te),
        ))
        cursor += timedelta(days=step_days)

    return WalkForwardResult(
        run_id=run_id, train_days=train_days, test_days=test_days, step_days=step_days,
        windows=windows, aggregate=_aggregate(windows),
    )


def render_walk_forward_markdown(result: WalkForwardResult, cfg: AppConfig) -> str:
    lines: list[str] = []
    lines.append(f"# Walk-forward — {result.run_id}")
    lines.append("")
    lines.append(f"**Train days:** {result.train_days}  ")
    lines.append(f"**Test days:** {result.test_days}  ")
    lines.append(f"**Step days:** {result.step_days}  ")
    lines.append(f"**Windows:** {len(result.windows)}  ")
    if result.aggregate:
        a = result.aggregate
        lines.append("")
        lines.append("## Aggregate (out-of-sample test slices)")
        lines.append("")
        lines.append(f"- mean return: **{a['mean_return']*100:+.2f}%**")
        lines.append(f"- best/worst: {a['best_return']*100:+.2f}% / {a['worst_return']*100:+.2f}%")
        lines.append(f"- mean max DD: {a['mean_max_dd']*100:.2f}%  (worst {a['worst_max_dd']*100:.2f}%)")
        lines.append(f"- mean win rate: {a['mean_win_rate']*100:.1f}%")
        lines.append(f"- mean PF: {a['mean_pf']:.2f}")
        lines.append(f"- positive windows: {a['n_positive']}/{a['n_windows']}")
    lines.append("")
    lines.append("## Per-window")
    lines.append("")
    lines.append("| test window | best params | test return | max DD | win% | PF | entries |")
    lines.append("|---|---|---|---|---|---|---|")
    for w in result.windows:
        params = ", ".join(f"{k}={v}" for k, v in w.best_params.items())
        t = w.test_result
        lines.append(
            f"| {w.test_start.date()}–{w.test_end.date()} | {params} "
            f"| {t['return_pct']*100:+.2f}% | {t['max_drawdown_pct']*100:.2f}% "
            f"| {t['win_rate']*100:.1f}% | {t['profit_factor']:.2f} | {t['entries']} |"
        )
    return "\n".join(lines)
