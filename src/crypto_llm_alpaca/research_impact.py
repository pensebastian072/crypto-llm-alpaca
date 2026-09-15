from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .backtest import MTF_LOOKBACK_HOURS, _strategy_feature_window
from .config import AppConfig
from .features import compute_strategy_features
from .quality_growth import fund_style_quality_score, quality_growth_score
from .research.narratives import aggregate_symbol_narrative
from .schemas import Bar, LlmSignal, NewsItem
from .sentiment_aggregator import build_sentiment_vector
from .storage import SQLiteStore


@dataclass
class _SymbolImpact:
    evaluated: int = 0
    eligible: int = 0
    fund_veto: int = 0
    fund_promotion: int = 0
    top_base: int = 0
    top_fund: int = 0
    helped_rank: int = 0
    hurt_rank: int = 0
    base_score_sum: float = 0.0
    research_score_sum: float = 0.0
    tradable_score_sum: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        denom = self.evaluated or 1
        return {
            "evaluated": self.evaluated,
            "eligible": self.eligible,
            "fund_veto": self.fund_veto,
            "fund_promotion": self.fund_promotion,
            "top_base": self.top_base,
            "top_fund": self.top_fund,
            "helped_rank": self.helped_rank,
            "hurt_rank": self.hurt_rank,
            "avg_base_score": self.base_score_sum / denom,
            "avg_research_score": self.research_score_sum / denom,
            "avg_tradable_score": self.tradable_score_sum / denom,
        }


@dataclass(frozen=True)
class _SnapshotIndex:
    asofs: list[datetime] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)

    def metrics_at(self, asof: datetime) -> dict[str, Any]:
        idx = bisect_right(self.asofs, asof) - 1
        if idx < 0:
            return {}
        row = self.rows[idx]
        return {
            key: row[key]
            for key in (
                "tvl",
                "tvl_growth",
                "tvl_growth_30d",
                "fees_7d",
                "fee_growth",
                "revenue_growth",
            )
            if row.get(key) is not None
        }


def run_research_impact(
    cfg: AppConfig,
    store: SQLiteStore,
    *,
    bars_by_symbol: dict[str, list[Bar]] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, Any]:
    """Measure whether fund-style research changes historical candidate selection.

    This is diagnostic only. It mirrors quality-growth backtest inputs but does
    not change entries, exits, sizing, or persisted backtest results.
    """

    qcfg = cfg.quality_growth
    if bars_by_symbol is None:
        bars_by_symbol = {
            symbol: store.list_bars(symbol, timeframe=cfg.data.timeframe)
            for symbol in cfg.universe.symbols
        }

    symbols = [
        symbol
        for symbol in cfg.universe.symbols
        if cfg.is_trade_enabled(symbol)
        and cfg.project_profiles.get(symbol)
        and bars_by_symbol.get(symbol)
    ]
    if not symbols:
        return _empty_result(cfg, start, end, reason="no_profiled_symbols_with_bars")

    news_by_symbol = {
        symbol: store.list_news_for_symbol(symbol, limit=1000)
        for symbol in symbols
    }
    signals_by_symbol = {
        symbol: store.latest_signals(symbol, limit=1000)
        for symbol in symbols
    }
    snapshots_by_symbol = {
        symbol: _build_snapshot_index(store.list_defi_snapshots(symbol=symbol, limit=20_000))
        for symbol in symbols
    }

    max_len = max(len(bars_by_symbol[symbol]) for symbol in symbols)
    decision_interval = max(1, int(qcfg.decision_interval_hours))
    fee_bps = cfg.costs.default_bps
    round_trip_fee_pct = 2.0 * fee_bps / 10_000.0

    decision_bars = 0
    eligible_bars = 0
    eligible_candidates = 0
    rank_changed_bars = 0
    fund_style_vetoes = 0
    fund_style_promotions = 0
    top_symbol_changes: Counter[str] = Counter()
    missing_data: Counter[str] = Counter()
    impacts: dict[str, _SymbolImpact] = {symbol: _SymbolImpact() for symbol in symbols}

    for idx in range(max_len):
        evaluated_this_bar: list[dict[str, Any]] = []
        eligible_this_bar: list[dict[str, Any]] = []

        for symbol in symbols:
            bars = bars_by_symbol[symbol]
            if idx >= len(bars):
                continue
            asof = _as_utc(bars[idx].timestamp)
            if start and asof < start:
                continue
            if end and asof > end:
                continue
            if int(asof.timestamp() // 3600) % decision_interval != 0:
                continue

            profile = cfg.project_profiles[symbol]
            try:
                features = compute_strategy_features(
                    _strategy_feature_window(
                        bars,
                        idx,
                        cfg.strategy,
                        extra_lookback_hours=MTF_LOOKBACK_HOURS,
                    ),
                    cfg.strategy,
                )
            except ValueError:
                continue

            signal_rows = _signals_at(signals_by_symbol[symbol], asof)
            vector = build_sentiment_vector(
                signal_rows,
                asof=asof,
                min_confidence=cfg.llm.min_confidence,
            )
            score = quality_growth_score(profile, features, qcfg, vector)
            expected_move_ok = (
                features.expected_move_pct
                >= round_trip_fee_pct * qcfg.min_expected_move_fee_multiple
            )
            news_rows = _news_at(
                news_by_symbol[symbol],
                asof,
                days=qcfg.backtest_news_window_days,
            )
            narrative = aggregate_symbol_narrative(
                symbol,
                news_rows,
                asof=asof,
                current_days=qcfg.narrative_current_days,
                previous_days=qcfg.narrative_previous_days,
            )
            signal_count_7d = sum(
                1 for signal in signal_rows
                if (asof - _as_utc(signal.created_at)) <= timedelta(days=7)
            )
            fund_style = fund_style_quality_score(
                profile,
                features,
                qcfg,
                vector,
                signal_count_7d=signal_count_7d,
                volume_usd=bars[idx].close * bars[idx].volume,
                defi_metrics=snapshots_by_symbol[symbol].metrics_at(asof),
                narrative=narrative,
            )

            technical_eligible = score.passes and expected_move_ok
            fund_tradable = fund_style.tradable_score > 0.0 and expected_move_ok
            impact = impacts[symbol]
            impact.evaluated += 1
            impact.base_score_sum += score.score
            impact.research_score_sum += fund_style.research_score
            impact.tradable_score_sum += fund_style.tradable_score
            missing_data.update(fund_style.missing_data)

            row = {
                "symbol": symbol,
                "base_score": score.score,
                "research_score": fund_style.research_score,
                "tradable_score": fund_style.tradable_score,
                "technical_eligible": technical_eligible,
                "fund_tradable": fund_tradable,
            }
            evaluated_this_bar.append(row)

            if technical_eligible:
                eligible_candidates += 1
                impact.eligible += 1
                eligible_this_bar.append(row)
                if fund_style.tradable_score <= 0.0:
                    fund_style_vetoes += 1
                    impact.fund_veto += 1
            elif fund_tradable:
                fund_style_promotions += 1
                impact.fund_promotion += 1

        if not evaluated_this_bar:
            continue
        decision_bars += 1
        if not eligible_this_bar:
            continue
        eligible_bars += 1

        base_order = sorted(
            eligible_this_bar,
            key=lambda item: (-item["base_score"], item["symbol"]),
        )
        fund_order = sorted(
            eligible_this_bar,
            key=lambda item: (
                -item["tradable_score"],
                -item["research_score"],
                item["symbol"],
            ),
        )
        base_symbols = [item["symbol"] for item in base_order]
        fund_symbols = [item["symbol"] for item in fund_order]
        impacts[base_symbols[0]].top_base += 1
        impacts[fund_symbols[0]].top_fund += 1
        if base_symbols != fund_symbols:
            rank_changed_bars += 1
        if base_symbols[0] != fund_symbols[0]:
            top_symbol_changes[f"{base_symbols[0]}->{fund_symbols[0]}"] += 1

        base_rank = {symbol: rank for rank, symbol in enumerate(base_symbols)}
        fund_rank = {symbol: rank for rank, symbol in enumerate(fund_symbols)}
        for symbol in set(base_rank) & set(fund_rank):
            if fund_rank[symbol] < base_rank[symbol]:
                impacts[symbol].helped_rank += 1
            elif fund_rank[symbol] > base_rank[symbol]:
                impacts[symbol].hurt_rank += 1

    rank_changed_pct = rank_changed_bars / eligible_bars if eligible_bars else 0.0
    result = {
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "timeframe": cfg.data.timeframe,
        "decision_interval_hours": decision_interval,
        "decision_bars": decision_bars,
        "eligible_bars": eligible_bars,
        "eligible_candidates": eligible_candidates,
        "rank_changed_bars": rank_changed_bars,
        "rank_changed_pct": rank_changed_pct,
        "fund_style_vetoes": fund_style_vetoes,
        "fund_style_promotions": fund_style_promotions,
        "top_symbol_changes": dict(top_symbol_changes.most_common()),
        "missing_data": dict(missing_data.most_common()),
        "symbol_impacts": {
            symbol: impact.as_dict()
            for symbol, impact in impacts.items()
            if impact.evaluated
        },
        "recommendation": _recommendation(
            eligible_candidates=eligible_candidates,
            rank_changed_bars=rank_changed_bars,
            fund_style_vetoes=fund_style_vetoes,
            fund_style_promotions=fund_style_promotions,
        ),
    }
    return result


def format_research_impact(result: dict[str, Any]) -> str:
    lines = [
        "# Research impact audit",
        f"timeframe: {result['timeframe']}",
        f"window: {result.get('start') or 'first bar'} -> {result.get('end') or 'last bar'}",
        "",
        f"decision bars: {result['decision_bars']}",
        f"eligible bars: {result['eligible_bars']}",
        f"eligible candidates: {result['eligible_candidates']}",
        f"rank changed bars: {result['rank_changed_bars']} ({result['rank_changed_pct']:.1%})",
        f"fund-style vetoes: {result['fund_style_vetoes']}",
        f"fund-style promotions: {result['fund_style_promotions']}",
        f"recommendation: {result['recommendation']}",
        "",
    ]
    changes = result.get("top_symbol_changes") or {}
    if changes:
        lines.append("Top symbol changes:")
        for label, count in list(changes.items())[:10]:
            lines.append(f"- {label}: {count}")
        lines.append("")

    impacts = result.get("symbol_impacts") or {}
    if impacts:
        lines.append("Symbol impact:")
        lines.append("  symbol      eval  elig  veto  prom  top_base  top_fund  avg_base  avg_research  avg_tradable")
        for symbol, row in sorted(
            impacts.items(),
            key=lambda item: (
                item[1].get("top_fund", 0),
                item[1].get("eligible", 0),
                item[1].get("avg_research_score", 0.0),
            ),
            reverse=True,
        ):
            lines.append(
                f"  {symbol.ljust(10)} "
                f"{int(row['evaluated']):>5} "
                f"{int(row['eligible']):>5} "
                f"{int(row['fund_veto']):>5} "
                f"{int(row['fund_promotion']):>5} "
                f"{int(row['top_base']):>9} "
                f"{int(row['top_fund']):>8} "
                f"{row['avg_base_score']:.3f}     "
                f"{row['avg_research_score']:.3f}         "
                f"{row['avg_tradable_score']:.3f}"
            )
        lines.append("")

    missing = result.get("missing_data") or {}
    if missing:
        lines.append("Missing data observations:")
        for label, count in list(missing.items())[:10]:
            lines.append(f"- {label}: {count}")
    return "\n".join(lines)


def _empty_result(
    cfg: AppConfig,
    start: datetime | None,
    end: datetime | None,
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "timeframe": cfg.data.timeframe,
        "decision_interval_hours": max(1, int(cfg.quality_growth.decision_interval_hours)),
        "decision_bars": 0,
        "eligible_bars": 0,
        "eligible_candidates": 0,
        "rank_changed_bars": 0,
        "rank_changed_pct": 0.0,
        "fund_style_vetoes": 0,
        "fund_style_promotions": 0,
        "top_symbol_changes": {},
        "missing_data": {},
        "symbol_impacts": {},
        "recommendation": reason,
    }


def _build_snapshot_index(rows: list[dict[str, Any]]) -> _SnapshotIndex:
    asofs: list[datetime] = []
    kept_rows: list[dict[str, Any]] = []
    for row in rows:
        raw = row.get("asof")
        if raw is None:
            continue
        asofs.append(_as_utc(datetime.fromisoformat(str(raw))))
        kept_rows.append(row)
    order = sorted(range(len(asofs)), key=lambda idx: asofs[idx])
    return _SnapshotIndex(
        asofs=[asofs[idx] for idx in order],
        rows=[kept_rows[idx] for idx in order],
    )


def _signals_at(signals: list[LlmSignal], asof: datetime) -> list[LlmSignal]:
    return [
        signal
        for signal in signals
        if _as_utc(signal.created_at) <= asof
    ]


def _news_at(items: list[NewsItem], asof: datetime, *, days: int) -> list[NewsItem]:
    start = asof - timedelta(days=max(days, 1))
    return [
        item
        for item in items
        if start <= _as_utc(item.published_at) <= asof
    ]


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _recommendation(
    *,
    eligible_candidates: int,
    rank_changed_bars: int,
    fund_style_vetoes: int,
    fund_style_promotions: int,
) -> str:
    if eligible_candidates == 0:
        return "no_technical_eligible_candidates"
    if rank_changed_bars == 0 and fund_style_vetoes == 0 and fund_style_promotions == 0:
        return "research_layer_neutral_current_gates"
    if fund_style_promotions > fund_style_vetoes:
        return "test_research_promotions_with_small_probe_size"
    return "test_research_ranked_entries_or_min_research_score"
