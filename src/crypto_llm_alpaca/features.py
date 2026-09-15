from __future__ import annotations

import math
from statistics import mean, pstdev

from .config import StrategyConfig
from .schemas import Bar, StrategyFeatures, TechnicalFeatures


def _returns(closes: list[float]) -> list[float]:
    return [(b / a) - 1.0 for a, b in zip(closes, closes[1:], strict=False) if a]


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains: list[float] = []
    losses: list[float] = []
    for prev, cur in zip(closes[-period - 1 : -1], closes[-period:], strict=True):
        delta = cur - prev
        gains.append(max(delta, 0.0))
        losses.append(abs(min(delta, 0.0)))
    avg_gain = mean(gains) if gains else 0.0
    avg_loss = mean(losses) if losses else 0.0
    if avg_loss == 0:
        return 100.0 if avg_gain else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _trend_slope(closes: list[float], window: int = 24) -> float:
    sample = closes[-window:]
    if len(sample) < 2 or sample[0] == 0:
        return 0.0
    return (sample[-1] - sample[0]) / sample[0]


def compute_features(bars: list[Bar]) -> TechnicalFeatures:
    if not bars:
        raise ValueError("at least one bar is required")
    closes = [bar.close for bar in bars]
    volumes = [bar.volume for bar in bars]
    rets = _returns(closes[-25:])
    momentum = (closes[-1] / closes[-24] - 1.0) if len(closes) >= 24 and closes[-24] else 0.0
    volatility = pstdev(rets) * math.sqrt(24.0) if len(rets) > 1 else 0.0
    if len(volumes) >= 24 and mean(volumes[-24:-1]) > 0:
        volume_change = volumes[-1] / mean(volumes[-24:-1]) - 1.0
    else:
        volume_change = 0.0
    return TechnicalFeatures(
        symbol=bars[-1].symbol,
        timestamp=bars[-1].timestamp,
        momentum=momentum,
        rsi=_rsi(closes),
        volatility=volatility,
        volume_change=volume_change,
        trend_slope=_trend_slope(closes),
    )


def normalize_momentum(value: float) -> float:
    return max(-1.0, min(1.0, value / 0.08))


def normalize_rsi(rsi: float) -> float:
    return max(-1.0, min(1.0, (rsi - 50.0) / 50.0))


def _avg(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _true_ranges(bars: list[Bar]) -> list[float]:
    if not bars:
        return []
    ranges: list[float] = []
    previous_close = bars[0].close
    for bar in bars:
        ranges.append(max(bar.high - bar.low, abs(bar.high - previous_close), abs(bar.low - previous_close)))
        previous_close = bar.close
    return ranges


def atr(bars: list[Bar], window: int = 14) -> float:
    ranges = _true_ranges(bars)
    if not ranges:
        return 0.0
    return _avg(ranges[-window:])


def _chunk_bars(bars: list[Bar], chunk_size: int) -> list[Bar]:
    chunks: list[Bar] = []
    for start in range(0, len(bars), chunk_size):
        chunk = bars[start : start + chunk_size]
        if len(chunk) < chunk_size:
            continue
        chunks.append(
            Bar(
                symbol=chunk[-1].symbol,
                timestamp=chunk[-1].timestamp,
                open=chunk[0].open,
                high=max(bar.high for bar in chunk),
                low=min(bar.low for bar in chunk),
                close=chunk[-1].close,
                volume=sum(bar.volume for bar in chunk),
                timeframe=f"{chunk_size}Hour",
                source=chunk[-1].source,
            )
        )
    return chunks


def higher_timeframe_trend(bars: list[Bar], lookback: int = 6, source_chunk_hours: int = 4) -> float:
    htf_bars = _chunk_bars(bars, source_chunk_hours)
    if len(htf_bars) < lookback + 1:
        return 0.0
    start = htf_bars[-lookback - 1].close
    if start == 0:
        return 0.0
    return htf_bars[-1].close / start - 1.0


def _close_offset_trend(bars: list[Bar], lookback_hours: int) -> float:
    if len(bars) < lookback_hours + 1:
        return 0.0
    start = bars[-lookback_hours - 1].close
    if start == 0:
        return 0.0
    return bars[-1].close / start - 1.0


def multi_timeframe_trends(bars: list[Bar]) -> tuple[float, float, float, float]:
    """Return 4H, daily, weekly trend plus a 0..1 alignment score.

    The repo stores hourly bars. These close-offset trends are intentionally
    cheap to compute during long backtests while still aligning entries with
    4-hour, daily, and weekly chart direction.
    """

    trend_4h = _close_offset_trend(bars, 4 * 12)      # last twelve 4H candles
    trend_1d = _close_offset_trend(bars, 24 * 14)     # last fourteen daily candles
    trend_1w = _close_offset_trend(bars, 24 * 7 * 8)  # last eight weekly candles

    def _score(value: float, scale: float) -> float:
        return max(0.0, min(1.0, value / scale))

    mtf_score = (
        0.30 * _score(trend_4h, 0.10)
        + 0.30 * _score(trend_1d, 0.20)
        + 0.40 * _score(trend_1w, 0.50)
    )
    return trend_4h, trend_1d, trend_1w, mtf_score


def compute_strategy_features(bars: list[Bar], cfg: StrategyConfig) -> StrategyFeatures:
    required = max(cfg.breakout_lookback, cfg.volume_lookback, cfg.momentum_lookback) + 1
    if len(bars) < required:
        raise ValueError(f"at least {required} bars are required")

    current = bars[-1]
    prior = bars[:-1]
    breakout_window = prior[-cfg.breakout_lookback :]
    volume_window = prior[-cfg.volume_lookback :]
    range_window = prior[-cfg.breakout_lookback :]
    breakout_level = max(bar.high for bar in breakout_window)
    breakout_pct = current.close / breakout_level - 1.0 if breakout_level else 0.0
    avg_volume = _avg([bar.volume for bar in volume_window])
    recent_avg_volume = _avg([bar.volume for bar in prior[-5:]])
    relative_volume = current.volume / avg_volume if avg_volume else 0.0
    volume_acceleration = current.volume / recent_avg_volume if recent_avg_volume else 0.0

    current_range = max(current.high - current.low, 0.0)
    avg_range = _avg(_true_ranges(range_window))
    range_expansion = current_range / avg_range if avg_range else 0.0
    close_location = (current.close - current.low) / current_range if current_range else 0.5

    momentum_start = bars[-cfg.momentum_lookback - 1].close
    momentum = current.close / momentum_start - 1.0 if momentum_start else 0.0
    htf_trend = higher_timeframe_trend(bars, cfg.htf_trend_lookback)
    trend_4h, trend_1d, trend_1w, mtf_trend_score = multi_timeframe_trends(bars)
    atr_value = atr(bars, cfg.atr_window)
    atr_short = atr(bars, cfg.atr_short_window)
    atr_long = atr(bars, cfg.atr_long_window)
    atr_pct = atr_value / current.close if current.close else 0.0
    atr_regime = atr_short / atr_long if atr_long else 0.0
    expected_move_pct = max(breakout_pct, atr_pct * range_expansion)
    passes_htf = htf_trend > 0.0 or not cfg.require_htf_trend
    is_breakout = current.close > breakout_level and close_location >= cfg.min_close_location
    passes_volume = relative_volume >= cfg.relative_volume_min
    passes_range = range_expansion >= cfg.range_expansion_min
    normalized_parts = (
        max(0.0, min(1.0, breakout_pct / 0.03)),
        max(0.0, min(1.0, (relative_volume - 1.0) / 2.0)),
        max(0.0, min(1.0, (range_expansion - 1.0) / 2.0)),
        max(0.0, min(1.0, momentum / 0.08)),
        max(0.0, min(1.0, htf_trend / 0.08)),
        mtf_trend_score,
        max(0.0, min(1.0, (atr_regime - 1.0) / 1.5)),
    )
    signal_strength = sum(normalized_parts) / len(normalized_parts)
    return StrategyFeatures(
        symbol=current.symbol,
        timestamp=current.timestamp,
        breakout_level=breakout_level,
        breakout_pct=breakout_pct,
        relative_volume=relative_volume,
        volume_acceleration=volume_acceleration,
        range_expansion=range_expansion,
        close_location=close_location,
        momentum=momentum,
        htf_trend=htf_trend,
        atr=atr_value,
        atr_pct=atr_pct,
        atr_short=atr_short,
        atr_long=atr_long,
        atr_regime=atr_regime,
        expected_move_pct=expected_move_pct,
        signal_strength=signal_strength,
        is_breakout=is_breakout,
        passes_volume=passes_volume,
        passes_range=passes_range,
        passes_htf=passes_htf,
        trend_4h=trend_4h,
        trend_1d=trend_1d,
        trend_1w=trend_1w,
        mtf_trend_score=mtf_trend_score,
    )
