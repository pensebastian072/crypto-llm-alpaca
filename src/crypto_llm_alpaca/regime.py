"""BTC-trend regime sensor.

Used by autopilot + autopilot_backtest as a portfolio-level risk gate.
risk_on = (BTC trend slope >= 0) AND (vol regime != "extreme").
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev

from .schemas import Bar


@dataclass(frozen=True)
class RegimeState:
    risk_on: bool
    trend_slope: float        # ratio of last close to EMA, minus 1
    vol_regime: str           # "low" | "normal" | "high" | "extreme"
    realized_vol: float
    samples: int
    reason: str


def _ema(values: list[float], window: int) -> float:
    if not values:
        return 0.0
    if len(values) <= window:
        return mean(values)
    alpha = 2.0 / (window + 1)
    ema = values[0]
    for v in values[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema


def compute_btc_regime(
    bars_btc: list[Bar],
    *,
    ema_window: int = 168,         # 1 week of hourly bars
    vol_window: int = 168,
    vol_extreme_threshold: float = 0.06,    # hourly stdev > 6% = extreme
    vol_high_threshold: float = 0.03,
    vol_low_threshold: float = 0.01,
) -> RegimeState:
    """Compute BTC regime state from a list of hourly bars.

    risk_on requires both BTC trend non-negative AND vol regime not extreme.
    Returns a permissive default when there isn't enough data.
    """
    if not bars_btc or len(bars_btc) < 25:
        return RegimeState(
            risk_on=True, trend_slope=0.0, vol_regime="unknown",
            realized_vol=0.0, samples=len(bars_btc),
            reason="insufficient_btc_bars; defaulting risk_on=True",
        )
    closes = [b.close for b in bars_btc]
    window_closes = closes[-max(ema_window, vol_window):]
    ema = _ema(window_closes, ema_window)
    last = window_closes[-1]
    trend_slope = (last / ema - 1.0) if ema else 0.0

    # realized vol = stdev of hourly log-ish returns
    rets = [(b / a) - 1.0 for a, b in zip(window_closes[-vol_window:], window_closes[-vol_window + 1:], strict=False) if a]
    realized_vol = pstdev(rets) if len(rets) > 1 else 0.0

    if realized_vol >= vol_extreme_threshold:
        vol_regime = "extreme"
    elif realized_vol >= vol_high_threshold:
        vol_regime = "high"
    elif realized_vol <= vol_low_threshold:
        vol_regime = "low"
    else:
        vol_regime = "normal"

    risk_on = (trend_slope >= 0.0) and (vol_regime != "extreme")
    if not risk_on:
        if trend_slope < 0 and vol_regime == "extreme":
            reason = f"BTC trend below EMA ({trend_slope:+.3f}) AND vol regime extreme ({realized_vol:.4f})"
        elif trend_slope < 0:
            reason = f"BTC trend below EMA ({trend_slope:+.3f})"
        else:
            reason = f"BTC vol regime extreme ({realized_vol:.4f})"
    else:
        reason = f"BTC trend {trend_slope:+.3f}; vol {vol_regime}"

    return RegimeState(
        risk_on=risk_on,
        trend_slope=trend_slope,
        vol_regime=vol_regime,
        realized_vol=realized_vol,
        samples=len(window_closes),
        reason=reason,
    )
