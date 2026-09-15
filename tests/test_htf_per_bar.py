from __future__ import annotations

from datetime import datetime, timedelta, timezone

from crypto_llm_alpaca.features import higher_timeframe_trend
from crypto_llm_alpaca.schemas import Bar


def _make_bar(idx: int, close: float) -> Bar:
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=idx)
    return Bar(
        symbol="BTC/USD",
        timestamp=ts,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


def test_htf_trend_recomputes_on_window_slice():
    """Regression: HTF must reflect the slice it's given, not a captured full series."""
    closes_down = [100.0 - i * 0.5 for i in range(60)]
    closes_up = [closes_down[-1] + i * 0.5 for i in range(60)]
    bars = [_make_bar(i, c) for i, c in enumerate(closes_down + closes_up)]

    early_window = bars[:60]
    late_window = bars

    early_trend = higher_timeframe_trend(early_window, lookback=6, source_chunk_hours=4)
    late_trend = higher_timeframe_trend(late_window, lookback=6, source_chunk_hours=4)

    assert early_trend < 0.0, "early window is downtrend; HTF should be negative"
    assert late_trend > 0.0, "later window flips to uptrend; HTF must reflect the slice"
