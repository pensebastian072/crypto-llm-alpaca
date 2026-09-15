from __future__ import annotations

from datetime import datetime, timedelta

from .schemas import Bar


def _parse_timeframe(timeframe: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    normalized = timeframe.strip()
    aliases = {
        "1Hour": TimeFrame.Hour,
        "1H": TimeFrame.Hour,
        "1Day": TimeFrame.Day,
        "1D": TimeFrame.Day,
        "1Min": TimeFrame.Minute,
        "1T": TimeFrame.Minute,
    }
    if normalized in aliases:
        return aliases[normalized]
    if normalized.endswith("Hour"):
        return TimeFrame(int(normalized.removesuffix("Hour")), TimeFrameUnit.Hour)
    if normalized.endswith("H"):
        return TimeFrame(int(normalized.removesuffix("H")), TimeFrameUnit.Hour)
    if normalized.endswith("Min"):
        return TimeFrame(int(normalized.removesuffix("Min")), TimeFrameUnit.Minute)
    if normalized.endswith("T"):
        return TimeFrame(int(normalized.removesuffix("T")), TimeFrameUnit.Minute)
    raise ValueError(f"unsupported timeframe: {timeframe}")


def _chunk_bounds(start: datetime, end: datetime, days: int = 60) -> list[tuple[datetime, datetime]]:
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=days), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def fetch_alpaca_crypto_bars(
    symbols: tuple[str, ...],
    start: datetime,
    end: datetime,
    timeframe: str = "1Hour",
) -> list[Bar]:
    """Fetch crypto bars from Alpaca's market data API.

    The import stays local so offline tests and mock workflows do not require alpaca-py.
    """
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest

    tf = _parse_timeframe(timeframe)
    client = CryptoHistoricalDataClient()
    bars: list[Bar] = []
    for symbol in symbols:
        for chunk_start, chunk_end in _chunk_bounds(start, end):
            request = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=tf,
                start=chunk_start,
                end=chunk_end,
            )
            frame = client.get_crypto_bars(request).df
            for row in frame.reset_index().itertuples(index=False):
                bars.append(
                    Bar(
                        symbol=str(row.symbol),
                        timestamp=row.timestamp.to_pydatetime(),
                        open=float(row.open),
                        high=float(row.high),
                        low=float(row.low),
                        close=float(row.close),
                        volume=float(row.volume),
                        timeframe=timeframe,
                        source="alpaca",
                    )
                )
    return bars
