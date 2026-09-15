from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .cache import load_cache, save_cache

BASE_URL = "https://api.llama.fi"

SYMBOL_TO_LLAMA_PROTOCOL = {
    "AAVE": "aave-v3",
    "UNI": "uniswap-v3",
    "CRV": "curve-dex",
    "SUSHI": "sushiswap",
    "YFI": "yearn-finance",
    "GRT": "the-graph",
    "ARB": "arbitrum-bridge",
}

SYMBOL_TO_LLAMA_FEES = {
    "AAVE": "aave",
    "UNI": "uniswap-v3",
    "CRV": "curve-dex",
    "SUSHI": "sushiswap",
    "YFI": "yearn-finance",
}

SYMBOL_TO_LLAMA_HISTORY = {
    "AAVE": ("aave-v3", "aave"),
    "UNI": ("uniswap", "uniswap-v3", "uniswap-v2"),
    "CRV": ("curve-finance", "curve-dex"),
    "SUSHI": ("sushiswap", "sushiswap-v3"),
    "YFI": ("yearn", "yearn-finance"),
    "GRT": ("the-graph",),
    "ARB": ("arbitrum-bridge",),
}


def _base_asset(symbol: str) -> str:
    return symbol.split("/")[0].upper()


def historical_slugs_for_symbol(symbol: str) -> tuple[str, ...]:
    asset = _base_asset(symbol)
    return SYMBOL_TO_LLAMA_HISTORY.get(asset, (SYMBOL_TO_LLAMA_PROTOCOL[asset],)) if asset in SYMBOL_TO_LLAMA_PROTOCOL else ()


def historical_slug_for_symbol(symbol: str) -> str:
    slugs = historical_slugs_for_symbol(symbol)
    return slugs[0] if slugs else ""


def _request_json(path: str, *, timeout_sec: int = 20) -> Any:
    request = Request(
        f"{BASE_URL}{path}",
        headers={"User-Agent": "crypto-llm-alpaca/0.1"},
    )
    with urlopen(request, timeout=timeout_sec) as response:
        return json.loads(response.read().decode("utf-8"))


def _cached_json(
    path: str,
    cache_key: str,
    *,
    max_age_sec: int,
    cache_dir: str | Path | None = None,
    timeout_sec: int = 20,
) -> Any | None:
    cached = load_cache(cache_key, max_age_sec, cache_dir)
    if cached is not None:
        return cached
    try:
        data = _request_json(path, timeout_sec=timeout_sec)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    save_cache(cache_key, data, cache_dir)
    return data


def get_protocols(
    *,
    cache_dir: str | Path | None = None,
    max_age_sec: int = 21_600,
) -> list[dict[str, Any]]:
    data = _cached_json(
        "/protocols",
        "defillama_protocols",
        max_age_sec=max_age_sec,
        cache_dir=cache_dir,
    )
    return data if isinstance(data, list) else []


def get_fees_summary(
    slug: str,
    *,
    cache_dir: str | Path | None = None,
    max_age_sec: int = 21_600,
) -> dict[str, Any]:
    data = _cached_json(
        f"/summary/fees/{slug}",
        f"defillama_fees_{slug}",
        max_age_sec=max_age_sec,
        cache_dir=cache_dir,
    )
    return data if isinstance(data, dict) else {}


def _find_protocol(slug: str, protocols: list[dict[str, Any]]) -> dict[str, Any] | None:
    for protocol in protocols:
        if protocol.get("slug") == slug:
            return protocol
    return None


def _pct(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value) / 100.0
    except (TypeError, ValueError):
        return None


def _fee_growth(summary: dict[str, Any]) -> float | None:
    current = summary.get("total24h")
    previous = summary.get("total48hto24h")
    try:
        current_f = float(current)
        previous_f = float(previous)
    except (TypeError, ValueError):
        return None
    if previous_f <= 0:
        return None
    return max(-1.0, min(1.0, current_f / previous_f - 1.0))


def extract_defi_metrics(
    symbol: str,
    *,
    protocols: list[dict[str, Any]] | None = None,
    fees_summary: dict[str, Any] | None = None,
    cache_dir: str | Path | None = None,
    max_age_sec: int = 21_600,
) -> dict[str, Any]:
    asset = _base_asset(symbol)
    slug = SYMBOL_TO_LLAMA_PROTOCOL.get(asset)
    if not slug:
        return {}

    protocol_list = get_protocols(cache_dir=cache_dir, max_age_sec=max_age_sec) if protocols is None else protocols
    protocol = _find_protocol(slug, protocol_list)
    if protocol is None:
        return {"defillama_slug": slug}

    fee_slug = SYMBOL_TO_LLAMA_FEES.get(asset, slug)
    fees = get_fees_summary(fee_slug, cache_dir=cache_dir, max_age_sec=max_age_sec) if fees_summary is None else fees_summary
    fee_growth = _fee_growth(fees)
    tvl_growth = _pct(protocol.get("change_7d"))
    tvl_growth_30d = _pct(protocol.get("change_1m"))

    metrics: dict[str, Any] = {
        "defillama_slug": slug,
        "defillama_fee_slug": fee_slug,
        "tvl": float(protocol.get("tvl") or 0.0),
        "tvl_growth": tvl_growth,
        "tvl_growth_30d": tvl_growth_30d,
        "fees_24h": float(fees.get("total24h") or 0.0) if fees else 0.0,
        "fees_7d": float(fees.get("total7d") or 0.0) if fees else 0.0,
        "fees_30d": float(fees.get("total30d") or 0.0) if fees else 0.0,
        "fee_growth": fee_growth,
        "revenue_growth": fee_growth if fee_growth is not None else tvl_growth,
    }
    return {key: value for key, value in metrics.items() if value is not None}


def fetch_historical_tvl(
    slug: str,
    *,
    cache_dir: str | Path | None = None,
    max_age_sec: int = 86_400 * 7,
) -> list[dict[str, Any]]:
    """Return historical TVL series for a protocol slug.

    DefiLlama `/protocol/{slug}` returns `tvl: [{date, totalLiquidityUSD}, ...]`.
    The endpoint is free and key-less. Cached aggressively (TVL series is monotonic).
    """
    data = _cached_json(
        f"/protocol/{slug}",
        f"defillama_protocol_history_{slug}",
        max_age_sec=max_age_sec,
        cache_dir=cache_dir,
    )
    if not isinstance(data, dict):
        return []
    series = data.get("tvl") or []
    out: list[dict[str, Any]] = []
    if not isinstance(series, list):
        return []
    for point in series:
        if not isinstance(point, dict):
            continue
        try:
            ts = float(point.get("date") or 0)
            tvl = float(point.get("totalLiquidityUSD") or 0)
        except (TypeError, ValueError):
            continue
        if ts <= 0:
            continue
        out.append({"date": ts, "tvl": tvl})
    out.sort(key=lambda p: p["date"])
    return out


def _tvl_at(series: list[dict[str, Any]], asof_ts: float) -> float | None:
    """Last TVL point with date <= asof_ts. None if none."""
    last: float | None = None
    for p in series:
        if p["date"] <= asof_ts:
            last = p["tvl"]
        else:
            break
    return last


def historical_metrics_at(
    symbol: str,
    asof,
    *,
    cache_dir: str | Path | None = None,
    max_age_sec: int = 86_400 * 7,
) -> dict[str, Any]:
    """Compute point-in-time DefiLlama metrics for `symbol` as of `asof` (datetime or unix ts).

    Uses the historical TVL series only (free endpoint). 7d/30d growth derived from the series.
    Returns empty dict if no data available.
    """
    asof_ts = asof.timestamp() if hasattr(asof, "timestamp") else float(asof)
    slugs = historical_slugs_for_symbol(symbol)
    if not slugs:
        return {}
    slug = slugs[0]
    series: list[dict[str, Any]] = []
    for candidate_slug in slugs:
        candidate_series = fetch_historical_tvl(
            candidate_slug,
            cache_dir=cache_dir,
            max_age_sec=max_age_sec,
        )
        if candidate_series:
            slug = candidate_slug
            series = candidate_series
            break
    if not series:
        return {"defillama_slug": slug}
    tvl = _tvl_at(series, asof_ts)
    tvl_7d_ago = _tvl_at(series, asof_ts - 86_400 * 7)
    tvl_30d_ago = _tvl_at(series, asof_ts - 86_400 * 30)
    metrics: dict[str, Any] = {
        "defillama_slug": slug,
        "tvl": tvl,
    }
    if tvl is not None and tvl_7d_ago and tvl_7d_ago > 0:
        metrics["tvl_growth"] = max(-1.0, min(5.0, tvl / tvl_7d_ago - 1.0))
    if tvl is not None and tvl_30d_ago and tvl_30d_ago > 0:
        metrics["tvl_growth_30d"] = max(-1.0, min(10.0, tvl / tvl_30d_ago - 1.0))
    return {k: v for k, v in metrics.items() if v is not None}


def extract_defi_metrics_at(
    symbol: str,
    asof,
    store=None,
    *,
    cache_dir: str | Path | None = None,
    max_age_sec: int = 86_400 * 7,
) -> dict[str, Any]:
    """Lookahead-free metric lookup. Prefers stored snapshot; falls back to API series.

    `store` is an SQLiteStore-like object with `latest_defi_snapshot_before(symbol, asof)`.
    """
    if store is not None:
        snap = store.latest_defi_snapshot_before(symbol, asof if hasattr(asof, "isoformat") else None)
        if snap:
            cleaned = {k: snap.get(k) for k in ("tvl", "tvl_growth", "tvl_growth_30d", "fees_7d", "fee_growth", "revenue_growth") if snap.get(k) is not None}
            if cleaned:
                cleaned["defillama_slug"] = historical_slug_for_symbol(symbol)
                return cleaned
    return historical_metrics_at(symbol, asof, cache_dir=cache_dir, max_age_sec=max_age_sec)


def enrich_with_defi(
    asset_data: dict[str, Any],
    symbol: str,
    *,
    cache_dir: str | Path | None = None,
    max_age_sec: int = 21_600,
) -> dict[str, Any]:
    enriched = dict(asset_data)
    enriched.update(extract_defi_metrics(symbol, cache_dir=cache_dir, max_age_sec=max_age_sec))
    return enriched
