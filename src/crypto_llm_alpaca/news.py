from __future__ import annotations

import hashlib
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape

from .schemas import NewsItem

_SYMBOL_KEYWORDS = {
    "BTC/USD": ("bitcoin", "btc"),
    "ETH/USD": ("ethereum", "ether", "eth"),
    "SOL/USD": ("solana", "sol"),
    "AVAX/USD": ("avalanche", "avax"),
    "LINK/USD": ("chainlink", "link"),
    "AAVE/USD": ("aave",),
    "UNI/USD": ("uniswap", "uni"),
    "DOT/USD": ("polkadot", "dot"),
    "GRT/USD": ("the graph", "graph protocol", "grt"),
    "ARB/USD": ("arbitrum", "arb"),
    "CRV/USD": ("curve", "curve dao", "crv"),
    "YFI/USD": ("yearn", "yearn finance", "yfi"),
    "SUSHI/USD": ("sushi", "sushiswap"),
    "BAT/USD": ("basic attention token", "brave", "bat"),
    "XTZ/USD": ("tezos", "xtz"),
}


def dedupe_hash(source: str, url: str, title: str, published_at: datetime) -> str:
    payload = f"{source}|{url}|{title}|{published_at.isoformat()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def map_symbols(text: str, universe: tuple[str, ...]) -> tuple[str, ...]:
    lowered = text.lower()
    found: list[str] = []
    for symbol in universe:
        keywords = _SYMBOL_KEYWORDS.get(symbol, (symbol.split("/")[0].lower(),))
        if any(re.search(rf"\b{re.escape(keyword)}\b", lowered) for keyword in keywords):
            found.append(symbol)
    return tuple(found)


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(tz=timezone.utc)
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return datetime.now(tz=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_rss(xml_text: str, source: str, universe: tuple[str, ...]) -> list[NewsItem]:
    root = ET.fromstring(xml_text)
    items: list[NewsItem] = []
    for node in root.findall(".//item"):
        title = unescape((node.findtext("title") or "").strip())
        url = (node.findtext("link") or "").strip()
        summary = unescape((node.findtext("description") or "").strip())
        published_at = _parse_datetime(node.findtext("pubDate"))
        symbols = map_symbols(f"{title} {summary}", universe)
        if not title or not symbols:
            continue
        digest = dedupe_hash(source, url, title, published_at)
        items.append(NewsItem(source, url, title, summary, published_at, symbols, digest))
    return items


def fetch_rss(url: str, universe: tuple[str, ...], timeout: float = 10.0) -> list[NewsItem]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "crypto-llm-alpaca/0.1 (+https://localhost)",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        xml_text = response.read().decode("utf-8", errors="replace")
    return parse_rss(xml_text, source=url, universe=universe)
