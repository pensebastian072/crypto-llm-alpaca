from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Protocol

from ..schemas import NewsItem


class Encoder(Protocol):
    def encode(self, text: str): ...


NARRATIVES: dict[str, tuple[str, ...]] = {
    "defi_lending": ("defi", "lending", "borrow", "collateral", "aave", "liquidity", "yield"),
    "dex_infrastructure": ("dex", "swap", "exchange", "amm", "fees", "uniswap", "curve", "sushiswap"),
    "layer2_scaling": ("layer 2", "l2", "rollup", "scaling", "arbitrum", "optimism", "base"),
    "interoperability": ("interoperability", "cross chain", "parachain", "polkadot", "bridge"),
    "data_indexing": ("data", "indexing", "subgraph", "the graph", "query", "ai"),
    "ai_data": ("ai", "artificial intelligence", "compute", "agent", "data"),
    "consumer_crypto": ("wallet", "browser", "consumer", "ad", "brave", "payments"),
    "governance_regulation": ("governance", "vote", "proposal", "sec", "regulation", "lawsuit"),
    "security_risk": ("hack", "exploit", "breach", "attack", "loss", "vulnerability"),
}


@dataclass(frozen=True)
class NarrativeMatch:
    label: str
    score: float


@dataclass(frozen=True)
class NarrativeProfile:
    label: str = ""
    score: float = 0.0
    momentum: float = 0.0
    current_score: float = 0.0
    previous_score: float = 0.0
    source_count: int = 0


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _vocabulary() -> tuple[str, ...]:
    terms: set[str] = set()
    for keywords in NARRATIVES.values():
        for keyword in keywords:
            terms.update(_tokens(keyword))
    return tuple(sorted(terms))


VOCABULARY = _vocabulary()


def keyword_embed_text(text: str) -> list[float]:
    counts = Counter(_tokens(text))
    return [float(counts.get(term, 0)) for term in VOCABULARY]


def embed_text(model: Encoder | None, text: str) -> list[float]:
    if model is not None:
        encoded = model.encode(text)
        return [float(value) for value in encoded]
    return keyword_embed_text(text)


def cosine_sim(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a)
    bv = list(b)
    if len(av) != len(bv) or not av:
        return 0.0
    dot = sum(x * y for x, y in zip(av, bv))
    norm_a = math.sqrt(sum(x * x for x in av))
    norm_b = math.sqrt(sum(y * y for y in bv))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def build_narrative_vectors(model: Encoder | None = None) -> dict[str, list[float]]:
    return {
        name: embed_text(model, " ".join(keywords))
        for name, keywords in NARRATIVES.items()
    }


def detect_narrative(
    text: str,
    narrative_vectors: dict[str, list[float]] | None = None,
    *,
    model: Encoder | None = None,
) -> NarrativeMatch:
    vectors = narrative_vectors or build_narrative_vectors(model)
    article_vector = embed_text(model, text)
    scores = {name: cosine_sim(article_vector, vector) for name, vector in vectors.items()}
    if not scores:
        return NarrativeMatch("", 0.0)
    label, score = max(scores.items(), key=lambda item: item[1])
    return NarrativeMatch(label if score > 0 else "", score)


def _published_at(item: NewsItem) -> datetime:
    ts = item.published_at
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def aggregate_narratives(
    items: Iterable[NewsItem],
    *,
    asof: datetime | None = None,
    model: Encoder | None = None,
    current_days: int = 7,
    previous_days: int = 7,
) -> dict[str, NarrativeProfile]:
    item_list = list(items)
    if not item_list:
        return {}
    asof = asof or max(_published_at(item) for item in item_list)
    if asof.tzinfo is None:
        asof = asof.replace(tzinfo=timezone.utc)
    current_start = asof - timedelta(days=current_days)
    previous_start = current_start - timedelta(days=previous_days)
    vectors = build_narrative_vectors(model)

    current: dict[str, list[NarrativeMatch]] = defaultdict(list)
    previous: dict[str, list[NarrativeMatch]] = defaultdict(list)
    all_matches: dict[str, list[NarrativeMatch]] = defaultdict(list)
    for item in item_list:
        match = detect_narrative(f"{item.title} {item.summary}", vectors, model=model)
        if match.score <= 0:
            continue
        ts = _published_at(item)
        for symbol in item.symbols:
            all_matches[symbol].append(match)
            if current_start < ts <= asof:
                current[symbol].append(match)
            elif previous_start < ts <= current_start:
                previous[symbol].append(match)

    profiles: dict[str, NarrativeProfile] = {}
    for symbol, matches in all_matches.items():
        cur = current.get(symbol, [])
        prev = previous.get(symbol, [])
        cur_scores = [match.score for match in cur]
        prev_scores = [match.score for match in prev]
        current_score = sum(cur_scores) / len(cur_scores) if cur_scores else 0.0
        previous_score = sum(prev_scores) / len(prev_scores) if prev_scores else 0.0
        label_pool = cur or matches
        label = Counter(match.label for match in label_pool if match.label).most_common(1)
        profiles[symbol] = NarrativeProfile(
            label=label[0][0] if label else "",
            score=max(current_score, sum(match.score for match in matches) / len(matches)),
            momentum=current_score - previous_score,
            current_score=current_score,
            previous_score=previous_score,
            source_count=len(cur) if cur else len(matches),
        )
    return profiles


@dataclass(frozen=True)
class SectorRotation:
    label: str
    current_score: float
    previous_score: float
    momentum: float
    current_articles: int
    previous_articles: int
    top_symbols: tuple[str, ...]


def aggregate_sector_rotation(
    items: Iterable[NewsItem],
    *,
    asof: datetime | None = None,
    model: Encoder | None = None,
    current_days: int = 7,
    previous_days: int = 7,
    top_symbols_per_sector: int = 5,
) -> list[SectorRotation]:
    """Roll up news by narrative label across all symbols. Returns sorted by momentum desc."""
    item_list = list(items)
    if not item_list:
        return []
    asof = asof or max(_published_at(item) for item in item_list)
    if asof.tzinfo is None:
        asof = asof.replace(tzinfo=timezone.utc)
    current_start = asof - timedelta(days=current_days)
    previous_start = current_start - timedelta(days=previous_days)
    vectors = build_narrative_vectors(model)

    current_scores: dict[str, list[float]] = defaultdict(list)
    previous_scores: dict[str, list[float]] = defaultdict(list)
    current_symbols: dict[str, Counter] = defaultdict(Counter)

    for item in item_list:
        match = detect_narrative(f"{item.title} {item.summary}", vectors, model=model)
        if match.score <= 0 or not match.label:
            continue
        ts = _published_at(item)
        if current_start < ts <= asof:
            current_scores[match.label].append(match.score)
            for sym in item.symbols:
                current_symbols[match.label][sym] += 1
        elif previous_start < ts <= current_start:
            previous_scores[match.label].append(match.score)

    labels = set(current_scores.keys()) | set(previous_scores.keys())
    results: list[SectorRotation] = []
    for label in labels:
        cur_list = current_scores.get(label, [])
        prev_list = previous_scores.get(label, [])
        cur_avg = sum(cur_list) / len(cur_list) if cur_list else 0.0
        prev_avg = sum(prev_list) / len(prev_list) if prev_list else 0.0
        results.append(SectorRotation(
            label=label,
            current_score=cur_avg,
            previous_score=prev_avg,
            momentum=cur_avg - prev_avg,
            current_articles=len(cur_list),
            previous_articles=len(prev_list),
            top_symbols=tuple(sym for sym, _ in current_symbols[label].most_common(top_symbols_per_sector)),
        ))
    results.sort(key=lambda r: r.momentum, reverse=True)
    return results


def aggregate_symbol_narrative(
    symbol: str,
    items: Iterable[NewsItem],
    *,
    asof: datetime | None = None,
    model: Encoder | None = None,
    current_days: int = 7,
    previous_days: int = 7,
) -> NarrativeProfile:
    return aggregate_narratives(
        items,
        asof=asof,
        model=model,
        current_days=current_days,
        previous_days=previous_days,
    ).get(symbol, NarrativeProfile())
