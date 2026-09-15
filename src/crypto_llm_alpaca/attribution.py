from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolReconciliation:
    symbols: tuple[str, ...]
    symbol_confidence: float
    corrected: bool
    fallback: bool


def _dedupe(items: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return tuple(out)


def reconcile_symbols(
    llm_symbols: tuple[str, ...] | list[str],
    article_symbols: tuple[str, ...] | list[str],
    allowed_symbols: tuple[str, ...] | list[str],
    *,
    primary_symbol_bias: bool = True,
) -> SymbolReconciliation:
    """Constrain provider symbols to curated article + universe truth."""

    llm = _dedupe(tuple(llm_symbols))
    article = _dedupe(tuple(article_symbols))
    allowed = set(allowed_symbols)

    if primary_symbol_bias and article and article[0] in allowed:
        final = (article[0],)
        return SymbolReconciliation(
            symbols=final,
            symbol_confidence=1.0,
            corrected=final != llm,
            fallback=not llm or not any(symbol in allowed for symbol in llm),
        )

    if not llm:
        fallback_symbols = tuple(symbol for symbol in article if symbol in allowed)
        return SymbolReconciliation(
            symbols=fallback_symbols,
            symbol_confidence=1.0 if fallback_symbols else 0.4,
            corrected=bool(fallback_symbols),
            fallback=True,
        )

    filtered = tuple(symbol for symbol in llm if symbol in allowed)
    if not filtered:
        fallback_symbols = tuple(symbol for symbol in article if symbol in allowed)
        return SymbolReconciliation(
            symbols=fallback_symbols,
            symbol_confidence=1.0 if fallback_symbols else 0.4,
            corrected=fallback_symbols != llm,
            fallback=True,
        )

    overlap = tuple(symbol for symbol in filtered if symbol in article)
    final = overlap or filtered
    return SymbolReconciliation(
        symbols=final,
        symbol_confidence=0.7 if overlap else 0.4,
        corrected=final != llm,
        fallback=False,
    )
