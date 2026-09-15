from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

from .schemas import LlmSignal, NewsItem


class LLMProvider(ABC):
    @abstractmethod
    def extract(self, item: NewsItem, symbol: str) -> LlmSignal:
        raise NotImplementedError


_HYPE_WORDS = ("surge", "rally", "approval", "inflow", "beat", "bull", "moon", "ath", "all-time")
_FEAR_WORDS = ("hack", "lawsuit", "outflow", "ban", "bear", "plunge", "crash", "exploit", "investigation")
_TOPIC_KEYWORDS = {
    "regulation": ("sec", "lawsuit", "ban", "regulatory", "approval"),
    "hack": ("hack", "exploit", "breach"),
    "etf": ("etf", "spot etf"),
    "adoption": ("adoption", "partnership", "integration"),
    "macro": ("fed", "cpi", "rates", "inflation"),
}


def _classify_topic(text: str) -> str:
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(k in text for k in keywords):
            return topic
    return ""


def _symbol_instructions(item: NewsItem, symbol: str) -> str:
    allowed_symbols = sorted(set(item.symbols) | {symbol})
    return (
        "SYSTEM:\n"
        "You are a crypto research analyst extracting structured trading signals from news.\n"
        "Your job is to identify whether the requested crypto asset is materially impacted, "
        "assign sentiment, and explain the reasoning.\n\n"
        "CRITICAL RULES:\n"
        "- Evaluate ONLY the Target Symbol.\n"
        "- The Target Symbol must be in Allowed Symbols.\n"
        "- Known Article Symbols are high confidence curated symbols.\n"
        "- If the article clearly focuses on one protocol, prioritize that protocol.\n"
        "- Do not default to BTC/USD or ETH/USD unless explicitly discussed in the article "
        "or listed in Known Article Symbols.\n"
        "- If the Target Symbol is not materially impacted, return sentiment 0.0 with low confidence.\n"
        "- Be conservative and precise.\n"
        "- Return JSON only matching the response schema.\n\n"
        "USER INPUT:\n"
        f"Target Symbol: {symbol}\n"
        f"Known Article Symbols: {list(item.symbols)}\n"
        f"Allowed Symbols: {allowed_symbols}\n"
    )


class MockLLMProvider(LLMProvider):
    def extract(self, item: NewsItem, symbol: str) -> LlmSignal:
        text = f"{item.title} {item.summary}".lower()
        positive = any(word in text for word in _HYPE_WORDS)
        negative = any(word in text for word in _FEAR_WORDS)
        sentiment = 0.0
        if positive:
            sentiment += 0.65
        if negative:
            sentiment -= 0.65
        hype = 0.6 if positive else 0.0
        fear = 0.6 if negative else 0.0
        topic = _classify_topic(text)
        return LlmSignal(
            news_hash=item.dedupe_hash,
            symbol=symbol,
            sentiment=max(-1.0, min(1.0, sentiment)),
            confidence=0.75 if sentiment else 0.35,
            horizon="short",
            event_type="mock_news_sentiment",
            rationale="Keyword-derived deterministic mock signal for offline tests.",
            provider="mock",
            model="mock",
            fear=fear,
            hype=hype,
            topic=topic,
            source_weight=1.0,
        )


class OpenAILLMProvider(LLMProvider):
    def __init__(self, model: str) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def extract(self, item: NewsItem, symbol: str) -> LlmSignal:
        schema = {
            "name": "crypto_news_signal",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "sentiment": {"type": "number", "minimum": -1, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "horizon": {"type": "string", "enum": ["short", "medium", "long"]},
                    "event_type": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["sentiment", "confidence", "horizon", "event_type", "rationale"],
            },
            "strict": True,
        }
        response = self.client.responses.create(
            model=self.model,
            input=(
                f"{_symbol_instructions(item, symbol)}\n"
                f"Title: {item.title}\nSummary: {item.summary}"
            ),
            response_format={"type": "json_schema", "json_schema": schema},
        )
        payload = json.loads(response.output_text)
        return LlmSignal(
            news_hash=item.dedupe_hash,
            symbol=symbol,
            sentiment=float(payload["sentiment"]),
            confidence=float(payload["confidence"]),
            horizon=str(payload["horizon"]),
            event_type=str(payload["event_type"]),
            rationale=str(payload["rationale"]),
            provider="openai",
            model=self.model,
        )


class HuggingFaceLLMProvider(LLMProvider):
    def __init__(self, model: str) -> None:
        token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("HUGGINGFACE_API_TOKEN or HF_TOKEN is required for Hugging Face provider")
        from openai import OpenAI

        self.client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=token)
        self.model = model

    def extract(self, item: NewsItem, symbol: str) -> LlmSignal:
        schema = {
            "name": "crypto_news_signal",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "sentiment": {"type": "number", "minimum": -1, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "horizon": {"type": "string", "enum": ["short", "medium", "long"]},
                    "event_type": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["sentiment", "confidence", "horizon", "event_type", "rationale"],
            },
            "strict": True,
        }
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{_symbol_instructions(item, symbol)}\n"
                        f"Title: {item.title}\nSummary: {item.summary}"
                    ),
                }
            ],
            response_format={"type": "json_schema", "json_schema": schema},
        )
        content = completion.choices[0].message.content or "{}"
        payload = json.loads(content)
        return LlmSignal(
            news_hash=item.dedupe_hash,
            symbol=symbol,
            sentiment=float(payload["sentiment"]),
            confidence=float(payload["confidence"]),
            horizon=str(payload["horizon"]),
            event_type=str(payload["event_type"]),
            rationale=str(payload["rationale"]),
            provider="huggingface",
            model=self.model,
        )


def build_provider(provider: str, model: str) -> LLMProvider:
    if provider == "mock":
        return MockLLMProvider()
    if provider == "openai":
        return OpenAILLMProvider(model=model)
    if provider == "huggingface":
        return HuggingFaceLLMProvider(model=model)
    raise ValueError(f"Unsupported LLM provider: {provider}")
