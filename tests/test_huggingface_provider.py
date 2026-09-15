import pytest

from crypto_llm_alpaca.llm import build_provider


def test_huggingface_provider_requires_env_token(monkeypatch):
    monkeypatch.delenv("HUGGINGFACE_API_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HUGGINGFACE_API_TOKEN"):
        build_provider("huggingface", "meta-llama/Llama-3.1-8B-Instruct:fastest")
