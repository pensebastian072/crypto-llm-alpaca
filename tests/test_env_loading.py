from __future__ import annotations

import os

import pytest

from crypto_llm_alpaca import env as env_module


@pytest.fixture(autouse=True)
def _reset():
    env_module.reset_for_tests()
    yield
    env_module.reset_for_tests()


def test_load_env_reads_file(tmp_path, monkeypatch):
    envfile = tmp_path / ".env"
    envfile.write_text("CLA_TEST_KEY=from_file\nCLA_OTHER=value2\n")
    monkeypatch.delenv("CLA_TEST_KEY", raising=False)
    monkeypatch.delenv("CLA_OTHER", raising=False)

    found = env_module.load_env(envfile)

    assert found is True
    assert os.environ.get("CLA_TEST_KEY") == "from_file"
    assert os.environ.get("CLA_OTHER") == "value2"


def test_load_env_does_not_override_existing(tmp_path, monkeypatch):
    envfile = tmp_path / ".env"
    envfile.write_text("CLA_PRESET=from_file\n")
    monkeypatch.setenv("CLA_PRESET", "from_process")

    env_module.load_env(envfile)

    assert os.environ["CLA_PRESET"] == "from_process"


def test_load_env_missing_file_is_noop(tmp_path):
    missing = tmp_path / "nope.env"
    found = env_module.load_env(missing)
    assert found is False


def test_load_env_idempotent(tmp_path, monkeypatch):
    envfile = tmp_path / ".env"
    envfile.write_text("CLA_ONCE=first\n")
    monkeypatch.delenv("CLA_ONCE", raising=False)

    env_module.load_env(envfile)
    envfile.write_text("CLA_ONCE=second\n")
    env_module.load_env(envfile)

    assert os.environ["CLA_ONCE"] == "first"
