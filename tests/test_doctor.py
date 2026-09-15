from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

from crypto_llm_alpaca import env as env_module
from crypto_llm_alpaca.cli import cmd_doctor, _doctor_checks


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    from crypto_llm_alpaca import _paths
    from crypto_llm_alpaca import env as env_mod
    from crypto_llm_alpaca import cli as cli_mod

    monkeypatch.chdir(tmp_path)
    _paths.find_repo_root.cache_clear()
    monkeypatch.setattr(_paths, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(env_mod, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(cli_mod, "find_repo_root", lambda: tmp_path)
    for var in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "HUGGINGFACE_API_TOKEN", "HF_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    env_module.reset_for_tests()
    yield
    env_module.reset_for_tests()


def _write_config(tmp_path: Path) -> Path:
    db = tmp_path / "test.sqlite3"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""
universe:
  symbols: ["BTC/USD"]
  quote_currency: USD
data:
  timeframe: "1Hour"
  lookback_bars: 200
  rss_feeds: []
llm:
  provider: mock
  model: mock
  min_confidence: 0.4
scoring:
  buy_threshold: 0.4
  sell_threshold: -0.4
  sentiment_weight: 1.0
  momentum_weight: 0.5
  volatility_weight: 0.2
strategy:
  breakout_lookback: 20
  volume_lookback: 20
  relative_volume_min: 1.2
  range_expansion_min: 1.2
  min_close_location: 0.5
  momentum_lookback: 5
  htf_timeframe: "4Hour"
  htf_trend_lookback: 20
  require_htf_trend: false
  atr_window: 14
  atr_short_window: 7
  atr_long_window: 28
  atr_stop_multiple: 1.5
  atr_trail_multiple: 2.5
  min_expected_move_fee_multiple: 1.5
  top_n_per_bar: 3
costs:
  taker_bps: 20
  maker_bps: 5
  default_fill_model: taker
risk:
  max_total_exposure_pct: 0.30
  min_positions: 1
  max_positions: 5
  stop_loss_pct: 0.01
  risk_per_trade_pct: 0.01
  max_symbol_exposure_pct: 0.10
  daily_drawdown_stop_pct: 0.05
  volatility_spike_limit: 0.08
  take_profit_levels:
    - gain_pct: 0.10
      sell_fraction: 0.25
broker:
  paper_base_url: https://paper-api.alpaca.markets
  live_trading_enabled: false
storage:
  path: "{db.as_posix()}"
"""
    )
    return cfg


def test_doctor_no_keys_warns_no_fail(tmp_path, capsys):
    cfg = _write_config(tmp_path)
    args = argparse.Namespace(config=str(cfg), json=False)
    rc = cmd_doctor(args)
    out = capsys.readouterr().out
    assert "Alpaca keys present" in out
    assert rc == 1  # missing alpaca keys = fail


def test_doctor_json_output(tmp_path, capsys):
    cfg = _write_config(tmp_path)
    args = argparse.Namespace(config=str(cfg), json=True)
    cmd_doctor(args)
    payload = json.loads(capsys.readouterr().out)
    names = [c["name"] for c in payload["checks"]]
    assert "Config loadable" in names
    assert "DB schema present" in names


def test_doctor_with_keys_passes_key_check(tmp_path, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "fake")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "fake")
    cfg = _write_config(tmp_path)
    args = argparse.Namespace(config=str(cfg), json=False)
    checks = _doctor_checks(args)
    by_name = {c["name"]: c for c in checks}
    assert by_name["Alpaca keys present"]["status"] == "ok"
    # reachable will warn or fail depending on network, but key presence is ok


def test_doctor_db_initialized(tmp_path):
    cfg_path = _write_config(tmp_path)
    args = argparse.Namespace(config=str(cfg_path), json=False)
    checks = _doctor_checks(args)
    by_name = {c["name"]: c for c in checks}
    assert by_name["DB schema present"]["status"] == "ok"
    assert by_name["DB has bars"]["status"] == "warn"  # empty DB
