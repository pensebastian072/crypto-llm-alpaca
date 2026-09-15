from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from crypto_llm_alpaca.broker import BrokerAccount, SubmittedOrder
from crypto_llm_alpaca.cli import main
from crypto_llm_alpaca.schemas import LlmSignal


def _write_cfg(tmp_path: Path) -> Path:
    db = tmp_path / "smoke.sqlite3"
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
  min_confidence: 0.0
scoring:
  buy_threshold: 0.0
  sell_threshold: -1.0
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
  htf_trend_lookback: 6
  require_htf_trend: false
  atr_window: 14
  atr_short_window: 6
  atr_long_window: 24
  atr_stop_multiple: 2.0
  atr_trail_multiple: 2.5
  min_expected_move_fee_multiple: 1.5
  top_n_per_bar: 3
costs:
  taker_bps: 25
  maker_bps: 15
  default_fill_model: taker
risk:
  max_total_exposure_pct: 0.30
  min_positions: 1
  max_positions: 5
  stop_loss_pct: 0.05
  risk_per_trade_pct: 0.01
  max_symbol_exposure_pct: 0.10
  daily_drawdown_stop_pct: 0.05
  volatility_spike_limit: 0.20
  take_profit_levels:
    - gain_pct: 0.10
      sell_fraction: 1.0
broker:
  paper_base_url: https://paper-api.alpaca.markets
  live_trading_enabled: false
storage:
  path: "{db.as_posix()}"
"""
    )
    return cfg


def test_cli_autopilot_dry_run_smoke(tmp_path, monkeypatch, capsys):
    cfg_path = _write_cfg(tmp_path)
    out_dir = tmp_path / "outputs"
    monkeypatch.chdir(tmp_path)

    from crypto_llm_alpaca import autopilot as ap_mod
    from crypto_llm_alpaca import _paths

    _paths.find_repo_root.cache_clear()
    monkeypatch.setattr(_paths, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(ap_mod, "find_repo_root", lambda: tmp_path)

    rc = main(["--config", str(cfg_path), "autopilot", "--dry-run", "--max-trades", "1", "--demo-only"])
    assert rc == 0
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["dry_run"] is True
    assert payload["n_orders"] == 0
    assert (tmp_path / "outputs").exists()
    md_files = list((tmp_path / "outputs").glob("*.md"))
    assert len(md_files) == 1
    assert "DRY RUN" in md_files[0].read_text(encoding="utf-8")
