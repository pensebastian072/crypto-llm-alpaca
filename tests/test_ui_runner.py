from __future__ import annotations

import json
from pathlib import Path

from crypto_llm_alpaca.ui._runner import active_config_path, run_cli


def _write_min_config(tmp_path: Path) -> Path:
    db = tmp_path / "ui_test.sqlite3"
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


def test_active_config_path_default(tmp_path, monkeypatch):
    from crypto_llm_alpaca import _paths

    _paths.find_repo_root.cache_clear()
    monkeypatch.setattr(_paths, "find_repo_root", lambda: tmp_path)
    from crypto_llm_alpaca.ui import _runner as runner_mod

    monkeypatch.setattr(runner_mod, "find_repo_root", lambda: tmp_path)
    assert active_config_path() == str(tmp_path / "config" / "default.yaml")


def test_active_config_path_uses_overrides(tmp_path, monkeypatch):
    from crypto_llm_alpaca import _paths
    from crypto_llm_alpaca.ui import _runner as runner_mod

    _paths.find_repo_root.cache_clear()
    monkeypatch.setattr(_paths, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(runner_mod, "find_repo_root", lambda: tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "ui_overrides.yaml").write_text("# stub\n")
    assert active_config_path().endswith("ui_overrides.yaml")


def test_all_ui_pages_parse():
    import ast
    from pathlib import Path

    from crypto_llm_alpaca import ui

    pages_dir = Path(ui.__file__).parent / "pages"
    pyfiles = sorted(pages_dir.glob("*.py"))
    assert len(pyfiles) >= 7, f"expected 7+ pages, got {len(pyfiles)}"
    for p in pyfiles:
        ast.parse(p.read_text(encoding="utf-8"))


def test_run_cli_invokes_report(tmp_path):
    cfg = _write_min_config(tmp_path)
    result = run_cli(["init-db"], config=str(cfg), timeout=60)
    assert result.ok, result.stderr
    result = run_cli(["report"], config=str(cfg), timeout=60)
    assert result.ok, result.stderr
    payload = json.loads(result.stdout)
    assert "counts" in payload
    assert payload["counts"]["bars"] == 0
