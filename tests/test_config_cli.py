from pathlib import Path

from crypto_llm_alpaca.cli import main
from crypto_llm_alpaca.config import load_config


def test_load_default_config():
    cfg = load_config(Path("config/default.yaml"))
    # BTC dropped from active trade universe in Phase 1; first trade symbol is ETH.
    assert cfg.universe.symbols[0] == "ETH/USD"
    assert "BTC/USD" not in cfg.universe.symbols
    assert cfg.is_trade_enabled("BTC/USD") is False  # but kept as regime sensor
    assert cfg.risk.max_total_exposure_pct == 0.30
    assert cfg.risk.take_profit_levels[0].gain_pct == 0.10
    assert cfg.strategy.breakout_lookback == 20
    assert cfg.strategy.atr_stop_multiple == 2.0
    assert cfg.risk.risk_per_trade_pct == 0.01
    assert cfg.costs.default_bps == 25.0
    assert cfg.symbol_override("ETH/USD").take_profit_pct == 0.15
    assert cfg.quality_growth.min_profile_score == 0.60


def test_cli_init_db_and_demo_backtest(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    db_path = tmp_path / "demo.sqlite3"
    cfg_path.write_text(
        f'''
storage:
  path: "{db_path.as_posix()}"
llm:
  provider: "mock"
''',
        encoding="utf-8",
    )
    assert main(["--config", str(cfg_path), "init-db"]) == 0
    assert main(["report", "--config", str(cfg_path)]) == 0
    assert db_path.exists()
    assert main(["--config", str(cfg_path), "extract-signals"]) == 0
    assert main(["--config", str(cfg_path), "backfill", "--demo"]) == 0
    assert main(["--config", str(cfg_path), "backtest", "--demo"]) == 0
    assert main(["--config", str(cfg_path), "backtest", "--demo", "--strategy", "breakout-volume", "--persist"]) == 0
    assert main(["--config", str(cfg_path), "backtest", "--demo", "--strategy", "adaptive-breakout", "--persist"]) == 0
    assert main(["--config", str(cfg_path), "backtest", "--demo", "--strategy", "quality-growth", "--persist"]) == 0
    assert main(["--config", str(cfg_path), "build-features"]) == 0
    assert main(["--config", str(cfg_path), "rank-setups"]) == 0
    assert main(["--config", str(cfg_path), "quality-growth-candidates"]) == 0
    assert main(["--config", str(cfg_path), "quality-growth-candidates", "--explain"]) == 0
    assert main(["--config", str(cfg_path), "research-impact", "--json"]) == 0
    assert main(["--config", str(cfg_path), "paper-candidates"]) == 0
    assert main(["--config", str(cfg_path), "paper-once"]) == 0
    assert main(["--config", str(cfg_path), "report"]) == 0


def test_ui_home_entrypoint_exists():
    from crypto_llm_alpaca import ui

    home = Path(ui.__file__).parent / "Home.py"
    assert home.exists(), "multipage UI entry point should be ui/Home.py"
