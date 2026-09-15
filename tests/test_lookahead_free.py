from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from crypto_llm_alpaca.defillama import _tvl_at, extract_defi_metrics_at
from crypto_llm_alpaca.storage import SQLiteStore


def test_tvl_at_returns_last_point_before_asof():
    series = [
        {"date": 1_700_000_000, "tvl": 10.0},
        {"date": 1_700_086_400, "tvl": 12.0},
        {"date": 1_700_172_800, "tvl": 15.0},
    ]
    assert _tvl_at(series, 1_699_999_999) is None
    assert _tvl_at(series, 1_700_000_000) == 10.0
    assert _tvl_at(series, 1_700_100_000) == 12.0
    assert _tvl_at(series, 9_999_999_999) == 15.0


def test_storage_snapshot_roundtrip(tmp_path):
    s = SQLiteStore(tmp_path / "x.sqlite3")
    s.init_schema()
    asof_a = datetime(2024, 6, 1, tzinfo=timezone.utc)
    asof_b = datetime(2025, 6, 1, tzinfo=timezone.utc)
    s.upsert_defi_snapshot("AAVE/USD", asof_a, {"tvl": 5e9, "tvl_growth": 0.05})
    s.upsert_defi_snapshot("AAVE/USD", asof_b, {"tvl": 14e9, "tvl_growth": 0.10})
    snaps = s.list_defi_snapshots(symbol="AAVE/USD")
    assert len(snaps) == 2
    assert snaps[0]["asof"] < snaps[1]["asof"]


def test_latest_snapshot_before_lookahead_free(tmp_path):
    """The whole point of Phase C: querying at 2024-06-15 must NOT see 2025-06-01 row."""
    s = SQLiteStore(tmp_path / "x.sqlite3")
    s.init_schema()
    s.upsert_defi_snapshot("AAVE/USD", datetime(2024, 6, 1, tzinfo=timezone.utc), {"tvl": 5e9, "tvl_growth": 0.05})
    s.upsert_defi_snapshot("AAVE/USD", datetime(2025, 6, 1, tzinfo=timezone.utc), {"tvl": 14e9, "tvl_growth": 0.10})

    # Query at 2024-06-15: should only see the June 2024 row.
    asof = datetime(2024, 6, 15, tzinfo=timezone.utc)
    snap = s.latest_defi_snapshot_before("AAVE/USD", asof)
    assert snap is not None
    assert snap["tvl"] == pytest.approx(5e9)
    assert snap["tvl_growth"] == pytest.approx(0.05)

    # Query at 2025-06-15: should now see the 2025 row.
    asof_later = datetime(2025, 6, 15, tzinfo=timezone.utc)
    snap_later = s.latest_defi_snapshot_before("AAVE/USD", asof_later)
    assert snap_later["tvl"] == pytest.approx(14e9)


def test_extract_defi_metrics_at_uses_store_first(tmp_path):
    """When store has a snapshot, extract should not hit the API."""
    s = SQLiteStore(tmp_path / "x.sqlite3")
    s.init_schema()
    asof = datetime(2024, 6, 15, tzinfo=timezone.utc)
    s.upsert_defi_snapshot("AAVE/USD", datetime(2024, 6, 1, tzinfo=timezone.utc), {"tvl": 5e9, "tvl_growth": 0.05})
    metrics = extract_defi_metrics_at("AAVE/USD", asof, store=s)
    assert metrics["tvl"] == pytest.approx(5e9)
    assert metrics["tvl_growth"] == pytest.approx(0.05)


def test_extract_defi_metrics_at_returns_empty_for_unknown_symbol(tmp_path):
    s = SQLiteStore(tmp_path / "x.sqlite3")
    s.init_schema()
    metrics = extract_defi_metrics_at("UNKNOWN/USD", datetime(2024, 6, 1, tzinfo=timezone.utc), store=s)
    # Either empty (unknown asset map) or has only slug
    assert "tvl" not in metrics or metrics.get("tvl") is None


def test_no_snapshot_returns_none(tmp_path):
    s = SQLiteStore(tmp_path / "x.sqlite3")
    s.init_schema()
    # No rows yet
    snap = s.latest_defi_snapshot_before("AAVE/USD", datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert snap is None
