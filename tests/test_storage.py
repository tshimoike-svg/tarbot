"""storage.py のテスト（インメモリ SQLite）。"""

from __future__ import annotations

import pandas as pd
import pytest

from data.storage import Storage
from execution.fill_monitor import FillMonitor, OrderIntent
from strategy.mean_reversion import Trade


@pytest.fixture
def db() -> Storage:
    return Storage(":memory:")


def _bars() -> pd.DataFrame:
    idx = pd.to_datetime(["2026-01-05 09:00", "2026-01-05 09:01"])
    return pd.DataFrame(
        {"open": [100, 101], "high": [101, 102], "low": [99, 100],
         "close": [100, 101], "volume": [10, 20]},
        index=idx, dtype="float64",
    )


# --- bars --------------------------------------------------------------------------
def test_insert_and_get_bars(db: Storage) -> None:
    db.insert_bars("1301", _bars())
    out = db.get_bars("1301")
    assert len(out) == 2
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.iloc[1]["close"] == 101


def test_bars_dedup_on_pk(db: Storage) -> None:
    db.insert_bars("1301", _bars())
    db.insert_bars("1301", _bars())  # 同じ (symbol, ts) は無視される
    assert len(db.get_bars("1301")) == 2


def test_get_bars_empty(db: Storage) -> None:
    out = db.get_bars("9999")
    assert out.empty
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_insert_bars_missing_column_raises(db: Storage) -> None:
    with pytest.raises(ValueError):
        db.insert_bars("1301", pd.DataFrame({"open": [1]}, index=pd.to_datetime(["2026-01-05"])))


# --- trades ------------------------------------------------------------------------
def test_insert_and_get_trade(db: Storage) -> None:
    tr = Trade(
        side="long",
        entry_time=pd.Timestamp("2026-01-05 09:30"),
        entry_price=1000.0,
        exit_time=pd.Timestamp("2026-01-05 10:00"),
        exit_price=1010.0,
        exit_reason="take_profit",
    )
    db.insert_trade("1301", tr, gross_return=0.01, cost=0.004, net_return=0.006)
    out = db.get_trades()
    assert len(out) == 1
    row = out.iloc[0]
    assert row["symbol"] == "1301"
    assert row["side"] == "long"
    assert row["net_return"] == pytest.approx(0.006)


# --- fills -------------------------------------------------------------------------
def test_insert_and_get_fill(db: Storage) -> None:
    mon = FillMonitor()
    mon.record_intent(
        OrderIntent(
            order_id="o1", symbol="1301", side="buy",
            limit_price=1000.0, shares=100, reference_price=1000.0,
            placed_at=pd.Timestamp("2026-01-05 09:30"),
        )
    )
    res = mon.record_fill("o1", fill_price=1000.2, filled_shares=100, filled_at=pd.Timestamp("2026-01-05 09:31"))
    db.insert_fill(res)
    out = db.get_fills()
    assert len(out) == 1
    assert out.iloc[0]["order_id"] == "o1"
    assert out.iloc[0]["status"] == "filled"
    assert out.iloc[0]["slippage_per_share"] == pytest.approx(0.2)


def test_context_manager_closes() -> None:
    with Storage(":memory:") as db:
        db.insert_bars("1301", _bars())
        assert len(db.get_bars("1301")) == 2


def test_file_backed_roundtrip(tmp_path: object) -> None:
    path = f"{tmp_path}/sub/forward.sqlite"  # type: ignore[str-bytes-safe]
    db = Storage(path)
    db.insert_bars("1301", _bars())
    db.close()
    # 再オープンしても残っている（永続）
    db2 = Storage(path)
    assert len(db2.get_bars("1301")) == 2
    db2.close()
