"""fill_monitor.py のテスト（API非依存・純ロジック）。

実測の積み上げが正しく集計されること：約定率・部分約定・不約定、滑りの符号（不利＝正）、
bps / tick 換算、未登録IDや重複のガードを固定する。
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from execution.fill_monitor import FillMonitor, OrderIntent, aggregate_fills


def _intent(
    oid: str,
    side: str,
    *,
    limit: float,
    ref: float,
    shares: int = 100,
    t: str = "2026-01-05 09:30",
) -> OrderIntent:
    return OrderIntent(
        order_id=oid,
        symbol="1301",
        side=side,  # type: ignore[arg-type]
        limit_price=limit,
        shares=shares,
        reference_price=ref,
        placed_at=pd.Timestamp(t),
    )


# --- 滑りの符号と換算 --------------------------------------------------------------
def test_buy_below_reference_is_favorable() -> None:
    mon = FillMonitor()
    mon.record_intent(_intent("1", "buy", limit=99.9, ref=100.0))
    res = mon.record_fill("1", fill_price=99.9, filled_shares=100, filled_at=pd.Timestamp("2026-01-05 09:31"))
    # 参照100より安く買えた → 滑りは負（有利）
    assert res.slippage_per_share == pytest.approx(-0.1)
    assert res.slippage_bps == pytest.approx(-0.1 / 100.0 * 1e4)
    assert res.slippage_ticks == pytest.approx(-0.1)  # price 100 → tick 1円
    assert res.status == "filled"
    assert res.fill_ratio == pytest.approx(1.0)


def test_buy_above_reference_is_adverse() -> None:
    mon = FillMonitor()
    mon.record_intent(_intent("1", "buy", limit=100.5, ref=100.0))
    res = mon.record_fill("1", fill_price=100.2, filled_shares=100, filled_at=pd.Timestamp("2026-01-05 09:31"))
    assert res.slippage_per_share == pytest.approx(0.2)  # 高く買った＝不利＝正


def test_sell_below_reference_is_adverse() -> None:
    mon = FillMonitor()
    mon.record_intent(_intent("1", "sell", limit=99.5, ref=100.0))
    res = mon.record_fill("1", fill_price=99.7, filled_shares=100, filled_at=pd.Timestamp("2026-01-05 09:31"))
    # 参照100より安く売った → 不利＝正
    assert res.slippage_per_share == pytest.approx(0.3)


# --- 部分約定・不約定 --------------------------------------------------------------
def test_partial_fill() -> None:
    mon = FillMonitor()
    mon.record_intent(_intent("1", "buy", limit=100, ref=100, shares=100))
    res = mon.record_fill("1", fill_price=100, filled_shares=40, filled_at=pd.Timestamp("2026-01-05 09:31"))
    assert res.status == "partial"
    assert res.fill_ratio == pytest.approx(0.4)


def test_unfilled() -> None:
    mon = FillMonitor()
    mon.record_intent(_intent("1", "buy", limit=99.0, ref=100.0))
    res = mon.record_unfilled("1")
    assert res.status == "unfilled"
    assert res.fill_price is None
    assert res.slippage_per_share is None
    assert res.slippage_bps is None
    assert res.fill_ratio == 0.0


# --- ガード ------------------------------------------------------------------------
def test_duplicate_intent_raises() -> None:
    mon = FillMonitor()
    mon.record_intent(_intent("1", "buy", limit=100, ref=100))
    with pytest.raises(ValueError):
        mon.record_intent(_intent("1", "buy", limit=100, ref=100))


def test_fill_unknown_order_raises() -> None:
    mon = FillMonitor()
    with pytest.raises(ValueError):
        mon.record_fill("x", fill_price=100, filled_shares=10, filled_at=pd.Timestamp("2026-01-05 09:31"))


def test_overfill_raises() -> None:
    mon = FillMonitor()
    mon.record_intent(_intent("1", "buy", limit=100, ref=100, shares=100))
    with pytest.raises(ValueError):
        mon.record_fill("1", fill_price=100, filled_shares=101, filled_at=pd.Timestamp("2026-01-05 09:31"))


def test_intent_validation() -> None:
    with pytest.raises(ValueError):
        _intent("1", "buy", limit=100, ref=100, shares=0)
    with pytest.raises(ValueError):
        _intent("1", "buy", limit=-1, ref=100)


# --- 集計・pending -----------------------------------------------------------------
def test_stats_aggregate() -> None:
    mon = FillMonitor()
    mon.record_intent(_intent("1", "buy", limit=100, ref=100, shares=100))
    mon.record_intent(_intent("2", "buy", limit=100, ref=100, shares=100))
    mon.record_intent(_intent("3", "buy", limit=99, ref=100, shares=100))
    mon.record_fill("1", fill_price=100.2, filled_shares=100, filled_at=pd.Timestamp("2026-01-05 09:31"))
    mon.record_fill("2", fill_price=100.0, filled_shares=50, filled_at=pd.Timestamp("2026-01-05 09:31"))
    mon.record_unfilled("3")

    stats = mon.stats()
    assert stats.n_orders == 3
    assert stats.n_filled == 2                  # filled + partial
    assert stats.fill_rate == pytest.approx(2 / 3)
    assert stats.unfilled_rate == pytest.approx(1 / 3)
    # 出来高ベース：(100+50+0)/(300) = 0.5
    assert stats.share_fill_rate == pytest.approx(0.5)
    # 滑りは約定分のみ平均：(0.2 + 0.0)/2 = 0.1
    assert stats.avg_slippage_per_share == pytest.approx(0.1)
    assert stats.by_status == {"filled": 1, "partial": 1, "unfilled": 1}


def test_pending_tracks_open_orders() -> None:
    mon = FillMonitor()
    mon.record_intent(_intent("1", "buy", limit=100, ref=100))
    mon.record_intent(_intent("2", "buy", limit=100, ref=100))
    mon.record_fill("1", fill_price=100, filled_shares=100, filled_at=pd.Timestamp("2026-01-05 09:31"))
    assert mon.pending() == ["2"]


def test_empty_aggregate_is_nan() -> None:
    stats = aggregate_fills([])
    assert stats.n_orders == 0
    assert math.isnan(stats.fill_rate)
    assert math.isnan(stats.avg_slippage_ticks)
