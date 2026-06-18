"""websocket_client.py のテスト（純ロジック・ネットワーク非依存）。"""

from __future__ import annotations

import pandas as pd

from data.storage import Storage
from data.websocket_client import Bar, BarAggregator, KabuRecorder


# --- BarAggregator -----------------------------------------------------------------
def test_aggregator_builds_minute_bars() -> None:
    bars: list[Bar] = []
    agg = BarAggregator(on_bar=bars.append)
    # 09:00 台に3ティック、09:01 台に1ティック
    agg.add_tick("A", pd.Timestamp("2026-01-05 09:00:10"), 100.0, 5)
    agg.add_tick("A", pd.Timestamp("2026-01-05 09:00:30"), 102.0, 3)
    agg.add_tick("A", pd.Timestamp("2026-01-05 09:00:50"), 99.0, 2)
    agg.add_tick("A", pd.Timestamp("2026-01-05 09:01:05"), 101.0, 4)  # ここで09:00足が確定
    assert len(bars) == 1
    b = bars[0]
    assert b.ts == pd.Timestamp("2026-01-05 09:00:00")
    assert (b.open, b.high, b.low, b.close) == (100.0, 102.0, 99.0, 99.0)
    assert b.volume == 10  # 5+3+2

    agg.flush()
    assert len(bars) == 2
    assert bars[1].ts == pd.Timestamp("2026-01-05 09:01:00")
    assert bars[1].close == 101.0


def test_aggregator_skips_nonpositive_price() -> None:
    bars: list[Bar] = []
    agg = BarAggregator(on_bar=bars.append)
    agg.add_tick("A", pd.Timestamp("2026-01-05 09:00:10"), 0.0, 5)  # スキップ
    agg.flush()
    assert bars == []


def test_aggregator_multi_symbol_independent() -> None:
    bars: list[Bar] = []
    agg = BarAggregator(on_bar=bars.append)
    agg.add_tick("A", pd.Timestamp("2026-01-05 09:00:10"), 100.0, 1)
    agg.add_tick("B", pd.Timestamp("2026-01-05 09:00:10"), 200.0, 1)
    agg.flush()
    syms = {b.symbol for b in bars}
    assert syms == {"A", "B"}


# --- KabuRecorder ------------------------------------------------------------------
def _msg(symbol: str, t: str, price: float, cum_vol: float) -> dict[str, object]:
    return {
        "Symbol": symbol,
        "CurrentPrice": price,
        "CurrentPriceTime": f"2026-01-05T{t}+09:00",
        "TradingVolume": cum_vol,
    }


def test_recorder_computes_volume_delta_from_cumulative() -> None:
    rec = KabuRecorder()
    bars = rec.run(
        [
            _msg("1301", "09:00:10", 100.0, 100),  # 初回 → delta 0
            _msg("1301", "09:00:40", 101.0, 130),  # +30
            _msg("1301", "09:01:10", 102.0, 150),  # 09:00足確定（vol=30）、+20 は次足
        ]
    )
    assert len(bars) == 2
    first = bars[0]
    assert first.volume == 30  # 0 + 30
    assert (first.open, first.close) == (100.0, 101.0)
    assert bars[1].volume == 20


def test_recorder_skips_messages_without_price() -> None:
    rec = KabuRecorder()
    bars = rec.run(
        [
            {"Symbol": "1301", "CurrentPriceTime": "2026-01-05T09:00:10+09:00"},  # 価格なし
            _msg("1301", "09:00:20", 100.0, 10),
        ]
    )
    assert len(bars) == 1


def test_recorder_persists_to_storage() -> None:
    db = Storage(":memory:")
    rec = KabuRecorder(storage=db)
    rec.run(
        [
            _msg("1301", "09:00:10", 100.0, 10),
            _msg("1301", "09:01:10", 101.0, 20),
        ]
    )
    stored = db.get_bars("1301")
    assert len(stored) == 2
    assert isinstance(stored.index, pd.DatetimeIndex)
    db.close()
