"""swing.py（共通エンジン）と swing_reversion / swing_momentum のテスト。"""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import SwingMomentumParams, SwingReversionParams
from strategy import swing_momentum, swing_reversion
from strategy.swing import walk_swing


def _df(opens: list[float], highs: list[float], lows: list[float], closes: list[float]) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-05", periods=len(closes))
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes}, index=idx, dtype="float64"
    )


def _series(vals: list[float], idx: pd.Index) -> pd.Series:
    return pd.Series(vals, index=idx, dtype="float64")


# --- walk_swing：エンジンの分岐 ----------------------------------------------------
def test_walk_long_target_exit() -> None:
    df = _df(
        opens=[100, 100, 100, 100],
        highs=[100, 101, 104, 104],
        lows=[100, 99, 100, 100],
        closes=[100, 100, 103, 103],
    )
    entries = _series([1, 0, 0, 0], df.index)  # d0 シグナル → d1 寄りで建て
    atr = _series([2, 2, 2, 2], df.index)
    target = _series([103, 103, 103, 103], df.index)
    [tr] = walk_swing(df, entries=entries, atr=atr, target=target, atr_stop_mult=1.0, max_holding_days=5)
    assert tr.side == "long"
    assert tr.entry_price == pytest.approx(100.0)  # d1 の寄り
    assert tr.exit_reason == "target"
    assert tr.exit_price == pytest.approx(103.0)
    assert tr.stop_price == pytest.approx(98.0)  # 100 - 1*2


def test_walk_long_stop_gap() -> None:
    df = _df(
        opens=[100, 100, 97, 100],   # d2 はギャップダウンでストップ(98)を割って寄る
        highs=[100, 101, 97.5, 100],
        lows=[100, 99, 96, 100],
        closes=[100, 100, 97, 100],
    )
    entries = _series([1, 0, 0, 0], df.index)
    atr = _series([2, 2, 2, 2], df.index)
    target = _series([110, 110, 110, 110], df.index)
    [tr] = walk_swing(df, entries=entries, atr=atr, target=target, atr_stop_mult=1.0, max_holding_days=5)
    assert tr.exit_reason == "stop"
    assert tr.exit_price == pytest.approx(97.0)  # ストップ98より不利な寄り値で約定（ギャップ）


def test_walk_time_stop() -> None:
    df = _df(
        opens=[100, 100, 100, 100],
        highs=[100, 101, 101, 101],
        lows=[100, 99, 99, 99],
        closes=[100, 100, 100, 100.5],
    )
    entries = _series([1, 0, 0, 0], df.index)
    atr = _series([2, 2, 2, 2], df.index)
    # max_holding_days=2 → d1,d2 保有して d2 終値でタイムストップ（目標なし）
    [tr] = walk_swing(df, entries=entries, atr=atr, target=None, atr_stop_mult=5.0, max_holding_days=2)
    assert tr.exit_reason == "time_stop"
    assert tr.exit_time == df.index[2]
    assert tr.exit_price == pytest.approx(100.0)  # d2 終値


def test_walk_short_target_exit() -> None:
    df = _df(
        opens=[100, 100, 100, 100],
        highs=[100, 101, 101, 101],
        lows=[100, 99, 96, 96],
        closes=[100, 100, 97, 97],
    )
    entries = _series([-1, 0, 0, 0], df.index)
    atr = _series([2, 2, 2, 2], df.index)
    target = _series([97, 97, 97, 97], df.index)
    [tr] = walk_swing(df, entries=entries, atr=atr, target=target, atr_stop_mult=2.0, max_holding_days=5)
    assert tr.side == "short"
    assert tr.exit_reason == "target"
    assert tr.exit_price == pytest.approx(97.0)
    assert tr.pnl_gross_per_share == pytest.approx(3.0)  # 100 → 97


def test_walk_skips_nan_atr_and_no_overlap() -> None:
    df = _df([100] * 6, [101] * 6, [99] * 6, [100] * 6)
    entries = _series([1, 0, 0, 1, 0, 0], df.index)
    atr = _series([float("nan"), 2, 2, 2, 2, 2], df.index)  # d0 の atr は NaN → スキップ
    # d0 はスキップされ、d3 のシグナルから1トレード（タイムストップ）
    trades = walk_swing(df, entries=entries, atr=atr, target=None, atr_stop_mult=5.0, max_holding_days=2)
    assert len(trades) == 1
    assert trades[0].entry_time == df.index[4]


# --- swing_reversion ---------------------------------------------------------------
def test_reversion_entry_fires_on_dip() -> None:
    closes = [100, 100, 100, 100, 100, 95, 100, 100]
    df = _df(closes, [c + 0.5 for c in closes], [c - 0.5 for c in closes], closes)
    p = SwingReversionParams(lookback=5, entry_z=1.5, atr_length=3, max_holding_days=5)
    sig = swing_reversion.compute_signals(df, p)
    assert (sig["entry"] == 1).any()  # 下振れでロングシグナル


def test_reversion_generates_trades() -> None:
    closes = [100, 100, 100, 100, 100, 95, 98, 100, 100, 100]
    df = _df(closes, [c + 0.5 for c in closes], [c - 0.5 for c in closes], closes)
    p = SwingReversionParams(lookback=5, entry_z=1.5, atr_length=3, atr_stop_mult=3.0, max_holding_days=5)
    trades = swing_reversion.generate_trades(df, p)
    assert len(trades) >= 1
    assert trades[0].side == "long"


# --- swing_momentum ----------------------------------------------------------------
def test_momentum_entry_on_breakout() -> None:
    closes = [10, 10, 10, 10, 15]  # d4 が直近3日高値を上抜け
    df = _df(closes, [c + 0.2 for c in closes], [c - 0.2 for c in closes], closes)
    p = SwingMomentumParams(breakout_lookback=3, atr_length=2, max_holding_days=3)
    sig = swing_momentum.compute_signals(df, p)
    assert sig["entry"].iloc[4] == 1


def test_momentum_requires_columns() -> None:
    with pytest.raises(ValueError):
        swing_momentum.compute_signals(pd.DataFrame({"close": [1, 2]}))


# --- 因果性 ------------------------------------------------------------------------
def test_reversion_signals_are_causal() -> None:
    closes = [100, 101, 99, 102, 98, 95, 100, 103, 100, 99]
    df = _df(closes, [c + 0.5 for c in closes], [c - 0.5 for c in closes], closes)
    p = SwingReversionParams(lookback=5, entry_z=1.5, atr_length=3)
    full = swing_reversion.compute_signals(df, p)
    for t in range(p.lookback, len(df)):
        sub = swing_reversion.compute_signals(df.iloc[: t + 1], p)
        a, b = sub["zscore"].iloc[t], full["zscore"].iloc[t]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == pytest.approx(b), f"zscore が t={t} で未来依存"
