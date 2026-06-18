"""indicators.py のテスト。

主眼は2つ：
1. 各指標の数値が手計算と一致すること。
2. **因果性（ルックアヘッド・バイアス回避）**：時点 t の指標値は、t より後の
   データに依存しない。系列を t で切り詰めても t の値が変わらないことを機械的に確認する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.indicators import (
    atr,
    bollinger_bands,
    rolling_zscore,
    true_range,
    typical_price,
    vwap,
)


# --- フィクスチャ ------------------------------------------------------------------
@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """JST 分足を模した 2 日分の小さな OHLCV。"""
    idx = pd.to_datetime(
        [
            "2026-01-05 09:00",
            "2026-01-05 09:01",
            "2026-01-05 09:02",
            "2026-01-05 09:03",
            "2026-01-06 09:00",
            "2026-01-06 09:01",
            "2026-01-06 09:02",
            "2026-01-06 09:03",
        ]
    )
    return pd.DataFrame(
        {
            "high": [101, 102, 103, 102, 201, 202, 203, 202],
            "low": [99, 100, 101, 100, 199, 200, 201, 200],
            "close": [100, 101, 102, 101, 200, 201, 202, 201],
            "volume": [10, 20, 30, 40, 10, 20, 30, 40],
        },
        index=idx,
        dtype="float64",
    )


# --- typical_price ----------------------------------------------------------------
def test_typical_price(ohlcv: pd.DataFrame) -> None:
    tp = typical_price(ohlcv["high"], ohlcv["low"], ohlcv["close"])
    # (101+99+100)/3 = 100
    assert tp.iloc[0] == pytest.approx(100.0)
    assert tp.iloc[2] == pytest.approx((103 + 101 + 102) / 3)


# --- VWAP -------------------------------------------------------------------------
def test_vwap_no_reset_cumulative(ohlcv: pd.DataFrame) -> None:
    v = vwap(ohlcv["close"], ohlcv["volume"])
    # 1本目は close と一致
    assert v.iloc[0] == pytest.approx(100.0)
    # 2本目 = (100*10 + 101*20) / 30
    assert v.iloc[1] == pytest.approx((100 * 10 + 101 * 20) / 30)


def test_vwap_daily_reset(ohlcv: pd.DataFrame) -> None:
    session = ohlcv.index.normalize()
    v = vwap(ohlcv["close"], ohlcv["volume"], session=pd.Series(session, index=ohlcv.index))
    # 2日目の先頭（index 4）はその日の最初なので close と一致＝リセットされている
    assert v.iloc[4] == pytest.approx(200.0)
    # 2日目2本目 = (200*10 + 201*20)/30、1日目の値を引きずらない
    assert v.iloc[5] == pytest.approx((200 * 10 + 201 * 20) / 30)


def test_vwap_zero_volume_is_nan() -> None:
    s = pd.Series([100.0, 101.0])
    vol = pd.Series([0.0, 0.0])
    v = vwap(s, vol)
    assert v.isna().all()


def test_vwap_index_mismatch_raises() -> None:
    a = pd.Series([1.0, 2.0], index=[0, 1])
    b = pd.Series([1.0, 2.0], index=[1, 2])
    with pytest.raises(ValueError):
        vwap(a, b)


# --- True Range / ATR -------------------------------------------------------------
def test_true_range_first_bar_is_high_low(ohlcv: pd.DataFrame) -> None:
    tr = true_range(ohlcv["high"], ohlcv["low"], ohlcv["close"])
    assert tr.iloc[0] == pytest.approx(101 - 99)  # 前終値なし → High-Low = 2


def test_true_range_uses_prev_close(ohlcv: pd.DataFrame) -> None:
    tr = true_range(ohlcv["high"], ohlcv["low"], ohlcv["close"])
    # index4: High=201, Low=199, prev_close=101 → max(2, |201-101|, |199-101|)=100
    assert tr.iloc[4] == pytest.approx(100.0)


def test_atr_warmup_is_nan(ohlcv: pd.DataFrame) -> None:
    a = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=3)
    assert a.iloc[:2].isna().all()
    assert not np.isnan(a.iloc[2])


def test_atr_sma_matches_manual(ohlcv: pd.DataFrame) -> None:
    tr = true_range(ohlcv["high"], ohlcv["low"], ohlcv["close"])
    a = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=3, method="sma")
    assert a.iloc[2] == pytest.approx(tr.iloc[:3].mean())


def test_atr_wilder_recursive_relation(ohlcv: pd.DataFrame) -> None:
    n = 3
    tr = true_range(ohlcv["high"], ohlcv["low"], ohlcv["close"])
    a = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=n, method="wilder")
    # warmup 後は Wilder の再帰式が成り立つ：ATR_t = (ATR_{t-1}*(n-1) + TR_t)/n
    for t in range(3, len(tr)):
        expected = (a.iloc[t - 1] * (n - 1) + tr.iloc[t]) / n
        assert a.iloc[t] == pytest.approx(expected)


def test_atr_rejects_bad_length(ohlcv: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=0)


# --- ボリンジャーバンド ------------------------------------------------------------
def test_bollinger_constant_series_zero_width() -> None:
    close = pd.Series([50.0] * 10)
    bb = bollinger_bands(close, length=5, num_std=2.0)
    last = bb.iloc[-1]
    assert last["middle"] == pytest.approx(50.0)
    assert last["upper"] == pytest.approx(50.0)
    assert last["lower"] == pytest.approx(50.0)
    assert last["bandwidth"] == pytest.approx(0.0)
    # 幅0のとき percent_b は NaN（0除算回避）
    assert np.isnan(last["percent_b"])


def test_bollinger_matches_manual() -> None:
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    bb = bollinger_bands(close, length=5, num_std=2.0, ddof=0)
    last = bb.iloc[-1]
    assert last["middle"] == pytest.approx(3.0)
    pop_std = np.std([1, 2, 3, 4, 5])  # 母標準偏差 ≈ 1.4142
    assert last["upper"] == pytest.approx(3.0 + 2.0 * pop_std)
    assert last["lower"] == pytest.approx(3.0 - 2.0 * pop_std)


# --- ローリング z スコア -----------------------------------------------------------
def test_rolling_zscore_basic() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = rolling_zscore(s, length=5, ddof=0)
    # 最終点：x=5, mean=3, pop_std=1.4142 → (5-3)/1.4142
    assert z.iloc[-1] == pytest.approx((5 - 3) / np.std([1, 2, 3, 4, 5]))


def test_rolling_zscore_constant_is_nan() -> None:
    s = pd.Series([7.0] * 6)
    z = rolling_zscore(s, length=3)
    assert z.iloc[-1] != z.iloc[-1] or np.isnan(z.iloc[-1])  # NaN


# --- 因果性（ルックアヘッド・バイアス回避）の機械的検証 ----------------------------
# 系列を時点 t で切り詰めても、t における指標値は変わってはいけない。
def test_atr_is_causal(ohlcv: pd.DataFrame) -> None:
    full = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=3)
    for t in range(2, len(ohlcv)):
        sub = ohlcv.iloc[: t + 1]
        truncated = atr(sub["high"], sub["low"], sub["close"], length=3)
        assert truncated.iloc[t] == pytest.approx(full.iloc[t]), f"ATR が t={t} で未来依存"


def test_bollinger_is_causal(ohlcv: pd.DataFrame) -> None:
    full = bollinger_bands(ohlcv["close"], length=3)
    for t in range(2, len(ohlcv)):
        sub = ohlcv["close"].iloc[: t + 1]
        truncated = bollinger_bands(sub, length=3)
        for col in ("middle", "upper", "lower"):
            a, b = truncated.iloc[t][col], full.iloc[t][col]
            assert a == pytest.approx(b), f"Bollinger.{col} が t={t} で未来依存"


def test_vwap_is_causal(ohlcv: pd.DataFrame) -> None:
    session = pd.Series(ohlcv.index.normalize(), index=ohlcv.index)
    full = vwap(ohlcv["close"], ohlcv["volume"], session=session)
    for t in range(len(ohlcv)):
        sub = ohlcv.iloc[: t + 1]
        sub_session = pd.Series(sub.index.normalize(), index=sub.index)
        truncated = vwap(sub["close"], sub["volume"], session=sub_session)
        assert truncated.iloc[t] == pytest.approx(full.iloc[t]), f"VWAP が t={t} で未来依存"
