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
    market_regime_mask,
    rolling_zscore,
    rsi,
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


# --- RSI -----------------------------------------------------------------------
def test_rsi_warmup_is_nan() -> None:
    close = pd.Series(range(1, 20), dtype="float64")
    r = rsi(close, length=14)
    assert r.iloc[:14].isna().all()
    assert not np.isnan(r.iloc[14])


def test_rsi_range_0_100() -> None:
    rng = np.random.default_rng(42)
    close = pd.Series(100.0 + rng.standard_normal(200).cumsum())
    r = rsi(close, length=14)
    valid = r.dropna()
    assert (valid >= 0.0).all() and (valid <= 100.0).all()


def test_rsi_only_gains_is_100() -> None:
    """単調増加 → avg_loss = 0 → RSI = 100。"""
    close = pd.Series(range(1, 50), dtype="float64")
    r = rsi(close, length=14)
    assert r.iloc[-1] == pytest.approx(100.0)


def test_rsi_only_losses_is_0() -> None:
    """単調減少 → avg_gain = 0 → RSI = 0。"""
    close = pd.Series(range(50, 1, -1), dtype="float64")
    r = rsi(close, length=14)
    assert r.iloc[-1] == pytest.approx(0.0)


def test_rsi_flat_is_nan() -> None:
    """完全フラット → avg_gain = avg_loss = 0 → NaN。"""
    close = pd.Series([100.0] * 20)
    r = rsi(close, length=14)
    assert r.iloc[-1] != r.iloc[-1]  # NaN


def test_rsi_rejects_bad_length() -> None:
    with pytest.raises(ValueError):
        rsi(pd.Series([1.0, 2.0]), length=0)


def test_rsi_is_causal() -> None:
    """truncation invariance: 系列を t で切り詰めても t の RSI 値が変わらない。"""
    rng = np.random.default_rng(7)
    close = pd.Series(100.0 + rng.standard_normal(40).cumsum())
    full = rsi(close, length=14)
    for t in range(14, len(close)):
        truncated = rsi(close.iloc[: t + 1], length=14)
        expected = full.iloc[t]
        actual = truncated.iloc[t]
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(expected, rel=1e-9), f"RSI が t={t} で未来依存"


def test_vwap_is_causal(ohlcv: pd.DataFrame) -> None:
    session = pd.Series(ohlcv.index.normalize(), index=ohlcv.index)
    full = vwap(ohlcv["close"], ohlcv["volume"], session=session)
    for t in range(len(ohlcv)):
        sub = ohlcv.iloc[: t + 1]
        sub_session = pd.Series(sub.index.normalize(), index=sub.index)
        truncated = vwap(sub["close"], sub["volume"], session=sub_session)
        assert truncated.iloc[t] == pytest.approx(full.iloc[t]), f"VWAP が t={t} で未来依存"


# --- market_regime_mask -------------------------------------------------------------
def _market_df(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"close": closes}, index=idx)


def test_regime_mask_range_market_allows_entry_when_not_inverted() -> None:
    # 一定値 → MA からの乖離が常に0 → レンジ判定 → invert=False で常に許可
    market = _market_df([100.0] * 30)
    target = market.index
    mask = market_regime_mask(market, target, ma_window=10, threshold=0.03, invert=False)
    assert bool(mask.iloc[-1])


def test_regime_mask_trending_market_blocks_entry_when_not_inverted() -> None:
    # 一貫して上昇 → 前日終値がMAより大きく乖離 → トレンド判定 → invert=False で不許可
    closes = [100.0 * (1.02**i) for i in range(30)]  # 複利2%/日の強いトレンド
    market = _market_df(closes)
    target = market.index
    mask = market_regime_mask(market, target, ma_window=10, threshold=0.03, invert=False)
    assert not bool(mask.iloc[-1])


def test_regime_mask_trending_market_allows_entry_when_inverted() -> None:
    closes = [100.0 * (1.02**i) for i in range(30)]
    market = _market_df(closes)
    target = market.index
    mask = market_regime_mask(market, target, ma_window=10, threshold=0.03, invert=True)
    assert bool(mask.iloc[-1])


def test_regime_mask_warmup_defaults_to_allowed() -> None:
    # MA計算に必要な本数が無い先頭付近はデータ不足 → 許可（True）にフォールバック
    market = _market_df([100.0, 101.0, 99.0])
    target = market.index
    mask = market_regime_mask(market, target, ma_window=60, threshold=0.03, invert=False)
    assert mask.iloc[0]


def test_regime_mask_no_lookahead() -> None:
    # 前日終値のみ参照するため、当日の急変が当日のマスクに影響しない
    closes = [100.0] * 20 + [200.0]  # 最終日だけ急騰
    market = _market_df(closes)
    target = market.index
    full = market_regime_mask(market, target, ma_window=10, threshold=0.03, invert=False)
    truncated = market_regime_mask(
        market.iloc[:-1], target[:-1], ma_window=10, threshold=0.03, invert=False
    )
    assert full.iloc[-2] == truncated.iloc[-1]


def test_regime_mask_ffill_for_missing_dates() -> None:
    # target_index に無い日は前方埋め（例：市場休場日の翌営業日でも直近の判定を引き継ぐ）
    market = _market_df([100.0 * (1.02**i) for i in range(30)])
    target = pd.date_range("2026-01-01", periods=32, freq="D")  # market に無い2日を含む
    mask = market_regime_mask(market, target, ma_window=10, threshold=0.03, invert=False)
    assert len(mask) == len(target)
    assert mask.iloc[-1] == mask.iloc[29]  # 最後の実データ日の値を引き継ぐ
