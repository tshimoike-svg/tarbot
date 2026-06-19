"""swing_reversion.py の季節・RSI・出来高フィルタのテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import SwingReversionParams
from strategy.swing_reversion import compute_signals


def _make_ohlcv(n: int = 60, *, volume: bool = True) -> pd.DataFrame:
    """単調減少→反転の日足ダミー（平均回帰エントリーが発生しやすい）。"""
    idx = pd.bdate_range("2024-01-02", periods=n)
    # 前半急落 → 後半反発
    closes = [100.0 - i * 0.5 for i in range(n // 2)] + [
        70.0 + i * 0.5 for i in range(n - n // 2)
    ]
    df: dict[str, list[float]] = {
        "open":  [c + 0.1 for c in closes],
        "high":  [c + 0.5 for c in closes],
        "low":   [c - 0.5 for c in closes],
        "close": closes,
    }
    if volume:
        df["volume"] = [float(1_000_000)] * n
    return pd.DataFrame(df, index=idx, dtype="float64")


# --- 季節フィルタ -------------------------------------------------------------------

def test_season_filter_blocks_avoided_months() -> None:
    """season_avoid_months に含まれる月はエントリーが 0 になる。"""
    df = _make_ohlcv(120)
    params = SwingReversionParams(
        lookback=20, entry_z=1.5,
        season_avoid_months=frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
    )
    sig = compute_signals(df, params)
    assert (sig["entry"] == 0).all(), "全月回避 → エントリーゼロであるべき"


def test_season_filter_passes_allowed_months() -> None:
    """season_avoid_months が空なら月フィルタは掛からない（エントリーが存在する）。"""
    df = _make_ohlcv(120)
    params = SwingReversionParams(lookback=20, entry_z=1.5)
    sig = compute_signals(df, params)
    assert (sig["entry"] != 0).any(), "フィルタなし → エントリーが存在するべき"


def test_season_filter_only_blocks_specified_months() -> None:
    """1月のみ回避 → 1月の行にエントリーがない、他はある可能性がある。"""
    df = _make_ohlcv(120)
    params = SwingReversionParams(lookback=20, entry_z=1.5, season_avoid_months=frozenset({1}))
    sig = compute_signals(df, params)
    jan_mask = pd.DatetimeIndex(df.index).month == 1
    assert (sig["entry"][jan_mask] == 0).all()


# --- RSI フィルタ ------------------------------------------------------------------

def test_rsi_filter_long_blocked_when_rsi_too_high() -> None:
    """RSI > rsi_entry_max の時はロングエントリーをブロック。"""
    df = _make_ohlcv(80)
    # rsi_entry_max=0 → RSI < 0 は絶対成立しない → ロングゼロ
    params_block = SwingReversionParams(
        lookback=20, entry_z=1.5,
        rsi_length=14, rsi_entry_max=0.0, allow_long=True, allow_short=False,
    )
    sig = compute_signals(df, params_block)
    assert (sig["entry"] <= 0).all(), "rsi_entry_max=0 → ロングなし"


def test_rsi_filter_disabled_when_100() -> None:
    """rsi_entry_max=100 → RSI フィルタ無効（デフォルト）。"""
    df = _make_ohlcv(80)
    params_on  = SwingReversionParams(lookback=20, entry_z=1.5, rsi_length=14, rsi_entry_max=100.0)
    params_off = SwingReversionParams(lookback=20, entry_z=1.5)
    assert (compute_signals(df, params_on)["entry"] == compute_signals(df, params_off)["entry"]).all()


def test_rsi_filter_short_blocked_when_rsi_too_low() -> None:
    """RSI < rsi_entry_min の時はショートエントリーをブロック。"""
    df = _make_ohlcv(80)
    # rsi_entry_min=100 → RSI > 100 は絶対成立しない → ショートゼロ
    params_block = SwingReversionParams(
        lookback=20, entry_z=1.5,
        rsi_length=14, rsi_entry_min=100.0, allow_long=False, allow_short=True,
    )
    sig = compute_signals(df, params_block)
    assert (sig["entry"] >= 0).all(), "rsi_entry_min=100 → ショートなし"


# --- 出来高フィルタ -----------------------------------------------------------------

def test_volume_filter_blocks_low_volume() -> None:
    """volume_ratio_min=100 → 出来高比 > 100 倍は無理 → エントリーゼロ。"""
    df = _make_ohlcv(80)
    params = SwingReversionParams(lookback=20, entry_z=1.5, volume_ratio_min=100.0)
    sig = compute_signals(df, params)
    assert (sig["entry"] == 0).all()


def test_volume_filter_skipped_without_volume_column() -> None:
    """volume 列がない場合は出来高フィルタをスキップして通常シグナルを返す。"""
    df_no_vol = _make_ohlcv(80, volume=False)
    df_with_vol = _make_ohlcv(80, volume=True)
    params = SwingReversionParams(lookback=20, entry_z=1.5, volume_ratio_min=1.0)
    sig_no  = compute_signals(df_no_vol, params)
    sig_off = compute_signals(df_with_vol, SwingReversionParams(lookback=20, entry_z=1.5))
    assert (sig_no["entry"] == sig_off["entry"]).all()


def test_volume_filter_disabled_when_zero() -> None:
    """volume_ratio_min=0 → フィルタ無効。デフォルトと同じ結果。"""
    df = _make_ohlcv(80)
    params_on  = SwingReversionParams(lookback=20, entry_z=1.5, volume_ratio_min=0.0)
    params_off = SwingReversionParams(lookback=20, entry_z=1.5)
    assert (compute_signals(df, params_on)["entry"] == compute_signals(df, params_off)["entry"]).all()


# --- 複合フィルタ -------------------------------------------------------------------

def test_combined_filters_reduce_signals() -> None:
    """複合フィルタはベースラインより少ないシグナルを返す（ゼロ以上）。"""
    df = _make_ohlcv(120)
    base = compute_signals(df, SwingReversionParams(lookback=20, entry_z=1.5))
    filtered = compute_signals(df, SwingReversionParams(
        lookback=20, entry_z=1.5,
        season_avoid_months=frozenset({7, 8, 9, 10, 11, 12}),
        rsi_length=14, rsi_entry_max=40.0, rsi_entry_min=60.0,
        volume_ratio_min=1.3,
    ))
    assert (filtered["entry"].abs() <= base["entry"].abs()).all()
    # 複合フィルタはサブセット（1を0に変えるだけで0を1には変えない）
    assert (filtered["entry"] != 0).sum() <= (base["entry"] != 0).sum()


# --- params バリデーション ---------------------------------------------------------

def test_invalid_season_month_raises() -> None:
    with pytest.raises(ValueError):
        SwingReversionParams(season_avoid_months=frozenset({0}))


def test_invalid_rsi_entry_max_raises() -> None:
    with pytest.raises(ValueError):
        SwingReversionParams(rsi_entry_max=150.0)


def test_invalid_rsi_entry_min_raises() -> None:
    with pytest.raises(ValueError):
        SwingReversionParams(rsi_entry_min=-5.0)
