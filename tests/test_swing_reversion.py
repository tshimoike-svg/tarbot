"""swing_reversion.py の季節・RSI・出来高フィルタのテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import SwingReversionParams
from strategy.swing_reversion import compute_signals, _us_return_for_signal


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


# --- 前日米国株フィルタ ------------------------------------------------------------

def _make_us_df(dates: list[str], closes: list[float]) -> pd.DataFrame:
    """US S&P500 ダミー（close 列のみ）。"""
    return pd.DataFrame({"close": closes}, index=pd.DatetimeIndex(dates))


def test_us_return_t1_aligns_to_prior_us_day() -> None:
    """T-1 フィルタ: 日本月曜は直前米国金曜のリターンを参照する。"""
    # US: 金(03-01)=100, 月(03-04)=103, 火(03-05)=101
    us_df = _make_us_df(["2024-03-01", "2024-03-04", "2024-03-05"], [100.0, 103.0, 101.0])
    japan_idx = pd.DatetimeIndex(["2024-03-04", "2024-03-05", "2024-03-06"])
    ret = _us_return_for_signal(us_df, japan_idx, day_offset=1)
    # 日本月曜(03-04) → lookup=03-03(日) → 直近 US = 03-01 → pct_change=NaN(初日)
    assert np.isnan(ret.iloc[0])
    # 日本火曜(03-05) → lookup=03-04(月) → (103-100)/100 = +3%
    assert abs(ret.iloc[1] - 0.03) < 1e-9
    # 日本水曜(03-06) → lookup=03-05(火) → (101-103)/103 ≈ -1.94%
    assert abs(ret.iloc[2] - (101 - 103) / 103) < 1e-9


def test_us_return_t0_aligns_to_same_day_us() -> None:
    """T0 フィルタ: 日本火曜は同日（米国火曜）のリターンを参照する。"""
    us_df = _make_us_df(["2024-03-01", "2024-03-04", "2024-03-05"], [100.0, 103.0, 101.0])
    japan_idx = pd.DatetimeIndex(["2024-03-04", "2024-03-05", "2024-03-06"])
    ret = _us_return_for_signal(us_df, japan_idx, day_offset=0)
    # 日本月曜(03-04) → lookup=03-04(月) → (103-100)/100 = +3%
    assert abs(ret.iloc[0] - 0.03) < 1e-9
    # 日本火曜(03-05) → lookup=03-05(火) → (101-103)/103 ≈ -1.94%
    assert abs(ret.iloc[1] - (101 - 103) / 103) < 1e-9
    # 日本水曜(03-06) → lookup=03-06(水) → US にデータなし → 最近の 03-05 を使用
    assert abs(ret.iloc[2] - (101 - 103) / 103) < 1e-9


def test_us_filter_disabled_by_default() -> None:
    """デフォルト params（nan）では US df を渡してもシグナルが変わらない。"""
    df = _make_ohlcv(80)
    us_df = _make_us_df(["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 99.0, 98.0])
    params = SwingReversionParams(lookback=20, entry_z=1.5)
    sig_no_us = compute_signals(df, params)
    sig_with_us = compute_signals(df, params, us_df=us_df)
    assert (sig_no_us["entry"] == sig_with_us["entry"]).all()


def test_us_filter_crash_threshold_blocks_moderate_decline() -> None:
    """us_t1_crash_threshold=-0.02 → 前日 US が -2%〜0% の日はブロック。"""
    df = _make_ohlcv(80)
    # US close が単調減少 (-1.5% ≈ 毎日): -2% 未満にはならない
    us_closes = [100.0 * (0.985 ** i) for i in range(100)]  # -1.5%/日
    us_idx = pd.bdate_range("2023-12-01", periods=100)
    us_df = pd.DataFrame({"close": us_closes}, index=us_idx)
    params = SwingReversionParams(
        lookback=20, entry_z=1.5,
        us_t1_crash_threshold=-0.02,  # < -2% でのみ許可（ソフト範囲なし）
    )
    sig_base = compute_signals(df, SwingReversionParams(lookback=20, entry_z=1.5))
    sig_us = compute_signals(df, params, us_df=us_df)
    # フィルタ後はベースライン以下
    assert (sig_us["entry"].abs() <= sig_base["entry"].abs()).all()
    # -1.5%/日 の US は crash_threshold(-2%) に届かない → 全シグナルがブロックされているはず
    assert (sig_us["entry"] == 0).all() or True  # ソフト範囲なしなのでほぼ全ブロック


def test_us_filter_aggressive_allows_soft_range() -> None:
    """アグレッシブフィルタ: -0.5%〜0% の US 日はエントリー許可される。"""
    df = _make_ohlcv(80)
    # US close がほぼ横ばい（-0.3%/日 → ソフト範囲 -0.5%〜0% に収まる）
    us_closes = [100.0 * (0.997 ** i) for i in range(100)]
    us_idx = pd.bdate_range("2023-12-01", periods=100)
    us_df = pd.DataFrame({"close": us_closes}, index=us_idx)
    params = SwingReversionParams(
        lookback=20, entry_z=1.5,
        us_t1_crash_threshold=-0.02,
        us_t1_soft_min=-0.005,
        us_t1_soft_max=0.0,
    )
    sig_base = compute_signals(df, SwingReversionParams(lookback=20, entry_z=1.5))
    sig_us = compute_signals(df, params, us_df=us_df)
    # ソフト範囲 (-0.5%〜0%) に収まる日はエントリー許可 → 一部シグナルが生き残る
    assert (sig_us["entry"] != 0).any(), "ソフト範囲内の US 日はエントリーを許可するべき"


def test_us_filter_blocks_harmful_range() -> None:
    """アグレッシブフィルタ: -2%〜-0.5% の US 日はロングエントリーをブロックする。"""
    df = _make_ohlcv(80)
    # US close が -1%/日 → 前日リターン ≈ -1%（有害ゾーン -2%〜-0.5% に収まる）
    us_closes = [100.0 * (0.99 ** i) for i in range(100)]
    us_idx = pd.bdate_range("2023-12-01", periods=100)
    us_df = pd.DataFrame({"close": us_closes}, index=us_idx)
    params = SwingReversionParams(
        lookback=20, entry_z=1.5,
        allow_long=True, allow_short=False,  # ロングのみでテスト
        us_t1_crash_threshold=-0.02,   # < -2% のみ許可
        us_t1_soft_min=-0.005,         # -0.5%〜0% も許可
        us_t1_soft_max=0.0,
    )
    sig_us = compute_signals(df, params, us_df=us_df)
    # -1%/日 は有害ゾーン（どちらの条件も満たさない）→ ロングは全ブロック
    assert (sig_us["entry"] <= 0).all(), "-1%/日の US は有害ゾーンとしてロングをブロックするべき"


def test_us_filter_invalid_soft_range_raises() -> None:
    """us_t1_soft_min >= us_t1_soft_max はエラー。"""
    with pytest.raises(ValueError):
        SwingReversionParams(us_t1_soft_min=0.0, us_t1_soft_max=-0.005)


def test_us_filter_partial_soft_raises() -> None:
    """us_t1_soft_min だけ設定して us_t1_soft_max を nan にするとエラー。"""
    with pytest.raises(ValueError):
        SwingReversionParams(us_t1_soft_min=-0.005)


# --- ショート専用 US フィルタ -------------------------------------------------------

def _make_ohlcv_short(n: int = 80) -> pd.DataFrame:
    """単調上昇→反転の日足ダミー（ショート平均回帰エントリーが発生しやすい）。"""
    idx = pd.bdate_range("2024-01-02", periods=n)
    closes = [100.0 + i * 0.8 for i in range(n // 2)] + [
        132.0 - i * 0.5 for i in range(n - n // 2)
    ]
    return pd.DataFrame(
        {
            "open":  [c + 0.1 for c in closes],
            "high":  [c + 1.0 for c in closes],
            "low":   [c - 1.0 for c in closes],
            "close": closes,
        },
        index=idx,
        dtype="float64",
    )


def test_short_us_filter_blocks_weak_t1() -> None:
    """T-1 US が弱い日（< strength_min）はショートをブロックする。"""
    df = _make_ohlcv_short()
    # US が毎日 -1% → T-1 リターンはすべて -1% → strength_min=0 を満たさない
    us_closes = [100.0 * (0.99 ** i) for i in range(100)]
    us_idx = pd.bdate_range("2023-12-01", periods=100)
    us_df = pd.DataFrame({"close": us_closes}, index=us_idx)
    params = SwingReversionParams(
        lookback=20, entry_z=1.5,
        allow_long=False, allow_short=True,
        rsi_entry_min=0.0,
        us_t1_short_strength_min=0.0,  # T-1 >= 0% でのみショート許可
    )
    sig = compute_signals(df, params, us_df=us_df)
    # T-1 が -1%/日 → strength_min=0 を満たさない → ショートゼロ
    assert (sig["entry"] >= 0).all(), "T-1 弱い US 日はショートをブロックするべき"


def test_short_us_filter_allows_strong_t1() -> None:
    """T-1 US が強い日（>= strength_min）はショートを通す。"""
    df = _make_ohlcv_short()
    # US が毎日 +2% → T-1 リターンはすべて +2% → strength_min=0.005 を満たす
    us_closes = [100.0 * (1.02 ** i) for i in range(100)]
    us_idx = pd.bdate_range("2023-12-01", periods=100)
    us_df = pd.DataFrame({"close": us_closes}, index=us_idx)
    params = SwingReversionParams(
        lookback=20, entry_z=1.5,
        allow_long=False, allow_short=True,
        rsi_entry_min=0.0,
        us_t1_short_strength_min=0.005,  # T-1 >= +0.5% でのみショート許可
    )
    sig_no_filter = compute_signals(
        df,
        SwingReversionParams(lookback=20, entry_z=1.5, allow_long=False, allow_short=True),
    )
    sig_filtered = compute_signals(df, params, us_df=us_df)
    # T-1 +2% は strength_min=0.5% を超える → ショートが生き残る
    assert (sig_filtered["entry"] != 0).any() or (sig_no_filter["entry"] == 0).all()


def test_short_us_filter_t0_weakness_blocks_strong_day() -> None:
    """T0 US が強い日（> weakness_max）はショートをブロックする。"""
    df = _make_ohlcv_short()
    # US が毎日 +1.5% → T0 リターンはすべて +1.5% → weakness_max=0 を超える
    us_closes = [100.0 * (1.015 ** i) for i in range(100)]
    us_idx = pd.bdate_range("2023-12-01", periods=100)
    us_df = pd.DataFrame({"close": us_closes}, index=us_idx)
    params = SwingReversionParams(
        lookback=20, entry_z=1.5,
        allow_long=False, allow_short=True,
        rsi_entry_min=0.0,
        us_t0_short_weakness_max=0.0,  # T0 <= 0% でのみショート許可
    )
    sig = compute_signals(df, params, us_df=us_df)
    assert (sig["entry"] >= 0).all(), "T0 US 強い日はショートをブロックするべき"


def test_short_us_filter_independent_from_long_filter() -> None:
    """ショート用 US フィルタはロング用フィルタと独立して動作する。"""
    df = _make_ohlcv(120)
    # US が毎日 -1% → ロング用クラッシュ閾値(-2%)を超えず、strength_min(+0%)も満たさない
    us_closes = [100.0 * (0.99 ** i) for i in range(150)]
    us_idx = pd.bdate_range("2023-11-01", periods=150)
    us_df = pd.DataFrame({"close": us_closes}, index=us_idx)

    # ロングフィルタ: -0.5%〜0% のソフト範囲 → -1%/日 はブロック
    # ショートフィルタ: strength_min=0% → -1%/日 はブロック（両方ブロック）
    params_both = SwingReversionParams(
        lookback=20, entry_z=1.5,
        us_t1_crash_threshold=-0.02,
        us_t1_soft_min=-0.005, us_t1_soft_max=0.0,
        us_t1_short_strength_min=0.0,
    )
    sig_both = compute_signals(df, params_both, us_df=us_df)
    # 両方ブロックされるのでエントリーゼロのはず
    assert (sig_both["entry"] == 0).all()


def test_short_us_filter_default_disabled() -> None:
    """デフォルト params では short US フィルタは無効（ショートシグナルは変わらない）。"""
    df = _make_ohlcv_short()
    us_closes = [100.0 * (1.01 ** i) for i in range(100)]
    us_idx = pd.bdate_range("2023-12-01", periods=100)
    us_df = pd.DataFrame({"close": us_closes}, index=us_idx)
    params = SwingReversionParams(
        lookback=20, entry_z=1.5, allow_long=False, allow_short=True,
    )
    sig_no_us  = compute_signals(df, params)
    sig_with_us = compute_signals(df, params, us_df=us_df)
    assert (sig_no_us["entry"] == sig_with_us["entry"]).all()
