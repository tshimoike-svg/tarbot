"""日足スイング平均回帰（押し目買い／戻り売り・トラックA）。

終値が移動平均から行き過ぎた（z スコアが ±entry_z 超）ところで逆張りし、移動平均への
回帰で利確、ATR で損切り、max_holding_days でタイムストップ。日足なので J-Quants の
無料/有料データで**そのままバックテスト可能**。

市場レジームフィルタ（params.enable_regime_filter=True + market_df を渡す）：
  市場指数（例: TOPIX ETF 1306）の MA(regime_ma_window) から ±regime_threshold% 以内の
  レンジ相場のときだけエントリーを許可。トレンド相場では個別株の平均回帰が成立しにくいため。

責務境界：本モジュールはグロス（コスト控除前）のトレード列を出すだけ。コスト控除後
（持ち越し金利含む）の合否は evaluator が判定する（絶対原則3）。
"""

from __future__ import annotations

import math

import pandas as pd

from config.settings import DEFAULT_SWING_REVERSION, SwingReversionParams
from strategy.indicators import atr, market_regime_mask, rolling_zscore, rsi as compute_rsi
from strategy.swing import walk_swing
from strategy.trade import Trade

__all__ = ["compute_signals", "generate_trades"]

_REQUIRED = ("open", "high", "low", "close")


def _check(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"必須列が不足: {missing}（必要: {_REQUIRED}）")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df は時刻昇順である必要があります")


def _us_return_for_signal(
    us_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    *,
    day_offset: int,
) -> pd.Series:
    """US リターンを日本取引日に対応付ける汎用ヘルパー。

    day_offset=1 （T-1）: 日本シグナル日 T の前日 US（= T 始値を動かした夜）。
      例: 日本火曜 T → US 月曜 → 月曜リターン（月曜終値/金曜終値-1）。
      用途: 押し目の起点確認。シグナル生成時点（T 終値 15:30）で既知。

    day_offset=0 （T0）: 日本シグナル日 T と同日の US（JST T+1 05:00 閉場）。
      例: 日本火曜 T → US 火曜 → 火曜リターン（火曜終値/月曜終値-1）。
      用途: エントリー当日 T+1 の始値を動かした US。T+1 09:00 より前に判明するため先読みなし。

    target_index は単調増加前提（_check で保証済み）。
    """
    close = us_df["close"]
    us_ret = close.pct_change().dropna()
    idx = pd.DatetimeIndex(us_ret.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    idx = idx.as_unit("ns")
    us_df_tmp = pd.DataFrame({"us_date": idx, "us_ret": us_ret.values}).sort_values("us_date")
    lookup = (pd.DatetimeIndex(target_index) - pd.Timedelta(days=day_offset)).as_unit("ns")
    japan_df = pd.DataFrame({"lookup_date": lookup})
    merged = pd.merge_asof(japan_df, us_df_tmp, left_on="lookup_date", right_on="us_date", direction="backward")
    return pd.Series(merged["us_ret"].values, index=target_index, name=f"us_ret_t{day_offset}")


def compute_signals(
    df: pd.DataFrame,
    params: SwingReversionParams = DEFAULT_SWING_REVERSION,
    *,
    market_df: pd.DataFrame | None = None,
    us_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """指標とエントリーシグナルを計算（因果・未来非参照）。

    Args:
        market_df: 市場レジームフィルタ用の指数 OHLCV（params.enable_regime_filter=True 時に使用）。
        us_df: 前日米国株リターンフィルタ用の S&P500 OHLCV（params.us_crash_threshold 等と組み合わせて使用）。

    Returns:
        DataFrame（列：ma, zscore, atr, entry, regime）。entry は +1/-1/0。
        regime は True=エントリー許可 / False=除外。
    """
    _check(df)
    ma = df["close"].rolling(window=params.lookback, min_periods=params.lookback).mean()
    z = rolling_zscore(df["close"], length=params.lookback)
    atr_series = atr(df["high"], df["low"], df["close"], length=params.atr_length)

    # RSI の事前計算（フィルタが有効なときのみ）
    need_rsi = (params.allow_long and params.rsi_entry_max < 100.0) or \
               (params.allow_short and params.rsi_entry_min > 0.0)
    rsi_series = compute_rsi(df["close"], length=params.rsi_length) if need_rsi else None

    # 出来高比率の事前計算（フィルタが有効かつ volume 列が存在するときのみ）
    need_vol = params.volume_ratio_min > 0.0 and "volume" in df.columns
    if need_vol:
        vol_ma = df["volume"].rolling(
            window=params.volume_ma_length,
            min_periods=params.volume_ma_length // 2,
        ).mean()
        vol_ratio = df["volume"] / vol_ma.replace(0.0, float("nan"))
    else:
        vol_ratio = None

    entry = pd.Series(0, index=df.index, dtype="int64")
    if params.allow_long:
        long_mask = z <= -params.entry_z
        if rsi_series is not None and params.rsi_entry_max < 100.0:
            long_mask = long_mask & (rsi_series < params.rsi_entry_max)
        if vol_ratio is not None:
            long_mask = long_mask & (vol_ratio > params.volume_ratio_min)
        entry = entry.mask(long_mask, 1)
    if params.allow_short:
        short_mask = z >= params.entry_z
        if rsi_series is not None and params.rsi_entry_min > 0.0:
            short_mask = short_mask & (rsi_series > params.rsi_entry_min)
        if vol_ratio is not None:
            short_mask = short_mask & (vol_ratio > params.volume_ratio_min)
        entry = entry.mask(short_mask, -1)

    # 季節フィルタ（指定月はエントリー禁止）
    if params.season_avoid_months:
        month_ok = ~pd.DatetimeIndex(df.index).month.isin(params.season_avoid_months)
        entry = entry.where(month_ok, other=0)

    # 米国株リターンフィルタ（ロング用と ショート用を方向別に独立して適用）
    if us_df is not None and not us_df.empty:
        japan_idx = pd.DatetimeIndex(df.index)

        # US リターンを必要なオフセット分だけ事前計算（重複取得を避ける）
        _need_t1 = any(not math.isnan(getattr(params, a)) for a in (
            "us_t1_crash_threshold", "us_t1_soft_min", "us_t1_recovery_min",
            "us_t1_short_strength_min",
        ))
        _need_t0 = any(not math.isnan(getattr(params, a)) for a in (
            "us_t0_crash_threshold", "us_t0_soft_min", "us_t0_recovery_min",
            "us_t0_short_weakness_max",
        ))
        us_t1 = _us_return_for_signal(us_df, japan_idx, day_offset=1) if _need_t1 else None
        us_t0 = _us_return_for_signal(us_df, japan_idx, day_offset=0) if _need_t0 else None

        # ---- ロング用フィルタ（crash / soft / recovery の OR 条件） ----
        long_us_allow = pd.Series(True, index=df.index)
        for us_ret_s, crash_attr, smin_attr, smax_attr, rec_attr in (
            (us_t1, "us_t1_crash_threshold", "us_t1_soft_min", "us_t1_soft_max", "us_t1_recovery_min"),
            (us_t0, "us_t0_crash_threshold", "us_t0_soft_min", "us_t0_soft_max", "us_t0_recovery_min"),
        ):
            if us_ret_s is None:
                continue
            _crash = getattr(params, crash_attr)
            _smin  = getattr(params, smin_attr)
            _rec   = getattr(params, rec_attr)
            _has_crash    = not math.isnan(_crash)
            _has_soft     = not math.isnan(_smin)
            _has_recovery = not math.isnan(_rec)
            if not _has_crash and not _has_soft and not _has_recovery:
                continue
            layer_allow = pd.Series(False, index=df.index)
            if _has_crash:
                layer_allow = layer_allow | (us_ret_s < _crash)
            if _has_soft:
                _smax = getattr(params, smax_attr)
                layer_allow = layer_allow | ((us_ret_s >= _smin) & (us_ret_s < _smax))
            if _has_recovery:
                layer_allow = layer_allow | (us_ret_s >= _rec)
            long_us_allow = long_us_allow & layer_allow.fillna(True)

        # ---- ショート用フィルタ（T-1 強さ AND T0 弱さ/平坦） ----
        short_us_allow = pd.Series(True, index=df.index)
        _t1_str  = params.us_t1_short_strength_min
        _t0_weak = params.us_t0_short_weakness_max
        if not math.isnan(_t1_str) and us_t1 is not None:
            short_us_allow = short_us_allow & (us_t1 >= _t1_str).fillna(True)
        if not math.isnan(_t0_weak) and us_t0 is not None:
            short_us_allow = short_us_allow & (us_t0 <= _t0_weak).fillna(True)

        # 方向別にフィルタを適用して再合成
        long_ok  = (entry == 1)  & long_us_allow.values
        short_ok = (entry == -1) & short_us_allow.values
        entry = pd.Series(0, index=df.index, dtype="int64")
        entry = entry.mask(long_ok,  1)
        entry = entry.mask(short_ok, -1)

    # 市場レジームフィルタ
    if market_df is not None and not market_df.empty and params.enable_regime_filter:
        regime = market_regime_mask(
            market_df, pd.DatetimeIndex(df.index),
            ma_window=params.regime_ma_window,
            threshold=params.regime_threshold,
            invert=params.regime_filter_invert,
        )
        entry = entry.where(regime, other=0)
    else:
        regime = pd.Series(True, index=df.index)

    return pd.DataFrame({"ma": ma, "zscore": z, "atr": atr_series, "entry": entry, "regime": regime})


def generate_trades(
    df: pd.DataFrame,
    params: SwingReversionParams = DEFAULT_SWING_REVERSION,
    *,
    market_df: pd.DataFrame | None = None,
    us_df: pd.DataFrame | None = None,
) -> list[Trade]:
    """日足 OHLC からスイング平均回帰のトレード列を生成（グロス）。

    Args:
        market_df: 市場レジームフィルタ用の指数 OHLCV（省略時はフィルタなし）。
        us_df: 前日米国株リターンフィルタ用の S&P500 OHLCV（省略時はフィルタなし）。
    """
    signals = compute_signals(df, params, market_df=market_df, us_df=us_df)
    return walk_swing(
        df,
        entries=signals["entry"],
        atr=signals["atr"],
        target=signals["ma"],  # 移動平均への回帰で利確
        atr_stop_mult=params.atr_stop_mult,
        max_holding_days=params.max_holding_days,
    )
