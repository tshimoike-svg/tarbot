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

import pandas as pd

from config.settings import DEFAULT_SWING_REVERSION, SwingReversionParams
from strategy.indicators import atr, rolling_zscore, rsi as compute_rsi
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


def _market_regime_mask(
    market_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    *,
    ma_window: int,
    threshold: float,
    invert: bool = False,
) -> pd.Series:
    """各日付について市場がエントリー許可レジームか（前日終値基準・ルックアヘッドなし）。

    shift(1) で前日の終値・MA を参照するため当日のデータを先読みしない。
    target_index に市場データが無い日（休場の翌日など）は前の値を前向きに埋め（ffill）、
    データ不足（MA計算前）はエントリー許可（True）とする。

    invert=False（既定）: MA60 ±threshold% 以内の「レンジ相場」のときエントリー許可。
    invert=True         : MA60 ±threshold% 外の「トレンド相場」のときエントリー許可。
    """
    close = market_df["close"]
    prev_close = close.shift(1)
    ma = prev_close.rolling(window=ma_window, min_periods=ma_window // 2).mean()
    in_range = ((prev_close / ma - 1).abs() < threshold).fillna(True)
    mask = (~in_range) if invert else in_range
    return mask.reindex(target_index, method="ffill").fillna(True)


def compute_signals(
    df: pd.DataFrame,
    params: SwingReversionParams = DEFAULT_SWING_REVERSION,
    *,
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """指標とエントリーシグナルを計算（因果・未来非参照）。

    Args:
        market_df: 市場レジームフィルタ用の指数 OHLCV（params.enable_regime_filter=True 時に使用）。

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

    # 市場レジームフィルタ
    if market_df is not None and not market_df.empty and params.enable_regime_filter:
        regime = _market_regime_mask(
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
) -> list[Trade]:
    """日足 OHLC からスイング平均回帰のトレード列を生成（グロス）。

    Args:
        market_df: 市場レジームフィルタ用の指数 OHLCV（省略時はフィルタなし）。
    """
    signals = compute_signals(df, params, market_df=market_df)
    return walk_swing(
        df,
        entries=signals["entry"],
        atr=signals["atr"],
        target=signals["ma"],  # 移動平均への回帰で利確
        atr_stop_mult=params.atr_stop_mult,
        max_holding_days=params.max_holding_days,
    )
