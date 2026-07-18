"""日足スイング・モメンタム（ブレイクアウト・トラックA）。

終値が直近 breakout_lookback 日の高値を上抜けでロング／安値を下抜けでショート。
固定目標は置かず、トレンドが続く限り保有し、ATR 損切り／タイムストップで出る。
比較対象として平均回帰（swing_reversion）と並走させる。

市場レジームフィルタ（params.enable_regime_filter=True + market_df を渡す）：
  市場指数（例: 日経225）が MA(regime_ma_window) から ±regime_threshold% 外の
  「トレンド相場」のときだけエントリーを許可する（regime_filter_invert=True運用）。
  2024-04〜2025-05の日経レンジ相場（円キャリー巻き戻し・トランプ関税ショック）で
  ブレイクアウトのダマシが多発し walk-forward が不安定だったことへの対処。

責務境界：グロスのトレード列のみ。コスト控除後の合否は evaluator が判定（絶対原則3）。
"""

from __future__ import annotations

import pandas as pd

from config.settings import DEFAULT_SWING_MOMENTUM, SwingMomentumParams
from strategy.indicators import atr, market_regime_mask
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


def compute_signals(
    df: pd.DataFrame,
    params: SwingMomentumParams = DEFAULT_SWING_MOMENTUM,
    *,
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """ブレイクアウト・シグナルを計算（前日までの窓の極値と比較＝未来非参照）。

    Args:
        market_df: 市場レジームフィルタ用の指数OHLCV（params.enable_regime_filter=True 時に使用）。

    Returns:
        DataFrame（列：prior_high, prior_low, atr, entry, regime）。entry は +1/-1/0。
        regime は True=エントリー許可 / False=除外。
    """
    _check(df)
    win = params.breakout_lookback
    # shift(1) で「前日まで」の窓の極値にする（当日終値との比較で未来を見ない）
    prior_high = df["high"].rolling(window=win, min_periods=win).max().shift(1)
    prior_low = df["low"].rolling(window=win, min_periods=win).min().shift(1)
    atr_series = atr(df["high"], df["low"], df["close"], length=params.atr_length)

    entry = pd.Series(0, index=df.index, dtype="int64")
    if params.allow_long:
        entry = entry.mask(df["close"] > prior_high, 1)
    if params.allow_short:
        entry = entry.mask(df["close"] < prior_low, -1)

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

    return pd.DataFrame(
        {
            "prior_high": prior_high, "prior_low": prior_low, "atr": atr_series,
            "entry": entry, "regime": regime,
        }
    )


def generate_trades(
    df: pd.DataFrame,
    params: SwingMomentumParams = DEFAULT_SWING_MOMENTUM,
    *,
    market_df: pd.DataFrame | None = None,
) -> list[Trade]:
    """日足 OHLC からブレイクアウトのトレード列を生成（グロス）。"""
    signals = compute_signals(df, params, market_df=market_df)
    return walk_swing(
        df,
        entries=signals["entry"],
        atr=signals["atr"],
        target=None,  # 固定目標なし（トレンド追随・損切り/タイムストップで出る）
        atr_stop_mult=params.atr_stop_mult,
        max_holding_days=params.max_holding_days,
    )
