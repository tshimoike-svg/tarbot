"""日足スイング平均回帰（押し目買い／戻り売り・トラックA）。

終値が移動平均から行き過ぎた（z スコアが ±entry_z 超）ところで逆張りし、移動平均への
回帰で利確、ATR で損切り、max_holding_days でタイムストップ。日足なので J-Quants の
無料/有料データで**そのままバックテスト可能**。

責務境界：本モジュールはグロス（コスト控除前）のトレード列を出すだけ。コスト控除後
（持ち越し金利含む）の合否は evaluator が判定する（絶対原則3）。
"""

from __future__ import annotations

import pandas as pd

from config.settings import DEFAULT_SWING_REVERSION, SwingReversionParams
from strategy.indicators import atr, rolling_zscore
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
    df: pd.DataFrame, params: SwingReversionParams = DEFAULT_SWING_REVERSION
) -> pd.DataFrame:
    """指標とエントリーシグナルを計算（因果・未来非参照）。

    Returns:
        DataFrame（列：ma, zscore, atr, entry）。entry は +1/-1/0。
    """
    _check(df)
    ma = df["close"].rolling(window=params.lookback, min_periods=params.lookback).mean()
    z = rolling_zscore(df["close"], length=params.lookback)
    atr_series = atr(df["high"], df["low"], df["close"], length=params.atr_length)

    entry = pd.Series(0, index=df.index, dtype="int64")
    if params.allow_long:
        entry = entry.mask(z <= -params.entry_z, 1)
    if params.allow_short:
        entry = entry.mask(z >= params.entry_z, -1)

    return pd.DataFrame({"ma": ma, "zscore": z, "atr": atr_series, "entry": entry})


def generate_trades(
    df: pd.DataFrame, params: SwingReversionParams = DEFAULT_SWING_REVERSION
) -> list[Trade]:
    """日足 OHLC からスイング平均回帰のトレード列を生成（グロス）。"""
    signals = compute_signals(df, params)
    return walk_swing(
        df,
        entries=signals["entry"],
        atr=signals["atr"],
        target=signals["ma"],  # 移動平均への回帰で利確
        atr_stop_mult=params.atr_stop_mult,
        max_holding_days=params.max_holding_days,
    )
