"""テクニカル指標（VWAP・ATR・ボリンジャーバンド・ローリングzスコア）。

設計対応（docs/trading_bot_design_v2.md §3, §5-1, §7-2）：
- 平均回帰戦略の主軸となる指標群。`pandas-ta` は numpy 2.x で import が壊れる
  既知の問題があるため使わず、pandas/numpy で自前実装する。
- **因果性（ルックアヘッド・バイアス回避）が最重要**：すべての指標は時点 t の値を
  「t 時点までに観測できる情報のみ」で計算する。未来のバーを一切参照しない。
  これは tests/test_indicators.py の「truncation invariance」テストで機械的に保証する。

入力は pandas.Series（価格・出来高）を直接受け取り、列名には依存しない。
時刻は JST 前提（VWAP の日次リセットは呼び出し側が JST の日付キーを渡す）。
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

__all__ = [
    "typical_price",
    "vwap",
    "true_range",
    "atr",
    "bollinger_bands",
    "rolling_zscore",
    "rsi",
    "market_regime_mask",
]

ATRMethod = Literal["wilder", "sma"]


def _validate_length(length: int) -> None:
    if not isinstance(length, int) or length < 1:
        raise ValueError(f"length は 1 以上の整数: {length!r}")


def typical_price(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """典型価格 (High + Low + Close) / 3。VWAP 等の価格代理に使う。"""
    return (high + low + close) / 3.0


def vwap(
    price: pd.Series,
    volume: pd.Series,
    *,
    session: pd.Series | None = None,
) -> pd.Series:
    """出来高加重平均価格（VWAP）。

    VWAP_t = Σ(price_i × volume_i) / Σ(volume_i)  （i は当該セッション開始〜t）

    累積和は時点 t までの観測のみを使うため、構造的にルックアヘッドしない。

    Args:
        price: 価格（典型価格や終値）。
        volume: 出来高。price と同じ index。
        session: 日次（または場）リセット用のグルーピングキー。
            例：`df.index.normalize()`（JST の日付）。None なら全期間で連続累積。
            日中平均回帰では **必ず日次キーを渡してセッション内 VWAP にする**こと。

    Returns:
        VWAP の Series（price と同じ index）。出来高ゼロ区間は NaN になり得る。
    """
    if not price.index.equals(volume.index):
        raise ValueError("price と volume の index が一致していません")

    pv = price * volume
    if session is None:
        cum_pv = pv.cumsum()
        cum_v = volume.cumsum()
    else:
        if not pd.Series(session, index=price.index).index.equals(price.index):
            raise ValueError("session の index が price と一致していません")
        grouper = pd.Series(session, index=price.index)
        cum_pv = pv.groupby(grouper).cumsum()
        cum_v = volume.groupby(grouper).cumsum()

    # 累積出来高 0 のとき 0 除算で inf にしないため NaN にしておく
    cum_v = cum_v.where(cum_v != 0)
    result = cum_pv / cum_v
    return result.rename("vwap")


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """真の値幅（True Range）。

    TR_t = max(High_t − Low_t, |High_t − Close_{t-1}|, |Low_t − Close_{t-1}|)

    前日終値（shift(1)）のみ参照するので未来を見ない。先頭バーは前終値が無いため
    High − Low になる。
    """
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    # 先頭バーは prev_close が NaN → skipna=True で High-Low が採用される
    return ranges.max(axis=1, skipna=True).rename("true_range")


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    length: int = 14,
    method: ATRMethod = "wilder",
) -> pd.Series:
    """平均真の値幅（ATR）。損切り幅（例：1.5×ATR）の基礎。

    Args:
        length: 平滑化期間。
        method: "wilder"（Wilder の RMA、業界標準）または "sma"（単純移動平均）。
            Wilder: ATR_t = (ATR_{t-1} × (n−1) + TR_t) / n（再帰・adjust=False の EMA）。

    Returns:
        ATR の Series。warmup（最初の length 本）は NaN。
    """
    _validate_length(length)
    tr = true_range(high, low, close)
    if method == "wilder":
        # Wilder の平滑化 = alpha=1/length の指数移動平均（adjust=False）
        return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean().rename("atr")
    if method == "sma":
        return tr.rolling(window=length, min_periods=length).mean().rename("atr")
    raise ValueError(f"未知の method: {method!r}")


def bollinger_bands(
    close: pd.Series,
    *,
    length: int = 20,
    num_std: float = 2.0,
    ddof: int = 0,
) -> pd.DataFrame:
    """ボリンジャーバンド。平均回帰の行き過ぎ判定に使う。

    中央線 = SMA(close, length)
    上限/下限 = 中央線 ± num_std × 標準偏差(close, length)

    rolling は時点 t までの length 本のみ使うため未来を見ない。

    Args:
        ddof: 標準偏差の自由度。ボリンジャー慣習に従い既定 0（母標準偏差）。

    Returns:
        DataFrame（列：middle, upper, lower, bandwidth, percent_b）。
        warmup は NaN。percent_b はバンド内位置（0=下限, 1=上限）。
    """
    _validate_length(length)
    if num_std < 0:
        raise ValueError("num_std は 0 以上")

    middle = close.rolling(window=length, min_periods=length).mean()
    std = close.rolling(window=length, min_periods=length).std(ddof=ddof)
    upper = middle + num_std * std
    lower = middle - num_std * std
    band = upper - lower
    percent_b = (close - lower) / band.where(band != 0)

    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "bandwidth": band,
            "percent_b": percent_b,
        }
    )


def rsi(close: pd.Series, *, length: int = 14) -> pd.Series:
    """RSI（Relative Strength Index）。Wilder の平滑化。

    Wilder の EWM（alpha=1/length, adjust=False）で平均利得・平均損失を計算し
    RSI = 100 × avg_gain / (avg_gain + avg_loss) で算出。
    avg_gain = avg_loss = 0（フラット系列）は NaN。warmup は NaN。

    Args:
        length: 平滑化期間（既定 14）。

    Returns:
        RSI の Series（0〜100 の範囲、warmup は NaN）。
    """
    _validate_length(length)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    denom = avg_gain + avg_loss
    return (100.0 * avg_gain / denom.replace(0.0, float("nan"))).rename("rsi")


def market_regime_mask(
    market_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    *,
    ma_window: int,
    threshold: float,
    invert: bool = False,
) -> pd.Series:
    """各日付について市場がエントリー許可レジームか（前日終値基準・ルックアヘッドなし）。

    reversion・momentum 共通の市場レジームフィルタ。shift(1) で前日の終値・MA を
    参照するため当日のデータを先読みしない。target_index に市場データが無い日
    （休場の翌日など）は前の値を前向きに埋め（ffill）、データ不足（MA計算前）は
    エントリー許可（True）とする。

    invert=False: MA(ma_window) ±threshold% 以内の「レンジ相場」のときエントリー許可
                  （平均回帰向き：個別株の押し目が効きやすい）。
    invert=True : MA(ma_window) ±threshold% 外の「トレンド相場」のときエントリー許可
                  （モメンタム向き：ブレイクアウトがダマシになりにくい）。
    """
    close = market_df["close"]
    prev_close = close.shift(1)
    ma = prev_close.rolling(window=ma_window, min_periods=ma_window // 2).mean()
    in_range = (prev_close / ma - 1).abs() < threshold
    mask = (~in_range) if invert else in_range
    # NaN < threshold は pandas では False（NaN ではない）になるため、fillna は効かない。
    # データ不足（ma が NaN）の行は invert に関わらず明示的に True にする。
    mask = mask.where(ma.notna(), other=True)
    return mask.reindex(target_index, method="ffill").fillna(True)


def rolling_zscore(series: pd.Series, *, length: int, ddof: int = 0) -> pd.Series:
    """ローリング z スコア = (x − 移動平均) / 移動標準偏差。

    平均回帰のエントリー判定（例：VWAP からの乖離が −Nσ）に使える汎用指標。
    `rolling_zscore(close - vwap, length=N)` のように差分系列へ適用する。

    Args:
        ddof: 標準偏差の自由度（既定 0＝母標準偏差）。

    Returns:
        z スコアの Series。標準偏差 0 の区間は NaN。warmup も NaN。
    """
    _validate_length(length)
    mean = series.rolling(window=length, min_periods=length).mean()
    std = series.rolling(window=length, min_periods=length).std(ddof=ddof)
    return ((series - mean) / std.where(std != 0)).rename("zscore")
