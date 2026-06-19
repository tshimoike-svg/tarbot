"""日中平均回帰戦略（トラックA・主力／バックテスト可能）。

docs/trading_bot_design_v2.md §5-1, §6, §12（Phase 0 の関門）に対応する初期仮説：
  エントリー：VWAP からの乖離 z スコアが ±entry_z 超で逆張り
  イグジット：VWAP 回帰で利確／ATR×倍率で損切り／当日強制クローズ

⚠️ 重要な責務の境界
- このモジュールは **シグナルと“グロス（コスト控除前）のトレード列”** を生成するだけ。
  期待値の合否は **コスト控除後**で評価する（絶対原則3）。コスト控除は
  `backtest/cost_model.py` を使って `backtest/evaluator.py` 側で行う。ここでは損益の
  期待値を主張しない。
- これはバックテスト用のロジック。**ライブ発注は別途 `risk_manager.py` を必ず通す**
  （絶対原則2）。本モジュールは発注を行わない。
- 因果性：シグナルは時点 t までの情報のみで決まる（indicators 側で保証）。エントリーは
  当該バーの終値で約定したと見なし、イグジット判定は**次バー以降**で行う（同バー内での
  未来参照を避ける）。指値の不約定・滑りは backtest 側でモデル化する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from config.settings import DEFAULT_MEAN_REVERSION, MeanReversionParams
from strategy.indicators import atr, rolling_zscore, typical_price, vwap

__all__ = ["Trade", "compute_signals", "generate_trades"]

Side = Literal["long", "short"]
ExitReason = Literal["take_profit", "stop", "force_close"]

_REQUIRED_COLUMNS = ("high", "low", "close", "volume")


@dataclass(frozen=True)
class Trade:
    """1往復の確定トレード（コスト控除前のグロス）。

    pnl_gross_per_share はスプレッド・滑り・手数料・金利を**含まない**。
    実効損益は cost_model で往復コストを引いて評価すること。
    """

    side: Side
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: ExitReason
    stop_price: float | None = None  # エントリー時の保護ストップ（リスク基準サイジングに使う）

    @property
    def pnl_gross_per_share(self) -> float:
        """1株あたりグロス損益（コスト控除前）。"""
        if self.side == "long":
            return self.exit_price - self.entry_price
        return self.entry_price - self.exit_price


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"必須列が不足: {missing}（必要: {_REQUIRED_COLUMNS}）")


def _session_key(df: pd.DataFrame, session: pd.Series | None) -> pd.Series:
    """VWAP 日次リセット用のセッションキー（既定は JST 日付）。"""
    if session is not None:
        return pd.Series(session, index=df.index)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("session 未指定時は DatetimeIndex が必要です（日次リセットのため）")
    return pd.Series(df.index.normalize(), index=df.index)


def compute_signals(
    df: pd.DataFrame,
    params: MeanReversionParams = DEFAULT_MEAN_REVERSION,
    *,
    session: pd.Series | None = None,
) -> pd.DataFrame:
    """指標とシグナル材料を計算して返す（因果・未来非参照）。

    Args:
        df: 列 high/low/close/volume を持つ分足 OHLCV。DatetimeIndex（JST）想定。
        params: 戦略パラメータ。
        session: VWAP リセットのセッションキー。None なら日次（index.normalize()）。

    Returns:
        DataFrame（列：vwap, zscore, atr）。warmup は NaN。
    """
    _check_columns(df)
    sess = _session_key(df, session)

    price = (
        typical_price(df["high"], df["low"], df["close"])
        if params.use_typical_price_for_vwap
        else df["close"]
    )
    vwap_series = vwap(price, df["volume"], session=sess)
    deviation = df["close"] - vwap_series
    zscore = rolling_zscore(deviation, length=params.zscore_length)
    atr_series = atr(df["high"], df["low"], df["close"], length=params.atr_length)

    return pd.DataFrame(
        {"vwap": vwap_series, "zscore": zscore, "atr": atr_series},
        index=df.index,
    )


def _walk_session(g: pd.DataFrame, params: MeanReversionParams) -> list[Trade]:
    """1セッション（1日）分を時系列に歩いて確定トレード列を作る。

    g は high/low/close/vwap/zscore/atr を持ち、時刻昇順の DatetimeIndex。
    単一ポジション制。手仕舞い後は同日中の再エントリーを許可する（頻回トレード）。
    """
    trades: list[Trade] = []
    pos: dict[str, object] | None = None
    n = len(g)

    for i in range(n):
        ts: pd.Timestamp = g.index[i]
        row = g.iloc[i]
        is_last = i == n - 1
        after_close = ts.time() >= params.force_close

        if pos is None:
            # 引け間際・最終バーでは新規建てしない
            if after_close or is_last:
                continue
            z = row["zscore"]
            a = row["atr"]
            if pd.isna(z) or pd.isna(a):
                continue
            entry = float(row["close"])
            if params.allow_long and z <= -params.entry_z:
                pos = {
                    "side": "long",
                    "entry_time": ts,
                    "entry_price": entry,
                    "stop": entry - params.atr_stop_mult * float(a),
                }
            elif params.allow_short and z >= params.entry_z:
                pos = {
                    "side": "short",
                    "entry_time": ts,
                    "entry_price": entry,
                    "stop": entry + params.atr_stop_mult * float(a),
                }
            # エントリーしたバーではイグジット判定をしない（次バー以降で評価）
            continue

        # --- ポジション保有中：イグジット判定 ---
        side = pos["side"]
        stop = float(pos["stop"])  # type: ignore[arg-type]
        target = float(row["vwap"])  # VWAP 回帰で利確
        exit_price: float | None = None
        reason: ExitReason | None = None

        if is_last or after_close:
            exit_price = float(row["close"])
            reason = "force_close"
        elif side == "long":
            # 保守的に損切りを優先（同バーで両方タッチしても不利側を採用）
            if float(row["low"]) <= stop:
                exit_price, reason = stop, "stop"
            elif not pd.isna(target) and float(row["high"]) >= target:
                exit_price, reason = target, "take_profit"
        else:  # short
            if float(row["high"]) >= stop:
                exit_price, reason = stop, "stop"
            elif not pd.isna(target) and float(row["low"]) <= target:
                exit_price, reason = target, "take_profit"

        if exit_price is not None and reason is not None:
            trades.append(
                Trade(
                    side=side,  # type: ignore[arg-type]
                    entry_time=pos["entry_time"],  # type: ignore[arg-type]
                    entry_price=float(pos["entry_price"]),  # type: ignore[arg-type]
                    exit_time=ts,
                    exit_price=exit_price,
                    exit_reason=reason,
                    stop_price=stop,
                )
            )
            pos = None

    return trades


def generate_trades(
    df: pd.DataFrame,
    params: MeanReversionParams = DEFAULT_MEAN_REVERSION,
    *,
    session: pd.Series | None = None,
) -> list[Trade]:
    """OHLCV からグロスの確定トレード列を生成する（コスト控除前）。

    Args:
        df: 列 high/low/close/volume を持つ分足 OHLCV（DatetimeIndex・JST・時刻昇順）。
        params: 戦略パラメータ。
        session: VWAP・セッション境界のキー。None なら日次。

    Returns:
        Trade のリスト（時系列順）。期待値評価は必ずコスト控除後で行うこと。
    """
    _check_columns(df)
    if not df.index.is_monotonic_increasing:
        raise ValueError("df は時刻昇順である必要があります")

    sess = _session_key(df, session)
    signals = compute_signals(df, params, session=sess)
    frame = pd.concat([df[["high", "low", "close"]], signals], axis=1)

    trades: list[Trade] = []
    for _, group in frame.groupby(sess, sort=False):
        trades.extend(_walk_session(group, params))
    return trades
