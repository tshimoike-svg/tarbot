"""日足クロスセクション平均回帰（市場中立・トラックA）。

同日にユニバース全銘柄のN日リターンをクロスセクションでz化し、
ピア対比で大きく劣後した銘柄をロング・大きく優位した銘柄をショート。
市場全体が上昇してもロング・ショート双方のcs_zは変わらないため
β（市場方向リスク）が消え、トレンド方向に依存しない。

時系列z（銘柄の自分自身のMAとの乖離）との違い：
  - 時系列z: 「この株は自分の歴史平均より高い/低い」（絶対的な過去比較）
  - クロスセクションz: 「この株は今日のピアより大きく上/下に外れた」（相対的な同日比較）
  ⇒ 強気相場では全銘柄が自分のMAを上回るため時系列z のショートが機能しなくなるが、
    クロスセクションz は相対的な外れ値のみを狙うため方向依存しない。

責務境界：グロスのトレード列のみ。コスト控除後の合否は evaluator が判定（絶対原則3）。
"""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import DEFAULT_CROSS_SECTION, SwingCrossSectionParams
from strategy.indicators import atr
from strategy.swing import walk_swing
from strategy.trade import Trade

__all__ = ["compute_cross_signals", "generate_trades"]

logger = logging.getLogger(__name__)

_REQUIRED = ("open", "high", "low", "close")


def _check(df: pd.DataFrame, symbol: str) -> bool:
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        logger.warning("銘柄 %s に必須列なし: %s", symbol, missing)
        return False
    if not df.index.is_monotonic_increasing:
        logger.warning("銘柄 %s のインデックスが昇順でない", symbol)
        return False
    return True


def compute_cross_signals(
    all_dfs: dict[str, pd.DataFrame],
    params: SwingCrossSectionParams = DEFAULT_CROSS_SECTION,
) -> dict[str, pd.Series]:
    """全銘柄のクロスセクションzスコアに基づくエントリーシグナルを計算。

    各日の全銘柄N日リターンをクロスセクション正規化し、外れ値銘柄を検出する。
    shift(1) で前日終値基準のN日リターンを使うためルックアヘッドなし。

    Args:
        all_dfs: symbol → 日足DataFrame（open/high/low/close 列必須）のdict。
        params: クロスセクション戦略パラメータ。

    Returns:
        symbol → エントリーシグナル Series（+1=ロング, -1=ショート, 0=なし）のdict。
    """
    valid = {sym: df for sym, df in all_dfs.items() if _check(df, sym)}
    if len(valid) < params.min_universe_size:
        logger.warning(
            "有効銘柄数 %d < min_universe_size %d。シグナルなし",
            len(valid), params.min_universe_size,
        )
        return {}

    # 全銘柄の終値パネルを共通インデックスで構築
    close_panel = pd.DataFrame({sym: df["close"] for sym, df in valid.items()})

    # N日リターン（前日終値基準 = shift(1) で翌日のopen建て発注でも先読みなし）
    # pct_change(n) は t の値を使うため、shift(1) で t-1 まで参照に変換
    ret = close_panel.shift(1).pct_change(params.return_lookback)

    # クロスセクション正規化（各日の横断的mean/std）
    cs_mean = ret.mean(axis=1)
    cs_std = ret.std(axis=1, ddof=1)
    # std が 0 の日（全銘柄リターンが同一）はNaNにして除外
    cs_std = cs_std.replace(0.0, float("nan"))
    cs_z = ret.sub(cs_mean, axis=0).div(cs_std, axis=0)

    signals: dict[str, pd.Series] = {}
    for sym, df in valid.items():
        sym_z = cs_z[sym].reindex(df.index)
        entry = pd.Series(0, index=df.index, dtype="int64")
        if params.allow_long:
            entry = entry.mask(sym_z <= -params.entry_z, 1)
        if params.allow_short:
            entry = entry.mask(sym_z >= params.entry_z, -1)
        signals[sym] = entry

    return signals


def generate_trades(
    all_dfs: dict[str, pd.DataFrame],
    params: SwingCrossSectionParams = DEFAULT_CROSS_SECTION,
) -> list[Trade]:
    """全銘柄の日足からクロスセクション平均回帰のトレード列を生成（グロス）。

    全銘柄を同時に受け取りクロスセクションzを計算するため、
    銘柄ごとに独立して呼ぶ従来戦略とは異なりdictで受け取る。

    Args:
        all_dfs: symbol → 日足DataFrame（open/high/low/close 列必須）のdict。
        params: クロスセクション戦略パラメータ。

    Returns:
        全銘柄分のトレードリスト（entry_time 昇順は呼び出し側で保証する）。
    """
    signals = compute_cross_signals(all_dfs, params)
    if not signals:
        return []

    trades: list[Trade] = []
    for sym, entry in signals.items():
        df = all_dfs[sym]
        atr_series = atr(df["high"], df["low"], df["close"], length=params.atr_length)
        sym_trades = walk_swing(
            df,
            entries=entry,
            atr=atr_series,
            target=None,   # 固定目標なし（クロスセクションのターゲットは相対的で追跡困難）
            atr_stop_mult=params.atr_stop_mult,
            max_holding_days=params.max_holding_days,
        )
        trades.extend(sym_trades)

    return trades
