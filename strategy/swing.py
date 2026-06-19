"""日足スイングの共通エンジン（保有・出口の状態機械）。

docs/trading_bot_design_v3.md に対応。平均回帰・モメンタムで共通する
「エントリー後、数日〜max_holding_days 保有し、目標／損切り／タイムストップで出る」
ロジックをここに集約する。各戦略は **エントリーシグナル**と**目標系列**だけを与える。

因果性・保守性の方針：
- シグナルは時点 t までの情報のみで決まる（各戦略が causal に算出）。約定は
  **翌営業日の寄り（open[t+1]）**で行い、同バー内の未来参照を避ける。
- **ギャップを保守的にモデル化**：損切りは「寄りが既にストップを割っていれば寄り値
  （より不利）で約定」。目標利確は逆に寄りで飛び越えても**目標値どまり**（楽観を避ける）。
- 同バーで損切りと目標が両方触れたら**損切り優先**。
- 単一ポジション制（ピラミッディングなし）。出口の翌バー以降で次のエントリーを探す。

オーバーナイトの持ち越し金利は cost_model が holding_days から計上する（evaluator 側）。
"""

from __future__ import annotations

import pandas as pd

from strategy.trade import Side, Trade

__all__ = ["walk_swing"]


def walk_swing(
    df: pd.DataFrame,
    *,
    entries: pd.Series,
    atr: pd.Series,
    target: pd.Series | None,
    atr_stop_mult: float,
    max_holding_days: int,
) -> list[Trade]:
    """エントリーシグナルからスイングの確定トレード列を生成する。

    Args:
        df: 列 open/high/low/close を持つ日足（DatetimeIndex・昇順）。
        entries: 各バーのシグナル（+1=ロング, -1=ショート, 0=なし）。df と同 index。
        atr: ATR 系列（損切り幅の基礎）。df と同 index。
        target: 利確目標の系列（平均回帰なら移動平均）。None なら目標なし
            （損切り／タイムストップのみ。モメンタム向け）。
        atr_stop_mult: 損切り幅 = atr_stop_mult × ATR(シグナル日)。
        max_holding_days: 最大保有バー数（タイムストップ）。

    Returns:
        Trade のリスト（時系列順・グロス）。
    """
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    sig = entries.to_numpy()
    atr_v = atr.to_numpy(dtype=float)
    tgt = target.to_numpy(dtype=float) if target is not None else None
    index = df.index
    n = len(df)

    trades: list[Trade] = []
    i = 0
    while i < n - 1:
        s = sig[i]
        if s == 0 or pd.isna(atr_v[i]) or pd.isna(opens[i + 1]):
            i += 1
            continue

        side: Side = "long" if s > 0 else "short"
        entry_idx = i + 1
        entry_price = opens[entry_idx]
        stop = (
            entry_price - atr_stop_mult * atr_v[i]
            if side == "long"
            else entry_price + atr_stop_mult * atr_v[i]
        )

        last = min(entry_idx + max_holding_days - 1, n - 1)
        exit_idx = last
        exit_price = closes[last]
        reason = "time_stop"

        for j in range(entry_idx, last + 1):
            if side == "long":
                if opens[j] <= stop:  # 寄りで既にストップ割れ（ギャップダウン）
                    exit_idx, exit_price, reason = j, opens[j], "stop"
                    break
                if lows[j] <= stop:
                    exit_idx, exit_price, reason = j, stop, "stop"
                    break
                if tgt is not None and not pd.isna(tgt[j]) and highs[j] >= tgt[j]:
                    exit_idx, exit_price, reason = j, tgt[j], "target"
                    break
            else:  # short
                if opens[j] >= stop:  # 寄りで既にストップ超え（ギャップアップ）
                    exit_idx, exit_price, reason = j, opens[j], "stop"
                    break
                if highs[j] >= stop:
                    exit_idx, exit_price, reason = j, stop, "stop"
                    break
                if tgt is not None and not pd.isna(tgt[j]) and lows[j] <= tgt[j]:
                    exit_idx, exit_price, reason = j, tgt[j], "target"
                    break

        trades.append(
            Trade(
                side=side,
                entry_time=index[entry_idx],
                entry_price=float(entry_price),
                exit_time=index[exit_idx],
                exit_price=float(exit_price),
                exit_reason=reason,  # type: ignore[arg-type]
                stop_price=float(stop),
            )
        )
        i = exit_idx  # 出口バー以降で次のエントリーを探す（重複保有なし）

    return trades
