"""トレードの共通データ型（戦略・評価・執行が共有する）。

戦略（平均回帰／モメンタム）が生成し、evaluator・dry_run・storage が消費する
1往復の確定トレード。コスト控除前（グロス）の損益のみを持つ。コスト控除後の評価は
evaluator（cost_model 経由）が行う（絶対原則3）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

__all__ = ["Side", "ExitReason", "Trade"]

Side = Literal["long", "short"]
# スイングの出口理由：目標到達 / 損切り / 保有日数上限（タイムストップ）
ExitReason = Literal["target", "stop", "time_stop"]


@dataclass(frozen=True)
class Trade:
    """1往復の確定トレード（コスト控除前のグロス）。

    pnl_gross_per_share はスプレッド・滑り・手数料・金利を含まない。
    実効損益は cost_model で往復コスト（持ち越し金利含む）を引いて評価する。
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
