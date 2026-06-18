"""約定率・滑りの実測（フォワード検証／コスト・キャリブレーションの心臓部）。

docs/trading_bot_design_v2.md §5-3, §10 / CLAUDE.md の方針に対応。

純フォワード方針では、コストを「仮定」ではなく「実測の積み上げ」で確定する。
本モジュールは発注を行わず、**発注意図（指値）とその結末（約定/不約定・約定価格）を
照合して、約定率と滑りを計測する**だけの純粋なロジック（API非依存・完全テスト可能）。

ここで集めた実測値（実効滑り bps / tick、約定率）を `config/costs.py` の想定に反映し、
`backtest/cost_model.py` の楽観バイアスを潰していく。

⚠️ これは計測専用。発注自体は `risk_manager.py` を通したうえで execution が行う（未実装）。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from config.costs import tick_size

__all__ = [
    "OrderSide",
    "FillStatus",
    "OrderIntent",
    "FillResult",
    "FillStats",
    "FillMonitor",
]

logger = logging.getLogger(__name__)

OrderSide = Literal["buy", "sell"]
FillStatus = Literal["filled", "partial", "unfilled"]


@dataclass(frozen=True)
class OrderIntent:
    """1つの発注意図（指値）。

    Attributes:
        order_id: 一意な注文ID。
        symbol: 銘柄コード。
        side: "buy"（買い／ロング新規・空売り返済）/ "sell"（売り／ロング返済・空売り新規）。
        limit_price: 指値。
        shares: 発注株数。
        reference_price: **意思決定時点の参照価格**（最良気配の仲値や直近約定値）。
            実装ショートフォール（本当の執行コスト）を測るための基準。
        placed_at: 発注時刻（JST）。
    """

    order_id: str
    symbol: str
    side: OrderSide
    limit_price: float
    shares: int
    reference_price: float
    placed_at: pd.Timestamp

    def __post_init__(self) -> None:
        if self.shares <= 0:
            raise ValueError("shares は正")
        if self.limit_price <= 0 or self.reference_price <= 0:
            raise ValueError("価格は正")


@dataclass(frozen=True)
class FillResult:
    """発注意図の結末（約定/部分約定/不約定）。"""

    intent: OrderIntent
    status: FillStatus
    fill_price: float | None
    filled_shares: int
    filled_at: pd.Timestamp | None

    @property
    def fill_ratio(self) -> float:
        """約定株数 / 発注株数（0〜1）。"""
        return self.filled_shares / self.intent.shares

    @property
    def slippage_per_share(self) -> float | None:
        """参照価格に対する**不利方向を正**とした1株あたり滑り（円）。

        buy:  fill − reference（高く買えば不利＝正）
        sell: reference − fill（安く売れば不利＝正）
        参照より有利に約定すれば負になる。不約定なら None。
        """
        if self.fill_price is None:
            return None
        ref = self.intent.reference_price
        if self.intent.side == "buy":
            return self.fill_price - ref
        return ref - self.fill_price

    @property
    def slippage_bps(self) -> float | None:
        """滑りを参照価格比の bps（万分率）で。"""
        sp = self.slippage_per_share
        if sp is None:
            return None
        return sp / self.intent.reference_price * 1e4

    @property
    def slippage_ticks(self) -> float | None:
        """滑りを呼値（tick）単位で（cost_model の slippage_ticks 想定に直接対応）。"""
        sp = self.slippage_per_share
        if sp is None:
            return None
        return sp / tick_size(self.intent.reference_price)


@dataclass(frozen=True)
class FillStats:
    """約定品質の集計（実測）。滑り系は約定（部分含む）のみで平均する。"""

    n_orders: int
    n_filled: int               # status が filled または partial
    fill_rate: float            # n_filled / n_orders（注文件数ベース）
    share_fill_rate: float      # 約定株数 / 発注株数（出来高ベース）
    avg_slippage_per_share: float
    avg_slippage_bps: float
    avg_slippage_ticks: float
    by_status: dict[str, int]

    @property
    def unfilled_rate(self) -> float:
        """不約定率 = 1 − fill_rate。指値戦略の機会損失の指標。"""
        return 1.0 - self.fill_rate if self.n_orders else math.nan


class FillMonitor:
    """発注意図と結末を記録し、約定率・滑りを集計する。

    使い方（ドライラン/ライブ共通）:
        mon.record_intent(intent)
        ... 約定したら ...
        mon.record_fill(order_id, fill_price=..., filled_shares=..., filled_at=...)
        ... 時間切れ/取消なら ...
        mon.record_unfilled(order_id)
        stats = mon.stats()
    """

    def __init__(self) -> None:
        self._intents: dict[str, OrderIntent] = {}
        self._results: dict[str, FillResult] = {}

    def record_intent(self, intent: OrderIntent) -> None:
        if intent.order_id in self._intents:
            raise ValueError(f"order_id が重複: {intent.order_id}")
        self._intents[intent.order_id] = intent
        logger.debug("intent recorded: %s", intent.order_id)

    def _intent(self, order_id: str) -> OrderIntent:
        try:
            return self._intents[order_id]
        except KeyError:
            raise ValueError(f"未登録の order_id: {order_id}") from None

    def record_fill(
        self,
        order_id: str,
        *,
        fill_price: float,
        filled_shares: int,
        filled_at: pd.Timestamp,
    ) -> FillResult:
        """約定（または部分約定）を記録する。"""
        intent = self._intent(order_id)
        if fill_price <= 0:
            raise ValueError("fill_price は正")
        if not 0 < filled_shares <= intent.shares:
            raise ValueError(f"filled_shares は 1〜{intent.shares}: {filled_shares}")
        status: FillStatus = "filled" if filled_shares == intent.shares else "partial"
        result = FillResult(
            intent=intent,
            status=status,
            fill_price=fill_price,
            filled_shares=filled_shares,
            filled_at=filled_at,
        )
        self._results[order_id] = result
        return result

    def record_unfilled(self, order_id: str) -> FillResult:
        """不約定（取消・時間切れ）を記録する。"""
        intent = self._intent(order_id)
        result = FillResult(
            intent=intent,
            status="unfilled",
            fill_price=None,
            filled_shares=0,
            filled_at=None,
        )
        self._results[order_id] = result
        return result

    @property
    def results(self) -> list[FillResult]:
        """記録済みの結末（記録順）。"""
        return list(self._results.values())

    def pending(self) -> list[str]:
        """結末未記録の order_id。"""
        return [oid for oid in self._intents if oid not in self._results]

    def stats(self) -> FillStats:
        """これまでの実測を集計する。結末未記録の注文は除外する。"""
        return aggregate_fills(self.results)


def aggregate_fills(results: Sequence[FillResult]) -> FillStats:
    """FillResult 列を集計して FillStats を返す（FillMonitor 非依存で再利用可能）。"""
    n_orders = len(results)
    if n_orders == 0:
        return FillStats(
            n_orders=0,
            n_filled=0,
            fill_rate=math.nan,
            share_fill_rate=math.nan,
            avg_slippage_per_share=math.nan,
            avg_slippage_bps=math.nan,
            avg_slippage_ticks=math.nan,
            by_status={},
        )

    filled = [r for r in results if r.fill_price is not None]
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1

    requested_shares = sum(r.intent.shares for r in results)
    filled_shares = sum(r.filled_shares for r in results)

    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else math.nan

    return FillStats(
        n_orders=n_orders,
        n_filled=len(filled),
        fill_rate=len(filled) / n_orders,
        share_fill_rate=(filled_shares / requested_shares) if requested_shares else math.nan,
        avg_slippage_per_share=_avg([r.slippage_per_share for r in filled]),  # type: ignore[misc]
        avg_slippage_bps=_avg([r.slippage_bps for r in filled]),  # type: ignore[misc]
        avg_slippage_ticks=_avg([r.slippage_ticks for r in filled]),  # type: ignore[misc]
        by_status=by_status,
    )
