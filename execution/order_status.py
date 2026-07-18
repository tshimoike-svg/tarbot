"""kabuステーションAPIの注文応答から約定状態を解釈する（API非依存の純粋ロジック）。

`KabuClient.send_order` は注文の**受理**を返すだけで、約定したかどうかは別途
`KabuClient.get_orders()` で照会する必要がある。本モジュールはその応答（dict）を
解釈して `OrderStatus` に正規化するだけで、HTTP呼び出しは一切行わない。

参照: kabucom/kabusapi リポジトリ reference/kabu_STATION_API.yaml の
OrdersSuccess スキーマ（State/OrderQty/CumQty/Details[].RecType/Details[].State）。
"""

from __future__ import annotations

from typing import Any, Literal

__all__ = ["OrderStatus", "interpret_order"]

OrderStatus = Literal["pending", "partial", "filled", "cancelled", "expired", "error", "unknown"]

_REC_TYPE_CANCELLED = 6
_REC_TYPE_EXPIRED = 3
_REC_TYPE_LAPSED = 7  # 失効
_DETAIL_STATE_ERROR = 4
_ORDER_STATE_WAITING = 1
_ORDER_STATE_PROCESSING = 2


def interpret_order(order: dict[str, Any]) -> OrderStatus:
    """GET /orders の1件の応答から約定状態を判定する。

    優先順位: エラー > 取消 > 期限切れ/失効 > 全約定 > 部分約定 > 処理中/待機 > 不明。
    """
    details = order.get("Details") or []

    if any(d.get("State") == _DETAIL_STATE_ERROR for d in details):
        return "error"

    rec_types = {d.get("RecType") for d in details}
    if _REC_TYPE_CANCELLED in rec_types:
        return "cancelled"
    if _REC_TYPE_EXPIRED in rec_types or _REC_TYPE_LAPSED in rec_types:
        return "expired"

    order_qty = order.get("OrderQty")
    cum_qty = order.get("CumQty")
    if isinstance(order_qty, (int, float)) and isinstance(cum_qty, (int, float)) and order_qty > 0:
        if cum_qty >= order_qty:
            return "filled"
        if cum_qty > 0:
            return "partial"

    state = order.get("State") or order.get("OrderState")
    if state in (_ORDER_STATE_WAITING, _ORDER_STATE_PROCESSING):
        return "pending"

    return "unknown"
