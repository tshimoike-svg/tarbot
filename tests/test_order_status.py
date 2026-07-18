"""order_status.py のテスト（GET /orders 応答 → OrderStatus への解釈ロジック）。"""

from __future__ import annotations

from execution.order_status import interpret_order


def _order(**kw: object) -> dict[str, object]:
    base = {"ID": "ORD1", "State": 3, "OrderQty": 100, "CumQty": 0, "Details": []}
    base.update(kw)
    return base


def test_full_fill() -> None:
    assert interpret_order(_order(OrderQty=100, CumQty=100)) == "filled"


def test_partial_fill() -> None:
    assert interpret_order(_order(OrderQty=100, CumQty=40)) == "partial"


def test_no_fill_yet_is_pending() -> None:
    assert interpret_order(_order(State=1, OrderQty=100, CumQty=0)) == "pending"


def test_processing_is_pending() -> None:
    assert interpret_order(_order(State=2, OrderQty=100, CumQty=0)) == "pending"


def test_cancelled_via_rectype() -> None:
    order = _order(OrderQty=100, CumQty=0, Details=[{"RecType": 6}])
    assert interpret_order(order) == "cancelled"


def test_expired_via_rectype() -> None:
    order = _order(OrderQty=100, CumQty=0, Details=[{"RecType": 3}])
    assert interpret_order(order) == "expired"


def test_lapsed_via_rectype() -> None:
    order = _order(OrderQty=100, CumQty=0, Details=[{"RecType": 7}])
    assert interpret_order(order) == "expired"


def test_error_via_detail_state() -> None:
    order = _order(OrderQty=100, CumQty=0, Details=[{"RecType": 4, "State": 4}])
    assert interpret_order(order) == "error"


def test_error_takes_priority_over_fill() -> None:
    # 実務上は同時発生しないはずだが、優先順位（エラー最優先）を固定しておく
    order = _order(OrderQty=100, CumQty=100, Details=[{"RecType": 4, "State": 4}])
    assert interpret_order(order) == "error"


def test_unknown_when_no_signal() -> None:
    order = _order(State=99, OrderQty=0, CumQty=0)
    assert interpret_order(order) == "unknown"
