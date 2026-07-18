"""order_engine.py のテスト（絶対原則4：これなしに変更しない）。

固定する不変条件：
- risk_manager が却下した注文は kabuへ一切送られない（絶対原則2）
- DRY_RUN=True のときは承認されてもkabuへ実送信しない（絶対原則1）
- DRY_RUN=False かつ承認時のみ実送信し、is_entry/side が正しくペイロードに反映される
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from config.settings import RiskParams
from execution.order_engine import OrderEngine
from strategy.risk_manager import OrderRequest, RiskManager


class FakeKabuClient:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._response = response or {"OrderId": "20260718-1"}

    def send_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        return self._response


def _rm(equity: float = 1_000_000, **param_kw: object) -> RiskManager:
    params = RiskParams(**param_kw) if param_kw else RiskParams()  # type: ignore[arg-type]
    rm = RiskManager(account_equity=equity, params=params)
    rm.start_day(date(2026, 1, 5))
    return rm


def _entry(symbol: str = "1301", side: str = "buy", shares: int = 100, price: float = 1000.0) -> OrderRequest:
    return OrderRequest(symbol=symbol, side=side, shares=shares, price=price, is_entry=True)  # type: ignore[arg-type]


def _exit(symbol: str = "1301", side: str = "sell", shares: int = 100, price: float = 1000.0) -> OrderRequest:
    return OrderRequest(symbol=symbol, side=side, shares=shares, price=price, is_entry=False)  # type: ignore[arg-type]


# --- 却下は一切送信されない --------------------------------------------------------
def test_rejected_order_is_not_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("execution.order_engine.DRY_RUN", False)
    client = FakeKabuClient()
    rm = _rm(max_positions=1)
    rm.on_open("9999", notional=100_000.0)  # 既に上限まで建玉保有 → 別銘柄の新規は却下
    engine = OrderEngine(client=client, risk_manager=rm)  # type: ignore[arg-type]

    result = engine.submit(_entry(symbol="1301"))

    assert not result.decision.approved
    assert not result.sent
    assert result.kabu_payload is None
    assert client.calls == []


# --- DRY_RUN=True は承認されても未送信 ---------------------------------------------
def test_dry_run_true_approves_but_does_not_send(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("execution.order_engine.DRY_RUN", True)
    client = FakeKabuClient()
    rm = _rm()
    engine = OrderEngine(client=client, risk_manager=rm)  # type: ignore[arg-type]

    result = engine.submit(_entry())

    assert result.decision.approved
    assert result.dry_run is True
    assert not result.sent
    assert result.kabu_payload is not None
    assert client.calls == []


# --- DRY_RUN=False かつ承認 → 実送信 ------------------------------------------------
def test_dry_run_false_sends_order_with_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("execution.order_engine.DRY_RUN", False)
    client = FakeKabuClient(response={"OrderId": "XYZ"})
    rm = _rm()
    engine = OrderEngine(client=client, risk_manager=rm)  # type: ignore[arg-type]

    result = engine.submit(_entry(symbol="7203", side="buy", shares=200))

    assert result.sent
    assert result.kabu_response == {"OrderId": "XYZ"}
    assert len(client.calls) == 1
    payload = client.calls[0]
    assert payload["Symbol"] == "7203"
    assert payload["Qty"] == 200
    assert payload["Side"] == "2"  # 買い
    assert "Password" not in payload  # Password 付与は KabuClient.send_order 側の責務


# --- 新規建て/決済で CashMargin が切り替わる ----------------------------------------
def test_entry_uses_margin_open_cash_margin_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("execution.order_engine.DRY_RUN", False)
    client = FakeKabuClient()
    rm = _rm()
    engine = OrderEngine(client=client, risk_manager=rm)  # type: ignore[arg-type]

    engine.submit(_entry())

    assert client.calls[0]["CashMargin"] == 2  # 信用新規


def test_exit_uses_margin_close_cash_margin_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("execution.order_engine.DRY_RUN", False)
    client = FakeKabuClient()
    rm = _rm()
    engine = OrderEngine(client=client, risk_manager=rm)  # type: ignore[arg-type]
    # 決済はゲートを素通りするので、事前の建て有無に関わらずテストできる
    engine.submit(_exit())

    assert client.calls[0]["CashMargin"] == 3  # 信用返済


# --- 一般信用（無期限）に統一されていること -----------------------------------------
def test_uses_general_unlimited_margin_trade_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("execution.order_engine.DRY_RUN", False)
    client = FakeKabuClient()
    rm = _rm()
    engine = OrderEngine(client=client, risk_manager=rm)  # type: ignore[arg-type]

    engine.submit(_entry())

    assert client.calls[0]["MarginTradeType"] == 3


# --- 売り方向のサイドコード ---------------------------------------------------------
def test_sell_side_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("execution.order_engine.DRY_RUN", False)
    client = FakeKabuClient()
    rm = _rm()
    engine = OrderEngine(client=client, risk_manager=rm)  # type: ignore[arg-type]

    engine.submit(_exit(side="sell"))

    assert client.calls[0]["Side"] == "1"
