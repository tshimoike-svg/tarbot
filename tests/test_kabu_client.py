"""kabu_client.py のテスト（トークン認証・読み取り専用エンドポイント。実通信なし）。"""

from __future__ import annotations

from typing import Any

import pytest

from execution.kabu_client import KabuAPIError, KabuAuthError, KabuClient, KabuError


class FakeResponse:
    def __init__(
        self, status_code: int, json_data: dict[str, Any] | list[Any] | None = None
    ) -> None:
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = str(json_data)

    def json(self) -> dict[str, Any] | list[Any]:
        return self._json


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self._responses:
            raise AssertionError(f"想定外の追加リクエスト: {method} {url}")
        return self._responses.pop(0)


def _client(session: FakeSession, **kw: Any) -> KabuClient:
    return KabuClient(
        api_password="PW", session=session, retry_backoff=0.0,  # type: ignore[arg-type]
        sleep=lambda _s: None, load_env=False, **kw,
    )


# --- 環境・ベースURL ---------------------------------------------------------------
def test_demo_env_uses_port_18081() -> None:
    client = _client(FakeSession([]))
    assert client.env == "demo"


def test_prod_env_uses_port_18080() -> None:
    client = _client(FakeSession([]), env="prod")
    assert "18080" in client._base_url  # noqa: SLF001


# --- 本番/検証で別パスワードを解決する ------------------------------------------------
def test_resolves_env_specific_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KABU_API_PASSWORD_PROD", "PROD_PW")
    monkeypatch.setenv("KABU_API_PASSWORD_DEMO", "DEMO_PW")
    prod = KabuClient(session=FakeSession([]), env="prod", load_env=False)  # type: ignore[arg-type]
    demo = KabuClient(session=FakeSession([]), env="demo", load_env=False)  # type: ignore[arg-type]
    assert prod._api_password == "PROD_PW"  # noqa: SLF001
    assert demo._api_password == "DEMO_PW"  # noqa: SLF001


def test_falls_back_to_shared_password_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KABU_API_PASSWORD_DEMO", raising=False)
    monkeypatch.setenv("KABU_API_PASSWORD", "SHARED_PW")
    client = KabuClient(session=FakeSession([]), env="demo", load_env=False)  # type: ignore[arg-type]
    assert client._api_password == "SHARED_PW"  # noqa: SLF001


# --- 認証 --------------------------------------------------------------------------
def test_get_token_sends_api_password_and_caches() -> None:
    session = FakeSession([FakeResponse(200, {"ResultCode": 0, "Token": "TOK123"})])
    client = _client(session)
    token = client.get_token()
    assert token == "TOK123"
    assert session.calls[0]["url"].endswith("/token")
    assert session.calls[0]["json"] == {"APIPassword": "PW"}

    # 2回目はキャッシュを返し、追加リクエストは発生しない
    assert client.get_token() == "TOK123"
    assert len(session.calls) == 1


def test_missing_api_password_raises() -> None:
    client = KabuClient(session=FakeSession([]), load_env=False)  # type: ignore[arg-type]
    with pytest.raises(KabuAuthError):
        client.get_token()


def test_token_result_code_nonzero_raises_auth_error() -> None:
    session = FakeSession([FakeResponse(200, {"ResultCode": 4001001, "Message": "NG"})])
    with pytest.raises(KabuAuthError):
        _client(session).get_token()


# --- 認証ヘッダの伝播 ----------------------------------------------------------------
def test_authenticated_call_sends_x_api_key_header() -> None:
    session = FakeSession(
        [
            FakeResponse(200, {"ResultCode": 0, "Token": "TOK123"}),
            FakeResponse(200, {"CurrentPrice": 1234.5}),
        ]
    )
    client = _client(session)
    board = client.get_board("1301")
    assert board["CurrentPrice"] == 1234.5
    call = session.calls[1]
    assert call["url"].endswith("/board/1301@1")
    assert call["headers"]["X-API-KEY"] == "TOK123"


# --- リトライ・エラー ----------------------------------------------------------------
def test_retry_on_503_then_success() -> None:
    session = FakeSession(
        [
            FakeResponse(200, {"ResultCode": 0, "Token": "TOK123"}),
            FakeResponse(503, {}),
            FakeResponse(200, {"StockAccountWallet": 1000000}),
        ]
    )
    cash = _client(session).get_wallet_cash()
    assert cash["StockAccountWallet"] == 1000000


def test_retry_exhausted_raises_api_error() -> None:
    session = FakeSession(
        [
            FakeResponse(200, {"ResultCode": 0, "Token": "TOK123"}),
            FakeResponse(503, {}),
            FakeResponse(503, {}),
        ]
    )
    with pytest.raises(KabuAPIError) as ei:
        _client(session, max_retries=1).get_wallet_cash()
    assert ei.value.status_code == 503


# --- 残高照会（get_positions） ------------------------------------------------------
def test_get_positions_returns_list() -> None:
    session = FakeSession(
        [
            FakeResponse(200, {"ResultCode": 0, "Token": "TOK123"}),
            FakeResponse(200, [{"ExecutionID": "E1", "Symbol": "1301"}]),
        ]
    )
    positions = _client(session).get_positions()
    assert positions == [{"ExecutionID": "E1", "Symbol": "1301"}]
    assert session.calls[1]["url"].endswith("/positions")


def test_get_positions_builds_query_string() -> None:
    session = FakeSession(
        [
            FakeResponse(200, {"ResultCode": 0, "Token": "TOK123"}),
            FakeResponse(200, []),
        ]
    )
    _client(session).get_positions(symbol="1301", product="2", side="2")
    url = session.calls[1]["url"]
    assert "symbol=1301" in url
    assert "product=2" in url
    assert "side=2" in url


# --- 注文約定照会（get_orders） ------------------------------------------------------
def test_get_orders_returns_list() -> None:
    session = FakeSession(
        [
            FakeResponse(200, {"ResultCode": 0, "Token": "TOK123"}),
            FakeResponse(200, [{"ID": "ORD1", "State": 5, "OrderQty": 100, "CumQty": 100}]),
        ]
    )
    orders = _client(session).get_orders()
    assert orders == [{"ID": "ORD1", "State": 5, "OrderQty": 100, "CumQty": 100}]
    assert session.calls[1]["url"].endswith("/orders")


def test_get_orders_builds_query_string() -> None:
    session = FakeSession(
        [
            FakeResponse(200, {"ResultCode": 0, "Token": "TOK123"}),
            FakeResponse(200, []),
        ]
    )
    _client(session).get_orders(order_id="ORD1", symbol="1301", state="5")
    url = session.calls[1]["url"]
    assert "id=ORD1" in url
    assert "symbol=1301" in url
    assert "state=5" in url


# --- 発注（send_order） ------------------------------------------------------------
def test_send_order_merges_order_password_and_posts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KABU_ORDER_PASSWORD_DEMO", "ORDER_PW")
    session = FakeSession(
        [
            FakeResponse(200, {"ResultCode": 0, "Token": "TOK123"}),
            FakeResponse(200, {"OrderId": "ABC"}),
        ]
    )
    client = KabuClient(
        api_password="PW", session=session, retry_backoff=0.0,  # type: ignore[arg-type]
        sleep=lambda _s: None, load_env=False,
    )
    result = client.send_order({"Symbol": "1301", "Qty": 100})
    assert result == {"OrderId": "ABC"}

    call = session.calls[1]
    assert call["url"].endswith("/sendorder")
    assert call["json"] == {"Symbol": "1301", "Qty": 100, "Password": "ORDER_PW"}
    assert call["headers"]["X-API-KEY"] == "TOK123"


def test_send_order_missing_order_password_raises() -> None:
    client = _client(FakeSession([]))
    with pytest.raises(KabuAuthError):
        client.send_order({"Symbol": "1301"})


def test_connection_error_raises_kabu_error() -> None:
    class RaisingSession:
        def request(self, *_a: Any, **_kw: Any) -> Any:
            import requests

            raise requests.exceptions.ConnectionError("refused")

    client = KabuClient(
        api_password="PW", session=RaisingSession(),  # type: ignore[arg-type]
        load_env=False,
    )
    with pytest.raises(KabuError):
        client.get_token()
