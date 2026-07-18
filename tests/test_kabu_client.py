"""kabu_client.py のテスト（トークン認証・読み取り専用エンドポイント。実通信なし）。"""

from __future__ import annotations

from typing import Any

import pytest

from execution.kabu_client import KabuAPIError, KabuAuthError, KabuClient, KabuError


class FakeResponse:
    def __init__(self, status_code: int, json_data: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._json = json_data or {}
        self.text = str(json_data)

    def json(self) -> dict[str, Any]:
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
