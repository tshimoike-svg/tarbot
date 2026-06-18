"""fetcher.py のテスト（実ネットワーク・実認証なし。フェイクセッションを注入）。

検証点：
- 認証フロー（auth_user → auth_refresh → Bearer 付与）
- ページネーション
- リトライ（429 → 200）とリトライ上限超過
- 401 での ID トークン再取得
- 日足の正規化（調整後優先・型・ソート・index）
- 認証情報不足のエラー
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from data.fetcher import (
    JQuantsAPIError,
    JQuantsAuthError,
    JQuantsClient,
)


class FakeResponse:
    def __init__(self, status_code: int, json_data: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._json = json_data or {}
        self.text = str(json_data)

    def json(self) -> dict[str, Any]:
        return self._json


class FakeSession:
    """request() ごとに用意したレスポンスを順に返す。呼び出しを記録する。"""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self._responses:
            raise AssertionError(f"想定外の追加リクエスト: {method} {url}")
        return self._responses.pop(0)


def _client(session: FakeSession, **kw: Any) -> JQuantsClient:
    return JQuantsClient(
        mailaddress="user@example.com",
        password="secret",
        session=session,  # type: ignore[arg-type]
        retry_backoff=0.0,
        sleep=lambda _s: None,
        load_env=False,
        **kw,
    )


_AUTH = [
    FakeResponse(200, {"refreshToken": "RT"}),
    FakeResponse(200, {"idToken": "IDT"}),
]


# --- 認証フロー --------------------------------------------------------------------
def test_auth_flow_and_bearer_header() -> None:
    session = FakeSession(
        [
            *_AUTH,
            FakeResponse(
                200,
                {
                    "daily_quotes": [
                        {"Date": "2026-01-05", "Code": "1301", "Close": 100, "Volume": 1000}
                    ]
                },
            ),
        ]
    )
    client = _client(session)
    df = client.get_daily_quotes(code="1301")

    # 1) auth_user → refreshToken, 2) auth_refresh(refreshtoken=RT), 3) daily_quotes(Bearer IDT)
    assert session.calls[0]["url"].endswith("/token/auth_user")
    assert session.calls[0]["json"] == {"mailaddress": "user@example.com", "password": "secret"}
    assert session.calls[1]["url"].endswith("/token/auth_refresh")
    assert session.calls[1]["params"] == {"refreshtoken": "RT"}
    assert session.calls[2]["headers"]["Authorization"] == "Bearer IDT"
    assert df.loc[df.index[0], "close"] == 100


def test_missing_credentials_raises() -> None:
    session = FakeSession([])
    client = JQuantsClient(session=session, load_env=False)  # type: ignore[arg-type]
    with pytest.raises(JQuantsAuthError):
        client.get_listed_info()


def test_refresh_token_skips_auth_user() -> None:
    session = FakeSession(
        [
            FakeResponse(200, {"idToken": "IDT"}),  # auth_refresh のみ（auth_user は呼ばれない）
            FakeResponse(200, {"info": [{"Code": "1301"}]}),
        ]
    )
    client = JQuantsClient(
        refresh_token="RT",
        session=session,  # type: ignore[arg-type]
        load_env=False,
        retry_backoff=0.0,
    )
    client.get_listed_info()
    assert session.calls[0]["url"].endswith("/token/auth_refresh")


# --- ページネーション --------------------------------------------------------------
def test_pagination_follows_key() -> None:
    session = FakeSession(
        [
            *_AUTH,
            FakeResponse(
                200,
                {
                    "daily_quotes": [{"Date": "2026-01-05", "Code": "1", "Close": 1, "Volume": 1}],
                    "pagination_key": "KEY2",
                },
            ),
            FakeResponse(
                200,
                {"daily_quotes": [{"Date": "2026-01-06", "Code": "1", "Close": 2, "Volume": 1}]},
            ),
        ]
    )
    client = _client(session)
    df = client.get_daily_quotes(code="1")
    assert len(df) == 2
    # 2ページ目のリクエストに pagination_key が乗る
    assert session.calls[-1]["params"].get("pagination_key") == "KEY2"


# --- リトライ ----------------------------------------------------------------------
def test_retry_on_429_then_success() -> None:
    session = FakeSession(
        [
            *_AUTH,
            FakeResponse(429, {}),
            FakeResponse(200, {"daily_quotes": []}),
        ]
    )
    client = _client(session)
    df = client.get_daily_quotes(code="1")
    assert df.empty
    assert list(df.columns) == ["date", "code", "open", "high", "low", "close", "volume"]


def test_retry_exhausted_raises() -> None:
    session = FakeSession([*_AUTH, FakeResponse(503, {}), FakeResponse(503, {})])
    client = _client(session, max_retries=1)
    with pytest.raises(JQuantsAPIError) as ei:
        client.get_daily_quotes(code="1")
    assert ei.value.status_code == 503


def test_401_triggers_reauth_and_retry() -> None:
    session = FakeSession(
        [
            *_AUTH,
            FakeResponse(401, {}),                      # 1回目の daily_quotes で失効
            FakeResponse(200, {"idToken": "IDT2"}),     # 強制 auth_refresh
            FakeResponse(200, {"daily_quotes": []}),    # 再試行成功
        ]
    )
    client = _client(session)
    client.get_daily_quotes(code="1")
    # 再認証後の Bearer が更新されている
    assert session.calls[-1]["headers"]["Authorization"] == "Bearer IDT2"


# --- 正規化 ------------------------------------------------------------------------
def test_normalization_prefers_adjusted_and_sorts() -> None:
    session = FakeSession(
        [
            *_AUTH,
            FakeResponse(
                200,
                {
                    "daily_quotes": [
                        {
                            "Date": "2026-01-06",
                            "Code": "1301",
                            "Open": 200, "High": 210, "Low": 190, "Close": 205, "Volume": 50,
                            "AdjustmentOpen": 100, "AdjustmentHigh": 105, "AdjustmentLow": 95,
                            "AdjustmentClose": 102, "AdjustmentVolume": 100,
                        },
                        {
                            "Date": "2026-01-05",
                            "Code": "1301",
                            "Open": 198, "High": 208, "Low": 188, "Close": 200, "Volume": 40,
                            # 調整後が欠落 → 素の値にフォールバック
                        },
                    ]
                },
            ),
        ]
    )
    client = _client(session)
    df = client.get_daily_quotes(code="1301")

    # Date 昇順、index は日付
    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.index) == [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")]
    # 1/6 は調整後優先（close=102, volume=100）
    assert df.loc[pd.Timestamp("2026-01-06"), "close"] == 102
    assert df.loc[pd.Timestamp("2026-01-06"), "volume"] == 100
    # 1/5 は素の値（close=200）
    assert df.loc[pd.Timestamp("2026-01-05"), "close"] == 200
    # 数値型
    assert pd.api.types.is_numeric_dtype(df["close"])


def test_daily_quotes_requires_code_or_date() -> None:
    session = FakeSession([*_AUTH])
    client = _client(session)
    with pytest.raises(ValueError):
        client.get_daily_quotes()


def test_listed_info_parsed() -> None:
    session = FakeSession(
        [
            *_AUTH,
            FakeResponse(
                200,
                {"info": [{"Code": "1301", "CompanyName": "極洋"}, {"Code": "1332"}]},
            ),
        ]
    )
    client = _client(session)
    info = client.get_listed_info(date_="2026-01-05")
    assert len(info) == 2
    assert "Code" in info.columns
