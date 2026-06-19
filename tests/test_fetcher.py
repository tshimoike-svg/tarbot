"""fetcher.py のテスト（J-Quants **v2**・x-api-key 認証。実通信なし）。"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from data.fetcher import JQuantsAPIError, JQuantsAuthError, JQuantsClient


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


def _client(session: FakeSession, **kw: Any) -> JQuantsClient:
    return JQuantsClient(
        api_key="KEY", session=session, retry_backoff=0.0,  # type: ignore[arg-type]
        sleep=lambda _s: None, load_env=False, **kw,
    )


# --- 認証ヘッダ・エンドポイント ----------------------------------------------------
def test_sends_x_api_key_to_v2_endpoint() -> None:
    session = FakeSession(
        [FakeResponse(200, {"data": [{"Date": "2026-01-05", "Code": "1301", "C": 100, "Vo": 1000}]})]
    )
    client = _client(session)
    df = client.get_daily_quotes(code="1301")
    call = session.calls[0]
    assert call["url"].endswith("/equities/bars/daily")
    assert call["headers"]["x-api-key"] == "KEY"
    assert "Authorization" not in call["headers"]  # 旧 Bearer は使わない
    assert df.loc[df.index[0], "close"] == 100


def test_missing_api_key_raises() -> None:
    client = JQuantsClient(session=FakeSession([]), load_env=False)  # type: ignore[arg-type]
    with pytest.raises(JQuantsAuthError):
        client.get_listed_info()


# --- ページネーション --------------------------------------------------------------
def test_pagination_follows_key() -> None:
    session = FakeSession(
        [
            FakeResponse(200, {"data": [{"Date": "2026-01-05", "Code": "1", "C": 1, "Vo": 1}],
                               "pagination_key": "K2"}),
            FakeResponse(200, {"data": [{"Date": "2026-01-06", "Code": "1", "C": 2, "Vo": 1}]}),
        ]
    )
    client = _client(session)
    df = client.get_daily_quotes(code="1")
    assert len(df) == 2
    assert session.calls[-1]["params"].get("pagination_key") == "K2"


# --- リトライ ----------------------------------------------------------------------
def test_retry_on_429_then_success() -> None:
    session = FakeSession([FakeResponse(429, {}), FakeResponse(200, {"data": []})])
    df = _client(session).get_daily_quotes(code="1")
    assert df.empty
    assert list(df.columns) == ["date", "code", "open", "high", "low", "close", "volume"]


def test_retry_exhausted_raises() -> None:
    session = FakeSession([FakeResponse(503, {}), FakeResponse(503, {})])
    with pytest.raises(JQuantsAPIError) as ei:
        _client(session, max_retries=1).get_daily_quotes(code="1")
    assert ei.value.status_code == 503


# --- 正規化（調整後優先） ----------------------------------------------------------
def test_normalization_prefers_adjusted_and_sorts() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {"data": [
                    {"Date": "2026-01-06", "Code": "1301",
                     "O": 200, "H": 210, "L": 190, "C": 205, "Vo": 50,
                     "AdjO": 100, "AdjH": 105, "AdjL": 95, "AdjC": 102, "AdjVo": 100},
                    {"Date": "2026-01-05", "Code": "1301",
                     "O": 198, "H": 208, "L": 188, "C": 200, "Vo": 40},  # 調整後欠落→素の値
                ]},
            )
        ]
    )
    df = _client(session).get_daily_quotes(code="1301")
    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.index) == [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")]
    assert df.loc[pd.Timestamp("2026-01-06"), "close"] == 102   # 調整後
    assert df.loc[pd.Timestamp("2026-01-06"), "volume"] == 100
    assert df.loc[pd.Timestamp("2026-01-05"), "close"] == 200   # 素の値
    assert pd.api.types.is_numeric_dtype(df["close"])


# --- パラメータ・日付正規化 --------------------------------------------------------
def test_date_params_normalized_to_yyyymmdd() -> None:
    session = FakeSession([FakeResponse(200, {"data": []})])
    _client(session).get_daily_quotes(code="1301", from_="2024-06-01", to="2026-06-01")
    params = session.calls[0]["params"]
    assert params["from"] == "20240601"
    assert params["to"] == "20260601"
    assert params["code"] == "1301"


def test_daily_quotes_requires_code_or_date() -> None:
    with pytest.raises(ValueError):
        _client(FakeSession([])).get_daily_quotes()


def test_listed_master_parsed() -> None:
    session = FakeSession(
        [FakeResponse(200, {"data": [{"Code": "1301", "CompanyName": "極洋"}, {"Code": "1332"}]})]
    )
    info = _client(session).get_listed_info(date_="2026-01-05")
    assert len(info) == 2
    assert session.calls[0]["url"].endswith("/equities/master")
    assert session.calls[0]["params"]["date"] == "20260105"
