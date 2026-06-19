"""J-Quants API クライアント（過去データ取得・**v2**）。

docs/trading_bot_design_v3.md §4 / CLAUDE.md「現在のタスク」に対応。

⚠️ J-Quants は 2025年末に **API v2** へ移行し、認証が大幅に簡略化された：
- ベースURL：`https://api.jquants.com/v2`
- 認証：ダッシュボードで発行した **APIキーを `x-api-key` ヘッダ**に付けるだけ
  （旧 v1 の mailaddress/password→refreshToken→idToken の流れは廃止）
- レスポンスは `data` 配列。列名は短縮（O/H/L/C/Vo、調整後 AdjO/AdjH/AdjL/AdjC/AdjVo）

APIキーは `.env` の `JQUANTS_API_KEY` から読む。コードやログに秘匿情報を残さない。
外部API規約（CLAUDE.md）に従いタイムアウト・指数バックオフ・レート制限・ページネーションを実装。
split バイアス回避のため調整後（Adj*）価格を優先して正規化する（§7）。

無料プラン：日足のみ・過去約2年・12週遅延（バックテストに遅延は無関係）。
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

__all__ = [
    "JQuantsError",
    "JQuantsAuthError",
    "JQuantsAPIError",
    "JQuantsClient",
]

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.jquants.com/v2"
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

# v2 の列名 → 正規化後の OHLCV（調整後を優先し、無ければ素の値にフォールバック）
_PRICE_FIELD_MAP = {
    "open": ("AdjO", "O"),
    "high": ("AdjH", "H"),
    "low": ("AdjL", "L"),
    "close": ("AdjC", "C"),
    "volume": ("AdjVo", "Vo"),
}


class JQuantsError(Exception):
    """J-Quants クライアントの基底例外。"""


class JQuantsAuthError(JQuantsError):
    """認証情報（APIキー）不足・認証失敗。"""


class JQuantsAPIError(JQuantsError):
    """API がエラー応答を返した。"""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"J-Quants API error {status_code}: {message}")
        self.status_code = status_code


def _ymd(value: str | None) -> str | None:
    """日付を v2 の YYYYMMDD 形式に正規化（ハイフン等を除去）。"""
    if value is None:
        return None
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits or None


class JQuantsClient:
    """J-Quants REST API v2 の薄いクライアント（APIキー認証）。

    APIキーは引数 > 環境変数 `JQUANTS_API_KEY`（.env）の順で解決する。
    ネットワーク I/O は `session`（requests.Session 互換）に委譲するため、テストでは
    フェイクセッションを注入できる。
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
        load_env: bool = True,
    ) -> None:
        if load_env:
            load_dotenv()
        self._api_key = api_key or os.getenv("JQUANTS_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._sleep = sleep

    def _require_api_key(self) -> str:
        if not self._api_key:
            raise JQuantsAuthError(
                "JQUANTS_API_KEY が未設定です。J-Quants ダッシュボードで発行した APIキーを "
                ".env の JQUANTS_API_KEY に設定してください（README 参照）。"
            )
        return self._api_key

    # --- 低レベル HTTP（タイムアウト・リトライ・x-api-key） ----------------------
    def _http(
        self, method: str, path: str, *, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        headers = {"x-api-key": self._require_api_key()}

        for attempt in range(self._max_retries + 1):
            response = self._session.request(
                method, url, params=params, headers=headers, timeout=self._timeout
            )
            status = response.status_code
            if status == 200:
                return response.json()
            if status in _RETRY_STATUS and attempt < self._max_retries:
                wait = self._retry_backoff * (2**attempt)
                logger.warning(
                    "J-Quants %s %s → %s。%.2fs 後に再試行 (%d/%d)",
                    method, path, status, wait, attempt + 1, self._max_retries,
                )
                self._sleep(wait)
                continue
            raise JQuantsAPIError(status, _safe_text(response))

        raise JQuantsAPIError(0, "リトライ上限に達しました")

    def _get_paginated(
        self, path: str, params: dict[str, Any], data_key: str = "data"
    ) -> list[dict[str, Any]]:
        """pagination_key を辿って全レコードを集める（v2 はデータを data 配列で返す）。"""
        records: list[dict[str, Any]] = []
        pagination_key: str | None = None
        while True:
            page_params = dict(params)
            if pagination_key:
                page_params["pagination_key"] = pagination_key
            data = self._http("GET", path, params=page_params)
            records.extend(data.get(data_key, []))
            pagination_key = data.get("pagination_key")
            if not pagination_key:
                break
        return records

    # --- 公開メソッド ------------------------------------------------------------
    def get_daily_quotes(
        self,
        *,
        code: str | None = None,
        date_: str | None = None,
        from_: str | None = None,
        to: str | None = None,
    ) -> pd.DataFrame:
        """日足株価（/equities/bars/daily）を取得し OHLCV へ正規化して返す。

        code か date_ のいずれかは必須。範囲は from_/to（YYYY-MM-DD / YYYYMMDD どちらも可）。
        価格は調整後（Adj*）を優先。

        Returns:
            列 open/high/low/close/volume/code を持つ DataFrame。code 単独指定時は
            Date を index にし昇順ソート。
        """
        if code is None and date_ is None:
            raise ValueError("code か date_ のいずれかを指定してください")
        params: dict[str, Any] = {}
        if code is not None:
            params["code"] = code
        if date_ is not None:
            params["date"] = _ymd(date_)
        if from_ is not None:
            params["from"] = _ymd(from_)
        if to is not None:
            params["to"] = _ymd(to)

        records = self._get_paginated("/equities/bars/daily", params)
        return _normalize_daily_quotes(records, single_symbol=code is not None and date_ is None)

    def get_listed_info(
        self, *, code: str | None = None, date_: str | None = None
    ) -> pd.DataFrame:
        """上場銘柄一覧（/equities/master）を取得。date_ でその日時点のユニバース。"""
        params: dict[str, Any] = {}
        if code is not None:
            params["code"] = code
        if date_ is not None:
            params["date"] = _ymd(date_)
        records = self._get_paginated("/equities/master", params)
        return pd.DataFrame(records)


def _safe_text(response: requests.Response) -> str:
    try:
        return response.text[:500]
    except Exception:  # noqa: BLE001 - ログ用途、失敗しても握りつぶす
        return "<本文取得不可>"


def _pick_field(record: dict[str, Any], candidates: Sequence[str]) -> Any:
    """調整後→素の順で最初に非 None の値を返す。"""
    for key in candidates:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _normalize_daily_quotes(
    records: list[dict[str, Any]], *, single_symbol: bool
) -> pd.DataFrame:
    """v2 の生レコードを OHLCV DataFrame に正規化（調整後価格を優先）。"""
    columns = ["date", "code", "open", "high", "low", "close", "volume"]
    if not records:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for rec in records:
        date_val: Any = rec.get("Date")
        row: dict[str, Any] = {"date": pd.to_datetime(date_val), "code": rec.get("Code")}
        for out_col, candidates in _PRICE_FIELD_MAP.items():
            row[out_col] = _pick_field(rec, candidates)
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    if single_symbol:
        df = df.set_index("date").drop(columns=["code"])
    return df
