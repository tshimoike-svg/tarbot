"""J-Quants API クライアント（過去データ取得）。

docs/trading_bot_design_v2.md §3, §13 / CLAUDE.md「現在のタスク 着手順序2」に対応。

設計方針：
- 認証は **固定APIキーではなく** メール＋パスワード →（1週間有効の）リフレッシュトークン
  →（約24時間有効の）ID トークン、の2段。トークンはキャッシュして自動更新する。
- 認証情報は `.env` から読む（python-dotenv）。コードやログに秘匿情報を残さない。
- 外部API規約（CLAUDE.md）に従い **タイムアウト・リトライ（指数バックオフ）・レート制限(429)・
  ページネーション** を実装する。
- 取得した日足は split を跨いだバイアスを避けるため **調整後（Adjustment*）価格を優先**して
  正規化する（§7 のバイアス対策）。
- サバイバーシップ・バイアス対策：`get_listed_info(date_)` は **その日時点の上場ユニバース**
  （のちに上場廃止となった銘柄を含む）を返す。過去日のユニバースを辿って和集合を取ることで
  廃止銘柄込みの検証母集団を構築できる。

無料プランの制約（2026-06 時点）：日足のみ・過去約2年・12週遅延。分足は Light 以上。
本クライアントは日足 (`/prices/daily_quotes`) と上場情報 (`/listed/info`) を実装する。
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

DEFAULT_BASE_URL = "https://api.jquants.com/v1"
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
# ID トークンの想定有効期間（約24h）。安全側に短めへ丸める。
_ID_TOKEN_TTL_SEC = 23 * 60 * 60

# 日足の正規化に使う列（調整後を優先し、無ければ素の値にフォールバック）
_PRICE_FIELD_MAP = {
    "open": ("AdjustmentOpen", "Open"),
    "high": ("AdjustmentHigh", "High"),
    "low": ("AdjustmentLow", "Low"),
    "close": ("AdjustmentClose", "Close"),
    "volume": ("AdjustmentVolume", "Volume"),
}


class JQuantsError(Exception):
    """J-Quants クライアントの基底例外。"""


class JQuantsAuthError(JQuantsError):
    """認証情報不足・認証失敗。"""


class JQuantsAPIError(JQuantsError):
    """API がエラー応答を返した。"""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"J-Quants API error {status_code}: {message}")
        self.status_code = status_code


class JQuantsClient:
    """J-Quants REST API の薄いクライアント。

    認証情報は引数 > 環境変数（.env）の順で解決する。
    - JQUANTS_REFRESH_TOKEN があればそれを使う（mail/pass 不要）
    - 無ければ JQUANTS_MAILADDRESS / JQUANTS_PASSWORD から取得する

    ネットワーク I/O は `session`（requests.Session 互換）に委譲するため、
    テストではフェイクセッションを注入できる。
    """

    def __init__(
        self,
        *,
        mailaddress: str | None = None,
        password: str | None = None,
        refresh_token: str | None = None,
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
        self._mail = mailaddress or os.getenv("JQUANTS_MAILADDRESS")
        self._password = password or os.getenv("JQUANTS_PASSWORD")
        self._refresh_token = refresh_token or os.getenv("JQUANTS_REFRESH_TOKEN")
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._sleep = sleep

        self._id_token: str | None = None
        self._id_token_expiry: float = 0.0

    # --- 認証 --------------------------------------------------------------------
    def _ensure_refresh_token(self, *, force: bool = False) -> str:
        if not force and self._refresh_token:
            return self._refresh_token
        if not (self._mail and self._password):
            raise JQuantsAuthError(
                "リフレッシュトークンが無く、JQUANTS_MAILADDRESS/PASSWORD も未設定です。"
                "（.env を確認してください）"
            )
        data = self._http(
            "POST",
            "/token/auth_user",
            json={"mailaddress": self._mail, "password": self._password},
            auth=False,
        )
        token = data.get("refreshToken")
        if not token:
            raise JQuantsAuthError("auth_user 応答に refreshToken がありません")
        self._refresh_token = token
        return token

    def _ensure_id_token(self, *, force: bool = False) -> str:
        now = time.monotonic()
        if not force and self._id_token and now < self._id_token_expiry:
            return self._id_token
        refresh_token = self._ensure_refresh_token()
        data = self._http(
            "POST",
            "/token/auth_refresh",
            params={"refreshtoken": refresh_token},
            auth=False,
        )
        token = data.get("idToken")
        if not token:
            raise JQuantsAuthError("auth_refresh 応答に idToken がありません")
        self._id_token = token
        self._id_token_expiry = time.monotonic() + _ID_TOKEN_TTL_SEC
        return token

    # --- 低レベル HTTP（タイムアウト・リトライ・401再認証） ----------------------
    def _http(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        auth: bool = True,
        _allow_reauth: bool = True,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        headers: dict[str, str] = {}
        if auth:
            headers["Authorization"] = f"Bearer {self._ensure_id_token()}"

        for attempt in range(self._max_retries + 1):
            response = self._session.request(
                method,
                url,
                params=params,
                json=json,
                headers=headers,
                timeout=self._timeout,
            )
            status = response.status_code
            if status == 200:
                return response.json()

            if status == 401 and auth and _allow_reauth:
                # ID トークン失効の可能性 → 強制更新して一度だけ再試行
                logger.info("J-Quants 401: ID トークンを再取得します")
                headers["Authorization"] = f"Bearer {self._ensure_id_token(force=True)}"
                _allow_reauth = False
                continue

            if status in _RETRY_STATUS and attempt < self._max_retries:
                wait = self._retry_backoff * (2**attempt)
                logger.warning(
                    "J-Quants %s %s → %s。%.2fs 後に再試行 (%d/%d)",
                    method,
                    path,
                    status,
                    wait,
                    attempt + 1,
                    self._max_retries,
                )
                self._sleep(wait)
                continue

            raise JQuantsAPIError(status, _safe_text(response))

        raise JQuantsAPIError(0, "リトライ上限に達しました")

    def _get_paginated(
        self, path: str, params: dict[str, Any], data_key: str
    ) -> list[dict[str, Any]]:
        """pagination_key を辿って全レコードを集める。"""
        records: list[dict[str, Any]] = []
        pagination_key: str | None = None
        while True:
            page_params = dict(params)
            if pagination_key:
                page_params["pagination_key"] = pagination_key
            data = self._http("GET", path, params=page_params, auth=True)
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
        """日足株価を取得し、OHLCV へ正規化して返す。

        code か date_ のいずれかは指定すること（J-Quants の仕様）。価格は調整後を優先。

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
            params["date"] = date_
        if from_ is not None:
            params["from"] = from_
        if to is not None:
            params["to"] = to

        records = self._get_paginated("/prices/daily_quotes", params, "daily_quotes")
        return _normalize_daily_quotes(records, single_symbol=code is not None and date_ is None)

    def get_listed_info(
        self, *, code: str | None = None, date_: str | None = None
    ) -> pd.DataFrame:
        """上場銘柄情報を取得。date_ を指定するとその日時点のユニバース（廃止銘柄含む）。"""
        params: dict[str, Any] = {}
        if code is not None:
            params["code"] = code
        if date_ is not None:
            params["date"] = date_
        records = self._get_paginated("/listed/info", params, "info")
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
    """生レコードを OHLCV DataFrame に正規化（調整後価格を優先）。"""
    columns = ["date", "code", "open", "high", "low", "close", "volume"]
    if not records:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for rec in records:
        date_val: Any = rec.get("Date")
        row: dict[str, Any] = {
            "date": pd.to_datetime(date_val),
            "code": rec.get("Code"),
        }
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
