"""kabuステーションAPI REST クライアント（認証・残高照会・板情報・発注のHTTPラッパー）。

kabuステーション（Windows GUIアプリ）がローカルで起動・ログイン中でないと
API（http://localhost:18080 または 18081 配下）は応答しない（CLAUDE.md ドメイン制約）。

環境は「本番」（ポート18080・実残高/実発注）と「検証」（ポート18081・常に固定値を
返す・実発注不可）の2つがあり、ポートと環境設定が一致しないと接続できない。

認証フロー：
  1. POST /token に {"APIPassword": <パスワード>} を送る
  2. レスポンスの Token を以後のリクエストの X-API-KEY ヘッダに使う
  3. トークンは kabuステーション再起動・PC再起動・有効期限切れで失効する

APIパスワードは本番用・検証用で別々に設定する仕様（kabuステーション システム設定
「APIタブ」）。.env は KABU_API_PASSWORD_PROD / KABU_API_PASSWORD_DEMO を使う。

発注（send_order）には別途「取引パスワード」（kabuステーションの注文設定で登録する、
APIパスワードとは異なるパスワード）が要る。.env は KABU_ORDER_PASSWORD_PROD /
KABU_ORDER_PASSWORD_DEMO を使う。

本モジュールは HTTP・認証・リトライを担うだけの薄いラッパー。**発注可否の判断
（risk_manager 経由・DRY_RUN確認）は一切行わない**（絶対原則1・2・6）。send_order を
いつ・どの内容で呼ぶかは execution/order_engine.py 側の責務。
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any, Literal
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

__all__ = [
    "KabuError",
    "KabuAuthError",
    "KabuAPIError",
    "KabuClient",
]

logger = logging.getLogger(__name__)

KabuEnv = Literal["prod", "demo"]

_PORT_BY_ENV: dict[KabuEnv, int] = {"prod": 18080, "demo": 18081}
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class KabuError(Exception):
    """kabuステーションAPIクライアントの基底例外。"""


class KabuAuthError(KabuError):
    """APIパスワード未設定・トークン発行失敗。"""


class KabuAPIError(KabuError):
    """APIがエラー応答を返した。"""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"kabuステーションAPI error {status_code}: {message}")
        self.status_code = status_code


class KabuClient:
    """kabuステーションAPI の薄いクライアント（トークン認証・読み取り専用メソッドのみ）。

    APIパスワードは引数 > 環境変数 `KABU_API_PASSWORD`（.env）の順で解決する。
    ネットワーク I/O は `session`（requests.Session 互換）に委譲するため、テストでは
    フェイクセッションを注入できる。
    """

    def __init__(
        self,
        *,
        api_password: str | None = None,
        order_password: str | None = None,
        env: KabuEnv = "demo",
        base_url: str | None = None,
        session: requests.Session | None = None,
        timeout: float = 10.0,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
        load_env: bool = True,
    ) -> None:
        if load_env:
            load_dotenv()
        env_var = "KABU_API_PASSWORD_PROD" if env == "prod" else "KABU_API_PASSWORD_DEMO"
        self._api_password = (
            api_password or os.getenv(env_var) or os.getenv("KABU_API_PASSWORD")
        )
        order_env_var = "KABU_ORDER_PASSWORD_PROD" if env == "prod" else "KABU_ORDER_PASSWORD_DEMO"
        self._order_password = (
            order_password or os.getenv(order_env_var) or os.getenv("KABU_ORDER_PASSWORD")
        )
        self._env: KabuEnv = env
        self._base_url = (base_url or f"http://localhost:{_PORT_BY_ENV[env]}/kabusapi").rstrip("/")
        self._session = session or requests.Session()
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._sleep = sleep
        self._token: str | None = None

    @property
    def env(self) -> KabuEnv:
        return self._env

    def _require_api_password(self) -> str:
        if not self._api_password:
            env_var = "KABU_API_PASSWORD_PROD" if self._env == "prod" else "KABU_API_PASSWORD_DEMO"
            raise KabuAuthError(
                f"{env_var} が未設定です。kabuステーションのシステム設定「APIタブ」で登録した"
                f"{'本番用' if self._env == 'prod' else '検証用'}パスワードを .env の {env_var} "
                "に設定してください（本番/検証で別パスワード）。"
            )
        return self._api_password

    def _require_order_password(self) -> str:
        if not self._order_password:
            env_var = "KABU_ORDER_PASSWORD_PROD" if self._env == "prod" else "KABU_ORDER_PASSWORD_DEMO"
            raise KabuAuthError(
                f"{env_var} が未設定です。kabuステーションの注文設定で登録した"
                "取引パスワード（APIパスワードとは別）を .env に設定してください。"
            )
        return self._order_password

    # --- 低レベル HTTP（タイムアウト・リトライ） --------------------------------------
    def _http(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        auth: bool = True,
    ) -> Any:
        url = f"{self._base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["X-API-KEY"] = self.get_token()

        for attempt in range(self._max_retries + 1):
            try:
                response = self._session.request(
                    method, url, json=json_body, headers=headers, timeout=self._timeout
                )
            except requests.exceptions.ConnectionError as exc:
                raise KabuError(
                    "kabuステーションAPIに接続できません。kabuステーションが起動・"
                    f"ログイン中か、環境({self._env})とポート({self._base_url})が"
                    "一致しているか確認してください。"
                ) from exc

            status = response.status_code
            if status == 200:
                return response.json()
            if status in _RETRY_STATUS and attempt < self._max_retries:
                wait = self._retry_backoff * (2**attempt)
                logger.warning(
                    "kabuステーションAPI %s %s → %s。%.2fs 後に再試行 (%d/%d)",
                    method, path, status, wait, attempt + 1, self._max_retries,
                )
                self._sleep(wait)
                continue
            raise KabuAPIError(status, _safe_text(response))

        raise KabuAPIError(0, "リトライ上限に達しました")

    # --- 認証 ------------------------------------------------------------------------
    def get_token(self, *, force_refresh: bool = False) -> str:
        """トークンを取得する（キャッシュ済みならそれを返す）。

        force_refresh=True で再発行する（kabuステーション再起動後の失効時など）。
        """
        if self._token is not None and not force_refresh:
            return self._token

        password = self._require_api_password()
        data = self._http(
            "POST", "/token", json_body={"APIPassword": password}, auth=False
        )
        result_code = data.get("ResultCode")
        token = data.get("Token")
        if result_code != 0 or not token:
            raise KabuAuthError(
                f"トークン発行に失敗しました（ResultCode={result_code}）。"
                "APIパスワードとkabuステーションのAPI設定を確認してください。"
            )
        self._token = token
        return token

    # --- 公開メソッド（読み取り専用） ---------------------------------------------------
    def get_wallet_cash(self) -> dict[str, Any]:
        """現物買付可能額（GET /wallet/cash）。"""
        return self._http("GET", "/wallet/cash")

    def get_positions(
        self,
        *,
        symbol: str | None = None,
        product: str | None = None,
        side: str | None = None,
    ) -> list[dict[str, Any]]:
        """保有建玉一覧（GET /positions）。信用返済の ClosePositions に使う建玉ID
        （レスポンスの ExecutionID）を調べるのに使う。

        Args:
            symbol: 銘柄コードで絞り込み（省略可）。
            product: "0"=すべて/"1"=現物/"2"=信用/"3"=先物/"4"=OP（省略可）。
            side: "1"=売/"2"=買で絞り込み（省略可）。
        """
        params = {"symbol": symbol, "product": product, "side": side}
        query = urlencode({k: v for k, v in params.items() if v is not None})
        path = f"/positions?{query}" if query else "/positions"
        result = self._http("GET", path)
        return result if isinstance(result, list) else []

    def get_wallet_margin(self) -> dict[str, Any]:
        """信用建余力（GET /wallet/margin）。"""
        return self._http("GET", "/wallet/margin")

    def get_board(self, symbol: str, exchange: int = 1) -> dict[str, Any]:
        """時価・板情報（GET /board/{symbol}@{exchange}）。exchange既定=1（東証）。"""
        return self._http("GET", f"/board/{symbol}@{exchange}")

    # --- 発注（POST /sendorder） -------------------------------------------------------
    def send_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        """注文を送信する（取引パスワードを自動付与してPOST /sendorder）。

        本メソッドは HTTP 送信のみを担う。**呼ぶかどうかの判断（risk_manager 通過・
        DRY_RUN確認）は呼び出し側（order_engine.py）の責務**であり、本メソッドは
        一切の可否判断をしない（絶対原則2・6：バイパスを作らないため、判断ロジックを
        ここに置かない）。

        Args:
            payload: Password 以外の sendorder ボディ（Symbol/Side/Qty 等）。
        """
        body = {**payload, "Password": self._require_order_password()}
        return self._http("POST", "/sendorder", json_body=body)


def _safe_text(response: requests.Response) -> str:
    try:
        return response.text[:500]
    except Exception:  # noqa: BLE001 - ログ用途、失敗しても握りつぶす
        return "<本文取得不可>"
