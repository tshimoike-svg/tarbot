"""実発注エンジン（絶対原則1・2・6が実際にコードとして交わる場所）。

新規建て・決済はすべてまず `risk_manager.check_order()` を通し、承認されたものだけを
kabuステーションAPI（KabuClient.send_order）へ渡す。バイパス経路は作らない（絶対原則2）。

`config.settings.DRY_RUN`（既定 True）のときは risk_manager の判定までは行うが、
kabuステーションへの実際のPOSTは一切行わない（絶対原則1）。DRY_RUN=False での実行は
コードとして存在しうるが、実際にそれを動かす判断・操作は常に人間が行う前提とする
（絶対原則6）。本モジュール自身は「実行する/しない」を選ばない・スケジュールしない。

信用取引区分は一般信用（無期限）に統一する。制度信用の空売りに伴う逆日歩
（品薄株での需給プレミアム。無制限・予測不能でコストモデルに織り込めない）を避けるため
（2026-07-18 合意）。

発注方法は寄成行（翌営業日の寄り執行、CLAUDE.md の日足約定基準と整合）に固定する。
指値・取消（cancelorder）は初期スコープ外。

⚠️ kabuステーションAPIの各コードは au カブコム/三菱UFJ eスマート証券の公式仕様書で
   要確認（本モジュールの値は設計時点の理解に基づく暫定値）。DRY_RUN=False で実際に
   送信する前に、検証環境（--env demo）で payload の形を必ず確認すること。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from config.settings import DRY_RUN
from execution.fill_monitor import OrderSide
from execution.kabu_client import KabuClient
from strategy.risk_manager import OrderRequest, RiskDecision, RiskManager

__all__ = ["OrderSubmission", "OrderEngine"]

logger = logging.getLogger(__name__)

# --- kabuステーションAPI 発注パラメータ ---------------------------------------------
# ⚠️ 要検証（公式仕様書で確認してからDRY_RUN=Falseで使うこと）
_EXCHANGE_TOSHO = 1
_SECURITY_TYPE_STOCK = 1
_SIDE_CODE: dict[OrderSide, str] = {"sell": "1", "buy": "2"}
_CASH_MARGIN_ENTRY = 2  # 信用新規
_CASH_MARGIN_EXIT = 3  # 信用返済
_MARGIN_TRADE_TYPE_GENERAL_UNLIMITED = 3  # 一般信用（無期限）
_FRONT_ORDER_TYPE_OPENING_MARKET = 13  # 寄成行
_ACCOUNT_TYPE_TOKUTEI = 2  # 特定口座
_EXPIRE_DAY_TODAY = 0  # 当日中


@dataclass(frozen=True)
class OrderSubmission:
    """発注試行の結果（却下・DRY_RUN見送り・実送信のいずれも記録する）。

    Attributes:
        request: 渡された発注リクエスト。
        decision: risk_manager の判定結果。
        dry_run: 実行時点の DRY_RUN 値。
        sent: kabuステーションへ実際にPOSTしたか（承認済みかつDRY_RUN=Falseのときのみ True）。
        kabu_payload: 送信（予定）した sendorder ボディ（Password は含まない）。却下時は None。
        kabu_response: 実送信した場合の応答。
    """

    request: OrderRequest
    decision: RiskDecision
    dry_run: bool
    sent: bool
    kabu_payload: dict[str, Any] | None = None
    kabu_response: dict[str, Any] | None = None


class OrderEngine:
    """risk_manager を必ず経由してkabuステーションAPIへ注文を送る執行エンジン。

    使い方:
        engine = OrderEngine(client=KabuClient(env="prod"), risk_manager=rm)
        rm.start_day(today)
        result = engine.submit(OrderRequest(...))
        if result.sent:
            ...  # 約定確認は別途 fill_monitor / orders 照会で行う
    """

    def __init__(
        self,
        *,
        client: KabuClient,
        risk_manager: RiskManager,
        account_type: int = _ACCOUNT_TYPE_TOKUTEI,
    ) -> None:
        self._client = client
        self._risk_manager = risk_manager
        self._account_type = account_type

    def submit(self, req: OrderRequest) -> OrderSubmission:
        """発注を試みる。risk_manager が却下した注文はkabuへ一切送らない（絶対原則2）。"""
        decision = self._risk_manager.check_order(req)
        if not decision.approved:
            logger.info(
                "発注却下 symbol=%s side=%s reason=%s", req.symbol, req.side, decision.reason
            )
            return OrderSubmission(request=req, decision=decision, dry_run=DRY_RUN, sent=False)

        payload = self._build_payload(req)

        if DRY_RUN:
            logger.info(
                "DRY_RUN: 発注をシミュレート（kabuへは未送信） symbol=%s payload=%s",
                req.symbol, payload,
            )
            return OrderSubmission(
                request=req, decision=decision, dry_run=True, sent=False, kabu_payload=payload,
            )

        response = self._client.send_order(payload)
        logger.warning(
            "実発注 送信済み symbol=%s side=%s shares=%d response=%s",
            req.symbol, req.side, req.shares, response,
        )
        return OrderSubmission(
            request=req, decision=decision, dry_run=False, sent=True,
            kabu_payload=payload, kabu_response=response,
        )

    def _build_payload(self, req: OrderRequest) -> dict[str, Any]:
        return {
            "Symbol": req.symbol,
            "Exchange": _EXCHANGE_TOSHO,
            "SecurityType": _SECURITY_TYPE_STOCK,
            "Side": _SIDE_CODE[req.side],
            "CashMargin": _CASH_MARGIN_ENTRY if req.is_entry else _CASH_MARGIN_EXIT,
            "MarginTradeType": _MARGIN_TRADE_TYPE_GENERAL_UNLIMITED,
            "AccountType": self._account_type,
            "Qty": req.shares,
            "FrontOrderType": _FRONT_ORDER_TYPE_OPENING_MARKET,
            "Price": 0,  # 成行系は0
            "ExpireDay": _EXPIRE_DAY_TODAY,
        }
