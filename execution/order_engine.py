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

kabuステーションAPIの各コード値（MarginTradeType/CashMargin/FrontOrderType/Side/
AccountType/DelivType）は2026-07-18時点で公式OpenAPI仕様書（kabucom/kabusapi リポジトリ
reference/kabu_STATION_API.yaml）で確認済み。

信用返済（決済注文）は sendorder の `ClosePositions`（返済建玉ID＝GET /positions の
ExecutionID）が必須のため、`submit()` は is_entry=False のとき `hold_id` を要求する。
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
# 2026-07-18 に公式リファレンス・複数の実装記事で相互確認済み（MarginTradeType/Side/
# CashMargin/FrontOrderType=10,13/AccountType）。DRY_RUN=Falseで初めて使う前には、
# 検証環境で実際のレスポンスを見て最終確認すること。
_EXCHANGE_TOSHO = 1
_SECURITY_TYPE_STOCK = 1
_SIDE_CODE: dict[OrderSide, str] = {"sell": "1", "buy": "2"}
_CASH_MARGIN_ENTRY = 2  # 信用新規
_CASH_MARGIN_EXIT = 3  # 信用返済
# MarginTradeType: 1=制度信用, 2=一般信用（長期＝無期限）, 3=一般信用（デイトレ・翌日持ち越し不可）
_MARGIN_TRADE_TYPE_GENERAL_UNLIMITED = 2  # 一般信用（長期＝無期限）
_FRONT_ORDER_TYPE_OPENING_MARKET = 13  # 寄成（前場）（10=成行, 13=寄成(前場), 14=寄成(後場)）
_ACCOUNT_TYPE_DEFAULT = 2  # 一般口座（2=一般, 4=特定, 12=法人。実口座に合わせて2026-07-18確認済み）
_EXPIRE_DAY_TODAY = 0  # 当日中
# DelivType: 信用新規は0(指定なし)固定。信用返済は指定必須で、auマネーコネクト未使用なら2(お預り金)。
_DELIV_TYPE_ENTRY = 0
_DELIV_TYPE_EXIT = 2


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
        result = engine.submit(entry_req)                       # 新規建て
        # 決済は返済建玉ID（GET /positions の ExecutionID）が必須
        result = engine.submit(exit_req, hold_id="E20260718...")
        if result.sent:
            ...  # 約定確認は別途 fill_monitor / orders 照会で行う
    """

    def __init__(
        self,
        *,
        client: KabuClient,
        risk_manager: RiskManager,
        account_type: int = _ACCOUNT_TYPE_DEFAULT,
    ) -> None:
        self._client = client
        self._risk_manager = risk_manager
        self._account_type = account_type

    def submit(self, req: OrderRequest, *, hold_id: str | None = None) -> OrderSubmission:
        """発注を試みる。risk_manager が却下した注文はkabuへ一切送らない（絶対原則2）。

        Args:
            hold_id: 決済（is_entry=False）時に必須。返す建玉のID
                （`KabuClient.get_positions()` の応答の `ExecutionID`）。
        """
        if not req.is_entry and not hold_id:
            raise ValueError(
                "決済注文には hold_id が必須です（GET /positions の ExecutionID を渡してください）"
            )

        decision = self._risk_manager.check_order(req)
        if not decision.approved:
            logger.info(
                "発注却下 symbol=%s side=%s reason=%s", req.symbol, req.side, decision.reason
            )
            return OrderSubmission(request=req, decision=decision, dry_run=DRY_RUN, sent=False)

        payload = self._build_payload(req, hold_id=hold_id)

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

    def _build_payload(self, req: OrderRequest, *, hold_id: str | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "Symbol": req.symbol,
            "Exchange": _EXCHANGE_TOSHO,
            "SecurityType": _SECURITY_TYPE_STOCK,
            "Side": _SIDE_CODE[req.side],
            "CashMargin": _CASH_MARGIN_ENTRY if req.is_entry else _CASH_MARGIN_EXIT,
            "MarginTradeType": _MARGIN_TRADE_TYPE_GENERAL_UNLIMITED,
            "DelivType": _DELIV_TYPE_ENTRY if req.is_entry else _DELIV_TYPE_EXIT,
            "AccountType": self._account_type,
            "Qty": req.shares,
            "FrontOrderType": _FRONT_ORDER_TYPE_OPENING_MARKET,
            "Price": 0,  # 成行系は0
            "ExpireDay": _EXPIRE_DAY_TODAY,
        }
        if not req.is_entry:
            payload["ClosePositions"] = [{"HoldID": hold_id, "Qty": req.shares}]
        return payload
