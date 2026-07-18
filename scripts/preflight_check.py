#!/usr/bin/env python3
"""本番環境での「発注一歩手前」リハーサル（実発注は絶対に行わない）。

残高照会 → 板情報取得 → サイジング → risk_manager → order_engine のペイロード構築
までを本番kabuステーションAPIに対して実際に流し、パイプライン全体の疎通・整合性を
確認する。DRY_RUN設定に関わらず、本スクリプトは KabuClient.send_order を物理的に
呼べないガード（_NoSendClient）越しにしか OrderEngine を使わない（二重の安全装置）。

また決済経路（find_hold_id_for_exit）も、実際の建玉があれば試す。建玉が無ければ
「対象なし」を期待どおりの結果として報告する（エラーではない）。

実行例:
    uv run python scripts/preflight_check.py --symbol 1301
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from config.settings import DEFAULT_RISK  # noqa: E402
from execution.kabu_client import KabuClient, KabuError  # noqa: E402
from execution.order_engine import (  # noqa: E402
    HoldIdLookupError,
    OrderEngine,
    find_hold_id_for_exit,
)
from strategy.position_sizer import size_position  # noqa: E402
from strategy.risk_manager import OrderRequest, RiskManager  # noqa: E402

_FALLBACK_EQUITY = 1_000_000.0  # 実残高が0円のときのみ使うリハーサル専用の仮の資金額
_ILLUSTRATIVE_STOP_RATIO = 0.95  # リハーサル用の仮ストップ（実指標ではない）


class _NoSendClient:
    """KabuClientをラップし send_order だけを物理的に無効化する二重の安全装置。

    DRY_RUN設定が万一 False になっていても、本スクリプト経由では実発注が
    絶対に発生しないようにする（読み取り系メソッドは実クライアントへ委譲）。
    """

    def __init__(self, client: KabuClient) -> None:
        self._client = client

    def get_wallet_cash(self) -> dict[str, Any]:
        return self._client.get_wallet_cash()

    def get_wallet_margin(self) -> dict[str, Any]:
        return self._client.get_wallet_margin()

    def get_board(self, symbol: str, exchange: int = 1) -> dict[str, Any]:
        return self._client.get_board(symbol, exchange)

    def get_positions(self, **kw: Any) -> list[dict[str, Any]]:
        return self._client.get_positions(**kw)

    def get_orders(self, **kw: Any) -> list[dict[str, Any]]:
        return self._client.get_orders(**kw)

    def send_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(
            "preflight_check.py は実発注を行いません（_NoSendClient による二重安全装置）"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本番リハーサル（実発注なし・読み取り+payload構築のみ）")
    parser.add_argument("--symbol", default="1301")
    parser.add_argument("--side", choices=["buy", "sell"], default="buy")
    parser.add_argument(
        "--equity", type=float, default=None,
        help="リハーサル用の資金額（円）。省略時は実残高、実残高が0円なら仮値を使う",
    )
    args = parser.parse_args(argv)

    print("=" * 60)
    print("本番環境プリフライトチェック（実発注は一切行いません）")
    print("=" * 60)

    real_client = KabuClient(env="prod")
    client = _NoSendClient(real_client)

    # --- 1. 残高照会（読み取りのみ） -------------------------------------------------
    print("\n[1/5] 残高照会")
    try:
        cash = client.get_wallet_cash()
        margin = client.get_wallet_margin()
    except KabuError as exc:
        print(f"  失敗: {exc}")
        return 1
    print(f"  現物買付可能額: {cash}")
    print(f"  信用建余力: {margin}")

    if args.equity is not None:
        account_equity = args.equity
        print(f"  ⚠️ --equity 指定によりリハーサル用の資金額 {account_equity:,.0f}円 を使用")
    else:
        margin_wallet = margin.get("MarginAccountWallet") or 0
        account_equity = float(margin_wallet) if margin_wallet else _FALLBACK_EQUITY
        if account_equity == _FALLBACK_EQUITY:
            print(f"  ⚠️ 実残高が0円のため、リハーサル用の仮資金額 {_FALLBACK_EQUITY:,.0f}円 を使用")

    # --- 2. 板情報（実際の市場価格） -------------------------------------------------
    print(f"\n[2/5] 板情報（{args.symbol}）")
    try:
        board = client.get_board(args.symbol)
    except KabuError as exc:
        print(f"  失敗: {exc}")
        return 1
    current_price = board.get("CurrentPrice")
    if not current_price:
        print(f"  ⚠️ 現在値を取得できません（市場時間外の可能性）: {board}")
        return 1
    print(f"  現在値: {current_price}")

    # --- 3. サイジング（仮のストップ幅・実シグナルではない） ---------------------------
    print("\n[3/5] サイジング（risk_manager 基準・ストップ幅は説明用の仮値）")
    stop_price = current_price * _ILLUSTRATIVE_STOP_RATIO
    sizing = size_position(
        account_equity=account_equity, entry_price=current_price,
        risk_params=DEFAULT_RISK, stop_price=stop_price,
    )
    print(f"  {sizing}")
    if sizing.shares < 1:
        print("  サイジング結果が0株。想定資金額に対して価格が高すぎるか設定を見直してください。")
        return 0

    # --- 4. risk_manager → order_engine（新規建て・実送信なし） -----------------------
    print("\n[4/5] 新規建てペイロード構築（risk_manager 経由・実送信なし）")
    rm = RiskManager(account_equity=account_equity, params=DEFAULT_RISK)
    rm.start_day(date.today())
    engine = OrderEngine(client=client, risk_manager=rm)  # type: ignore[arg-type]

    entry_req = OrderRequest(
        symbol=args.symbol, side=args.side, shares=sizing.shares, price=current_price,
        is_entry=True, risk_amount=sizing.risk_amount,
    )
    result = engine.submit(entry_req)
    print(f"  risk_manager 判定: approved={result.decision.approved} reason={result.decision.reason}")
    print(f"  構築されたpayload: {result.kabu_payload}")
    print(f"  実送信されたか: {result.sent}（常にFalseのはず）")

    # --- 5. 決済経路（実建玉があれば hold_id 解決を試す） -----------------------------
    print(f"\n[5/5] 決済建玉ID解決の疎通確認（{args.symbol}）")
    exit_side = "sell" if args.side == "buy" else "buy"
    try:
        hold_id = find_hold_id_for_exit(client, symbol=args.symbol, exit_side=exit_side)  # type: ignore[arg-type]
        print(f"  見つかった建玉ID: {hold_id}")
    except HoldIdLookupError as exc:
        print(f"  対象建玉なし（現状 保有ゼロなら想定どおり）: {exc}")

    print("\n" + "=" * 60)
    print("プリフライトチェック完了（実発注は一切行っていません）")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
