#!/usr/bin/env python3
"""order_engine.py の疎通確認スクリプト（検証環境のみ・実約定なし）。

検証環境（--env demo、ポート18081）は常に固定値応答で実発注が起きないため、
risk_manager → order_engine → kabu_client.send_order の一連のコードパスを
安全に確認できる。本番環境（--env prod）は誤操作防止のため意図的にサポートしない
（本番での実発注はユーザー自身が判断・実行する。絶対原則6）。

実行例:
    uv run python scripts/order_engine_ping.py --symbol 1301 --side buy --shares 100
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from config.settings import DEFAULT_RISK  # noqa: E402
from execution.kabu_client import KabuClient, KabuError  # noqa: E402
from execution.order_engine import OrderEngine  # noqa: E402
from strategy.risk_manager import OrderRequest, RiskManager  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="order_engine.py 疎通確認（検証環境限定）")
    parser.add_argument("--symbol", default="1301")
    parser.add_argument("--side", choices=["buy", "sell"], default="buy")
    parser.add_argument("--shares", type=int, default=100)
    parser.add_argument("--price", type=float, default=1000.0)
    parser.add_argument("--exit", action="store_true", help="決済注文として送る（既定は新規建て）")
    parser.add_argument(
        "--hold-id", default=None,
        help="決済時必須。返済建玉ID（GET /positions の ExecutionID。例: E20260718001）",
    )
    args = parser.parse_args(argv)
    if args.exit and not args.hold_id:
        parser.error("--exit を指定する場合は --hold-id が必須です")

    print("環境: demo（検証 18081・実発注は起きません）")
    print("DRY_RUN の値は config.settings.DRY_RUN を参照（.env の DRY_RUN で上書き可）")

    client = KabuClient(env="demo")
    rm = RiskManager(account_equity=1_000_000, params=DEFAULT_RISK)
    rm.start_day(date.today())
    engine = OrderEngine(client=client, risk_manager=rm)

    req = OrderRequest(
        symbol=args.symbol, side=args.side, shares=args.shares, price=args.price,
        is_entry=not args.exit,
    )

    try:
        result = engine.submit(req, hold_id=args.hold_id)
    except KabuError as exc:
        print(f"\n❌ 疎通確認 失敗: {exc}")
        return 1

    print(f"\nrisk_manager 判定: approved={result.decision.approved} reason={result.decision.reason}")
    print(f"DRY_RUN: {result.dry_run}")
    print(f"kabuへ送信したか: {result.sent}")
    print(f"送信payload: {result.kabu_payload}")
    print(f"kabu応答: {result.kabu_response}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
