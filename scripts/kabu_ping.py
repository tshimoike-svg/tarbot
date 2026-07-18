#!/usr/bin/env python3
"""kabuステーションAPI 疎通確認スクリプト（読み取り専用・発注なし）。

前提: kabuステーション（Windows GUIアプリ）が起動・ログイン中であること。
既定は検証環境（ポート18081・常に固定値応答・実発注不可）。本番残高を見る場合のみ
明示的に --env prod を指定する（それでも本スクリプトは読み取りしか行わない）。

実行例:
    uv run python scripts/kabu_ping.py            # 検証環境
    uv run python scripts/kabu_ping.py --env prod # 本番環境（残高は読み取りのみ）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from execution.kabu_client import KabuClient, KabuError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="kabuステーションAPI 疎通確認（読み取り専用）")
    parser.add_argument("--env", choices=["prod", "demo"], default="demo")
    parser.add_argument("--symbol", default="1301", help="板情報を確認する銘柄コード（既定: 1301）")
    args = parser.parse_args(argv)

    client = KabuClient(env=args.env)

    print(f"環境: {args.env}（{'本番 18080' if args.env == 'prod' else '検証 18081'}）")
    try:
        client.get_token()
        print("✅ トークン発行 OK")

        cash = client.get_wallet_cash()
        print(f"✅ 現物買付可能額: {cash}")

        margin = client.get_wallet_margin()
        print(f"✅ 信用建余力: {margin}")

        board = client.get_board(args.symbol)
        print(f"✅ 板情報({args.symbol}): {board.get('CurrentPrice', board)}")

        print("\n疎通確認 成功（発注は一切行っていません）")
        return 0
    except KabuError as exc:
        print(f"\n❌ 疎通確認 失敗: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
