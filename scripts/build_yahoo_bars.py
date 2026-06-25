#!/usr/bin/env python3
"""バックテスト用に Yahoo Finance の日足を全銘柄分まとめて取得し、simulate_leverage が
読める命名（`{code}_{from}_{to}.pkl`）で保存する。

J-Quants 無料は約3ヶ月遅延するため bars_cache は 2026-03-27 で止まっている。
Yahoo は当日まで取れるので、ロジックの「直近未使用区間（OOS）」検証に使う。

調整方式は data.yahoo_loader と同じく auto_adjust=False の Close（分割調整のみ）で、
J-Quants 調整後と一致する（実測 max 0.00% / リターン相関 1.0000）。

実行例:
    uv run python scripts/build_yahoo_bars.py --from 2024-03-28
    uv run python scripts/build_yahoo_bars.py --from 2024-03-28 --to 2026-06-24
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from config.symbols import SYMBOLS
from data.yahoo_loader import _normalize_yahoo, to_yahoo_ticker

logger = logging.getLogger(__name__)

_DEFAULT_OUT = _ROOT / "data/db/bars_cache_yahoo"


def _all_symbols() -> list[str]:
    raw = [str(s) for s in SYMBOLS]
    try:
        from config.symbols_cs import SYMBOLS_CS  # noqa: PLC0415
        raw += [str(s) for s in SYMBOLS_CS]
    except ImportError:
        pass
    seen: set[str] = set()
    out: list[str] = []
    for s in raw:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Yahoo 日足を全銘柄取得して bars_cache 形式で保存")
    p.add_argument("--from", dest="from_", default="2024-03-28", help="開始日 YYYY-MM-DD")
    p.add_argument("--to", dest="to", default=None, help="終了日 YYYY-MM-DD（既定: 前営業日相当=昨日）")
    p.add_argument("--out-dir", default=str(_DEFAULT_OUT), help="保存先ディレクトリ")
    args = p.parse_args(argv)

    from_ = args.from_
    to = args.to or (date.today() - timedelta(days=1)).isoformat()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fetch_end = (pd.Timestamp(to) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end は排他
    tag = f"{pd.Timestamp(from_).strftime('%Y%m%d')}_{pd.Timestamp(to).strftime('%Y%m%d')}"

    import yfinance as yf  # noqa: PLC0415

    symbols = _all_symbols()
    logger.info("取得対象: %d 銘柄  期間 %s..%s → %s", len(symbols), from_, to, out_dir)

    ok = empty = fail = 0
    for i, code in enumerate(symbols, 1):
        ticker = to_yahoo_ticker(code)
        try:
            raw = yf.download(
                ticker, start=from_, end=fetch_end,
                auto_adjust=False, progress=False, threads=False,
            )
            df = _normalize_yahoo(raw)
        except Exception as exc:
            logger.warning("[%d/%d] %s 取得失敗: %s", i, len(symbols), ticker, exc)
            fail += 1
            continue
        if df.empty:
            logger.warning("[%d/%d] %s データ空", i, len(symbols), ticker)
            empty += 1
            continue
        with (out_dir / f"{code}_{tag}.pkl").open("wb") as f:
            pickle.dump(df, f)
        ok += 1
        if i % 30 == 0:
            logger.info("  …%d/%d 完了", i, len(symbols))

    logger.info("完了: 成功 %d / 空 %d / 失敗 %d（保存先 %s）", ok, empty, fail, out_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
