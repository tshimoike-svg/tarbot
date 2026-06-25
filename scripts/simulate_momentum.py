#!/usr/bin/env python3
"""モメンタム（ブレイクアウト）トラックのバックテスト → CSV出力。

simulate_leverage.py（反転）と同じ CSV スキーマで出力するので、同じ OOS 分離分析に
かけられる。swing_momentum は US フィルタを持たない純ブレイクアウト（close が直近
breakout_lookback 日高値を上抜けでロング）。反転トラックがレジーム（資金ローテーション）で
死んだ局面を、モメンタムが拾えるかを測る。

実行例:
    uv run python scripts/simulate_momentum.py --cache-dir data/db/bars_cache_yahoo \\
        --out data/db/bt_momentum_yahoo.csv --lookbacks 20 40 60
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from backtest.cost_model import CostModel
from backtest.evaluator import compute_trade_results
from config.costs import get_cost_params
from config.settings import SwingMomentumParams
from config.symbols import SYMBOLS
from strategy.swing_momentum import generate_trades

logger = logging.getLogger(__name__)

_CACHE_DIR = _ROOT / "data/db/bars_cache_yahoo"
_OUT_CSV = _ROOT / "data/db/bt_momentum_yahoo.csv"


def _find_cache(symbol: str, cache_dir: Path) -> Path | None:
    cands = sorted(cache_dir.glob(f"{symbol}_*.pkl"), reverse=True)
    return cands[0] if cands else None


def _load_cached(symbol: str, cache_dir: Path) -> pd.DataFrame:
    pkl = _find_cache(symbol, cache_dir)
    if pkl is None:
        return pd.DataFrame()
    with pkl.open("rb") as f:
        return pickle.load(f)  # noqa: S301


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
    p = argparse.ArgumentParser(description="モメンタム（ブレイクアウト）バックテスト → CSV")
    p.add_argument("--lookbacks", nargs="+", type=int, default=[20, 40, 60],
                   help="breakout_lookback の比較セット")
    p.add_argument("--allow-short", action="store_true", help="ショートも許可（既定はロングのみ）")
    p.add_argument("--cache-dir", default=str(_CACHE_DIR))
    p.add_argument("--out", default=str(_OUT_CSV))
    args = p.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    out_csv = Path(args.out)
    symbols = [s for s in _all_symbols() if _find_cache(s, cache_dir) is not None]
    logger.info("対象銘柄: %d 件  ロング%s", len(symbols), "+ショート" if args.allow_short else "のみ")

    cost_model = CostModel(get_cost_params("mid"))
    rows: list[dict] = []

    for lb in args.lookbacks:
        params = SwingMomentumParams(
            breakout_lookback=lb, atr_stop_mult=2.0, max_holding_days=10,
            allow_long=True, allow_short=args.allow_short,
        )
        cfg = f"mom_lb{lb}{'_ls' if args.allow_short else ''}"
        logger.info("=== %s ===", cfg)
        n = 0
        for symbol in symbols:
            df = _load_cached(symbol, cache_dir)
            if df.empty or len(df) < 50:
                continue
            trades = generate_trades(df, params)
            if not trades:
                continue
            for tr in compute_trade_results(trades, cost_model):
                t = tr.trade
                rows.append({
                    "config_name": cfg, "symbol": symbol,
                    "signal_date": (t.entry_time - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    "entry_date": t.entry_time.strftime("%Y-%m-%d"),
                    "exit_date": t.exit_time.strftime("%Y-%m-%d"),
                    "entry_price": t.entry_price, "exit_price": t.exit_price,
                    "stop_price": t.stop_price or 0.0, "exit_reason": t.exit_reason,
                    "gross_return": tr.gross_return, "net_return": tr.net_return,
                    "holding_days": tr.holding_days, "status": "closed",
                    "side": t.side, "priority": 1,
                })
                n += 1
        logger.info("  → %d トレード", n)

    if not rows:
        logger.error("トレードが0件。キャッシュを確認。")
        return 1

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    logger.info("保存完了: %s (%d 行)", out_csv, len(out))

    print("\n=== サマリ（コスト控除後 net）===")
    for cfg, g in out.groupby("config_name"):
        e = g["net_return"].mean() * 100
        wr = (g["net_return"] > 0).mean()
        print(f"  {cfg}: n={len(g)}  E={e:+.2f}%  WR={wr:.0%}  hold={g['holding_days'].mean():.1f}d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
