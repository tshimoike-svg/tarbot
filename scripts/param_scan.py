"""平均回帰戦略のパラメータ感度スキャン。

Phase 0 の「区間安定性」と「トレード数」を確保するために、
swing_reversion の主要パラメータを格子探索して期待値の頑健性を見る。

探索対象パラメータ（CLAUDE.md §7 感度チェック）：
  - lookback    : 移動平均・zスコアの窓 [10, 15, 20, 30]
  - entry_z     : エントリーzスコア閾値 [1.5, 2.0, 2.5]
  - atr_stop_mult: ATR損切り乗数 [1.5, 2.0, 2.5]
  - max_holding : 最大保有日数 [7, 10, 14]

大きな格子探索はキャッシュがあればAPIを叩かない（disk_cached_loader）。

実行例：
    uv run python scripts/param_scan.py --top 10
    uv run python scripts/param_scan.py --fixed-z 2.0 --fixed-stop 2.0
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---- デフォルト探索グリッド ---------------------------------------------------
_LOOKBACKS = [10, 15, 20, 30]
_ENTRY_ZS = [1.5, 2.0, 2.5]
_ATR_MULTS = [1.5, 2.0, 2.5]
_MAX_HOLDINGS = [7, 10, 14]


def scan(
    *,
    from_: str,
    to: str,
    tier: str,
    min_trades: int,
    cache_dir: str,
    lookbacks: list[int],
    entry_zs: list[float],
    atr_mults: list[float],
    max_holdings: list[int],
) -> list[dict]:
    """パラメータグリッドを総当たりし結果リストを返す（キャッシュ必須）。"""
    from config.costs import LiquidityTier, get_cost_params
    from config.settings import SwingReversionParams
    from config.symbols import SYMBOLS
    from backtest.cost_model import CostModel
    from backtest.runner import disk_cached_loader, jquants_loader
    from strategy import swing_reversion

    # キャッシュ経由ローダー（1度でもAPIから取得済みなら即返す）
    raw = jquants_loader(from_=from_, to=to)
    load_bars = disk_cached_loader(raw, from_=from_, to=to, cache_dir=cache_dir)

    liq_tier: LiquidityTier = tier  # type: ignore[assignment]
    cost_model = CostModel(get_cost_params(liq_tier))

    results = []
    combos = list(itertools.product(lookbacks, entry_zs, atr_mults, max_holdings))
    print(f"探索点数: {len(combos)}")

    for i, (lb, ez, am, mh) in enumerate(combos, 1):
        params = SwingReversionParams(
            lookback=lb,
            entry_z=ez,
            atr_stop_mult=am,
            max_holding_days=mh,
        )

        def _gen(df, p=params):  # noqa: ANN001
            return swing_reversion.generate_trades(df, params=p)

        trades = []
        for sym in SYMBOLS:
            df = load_bars(sym)
            if not df.empty:
                trades.extend(_gen(df))
        trades.sort(key=lambda t: t.entry_time)

        from backtest.evaluator import evaluate_trades, walk_forward, check_phase0_gate
        ev = evaluate_trades(trades, cost_model)
        folds = walk_forward(trades, cost_model, n_splits=4)
        gate = check_phase0_gate(ev, folds, min_trades=min_trades)

        row = {
            "lookback": lb, "entry_z": ez, "atr_mult": am, "max_hold": mh,
            "n_trades": ev.n_trades,
            "expectancy": ev.expectancy,
            "gross_exp": ev.gross_expectancy,
            "win_rate": ev.win_rate,
            "pf": ev.profit_factor,
            "proxy_dd": ev.max_drawdown,
            "sized_dd": ev.sized_max_drawdown,
            "wf_stable": gate.walkforward_stable,
            "passed": gate.passed,
        }
        results.append(row)

        if i % 10 == 0 or i == len(combos):
            print(f"  [{i}/{len(combos)}] lb={lb} z={ez} atr={am} hold={mh}"
                  f" → n={ev.n_trades} E={ev.expectancy:+.4f} pass={gate.passed}")

    return results


def print_top(results: list[dict], n: int) -> None:
    """期待値上位 n 件を表示。"""
    ranked = sorted(
        [r for r in results if r["n_trades"] > 0],
        key=lambda r: r["expectancy"],
        reverse=True,
    )
    print(f"\n=== 期待値上位 {n} パラメータセット ===")
    header = (
        f"{'lb':>4} {'z':>4} {'atr':>4} {'hold':>4}"
        f" {'n':>5} {'E':>8} {'WR':>5} {'PF':>5}"
        f" {'SzDD':>6} {'WFok':>5} {'pass':>5}"
    )
    print(header)
    for r in ranked[:n]:
        print(
            f"{r['lookback']:>4} {r['entry_z']:>4} {r['atr_mult']:>4} {r['max_hold']:>4}"
            f" {r['n_trades']:>5} {r['expectancy']:>+8.4f}"
            f" {r['win_rate']:>5.1%} {r['pf']:>5.2f}"
            f" {r['sized_dd']:>6.2%} {str(r['wf_stable']):>5} {str(r['passed']):>5}"
        )


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="swing_reversion パラメータ感度スキャン")
    parser.add_argument("--from", dest="from_", default="2024-03-28")
    parser.add_argument("--to", default="2026-03-27")
    parser.add_argument("--tier", default="mid")
    parser.add_argument("--min-trades", type=int, default=300)
    parser.add_argument("--cache-dir", default="data/db/bars_cache")
    parser.add_argument("--top", type=int, default=15, help="上位N件を表示")
    # 固定オプション（特定パラメータを固定して他を変化）
    parser.add_argument("--fixed-z", type=float, default=None)
    parser.add_argument("--fixed-stop", type=float, default=None)
    parser.add_argument("--fixed-hold", type=int, default=None)
    parser.add_argument("--fixed-lookback", type=int, default=None)
    args = parser.parse_args()

    lookbacks = [args.fixed_lookback] if args.fixed_lookback else _LOOKBACKS
    entry_zs = [args.fixed_z] if args.fixed_z else _ENTRY_ZS
    atr_mults = [args.fixed_stop] if args.fixed_stop else _ATR_MULTS
    max_holdings = [args.fixed_hold] if args.fixed_hold else _MAX_HOLDINGS

    results = scan(
        from_=args.from_,
        to=args.to,
        tier=args.tier,
        min_trades=args.min_trades,
        cache_dir=args.cache_dir,
        lookbacks=lookbacks,
        entry_zs=entry_zs,
        atr_mults=atr_mults,
        max_holdings=max_holdings,
    )
    print_top(results, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
