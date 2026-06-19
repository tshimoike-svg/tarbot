"""バックテスト司令塔（複数銘柄×両戦略を束ねて Phase 0 を判定）。

docs/trading_bot_design_v3.md §5,§7 / CLAUDE.md「現在のタスク」に対応。
各銘柄の日足から戦略（swing_reversion / swing_momentum）のトレードを生成し、
**コスト控除後**（持ち越し金利込み）で集計・ウォークフォワード・Phase0ゲートを判定する。
2戦略を同条件で回して比較する `compare_strategies` を提供する。

データ供給は `load_bars`（symbol→日足DataFrame）に抽象化してあり、テストはオフラインの
ダミー供給を注入できる。実データは `jquants_loader`（.env の認証で J-Quants 日足を取得）。

CLI:
    uv run python -m backtest.runner --strategy swing_reversion --symbols 7203,6758
    uv run python -m backtest.runner --compare --from 2024-06-01 --to 2026-06-01
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from backtest.cost_model import CostModel
from backtest.evaluator import (
    EvaluationResult,
    GateReport,
    check_phase0_gate,
    evaluate_trades,
    walk_forward,
)
from config.costs import LiquidityTier, get_cost_params
from strategy import swing_momentum, swing_reversion
from strategy.trade import Trade

__all__ = [
    "LoadBars",
    "BacktestResult",
    "STRATEGIES",
    "run_strategy",
    "compare_strategies",
    "format_result",
    "jquants_loader",
    "main",
]

logger = logging.getLogger(__name__)

LoadBars = Callable[[str], pd.DataFrame]

# 戦略名 → トレード生成関数
STRATEGIES: dict[str, Callable[[pd.DataFrame], list[Trade]]] = {
    "swing_reversion": swing_reversion.generate_trades,
    "swing_momentum": swing_momentum.generate_trades,
}


@dataclass(frozen=True)
class BacktestResult:
    """1戦略のバックテスト結果（コスト控除後）。"""

    strategy: str
    n_symbols: int
    evaluation: EvaluationResult
    walk_forward: list[EvaluationResult]
    gate: GateReport


def _collect_trades(
    symbols: Sequence[str],
    load_bars: LoadBars,
    generate: Callable[[pd.DataFrame], list[Trade]],
) -> list[Trade]:
    """全銘柄のトレードを集めて entry_time 昇順に並べる。"""
    trades: list[Trade] = []
    for symbol in symbols:
        df = load_bars(symbol)
        if df.empty:
            logger.warning("銘柄 %s の日足が空。スキップ", symbol)
            continue
        trades.extend(generate(df))
    trades.sort(key=lambda t: t.entry_time)
    return trades


def run_strategy(
    symbols: Sequence[str],
    load_bars: LoadBars,
    *,
    strategy: str,
    cost_model: CostModel,
    n_splits: int = 4,
    min_trades: int = 300,
    max_drawdown_threshold: float = 0.15,
) -> BacktestResult:
    """1戦略を全銘柄で回し、コスト控除後で評価・ゲート判定する。"""
    if strategy not in STRATEGIES:
        raise ValueError(f"未知の戦略: {strategy}（{list(STRATEGIES)}）")
    generate = STRATEGIES[strategy]
    trades = _collect_trades(symbols, load_bars, generate)

    evaluation = evaluate_trades(trades, cost_model)
    folds = walk_forward(trades, cost_model, n_splits=n_splits)
    gate = check_phase0_gate(
        evaluation, folds, min_trades=min_trades, max_drawdown_threshold=max_drawdown_threshold
    )
    return BacktestResult(
        strategy=strategy,
        n_symbols=len(symbols),
        evaluation=evaluation,
        walk_forward=folds,
        gate=gate,
    )


def _memoize_loader(load_bars: LoadBars) -> LoadBars:
    """銘柄ごとの取得を1回にキャッシュする（compare で二度取りしない＝API節約）。"""
    cache: dict[str, pd.DataFrame] = {}

    def _load(symbol: str) -> pd.DataFrame:
        if symbol not in cache:
            cache[symbol] = load_bars(symbol)
        return cache[symbol]

    return _load


def compare_strategies(
    symbols: Sequence[str],
    load_bars: LoadBars,
    *,
    cost_model: CostModel,
    n_splits: int = 4,
    min_trades: int = 300,
    max_drawdown_threshold: float = 0.15,
) -> dict[str, BacktestResult]:
    """両戦略を同条件で回して比較結果を返す。"""
    cached = _memoize_loader(load_bars)
    return {
        name: run_strategy(
            symbols,
            cached,
            strategy=name,
            cost_model=cost_model,
            n_splits=n_splits,
            min_trades=min_trades,
            max_drawdown_threshold=max_drawdown_threshold,
        )
        for name in STRATEGIES
    }


def format_result(result: BacktestResult) -> str:
    """結果を人間可読のテキストに整形する。"""
    e = result.evaluation
    g = result.gate
    lines = [
        f"=== {result.strategy}（{result.n_symbols} 銘柄）===",
    ]
    if e.is_empty:
        lines.append("  トレードなし（シグナル未発生 or データ不足）")
        return "\n".join(lines)
    fold_exp = ", ".join(
        f"{f.expectancy:+.4f}" if not f.is_empty else "—" for f in result.walk_forward
    )
    lines += [
        f"  トレード数            : {e.n_trades}",
        f"  期待値(コスト控除後)  : {e.expectancy:+.5f}  （グロス {e.gross_expectancy:+.5f}）",
        f"  平均コスト            : {e.mean_cost:.5f}",
        f"  勝率 / PF             : {e.win_rate:.1%} / {e.profit_factor:.2f}",
        f"  最大DD(プロキシ)      : {e.max_drawdown:.2%}",
        f"  出口理由              : {e.by_exit_reason}",
        f"  ウォークフォワード    : [{fold_exp}]",
        f"  Phase0ゲート          : {'PASS' if g.passed else 'FAIL'} "
        f"(期待値+:{g.expectancy_positive} 回数:{g.enough_trades} "
        f"DD:{g.drawdown_ok} 区間安定:{g.walkforward_stable})",
    ]
    return "\n".join(lines)


def jquants_loader(
    *, from_: str, to: str, base_url: str | None = None
) -> LoadBars:
    """J-Quants から日足を取得する load_bars を返す（.env の認証を使用）。

    実データ取得用。認証情報が無ければ最初の呼び出しで JQuantsAuthError。
    """
    from data.fetcher import JQuantsClient  # 遅延 import（テストは認証不要のまま）

    # レート制限対策：リクエスト間隔を空け、リトライを厚めに
    kwargs: dict[str, object] = {"min_interval": 1.0, "max_retries": 5, "retry_backoff": 2.0}
    client = (
        JQuantsClient(base_url=base_url, **kwargs)  # type: ignore[arg-type]
        if base_url
        else JQuantsClient(**kwargs)  # type: ignore[arg-type]
    )

    def _load(symbol: str) -> pd.DataFrame:
        return client.get_daily_quotes(code=symbol, from_=from_, to=to)

    return _load


def _default_dates() -> tuple[str, str]:
    """既定の期間：直近約2年（無料プランの保持期間に合わせる）。"""
    today = date.today()
    return (today - timedelta(days=730)).isoformat(), today.isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI エントリポイント。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from_default, to_default = _default_dates()
    parser = argparse.ArgumentParser(description="日足スイングのバックテスト（Phase 0）")
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="swing_reversion")
    parser.add_argument("--compare", action="store_true", help="両戦略を比較")
    parser.add_argument("--symbols", default=None, help="カンマ区切りの銘柄コード（省略時 config/symbols.py）")
    parser.add_argument("--from", dest="from_", default=from_default, help="開始日 YYYY-MM-DD")
    parser.add_argument("--to", default=to_default, help="終了日 YYYY-MM-DD")
    parser.add_argument("--tier", choices=["large", "mid", "small"], default="mid", help="コスト想定ティア")
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--min-trades", type=int, default=300)
    args = parser.parse_args(argv)

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        from config.symbols import SYMBOLS

        symbols = SYMBOLS

    tier: LiquidityTier = args.tier
    cost_model = CostModel(get_cost_params(tier))

    try:
        load_bars = jquants_loader(from_=args.from_, to=args.to)
        if args.compare:
            results = compare_strategies(
                symbols, load_bars, cost_model=cost_model,
                n_splits=args.n_splits, min_trades=args.min_trades,
            )
            for res in results.values():
                print(format_result(res))
        else:
            res = run_strategy(
                symbols, load_bars, strategy=args.strategy, cost_model=cost_model,
                n_splits=args.n_splits, min_trades=args.min_trades,
            )
            print(format_result(res))
    except Exception as exc:  # noqa: BLE001 - CLI 最外殻で握って案内
        from data.fetcher import JQuantsAuthError

        if isinstance(exc, JQuantsAuthError):
            print(
                "J-Quants 認証情報がありません。.env に JQUANTS_API_KEY を設定してください"
                "（README 参照）。"
            )
            return 2
        raise
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
