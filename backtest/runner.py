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
import pickle
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

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
from strategy import swing_cross_section, swing_momentum, swing_reversion
from strategy.trade import Trade

__all__ = [
    "LoadBars",
    "BacktestResult",
    "STRATEGIES",
    "run_strategy",
    "run_cross_section_strategy",
    "compare_strategies",
    "format_result",
    "jquants_loader",
    "disk_cached_loader",
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
    *,
    generate_kwargs: dict[str, object] | None = None,
) -> list[Trade]:
    """全銘柄のトレードを集めて entry_time 昇順に並べる。"""
    trades: list[Trade] = []
    kw = generate_kwargs or {}
    for symbol in symbols:
        df = load_bars(symbol)
        if df.empty:
            logger.warning("銘柄 %s の日足が空。スキップ", symbol)
            continue
        trades.extend(generate(df, **kw))
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
    market_df: pd.DataFrame | None = None,
    rev_params: object | None = None,
    us_df: pd.DataFrame | None = None,
) -> BacktestResult:
    """1戦略を全銘柄で回し、コスト控除後で評価・ゲート判定する。

    market_df: 市場レジームフィルタ用の指数 OHLCV（swing_reversion のみ有効）。
    rev_params: SwingReversionParams の上書き（フィルタ付きパラメータを渡す）。
    us_df: 前日米国株リターンフィルタ用の S&P500 DataFrame（swing_reversion のみ有効）。
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"未知の戦略: {strategy}（{list(STRATEGIES)}）")
    generate = STRATEGIES[strategy]
    generate_kwargs: dict[str, object] = {}
    if market_df is not None:
        generate_kwargs["market_df"] = market_df
    if rev_params is not None and strategy == "swing_reversion":
        generate_kwargs["params"] = rev_params
    if us_df is not None and strategy == "swing_reversion":
        generate_kwargs["us_df"] = us_df
    trades = _collect_trades(
        symbols, load_bars, generate,
        generate_kwargs=generate_kwargs if generate_kwargs else None,
    )

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


def run_cross_section_strategy(
    symbols: Sequence[str],
    load_bars: LoadBars,
    *,
    cost_model: CostModel,
    n_splits: int = 4,
    min_trades: int = 300,
    max_drawdown_threshold: float = 0.15,
    cs_params: object | None = None,
) -> BacktestResult:
    """クロスセクション平均回帰戦略を全銘柄で回し、コスト控除後で評価・ゲート判定する。

    全銘柄データを一括ロードしてからクロスセクションzを計算するため、
    run_strategy とは異なるデータフロー（dict渡し）を使う。
    """
    from config.settings import DEFAULT_CROSS_SECTION, SwingCrossSectionParams

    params = cs_params if isinstance(cs_params, SwingCrossSectionParams) else DEFAULT_CROSS_SECTION

    all_dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_bars(sym)
        if not df.empty:
            all_dfs[sym] = df
        else:
            logger.warning("銘柄 %s の日足が空。スキップ", sym)

    trades = swing_cross_section.generate_trades(all_dfs, params)
    trades.sort(key=lambda t: t.entry_time)

    evaluation = evaluate_trades(trades, cost_model)
    folds = walk_forward(trades, cost_model, n_splits=n_splits)
    gate = check_phase0_gate(
        evaluation, folds, min_trades=min_trades, max_drawdown_threshold=max_drawdown_threshold
    )
    return BacktestResult(
        strategy="swing_cross_section",
        n_symbols=len(all_dfs),
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
    market_df: pd.DataFrame | None = None,
    rev_params: object | None = None,
    us_df: pd.DataFrame | None = None,
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
            market_df=market_df,
            rev_params=rev_params,
            us_df=us_df,
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
        f"  最大DD(プロキシ)      : {e.max_drawdown:.2%}  口座DD(0.5%リスク): {e.sized_max_drawdown:.2%}",
        f"  出口理由              : {e.by_exit_reason}",
        f"  ウォークフォワード    : [{fold_exp}]",
        f"  Phase0ゲート          : {'PASS' if g.passed else 'FAIL'} "
        f"(期待値+:{g.expectancy_positive} 回数:{g.enough_trades} "
        f"DD:{g.drawdown_ok}[{g.sized_max_drawdown:.2%}<{g.max_drawdown_threshold:.0%}] "
        f"区間安定:{g.walkforward_stable})",
    ]
    return "\n".join(lines)


def jquants_loader(
    *, from_: str, to: str, base_url: str | None = None, min_interval: float = 13.0
) -> LoadBars:
    """J-Quants から日足を取得する load_bars を返す（.env の認証を使用）。

    実データ取得用。認証情報が無ければ最初の呼び出しで JQuantsAuthError。

    プラン別レート制限（https://jpx-jquants.com/ja/spec/rate-limits）:
      Free    : 5 req/min  → min_interval=13.0 (余裕を持ち 4.6req/min)
      Light   : 60 req/min → min_interval=1.0
      Standard: 120 req/min→ min_interval=0.5
    大幅超過時は最大5分間ブロックされるため、disk_cached_loader との併用を推奨。
    """
    from data.fetcher import JQuantsClient  # 遅延 import（テストは認証不要のまま）

    # retry_backoff=5.0 / max_retries=8 で最大累積待機 ≈ 1275s（5分ブロック対応）
    kwargs: dict[str, object] = {"min_interval": min_interval, "max_retries": 8, "retry_backoff": 5.0}
    client = (
        JQuantsClient(base_url=base_url, **kwargs)  # type: ignore[arg-type]
        if base_url
        else JQuantsClient(**kwargs)  # type: ignore[arg-type]
    )

    def _load(symbol: str) -> pd.DataFrame:
        return client.get_daily_quotes(code=symbol, from_=from_, to=to)

    return _load


def disk_cached_loader(
    base_loader: LoadBars,
    *,
    from_: str,
    to: str,
    cache_dir: str | Path = "data/db/bars_cache",
) -> LoadBars:
    """ディスクキャッシュ（pickle）付きローダー。

    同一 (symbol, from_, to_) の日足は1度だけ取得してローカルに保存する。
    繰り返しバックテストで J-Quants API 呼び出しを減らし、レート制限を回避する。
    キャッシュキーに from_/to を含むので期間変更時は自動的に再取得する。
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    key = f"{from_}_{to}".replace("-", "")

    def _load(symbol: str) -> pd.DataFrame:
        pkl = cache_path / f"{symbol}_{key}.pkl"
        if pkl.exists():
            logger.debug("キャッシュから読み込み: %s", symbol)
            with pkl.open("rb") as f:
                return pickle.load(f)  # noqa: S301 - ローカル自前生成ファイルのみ
        df = base_loader(symbol)
        with pkl.open("wb") as f:
            pickle.dump(df, f)
        return df

    return _load


def _default_dates() -> tuple[str, str]:
    """既定の期間：直近約2年（無料プランの保持期間に合わせる）。"""
    today = date.today()
    return (today - timedelta(days=730)).isoformat(), today.isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI エントリポイント。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from_default, to_default = _default_dates()
    _all_strategies = list(STRATEGIES) + ["swing_cross_section"]
    parser = argparse.ArgumentParser(description="日足スイングのバックテスト（Phase 0）")
    parser.add_argument("--strategy", choices=_all_strategies, default="swing_reversion")
    parser.add_argument("--compare", action="store_true", help="両戦略を比較")
    parser.add_argument("--symbols", default=None, help="カンマ区切りの銘柄コード（省略時 config/symbols.py）")
    parser.add_argument("--from", dest="from_", default=from_default, help="開始日 YYYY-MM-DD")
    parser.add_argument("--to", default=to_default, help="終了日 YYYY-MM-DD")
    parser.add_argument("--tier", choices=["large", "mid", "small"], default="mid", help="コスト想定ティア")
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--min-trades", type=int, default=300)
    parser.add_argument("--cache-dir", default="data/db/bars_cache",
                        help="日足キャッシュdir（既定: data/db/bars_cache）。--no-cache で無効化")
    parser.add_argument("--no-cache", action="store_true", help="キャッシュを使わず毎回取得")
    parser.add_argument("--min-interval", type=float, default=13.0,
                        help="APIリクエスト間隔(秒)（Free=5req/min → 13秒推奨。Light=60req/min → 1秒）")
    parser.add_argument("--regime", action="store_true",
                        help="市場レジームフィルタを有効化（レンジ相場のときだけエントリー）")
    parser.add_argument("--regime-symbol", default="1306",
                        help="レジーム判定に使う指数 ETF コード（既定 1306=TOPIX ETF）")
    parser.add_argument("--regime-window", type=int, default=60,
                        help="レジーム判定の MA 窓（日）")
    parser.add_argument("--regime-threshold", type=float, default=0.03,
                        help="レンジ判定の閾値（±X% 以内 → レンジ）")
    parser.add_argument("--regime-invert", action="store_true",
                        help="レジームフィルタを反転（トレンド期のみエントリー）")
    parser.add_argument("--us-filter", action="store_true",
                        help="前日米国株リターンフィルタを有効化（アグレッシブ設定）")
    parser.add_argument("--us-crash", type=float, default=-0.02,
                        help="US クラッシュ閾値: 前日 S&P500 < この値でエントリー許可（既定 -0.02 = -2%%）")
    parser.add_argument("--us-soft-min", type=float, default=-0.005,
                        help="US ソフト許可範囲の下限（既定 -0.005 = -0.5%%）")
    parser.add_argument("--us-soft-max", type=float, default=0.0,
                        help="US ソフト許可範囲の上限（既定 0.0 = 0%%）")
    args = parser.parse_args(argv)

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        from config.symbols import SYMBOLS

        symbols = SYMBOLS

    tier: LiquidityTier = args.tier
    cost_model = CostModel(get_cost_params(tier))

    try:
        raw_loader = jquants_loader(
            from_=args.from_, to=args.to, min_interval=args.min_interval
        )
        load_bars = raw_loader if args.no_cache else disk_cached_loader(
            raw_loader, from_=args.from_, to=args.to, cache_dir=args.cache_dir
        )

        # 市場レジームフィルタ用データ取得
        market_df: pd.DataFrame | None = None
        rev_params: object | None = None
        us_df: pd.DataFrame | None = None
        if args.regime:
            from config.settings import SwingReversionParams as _RevP
            logger.info("レジームフィルタ用データを取得中（%s）...", args.regime_symbol)
            market_df = load_bars(args.regime_symbol)
            if market_df.empty:
                logger.warning("レジーム用データが空。フィルタなしで実行します")
                market_df = None
            else:
                logger.info(
                    "レジームフィルタ有効: MA%d ±%.0f%%",
                    args.regime_window, args.regime_threshold * 100,
                )
                rev_params = _RevP(
                    lookback=30, entry_z=2.5, atr_stop_mult=2.5, max_holding_days=14,
                    enable_regime_filter=True,
                    regime_ma_window=args.regime_window,
                    regime_threshold=args.regime_threshold,
                    regime_filter_invert=args.regime_invert,
                )

        # 前日米国株リターンフィルタ用データ取得
        if args.us_filter:
            from config.settings import SwingReversionParams as _RevP
            from data.us_loader import load_spx

            logger.info("S&P500 データを取得中（アグレッシブ US フィルタ用）...")
            us_df = load_spx(args.from_, args.to)
            if us_df.empty:
                logger.warning("S&P500 データが空。US フィルタなしで実行します")
                us_df = None
            else:
                logger.info(
                    "US フィルタ有効: クラッシュ<%.1f%%, ソフト[%.1f%%,%.1f%%)",
                    args.us_crash * 100, args.us_soft_min * 100, args.us_soft_max * 100,
                )
                # rev_params が未設定の場合はベスト設定（lb30/z2.5/rsi<30/long-only）を使う
                base = rev_params if rev_params is not None else _RevP(
                    lookback=30, entry_z=2.5, atr_stop_mult=2.5, max_holding_days=14,
                    allow_long=True, allow_short=False,
                    rsi_entry_max=30.0,
                )
                from dataclasses import replace as _replace
                rev_params = _replace(
                    base,
                    us_t1_crash_threshold=args.us_crash,
                    us_t1_soft_min=args.us_soft_min,
                    us_t1_soft_max=args.us_soft_max,
                    us_t0_crash_threshold=args.us_crash,
                    us_t0_soft_min=args.us_soft_min,
                    us_t0_soft_max=args.us_soft_max,
                ) if isinstance(base, _RevP) else base

        if args.compare:
            results = compare_strategies(
                symbols, load_bars, cost_model=cost_model,
                n_splits=args.n_splits, min_trades=args.min_trades,
                market_df=market_df, rev_params=rev_params, us_df=us_df,
            )
            for res in results.values():
                print(format_result(res))
        elif args.strategy == "swing_cross_section":
            res = run_cross_section_strategy(
                symbols, load_bars, cost_model=cost_model,
                n_splits=args.n_splits, min_trades=args.min_trades,
            )
            print(format_result(res))
        else:
            res = run_strategy(
                symbols, load_bars, strategy=args.strategy, cost_model=cost_model,
                n_splits=args.n_splits, min_trades=args.min_trades,
                market_df=market_df, rev_params=rev_params, us_df=us_df,
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
