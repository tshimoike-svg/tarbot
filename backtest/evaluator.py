"""バックテスト評価（Phase 0 の合否判定）。

docs/trading_bot_design_v2.md §7 / CLAUDE.md「現在のタスク」に対応。
戦略（swing_reversion / swing_momentum）が出す**グロス（コスト控除前）**のトレード列に、
`backtest.cost_model.CostModel` の往復コストを差し引いて、**コスト控除後の期待値**で評価する。

絶対原則3「バックテスト評価は必ずコスト控除後で行う」を構造的に強制するため、
本モジュールの評価関数は `CostModel` を必須引数に取り、コストを引かずに期待値を出す経路を持たない。

§7 バイアスチェックリストへの対応状況：
  [x] ルックアヘッド・バイアス … indicators が因果的（test_*_is_causal で保証）
  [x] コスト控除後で評価       … 本モジュールが CostModel を必須にして強制
  [x] イン/アウトサンプル分離   … walk_forward() が時系列で分割
  [x] 統計的十分性             … check_phase0_gate(min_trades=...) で判定
  [~] 滑り・約定率             … 滑りは cost_model で計上。指値の**不約定**は
                                 generate_trades が「終値で約定」と理想化しており未モデル化
                                 （→ §13 タスク。assume_fill=True を明示し楽観性を申し送る）
  [~] マーケットインパクト     … adv_shares を渡せば cost_model が平方根モデルで計上
  [ ] サバイバーシップ         … データ層（data/fetcher の廃止銘柄取得）の責務。ここでは検査不能

最大ドローダウンは「各トレード同一ノートーション」を仮定した累積リターン曲線上の値（プロキシ）。
口座レベルの真のDDはポジションサイジング（risk_manager、未実装）に依存する。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from backtest.cost_model import CostModel
from strategy.trade import Side, Trade

__all__ = [
    "TradeResult",
    "EvaluationResult",
    "GateReport",
    "compute_trade_results",
    "evaluate_trades",
    "walk_forward",
    "max_drawdown",
    "check_phase0_gate",
]


@dataclass(frozen=True)
class TradeResult:
    """1トレードのコスト控除後リターン（notional 比率）。"""

    trade: Trade
    gross_return: float  # (グロス損益 / エントリー価格)
    cost: float          # 往復コスト比率（>0 が費用、capture>0.5 だと負＝利得）
    net_return: float    # gross_return − cost
    holding_days: int


def _holding_days(trade: Trade) -> int:
    """保有日数（暦日）。当日手仕舞いなら 0 → 信用コストは効かない。"""
    return int((trade.exit_time.normalize() - trade.entry_time.normalize()).days)


def compute_trade_results(
    trades: Sequence[Trade],
    cost_model: CostModel,
    *,
    shares: int = 1,
    adv_shares: float | None = None,
) -> list[TradeResult]:
    """各トレードにコストを差し引いた結果列を返す。

    リターンは notional 比率（= 期待値ゲートと同じ単位）。spread・slippage・financing は
    shares に依らず比率一定。インパクト（adv_shares 指定時）と固定手数料のみ shares 依存。

    Args:
        shares: コスト計算上の株数（インパクト/固定手数料に影響。既定 1）。
        adv_shares: 1日平均出来高。None ならインパクト非計上（保守性は別途要検討）。
    """
    results: list[TradeResult] = []
    for tr in trades:
        if tr.entry_price <= 0:
            raise ValueError(f"entry_price は正である必要があります: {tr.entry_price}")
        side: Side = tr.side
        days = _holding_days(tr)
        gross = tr.pnl_gross_per_share / tr.entry_price
        cost = cost_model.round_trip_cost_fraction(
            price=tr.entry_price,
            shares=shares,
            holding_days=days,
            side=side,
            adv_shares=adv_shares,
        )
        results.append(
            TradeResult(
                trade=tr,
                gross_return=gross,
                cost=cost,
                net_return=gross - cost,
                holding_days=days,
            )
        )
    return results


def max_drawdown(returns: Sequence[float]) -> float:
    """各トレード同一ノートーションを仮定した累積リターン曲線の最大ドローダウン。

    equity_t = Σ_{i<=t} return_i（加算）。最大DD = max(ピーク − equity)。
    返り値は notional 比率（例：0.15 = 15%）。トレードが無ければ 0.0。
    """
    peak = 0.0
    equity = 0.0
    max_dd = 0.0
    for r in returns:
        equity += r
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


@dataclass(frozen=True)
class EvaluationResult:
    """集計結果（すべてコスト控除後）。"""

    n_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float           # コスト控除後 E[1トレード]（= net_return の平均）
    gross_expectancy: float     # 参考：コスト控除前の平均
    mean_cost: float            # 平均往復コスト比率
    total_net_return: float     # net_return の総和
    profit_factor: float        # 総利益 / |総損失|（損失ゼロなら inf）
    max_drawdown: float         # プロキシ（上記仮定）
    by_exit_reason: dict[str, int]

    @property
    def is_empty(self) -> bool:
        return self.n_trades == 0


def _aggregate(results: Sequence[TradeResult]) -> EvaluationResult:
    n = len(results)
    if n == 0:
        return EvaluationResult(
            n_trades=0,
            win_rate=math.nan,
            avg_win=math.nan,
            avg_loss=math.nan,
            expectancy=math.nan,
            gross_expectancy=math.nan,
            mean_cost=math.nan,
            total_net_return=0.0,
            profit_factor=math.nan,
            max_drawdown=0.0,
            by_exit_reason={},
        )
    nets = [r.net_return for r in results]
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x < 0]
    gross_profit = sum(wins)
    gross_loss = sum(losses)  # <= 0
    profit_factor = math.inf if gross_loss == 0 else gross_profit / abs(gross_loss)
    reasons: Counter[str] = Counter(r.trade.exit_reason for r in results)

    return EvaluationResult(
        n_trades=n,
        win_rate=len(wins) / n,
        avg_win=(gross_profit / len(wins)) if wins else 0.0,
        avg_loss=(gross_loss / len(losses)) if losses else 0.0,
        expectancy=sum(nets) / n,
        gross_expectancy=sum(r.gross_return for r in results) / n,
        mean_cost=sum(r.cost for r in results) / n,
        total_net_return=sum(nets),
        profit_factor=profit_factor,
        max_drawdown=max_drawdown(nets),
        by_exit_reason=dict(reasons),
    )


def evaluate_trades(
    trades: Sequence[Trade],
    cost_model: CostModel,
    *,
    shares: int = 1,
    adv_shares: float | None = None,
) -> EvaluationResult:
    """トレード列をコスト控除後で集計評価する。"""
    results = compute_trade_results(
        trades, cost_model, shares=shares, adv_shares=adv_shares
    )
    return _aggregate(results)


def walk_forward(
    trades: Sequence[Trade],
    cost_model: CostModel,
    *,
    n_splits: int,
    shares: int = 1,
    adv_shares: float | None = None,
) -> list[EvaluationResult]:
    """トレードを時系列で n_splits 個の連続区間に分け、各区間を評価する。

    特定期間依存でないか（ウォークフォワードの安定性）を見るための分割。
    trades は時系列順であること（generate_trades の出力はその順序）。
    """
    if n_splits < 1:
        raise ValueError("n_splits は 1 以上")
    n = len(trades)
    if n == 0:
        return []
    # ほぼ均等に連続分割（端数は前半の区間に1つずつ寄せる）
    base, extra = divmod(n, n_splits)
    folds: list[EvaluationResult] = []
    start = 0
    for k in range(n_splits):
        size = base + (1 if k < extra else 0)
        if size == 0:
            continue
        chunk = trades[start : start + size]
        folds.append(
            evaluate_trades(chunk, cost_model, shares=shares, adv_shares=adv_shares)
        )
        start += size
    return folds


@dataclass(frozen=True)
class GateReport:
    """Phase 0 完了条件（DoD）の判定結果。"""

    expectancy_positive: bool      # コスト控除後 E[1トレード] > 0
    enough_trades: bool            # 統計的に十分なトレード数
    drawdown_ok: bool              # 最大DD < しきい値（プロキシ）
    walkforward_stable: bool       # 各区間でも期待値が正
    n_trades: int
    expectancy: float
    max_drawdown: float
    min_trades: int
    max_drawdown_threshold: float

    @property
    def passed(self) -> bool:
        return (
            self.expectancy_positive
            and self.enough_trades
            and self.drawdown_ok
            and self.walkforward_stable
        )


def check_phase0_gate(
    result: EvaluationResult,
    walk_forward_results: Sequence[EvaluationResult],
    *,
    min_trades: int = 300,
    max_drawdown_threshold: float = 0.15,
) -> GateReport:
    """CLAUDE.md の Phase 0 完了条件を判定する。

    - コスト控除後 E[1トレード] > 0
    - ウォークフォワードの各区間で安定（各区間の期待値も正）
    - 最大ドローダウン < しきい値（既定 15%・プロキシ）
    - 統計的に十分なトレード数（既定 300）
    """
    expectancy_positive = (not result.is_empty) and result.expectancy > 0.0
    enough_trades = result.n_trades >= min_trades
    drawdown_ok = result.max_drawdown < max_drawdown_threshold
    walkforward_stable = len(walk_forward_results) > 0 and all(
        (not f.is_empty) and f.expectancy > 0.0 for f in walk_forward_results
    )
    return GateReport(
        expectancy_positive=expectancy_positive,
        enough_trades=enough_trades,
        drawdown_ok=drawdown_ok,
        walkforward_stable=walkforward_stable,
        n_trades=result.n_trades,
        expectancy=result.expectancy,
        max_drawdown=result.max_drawdown,
        min_trades=min_trades,
        max_drawdown_threshold=max_drawdown_threshold,
    )
