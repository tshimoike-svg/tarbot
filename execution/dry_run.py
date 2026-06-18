"""ドライラン・ハーネス（フォワード検証の司令塔・発注なし）。

docs/trading_bot_design_v2.md §10 / CLAUDE.md（execution/dry_run.py）に対応。
分足をストリーム的に流し、戦略→**risk_manager（関門）**→疑似約定→**fill_monitor（実測）**
→evaluator→storage を1本に束ねる。**実発注は一切しない**（絶対原則1・6）。

純フォワード方針での位置づけ：口座開通後、本ハーネスをライブ分足につなぎ、
- risk_manager で全発注を関門に通し（絶対原則2）
- fill_monitor で約定率・滑りを実測（コスト・キャリブレーション）
- storage に蓄積
する。現状はリプレイ（既存の分足 DataFrame）で全結線を検証できる。

⚠️ 疑似約定は「指値どおり全量約定・滑りゼロ」の理想化。実約定率・実滑りは
   ライブ運用で fill_monitor が計測し、ここを置き換える（楽観バイアスの是正）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from backtest.cost_model import CostModel
from backtest.evaluator import EvaluationResult, evaluate_trades
from config.settings import (
    DEFAULT_MEAN_REVERSION,
    DEFAULT_RISK,
    DRY_RUN,
    MeanReversionParams,
    RiskParams,
)
from execution.fill_monitor import FillMonitor, FillStats, OrderIntent, OrderSide
from strategy.mean_reversion import Trade, generate_trades
from strategy.risk_manager import OrderRequest, RiskManager

if TYPE_CHECKING:
    from data.storage import Storage

__all__ = ["DryRunReport", "DryRunHarness"]

logger = logging.getLogger(__name__)


@dataclass
class DryRunReport:
    """ドライランの結果。"""

    executed_trades: list[Trade] = field(default_factory=list)
    rejected: list[tuple[Trade, str]] = field(default_factory=list)
    fill_stats: FillStats | None = None
    evaluation: EvaluationResult | None = None
    final_equity: float = 0.0
    n_signals: int = 0

    @property
    def n_executed(self) -> int:
        return len(self.executed_trades)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)


def _sides(trade_side: str) -> tuple[OrderSide, OrderSide]:
    """戦略のサイド（long/short）→ (エントリー, イグジット) の売買方向。"""
    if trade_side == "long":
        return "buy", "sell"
    return "sell", "buy"


class DryRunHarness:
    """1銘柄分の分足をリプレイし、フォワードの全工程を結線する。

    Args:
        account_equity: 初期資金（円）。
        cost_model: 往復コストモデル。疑似損益の控除に使う。
        mr_params / risk_params: 戦略・リスクのパラメータ。
        storage: 指定すれば確定トレード・約定実測を蓄積する。
        target_notional_frac: 1トレードの目標ノートーション（資金比）。簡易サイジング。
    """

    def __init__(
        self,
        *,
        account_equity: float,
        cost_model: CostModel,
        mr_params: MeanReversionParams = DEFAULT_MEAN_REVERSION,
        risk_params: RiskParams = DEFAULT_RISK,
        storage: Storage | None = None,
        target_notional_frac: float = 0.1,
    ) -> None:
        if not 0.0 < target_notional_frac <= 1.0:
            raise ValueError("target_notional_frac は (0, 1]")
        self.account_equity = account_equity
        self.cost_model = cost_model
        self.mr_params = mr_params
        self.risk_params = risk_params
        self.storage = storage
        self.target_notional_frac = target_notional_frac

    def run(
        self, df: pd.DataFrame, *, symbol: str, session: pd.Series | None = None
    ) -> DryRunReport:
        """分足 df（単一銘柄）をリプレイしてフォワード結果を返す。"""
        logger.info("dry_run 開始 symbol=%s DRY_RUN=%s（実発注なし）", symbol, DRY_RUN)
        rm = RiskManager(account_equity=self.account_equity, params=self.risk_params)
        monitor = FillMonitor()
        report = DryRunReport(final_equity=self.account_equity)

        if session is not None:
            sess = session
        else:
            sess = pd.Series(pd.DatetimeIndex(df.index).normalize(), index=df.index)
        executed_net_returns: list[float] = []
        order_seq = 0

        for day, day_idx in _iter_days(df, sess):
            rm.start_day(pd.Timestamp(day).date())
            day_df = df.loc[day_idx]
            trades = generate_trades(day_df, self.mr_params, session=sess.loc[day_idx])
            report.n_signals += len(trades)

            for trade in trades:
                order_seq += 1
                outcome = self._process_trade(
                    trade, symbol, order_seq, rm, monitor, executed_net_returns, report
                )
                if outcome is None:
                    continue
                report.executed_trades.append(trade)
                # 直近期待値で期待値ゲートを更新（負ならその後の新規建てを止める）
                rm.set_recent_expectancy(
                    sum(executed_net_returns) / len(executed_net_returns)
                )

        report.fill_stats = monitor.stats()
        report.evaluation = evaluate_trades(report.executed_trades, self.cost_model)
        report.final_equity = self.account_equity + sum(
            r * (self.account_equity * self.target_notional_frac) for r in executed_net_returns
        )
        return report

    # --- 内部 --------------------------------------------------------------------
    def _process_trade(
        self,
        trade: Trade,
        symbol: str,
        seq: int,
        rm: RiskManager,
        monitor: FillMonitor,
        net_returns: list[float],
        report: DryRunReport,
    ) -> Trade | None:
        """1トレードを関門に通し、承認なら疑似約定・実測・蓄積する。"""
        target_notional = self.target_notional_frac * self.account_equity
        shares = int(target_notional // trade.entry_price)
        if shares < 1:
            report.rejected.append((trade, "size_too_small"))
            return None

        entry_side, exit_side = _sides(trade.side)
        notional = shares * trade.entry_price

        entry_req = OrderRequest(
            symbol=symbol, side=entry_side, shares=shares, price=trade.entry_price, is_entry=True
        )
        decision = rm.check_order(entry_req)
        if not decision.approved:
            report.rejected.append((trade, decision.reason))
            return None

        # 疑似約定（理想化：指値どおり全量・滑り0）。実測値は本番で fill_monitor が上書き。
        entry_id = f"{symbol}-{seq}-entry"
        monitor.record_intent(
            OrderIntent(
                order_id=entry_id, symbol=symbol, side=entry_side,
                limit_price=trade.entry_price, shares=shares,
                reference_price=trade.entry_price, placed_at=trade.entry_time,
            )
        )
        entry_fill = monitor.record_fill(
            entry_id, fill_price=trade.entry_price, filled_shares=shares, filled_at=trade.entry_time
        )
        rm.on_open(symbol, notional=notional)

        # コスト控除後の損益（比率→円）
        cost_frac = self.cost_model.round_trip_cost_fraction(
            price=trade.entry_price, shares=1, holding_days=0, side=trade.side
        )
        gross_frac = trade.pnl_gross_per_share / trade.entry_price
        net_frac = gross_frac - cost_frac
        realized_yen = net_frac * notional

        exit_id = f"{symbol}-{seq}-exit"
        monitor.record_intent(
            OrderIntent(
                order_id=exit_id, symbol=symbol, side=exit_side,
                limit_price=trade.exit_price, shares=shares,
                reference_price=trade.exit_price, placed_at=trade.exit_time,
            )
        )
        exit_fill = monitor.record_fill(
            exit_id, fill_price=trade.exit_price, filled_shares=shares, filled_at=trade.exit_time
        )
        rm.on_close(symbol, realized_pnl=realized_yen)

        net_returns.append(net_frac)

        if self.storage is not None:
            self.storage.insert_trade(
                symbol, trade, gross_return=gross_frac, cost=cost_frac, net_return=net_frac
            )
            self.storage.insert_fill(entry_fill)
            self.storage.insert_fill(exit_fill)

        return trade


def _iter_days(df: pd.DataFrame, sess: pd.Series) -> list[tuple[Any, Any]]:
    """セッションキーごとに (key, そのインデックス) を時系列順で返す。"""
    groups: list[tuple[Any, Any]] = []
    for key, idx in df.groupby(sess, sort=True).groups.items():
        groups.append((key, idx))
    return groups
