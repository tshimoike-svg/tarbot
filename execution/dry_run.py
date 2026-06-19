"""ドライラン・ハーネス（フォワード検証の司令塔・発注なし）。

docs/trading_bot_design_v3.md / CLAUDE.md（execution/dry_run.py）に対応。
スイング戦略の確定トレードを**時系列イベント（建て→手仕舞い）として再生**し、
建て時に risk_manager（関門）→position_sizer→疑似約定→fill_monitor（実測）を通し、
手仕舞い時に損益を確定して evaluator・storage へ流す。**実発注は一切しない**（絶対原則1・6）。

戦略非依存：`generate`（df→list[Trade]）を渡すので、swing_reversion / swing_momentum の
どちらでも回せる（比較に使う）。オーバーナイトで複数銘柄日跨ぎ保有が起こり得るため、
建て/手仕舞いを時刻順にマージして処理し、サーキットブレーカーの日次集計と同時保有数を
正しく扱う。

⚠️ 疑似約定は「寄り/指値どおり約定・滑り0」の理想化。実約定率・実滑りはライブ運用で
   fill_monitor が計測して置き換える。持ち越し金利は cost_model が holding_days で計上する。
"""

from __future__ import annotations

import heapq
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from backtest.cost_model import CostModel
from backtest.evaluator import EvaluationResult, evaluate_trades
from config.settings import DEFAULT_RISK, DRY_RUN, RiskParams
from execution.fill_monitor import FillMonitor, FillStats, OrderIntent, OrderSide
from strategy.position_sizer import size_position
from strategy.risk_manager import OrderRequest, RiskManager
from strategy.trade import Trade

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


def _holding_days(trade: Trade) -> int:
    return int((trade.exit_time.normalize() - trade.entry_time.normalize()).days)


@dataclass(order=True)
class _OpenPosition:
    """建玉のコンテキスト（手仕舞いイベントで使う）。order=True で heap 比較可能に。"""

    exit_time: pd.Timestamp
    seq: int
    trade: Trade = field(compare=False)
    symbol: str = field(compare=False)
    shares: int = field(compare=False)
    notional: float = field(compare=False)
    net_frac: float = field(compare=False)
    realized_yen: float = field(compare=False)
    exit_side: OrderSide = field(compare=False)


class DryRunHarness:
    """戦略のトレード列をイベント再生し、フォワードの全工程を結線する。

    Args:
        account_equity: 初期資金（円）。
        cost_model: 往復コストモデル（持ち越し金利を含む）。
        generate: df → list[Trade] の戦略関数（swing_*.generate_trades を partial 等で）。
        risk_params: リスク設定。
        storage: 指定すれば確定トレード・約定実測を蓄積する。
        unit: 単元株数（既定100）。
    """

    def __init__(
        self,
        *,
        account_equity: float,
        cost_model: CostModel,
        generate: Callable[[pd.DataFrame], list[Trade]],
        risk_params: RiskParams = DEFAULT_RISK,
        storage: Storage | None = None,
        unit: int = 100,
    ) -> None:
        if unit < 1:
            raise ValueError("unit は 1 以上")
        self.account_equity = account_equity
        self.cost_model = cost_model
        self.generate = generate
        self.risk_params = risk_params
        self.storage = storage
        self.unit = unit

    def run(self, df: pd.DataFrame, *, symbol: str) -> DryRunReport:
        """日足 df（単一銘柄）から戦略のトレードを生成し、再生する。"""
        logger.info("dry_run 開始 symbol=%s DRY_RUN=%s（実発注なし）", symbol, DRY_RUN)
        rm = RiskManager(account_equity=self.account_equity, params=self.risk_params)
        monitor = FillMonitor()
        report = DryRunReport(final_equity=self.account_equity)

        trades = sorted(self.generate(df), key=lambda t: t.entry_time)
        report.n_signals = len(trades)

        net_returns: list[float] = []
        realized_yens: list[float] = []
        current_day = None
        seq = 0
        open_idx = 0
        close_heap: list[_OpenPosition] = []

        def roll(ts: pd.Timestamp) -> None:
            nonlocal current_day
            day = ts.date()
            if day != current_day:
                rm.start_day(day)
                current_day = day

        while open_idx < len(trades) or close_heap:
            next_open = trades[open_idx].entry_time if open_idx < len(trades) else None
            next_close = close_heap[0].exit_time if close_heap else None
            # 同時刻なら手仕舞いを先に（建玉・損益を先に解放してから新規を判定）
            do_close = next_open is None or (next_close is not None and next_close <= next_open)

            if do_close:
                pos = heapq.heappop(close_heap)
                roll(pos.exit_time)
                self._close(pos, rm, monitor, report, net_returns, realized_yens)
                rm.set_recent_expectancy(sum(net_returns) / len(net_returns))
            else:
                trade = trades[open_idx]
                open_idx += 1
                seq += 1
                roll(trade.entry_time)
                self._open(trade, symbol, seq, rm, monitor, report, close_heap)

        report.fill_stats = monitor.stats()
        report.evaluation = evaluate_trades(report.executed_trades, self.cost_model)
        report.final_equity = self.account_equity + sum(realized_yens)
        return report

    # --- 建て -------------------------------------------------------------------
    def _open(
        self,
        trade: Trade,
        symbol: str,
        seq: int,
        rm: RiskManager,
        monitor: FillMonitor,
        report: DryRunReport,
        close_heap: list[_OpenPosition],
    ) -> None:
        sizing = size_position(
            account_equity=self.account_equity,
            entry_price=trade.entry_price,
            risk_params=self.risk_params,
            stop_price=trade.stop_price,
            unit=self.unit,
        )
        if sizing.shares < 1:
            report.rejected.append((trade, "size_too_small"))
            return

        entry_side, exit_side = _sides(trade.side)
        entry_req = OrderRequest(
            symbol=symbol, side=entry_side, shares=sizing.shares, price=trade.entry_price,
            is_entry=True, risk_amount=sizing.risk_amount,
        )
        decision = rm.check_order(entry_req)
        if not decision.approved:
            report.rejected.append((trade, decision.reason))
            return

        entry_id = f"{symbol}-{seq}-entry"
        monitor.record_intent(
            OrderIntent(
                order_id=entry_id, symbol=symbol, side=entry_side,
                limit_price=trade.entry_price, shares=sizing.shares,
                reference_price=trade.entry_price, placed_at=trade.entry_time,
            )
        )
        entry_fill = monitor.record_fill(
            entry_id, fill_price=trade.entry_price, filled_shares=sizing.shares,
            filled_at=trade.entry_time,
        )
        rm.on_open(symbol, notional=sizing.notional)
        if self.storage is not None:
            self.storage.insert_fill(entry_fill)

        # コスト控除後の損益（持ち越し金利込み）
        cost_frac = self.cost_model.round_trip_cost_fraction(
            price=trade.entry_price, shares=1, holding_days=_holding_days(trade), side=trade.side
        )
        gross_frac = trade.pnl_gross_per_share / trade.entry_price
        net_frac = gross_frac - cost_frac

        heapq.heappush(
            close_heap,
            _OpenPosition(
                exit_time=trade.exit_time, seq=seq, trade=trade, symbol=symbol,
                shares=sizing.shares, notional=sizing.notional, net_frac=net_frac,
                realized_yen=net_frac * sizing.notional, exit_side=exit_side,
            ),
        )

    # --- 手仕舞い ---------------------------------------------------------------
    def _close(
        self,
        pos: _OpenPosition,
        rm: RiskManager,
        monitor: FillMonitor,
        report: DryRunReport,
        net_returns: list[float],
        realized_yens: list[float],
    ) -> None:
        exit_id = f"{pos.symbol}-{pos.seq}-exit"
        monitor.record_intent(
            OrderIntent(
                order_id=exit_id, symbol=pos.symbol, side=pos.exit_side,
                limit_price=pos.trade.exit_price, shares=pos.shares,
                reference_price=pos.trade.exit_price, placed_at=pos.trade.exit_time,
            )
        )
        exit_fill = monitor.record_fill(
            exit_id, fill_price=pos.trade.exit_price, filled_shares=pos.shares,
            filled_at=pos.trade.exit_time,
        )
        rm.on_close(pos.symbol, realized_pnl=pos.realized_yen)

        net_returns.append(pos.net_frac)
        realized_yens.append(pos.realized_yen)
        report.executed_trades.append(pos.trade)

        if self.storage is not None:
            gross_frac = pos.trade.pnl_gross_per_share / pos.trade.entry_price
            self.storage.insert_trade(
                pos.symbol, pos.trade, gross_return=gross_frac,
                cost=gross_frac - pos.net_frac, net_return=pos.net_frac,
            )
            self.storage.insert_fill(exit_fill)
