"""リスク管理の関門（全発注が必ず通る／絶対原則2）。

docs/trading_bot_design_v2.md §6 / CLAUDE.md 絶対原則2 に対応。
成行・指値・取消を問わず、**新規建ては必ず本モジュールの `check_order` を通す**。
本モジュールは発注しない（承認/拒否を判定するだけ）。発注は execution 層が、
DRY_RUN を確認したうえで行う。

強制する事前チェック（新規建て＝is_entry に対して）：
  1. サーキットブレーカー（当日損失が上限到達 → 当日の新規建て停止）
  2. 期待値ゲート（直近の実効期待値がコスト割れ → 新規建て停止）
  3. 1日の最大トレード回数
  4. 同時保有最大銘柄数
  5. 1銘柄あたり最大投資比率
  6. 1トレードの最大リスク（損切り到達時の想定損失額）

**決済・手仕舞い（is_entry=False）は、リスク低減のため上記ゲートを素通りで承認する**
（サーキットブレーカー作動中でもポジションは閉じられねばならない）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from config.settings import DEFAULT_RISK, RiskParams
from execution.fill_monitor import OrderSide

__all__ = ["OrderRequest", "RiskDecision", "RiskManager", "DenyReason"]

logger = logging.getLogger(__name__)

DenyReason = Literal[
    "ok",
    "invalid_order",
    "circuit_breaker",
    "expectancy_gate",
    "max_trades_reached",
    "max_positions_reached",
    "symbol_exposure_exceeded",
    "risk_per_trade_exceeded",
]


@dataclass(frozen=True)
class OrderRequest:
    """発注リクエスト（関門に渡す）。

    Attributes:
        symbol: 銘柄コード。
        side: "buy" / "sell"。
        shares: 株数。
        price: 想定約定価格（指値）。
        is_entry: True=新規建て（全ゲート対象）/ False=決済・手仕舞い（素通り承認）。
        risk_amount: この注文の想定最大損失額（円）。損切り幅×株数など。
            指定時のみ「1トレードの最大リスク」を検査する。
    """

    symbol: str
    side: OrderSide
    shares: int
    price: float
    is_entry: bool
    risk_amount: float | None = None


@dataclass(frozen=True)
class RiskDecision:
    """関門の判定結果。"""

    approved: bool
    reason: DenyReason
    message: str = ""

    def __bool__(self) -> bool:
        return self.approved


@dataclass
class _DailyState:
    day: date
    trades: int = 0
    realized_pnl: float = 0.0
    halted: bool = False  # サーキットブレーカー作動フラグ（当日限り）


@dataclass
class RiskManager:
    """発注前チェックと当日状態の管理。

    使い方：
        rm = RiskManager(account_equity=1_000_000)
        rm.start_day(date(2026, 1, 5))
        if rm.check_order(req):
            ... 執行（DRY_RUN 確認は execution 層） ...
            rm.on_open(symbol, notional)         # 約定で建ったら
            ...
            rm.on_close(symbol, realized_pnl)    # 手仕舞いしたら
        rm.set_recent_expectancy(e)              # 期待値ゲート更新
    """

    account_equity: float
    params: RiskParams = DEFAULT_RISK
    _positions: dict[str, float] = field(default_factory=dict)  # symbol -> notional(円)
    _expectancy_halt: bool = False
    _state: _DailyState | None = field(default=None)

    def __post_init__(self) -> None:
        if self.account_equity <= 0:
            raise ValueError("account_equity は正")

    # --- 日次状態 ----------------------------------------------------------------
    def start_day(self, day: date) -> None:
        """新しい取引日を開始（当日カウンタ・サーキットブレーカーをリセット）。

        期待値ゲートは日跨ぎで持続する（エッジ消失は1日で回復しないため）。
        """
        self._state = _DailyState(day=day)

    def _require_state(self) -> _DailyState:
        if self._state is None:
            raise RuntimeError("start_day(day) を先に呼んでください")
        return self._state

    # --- 期待値ゲート ------------------------------------------------------------
    def set_recent_expectancy(self, expectancy: float) -> None:
        """直近の実効期待値（コスト控除後）を更新。負ならゲートを閉じる。"""
        self._expectancy_halt = expectancy < 0.0
        if self._expectancy_halt:
            logger.warning("期待値ゲート作動：直近期待値=%.5f が負", expectancy)

    # --- 約定の反映 --------------------------------------------------------------
    def on_open(self, symbol: str, notional: float) -> None:
        """新規建ての約定を反映（建玉計上＋当日トレード数加算）。"""
        if notional <= 0:
            raise ValueError("notional は正")
        state = self._require_state()
        self._positions[symbol] = self._positions.get(symbol, 0.0) + notional
        state.trades += 1

    def on_close(self, symbol: str, realized_pnl: float) -> None:
        """手仕舞いの約定を反映（建玉解消＋当日損益更新＋ブレーカー判定）。"""
        state = self._require_state()
        self._positions.pop(symbol, None)
        state.realized_pnl += realized_pnl
        if state.realized_pnl <= -self.params.max_daily_loss * self.account_equity:
            if not state.halted:
                logger.warning(
                    "サーキットブレーカー作動：当日損益=%.0f が上限超過", state.realized_pnl
                )
            state.halted = True

    # --- 状態参照 ----------------------------------------------------------------
    @property
    def open_positions(self) -> dict[str, float]:
        return dict(self._positions)

    @property
    def trades_today(self) -> int:
        return self._require_state().trades

    @property
    def is_halted(self) -> bool:
        """新規建てが停止状態か（ブレーカーまたは期待値ゲート）。"""
        return self._require_state().halted or self._expectancy_halt

    # --- 関門本体 ----------------------------------------------------------------
    def check_order(self, req: OrderRequest) -> RiskDecision:
        """発注可否を判定する。決済（is_entry=False）は素通り承認。"""
        if req.shares <= 0 or req.price <= 0:
            return RiskDecision(False, "invalid_order", "株数・価格は正である必要があります")

        # 決済・手仕舞いはリスク低減なので常に許可（ブレーカー作動中でも閉じられる）
        if not req.is_entry:
            return RiskDecision(True, "ok", "決済注文（ゲート対象外）")

        state = self._require_state()

        if state.halted:
            return RiskDecision(False, "circuit_breaker", "当日損失上限到達。新規建て停止中")
        if self._expectancy_halt:
            return RiskDecision(False, "expectancy_gate", "直近期待値が負。新規建て停止中")
        if state.trades >= self.params.max_trades_per_day:
            return RiskDecision(
                False, "max_trades_reached", f"1日の最大トレード回数 {self.params.max_trades_per_day} 到達"
            )

        new_symbol = req.symbol not in self._positions
        if new_symbol and len(self._positions) >= self.params.max_positions:
            return RiskDecision(
                False, "max_positions_reached", f"同時保有上限 {self.params.max_positions} 銘柄到達"
            )

        notional = req.shares * req.price
        existing = self._positions.get(req.symbol, 0.0)
        if existing + notional > self.params.max_symbol_ratio * self.account_equity:
            return RiskDecision(
                False,
                "symbol_exposure_exceeded",
                f"1銘柄上限 {self.params.max_symbol_ratio:.0%} 超過",
            )

        if req.risk_amount is not None:
            if req.risk_amount > self.params.max_risk_per_trade * self.account_equity:
                return RiskDecision(
                    False,
                    "risk_per_trade_exceeded",
                    f"1トレード最大リスク {self.params.max_risk_per_trade:.1%} 超過",
                )

        return RiskDecision(True, "ok", "承認")
