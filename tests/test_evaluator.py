"""evaluator.py のテスト。

主眼：
- コストを差し引いた net リターンが正しく、**グロスがプラスでもコストで負ける**ことを示す。
- 集計（期待値・勝率・PF・最大DD）の数値。
- ウォークフォワード分割。
- Phase 0 ゲート判定の pass/fail。
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from backtest.cost_model import CostModel
from backtest.evaluator import (
    check_phase0_gate,
    compute_trade_results,
    evaluate_trades,
    max_drawdown,
    walk_forward,
)
from config.costs import CostParams
from strategy.mean_reversion import Side, Trade


def _trade(
    side: Side,
    entry_price: float,
    exit_price: float,
    *,
    day: str = "2026-01-05",
    exit_day: str | None = None,
    reason: str = "take_profit",
) -> Trade:
    return Trade(
        side=side,
        entry_time=pd.Timestamp(f"{day} 09:30"),
        entry_price=entry_price,
        exit_time=pd.Timestamp(f"{exit_day or day} 10:00"),
        exit_price=exit_price,
        exit_reason=reason,  # type: ignore[arg-type]
    )


# spread 2tick + slippage 2tick(往復)、price=1000→tick=1 → cost fraction = 0.004、財務/手数料0
_NOIMPACT = CostParams(
    spread_ticks=2.0,
    slippage_ticks_per_side=1.0,
    impact_coefficient=0.0,
)
_MODEL = CostModel(_NOIMPACT)


# --- コスト控除：グロス＋でもコストで負ける --------------------------------------
def test_cost_turns_marginal_winner_negative() -> None:
    # long 1000→1003：グロス +0.3%、往復コスト 0.4% → net −0.1%
    [res] = compute_trade_results([_trade("long", 1000, 1003)], _MODEL)
    assert res.gross_return == pytest.approx(0.003)
    assert res.cost == pytest.approx(0.004)
    assert res.net_return == pytest.approx(-0.001)
    assert res.holding_days == 0


def test_short_gross_sign() -> None:
    # short 1000→997：グロス +0.3%
    [res] = compute_trade_results([_trade("short", 1000, 997)], _MODEL)
    assert res.gross_return == pytest.approx(0.003)
    assert res.net_return == pytest.approx(-0.001)


def test_overnight_adds_financing() -> None:
    # 翌日決済 → 1日分の金利が乗る（long: margin_annual_rate 既定 2.8%）
    params = CostParams(spread_ticks=0.0, slippage_ticks_per_side=0.0, margin_annual_rate=0.0365)
    model = CostModel(params)
    [res] = compute_trade_results(
        [_trade("long", 1000, 1000, exit_day="2026-01-06")], model
    )
    assert res.holding_days == 1
    assert res.cost == pytest.approx(0.0365 / 365)  # 1日分 = 0.01%


# --- 集計 --------------------------------------------------------------------------
def test_expectancy_winrate_aggregate() -> None:
    # コスト無しモデルで純粋にグロスを見る
    free = CostModel(CostParams(spread_ticks=0.0, slippage_ticks_per_side=0.0))
    trades = [
        _trade("long", 1000, 1020),  # +2%
        _trade("long", 1000, 990),   # -1%
    ]
    res = evaluate_trades(trades, free)
    assert res.n_trades == 2
    assert res.win_rate == pytest.approx(0.5)
    assert res.avg_win == pytest.approx(0.02)
    assert res.avg_loss == pytest.approx(-0.01)
    assert res.expectancy == pytest.approx((0.02 - 0.01) / 2)
    assert res.profit_factor == pytest.approx(0.02 / 0.01)
    assert res.by_exit_reason == {"take_profit": 2}


def test_empty_trades() -> None:
    res = evaluate_trades([], _MODEL)
    assert res.is_empty
    assert res.n_trades == 0
    assert math.isnan(res.expectancy)
    assert res.max_drawdown == 0.0


# --- 最大ドローダウン --------------------------------------------------------------
def test_max_drawdown_known_sequence() -> None:
    # equity: +0.1→0.1, -0.3→-0.2, +0.05→-0.15。ピーク0.1からトラフ-0.2 → 最大下落=0.3
    assert max_drawdown([0.1, -0.3, 0.05]) == pytest.approx(0.3)


def test_max_drawdown_monotonic_up_is_zero() -> None:
    assert max_drawdown([0.1, 0.2, 0.05]) == pytest.approx(0.0)


# --- ウォークフォワード ------------------------------------------------------------
def test_walk_forward_splits_contiguously() -> None:
    free = CostModel(CostParams(spread_ticks=0.0, slippage_ticks_per_side=0.0))
    trades = [_trade("long", 1000, 1000 + i) for i in range(10)]
    folds = walk_forward(trades, free, n_splits=2)
    assert len(folds) == 2
    assert folds[0].n_trades == 5
    assert folds[1].n_trades == 5
    # 端数：7件を3分割 → 3,2,2
    folds3 = walk_forward(trades[:7], free, n_splits=3)
    assert [f.n_trades for f in folds3] == [3, 2, 2]


def test_walk_forward_empty() -> None:
    assert walk_forward([], _MODEL, n_splits=3) == []


# --- Phase 0 ゲート ----------------------------------------------------------------
def test_gate_passes_when_all_conditions_met() -> None:
    free = CostModel(CostParams(spread_ticks=0.0, slippage_ticks_per_side=0.0))
    # 全勝の +0.5% トレードを十分数。DD ほぼ無し
    trades = [_trade("long", 1000, 1005) for _ in range(300)]
    res = evaluate_trades(trades, free)
    folds = walk_forward(trades, free, n_splits=3)
    gate = check_phase0_gate(res, folds, min_trades=300, max_drawdown_threshold=0.15)
    assert gate.expectancy_positive
    assert gate.enough_trades
    assert gate.drawdown_ok
    assert gate.walkforward_stable
    assert gate.passed


def test_gate_fails_on_negative_expectancy_after_cost() -> None:
    # グロス +0.3% だがコスト 0.4% → 期待値マイナス。回数も不足
    trades = [_trade("long", 1000, 1003) for _ in range(10)]
    res = evaluate_trades(trades, _MODEL)
    folds = walk_forward(trades, _MODEL, n_splits=2)
    gate = check_phase0_gate(res, folds, min_trades=300)
    assert not gate.expectancy_positive
    assert not gate.enough_trades
    assert not gate.passed


def test_gate_fails_when_a_fold_is_unstable() -> None:
    free = CostModel(CostParams(spread_ticks=0.0, slippage_ticks_per_side=0.0))
    # 勝ち150・負け50。4分割すると最終区間(=末尾50件)は負けだけ → 区間安定性で落ちる
    winners = [_trade("long", 1000, 1010, day="2026-01-05") for _ in range(150)]  # +1%
    losers = [_trade("long", 1000, 990, day="2026-01-06") for _ in range(50)]     # -1%
    trades = winners + losers
    res = evaluate_trades(trades, free)
    folds = walk_forward(trades, free, n_splits=4)
    gate = check_phase0_gate(res, folds, min_trades=100)
    assert res.expectancy > 0  # 全体ではプラス（(150*0.01 - 50*0.01)/200 = 0.005）
    assert folds[-1].expectancy < 0  # 末尾区間は負けのみ
    assert not gate.walkforward_stable
    assert not gate.passed
