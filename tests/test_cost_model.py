"""cost_model.py / costs.py のテスト（絶対原則4：これなしに変更しない）。

コストの“過小評価”は本番での負けに直結するため、
- 呼値テーブルの境界
- スプレッド capture の符号（0/0.5/1.0）
- 滑り・手数料・信用コスト・インパクトの各要素
- 合計と notional 比率
を明示的に固定する。
"""

from __future__ import annotations

import math

import pytest

from backtest.cost_model import CostModel
from config.costs import (
    CONSERVATIVE_DEFAULT,
    CostParams,
    get_cost_params,
    tick_size,
)


# --- tick_size --------------------------------------------------------------------
@pytest.mark.parametrize(
    ("price", "expected"),
    [
        (1, 1),
        (3_000, 1),       # 境界（含む）
        (3_001, 5),       # 次段へ
        (5_000, 5),
        (5_001, 10),
        (30_000, 10),
        (50_000, 50),
        (300_000, 100),
        (500_000, 500),
        (3_000_000, 1_000),
        (5_000_000, 5_000),
        (30_000_000, 10_000),
        (50_000_000, 50_000),
        (50_000_001, 100_000),
    ],
)
def test_tick_size_boundaries(price: float, expected: float) -> None:
    assert tick_size(price) == expected


def test_tick_size_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        tick_size(0)
    with pytest.raises(ValueError):
        tick_size(-100)


# --- effective spread -------------------------------------------------------------
def test_effective_spread_uses_ticks_times_ticksize() -> None:
    params = CostParams(spread_ticks=2.0, slippage_ticks_per_side=0.0)
    model = CostModel(params)
    # price=1000 → 呼値1円。spread = 2 * 1 = 2 円/株
    assert model.effective_spread_yen(1000.0) == pytest.approx(2.0)
    # price=4000 → 呼値5円。spread = 2 * 5 = 10 円/株
    assert model.effective_spread_yen(4000.0) == pytest.approx(10.0)


# --- spread capture の符号 --------------------------------------------------------
def _spread_only_model(capture: float) -> CostModel:
    return CostModel(
        CostParams(
            spread_ticks=2.0,
            slippage_ticks_per_side=0.0,
            spread_capture_ratio=capture,
        )
    )


def test_spread_capture_zero_pays_full_spread() -> None:
    # capture=0 → 往復で full-spread を払う = effective_spread * shares
    model = _spread_only_model(0.0)
    price, shares = 1000.0, 100
    full_spread = model.effective_spread_yen(price) * shares  # 2円 * 100 = 200
    assert model.round_trip_spread_cost_yen(price, shares) == pytest.approx(200.0)
    assert model.round_trip_spread_cost_yen(price, shares) == pytest.approx(full_spread)


def test_spread_capture_half_is_zero() -> None:
    model = _spread_only_model(0.5)
    assert model.round_trip_spread_cost_yen(1000.0, 100) == pytest.approx(0.0)


def test_spread_capture_full_is_negative_gain() -> None:
    model = _spread_only_model(1.0)
    # capture=1 → 往復で −full-spread（利得）
    assert model.round_trip_spread_cost_yen(1000.0, 100) == pytest.approx(-200.0)


# --- 滑り --------------------------------------------------------------------------
def test_slippage_is_two_sided_and_nonnegative() -> None:
    params = CostParams(spread_ticks=0.0, slippage_ticks_per_side=1.0)
    model = CostModel(params)
    # price=1000 → 呼値1円。往復滑り = 2 * 1 * 100 = 200
    assert model.round_trip_slippage_cost_yen(1000.0, 100) == pytest.approx(200.0)


# --- 手数料 ------------------------------------------------------------------------
def test_commission_zero_by_default_sor_free() -> None:
    model = CostModel(get_cost_params("mid"))
    assert model.commission_yen(1000.0, 100) == 0.0


def test_commission_rate_and_fixed_are_round_trip() -> None:
    params = CostParams(
        spread_ticks=0.0,
        slippage_ticks_per_side=0.0,
        commission_rate=0.001,
        commission_per_trade_yen=50.0,
    )
    model = CostModel(params)
    # notional = 1000*100 = 100_000。比例 = 0.001 * 100_000 * 2 = 200。固定 = 50*2 = 100
    assert model.commission_yen(1000.0, 100) == pytest.approx(300.0)


# --- 信用コスト --------------------------------------------------------------------
def test_financing_zero_for_day_trade() -> None:
    model = CostModel(CONSERVATIVE_DEFAULT)
    assert model.financing_cost_yen(1000.0, 100, holding_days=0, side="long") == 0.0


def test_financing_scales_with_days_and_uses_side_rate() -> None:
    params = CostParams(
        spread_ticks=0.0,
        slippage_ticks_per_side=0.0,
        margin_annual_rate=0.0365,        # 年率3.65% → 1日0.01%
        short_lending_annual_rate=0.0730,  # 年率7.30% → 1日0.02%
        financing_days_per_year=365,
    )
    model = CostModel(params)
    notional = 1000.0 * 100  # 100_000
    long_1d = model.financing_cost_yen(1000.0, 100, holding_days=1, side="long")
    short_1d = model.financing_cost_yen(1000.0, 100, holding_days=1, side="short")
    assert long_1d == pytest.approx(notional * 0.0365 / 365)   # = 10円
    assert short_1d == pytest.approx(notional * 0.0730 / 365)  # = 20円
    # 日数に線形
    assert model.financing_cost_yen(1000.0, 100, holding_days=3, side="long") == pytest.approx(
        3 * long_1d
    )


# --- マーケットインパクト ----------------------------------------------------------
def test_impact_zero_without_volume() -> None:
    model = CostModel(get_cost_params("small"))
    assert model.market_impact_yen(1000.0, 100, adv_shares=None) == 0.0
    assert model.market_impact_yen(1000.0, 100, adv_shares=0) == 0.0


def test_impact_sqrt_model_two_sided() -> None:
    params = CostParams(
        spread_ticks=0.0,
        slippage_ticks_per_side=0.0,
        impact_coefficient=0.1,
    )
    model = CostModel(params)
    # participation = 100/10_000 = 0.01 → sqrt=0.1 → one_way_fraction = 0.1*0.1 = 0.01
    # notional=100_000 → one_way=1000、往復=2000
    impact = model.market_impact_yen(1000.0, 100, adv_shares=10_000)
    assert impact == pytest.approx(100_000 * 0.1 * math.sqrt(0.01) * 2.0)
    assert impact == pytest.approx(2000.0)


# --- 統合：合計と比率 --------------------------------------------------------------
def test_total_is_sum_of_components() -> None:
    model = CostModel(get_cost_params("small"))
    bd = model.round_trip_cost(
        price=1000.0, shares=100, holding_days=1, side="long", adv_shares=50_000
    )
    assert bd.total == pytest.approx(
        bd.spread + bd.slippage + bd.commission + bd.financing + bd.impact
    )
    assert bd.notional == pytest.approx(100_000.0)
    assert bd.total_fraction == pytest.approx(bd.total / bd.notional)


def test_conservative_default_round_trip_is_costly() -> None:
    # 最保守設定では往復コスト比率は明確に正（タダではない）
    model = CostModel(CONSERVATIVE_DEFAULT)
    frac = model.round_trip_cost_fraction(price=1000.0, shares=100)
    assert frac > 0.0


def test_fraction_helper_matches_breakdown() -> None:
    model = CostModel(get_cost_params("mid"))
    bd = model.round_trip_cost(price=2500.0, shares=200, holding_days=2, side="short")
    frac = model.round_trip_cost_fraction(
        price=2500.0, shares=200, holding_days=2, side="short"
    )
    assert frac == pytest.approx(bd.total_fraction)


# --- 入力バリデーション ------------------------------------------------------------
def test_round_trip_rejects_bad_inputs() -> None:
    model = CostModel(CONSERVATIVE_DEFAULT)
    with pytest.raises(ValueError):
        model.round_trip_cost(price=0.0, shares=100)
    with pytest.raises(ValueError):
        model.round_trip_cost(price=1000.0, shares=0)


def test_costparams_validation() -> None:
    with pytest.raises(ValueError):
        CostParams(spread_ticks=-1.0, slippage_ticks_per_side=0.0)
    with pytest.raises(ValueError):
        CostParams(spread_ticks=1.0, slippage_ticks_per_side=0.0, spread_capture_ratio=1.5)
