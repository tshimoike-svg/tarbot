"""position_sizer.py のテスト。"""

from __future__ import annotations

import pytest

from config.settings import RiskParams
from strategy.position_sizer import size_position, size_position_fixed_fraction

_RP = RiskParams(max_risk_per_trade=0.005, max_symbol_ratio=0.20)


def test_exposure_bound_without_stop() -> None:
    # stop なし → エクスポージャ上限のみ。0.2*1e6/1000 = 200株
    r = size_position(account_equity=1_000_000, entry_price=1000, risk_params=_RP)
    assert r.shares == 200
    assert r.binding == "exposure"
    assert r.risk_amount is None
    assert r.notional == pytest.approx(200_000)


def test_risk_bound_with_tight_stop() -> None:
    # stop 950（損切り幅50）→ risk上限 = 5000/50 = 100株（exposure 200株より小）
    r = size_position(account_equity=1_000_000, entry_price=1000, risk_params=_RP, stop_price=950)
    assert r.shares == 100
    assert r.binding == "risk"
    assert r.risk_amount == pytest.approx(100 * 50)  # 5,000円 = 資金の0.5%
    assert r.risk_amount <= _RP.max_risk_per_trade * 1_000_000


def test_exposure_bound_when_stop_loose() -> None:
    # 損切り幅が狭い（広い株数を許す）ときはエクスポージャが律速
    # stop 999（幅1）→ risk上限 5000/1=5000株、exposure 200株 → 200株
    r = size_position(account_equity=1_000_000, entry_price=1000, risk_params=_RP, stop_price=999)
    assert r.shares == 200
    assert r.binding == "exposure"


def test_rounds_down_to_unit() -> None:
    # exposure cap = 0.2*1e6/1000 = 200 → ちょうど。unit=100 の倍数に切り捨て
    r = size_position(account_equity=1_000_000, entry_price=1000, risk_params=_RP, unit=100)
    assert r.shares % 100 == 0
    # 単元を大きくすると切り捨てが効く（150株相当 → 100株）
    r2 = size_position(account_equity=750_000, entry_price=1000, risk_params=_RP, unit=100)
    # exposure cap = 0.2*750000/1000 = 150 → floor100 = 100
    assert r2.shares == 100


def test_untradable_when_too_small() -> None:
    # 資金が小さく単元すら買えない
    r = size_position(account_equity=1_000, entry_price=1000, risk_params=_RP)
    assert r.shares == 0
    assert r.binding == "untradable"
    assert r.risk_amount is None


def test_validation() -> None:
    with pytest.raises(ValueError):
        size_position(account_equity=0, entry_price=1000, risk_params=_RP)
    with pytest.raises(ValueError):
        size_position(account_equity=1_000_000, entry_price=0, risk_params=_RP)
    with pytest.raises(ValueError):
        size_position(account_equity=1_000_000, entry_price=1000, risk_params=_RP, unit=0)


# --- size_position_fixed_fraction（2026-07-18決定：config_v + mom_lb60_filtered運用） ---
# 予算 = account_equity(100万) × leverage(3) × usage_rate(0.8) × per_symbol_share(0.2) = 48万円

def _combo(**kw: object) -> dict:
    base: dict = dict(
        account_equity=1_000_000, entry_price=1000, leverage=3.0, usage_rate=0.8,
        per_symbol_share=0.20,
    )
    base.update(kw)
    return base


def test_fixed_fraction_fills_budget_with_cheap_stock() -> None:
    # 予算48万・株価1000円→1単元10万円 → 4単元(40万)まで買える
    r = size_position_fixed_fraction(**_combo(entry_price=1000))
    assert r.shares == 400
    assert r.notional == pytest.approx(400_000)
    assert r.binding == "exposure"
    assert r.risk_amount is None


def test_fixed_fraction_buys_one_lot_when_lot_exceeds_budget() -> None:
    # 株価6000円→1単元60万円（予算48万円を超える）→ それでも1単元(100株)は買う
    r = size_position_fixed_fraction(**_combo(entry_price=6000))
    assert r.shares == 100
    assert r.notional == pytest.approx(600_000)
    assert r.binding == "exposure"


def test_fixed_fraction_exact_budget_boundary() -> None:
    # 1単元がちょうど予算と同額 → 1単元買う（境界値）
    r = size_position_fixed_fraction(**_combo(entry_price=4800))  # 100株=48万円
    assert r.shares == 100


def test_fixed_fraction_untradable_when_lot_exceeds_equity() -> None:
    # 1単元の必要額が資金そのものを超える → さすがに買えない
    r = size_position_fixed_fraction(**_combo(entry_price=20_000))  # 100株=200万円 > 資金100万円
    assert r.shares == 0
    assert r.binding == "untradable"


def test_fixed_fraction_scales_with_equity() -> None:
    # 資金が2倍なら予算も2倍・買える株数も2倍（entry_price=800円は端数なく割り切れる設定）
    r1 = size_position_fixed_fraction(**_combo(entry_price=800))
    r2 = size_position_fixed_fraction(**_combo(account_equity=2_000_000, entry_price=800))
    assert r2.shares == r1.shares * 2


def test_fixed_fraction_validation() -> None:
    with pytest.raises(ValueError):
        size_position_fixed_fraction(**_combo(account_equity=0))
    with pytest.raises(ValueError):
        size_position_fixed_fraction(**_combo(entry_price=0))
    with pytest.raises(ValueError):
        size_position_fixed_fraction(**_combo(leverage=0))
    with pytest.raises(ValueError):
        size_position_fixed_fraction(**_combo(usage_rate=0))
    with pytest.raises(ValueError):
        size_position_fixed_fraction(**_combo(usage_rate=1.5))
    with pytest.raises(ValueError):
        size_position_fixed_fraction(**_combo(per_symbol_share=0))
    with pytest.raises(ValueError):
        size_position_fixed_fraction(**_combo(unit=0))
