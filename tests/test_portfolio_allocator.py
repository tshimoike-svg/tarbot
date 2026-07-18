"""portfolio_allocator.py のテスト（2026-07-18決定：config_v + mom_lb60_filtered運用）。"""

from __future__ import annotations

from datetime import date

from config.settings import COMBO_V_MOM60_RISK, RiskParams
from strategy.portfolio_allocator import Candidate, allocate_daily_entries
from strategy.risk_manager import RiskManager

_EQUITY = 1_000_000.0
_KW = dict(leverage=3.0, usage_rate=0.8, per_symbol_share=0.20)  # 予算48万円/銘柄


def _rm(params: RiskParams = COMBO_V_MOM60_RISK) -> RiskManager:
    rm = RiskManager(account_equity=_EQUITY, params=params)
    rm.start_day(date(2026, 1, 5))
    return rm


def test_all_candidates_approved_when_under_daily_cap() -> None:
    rm = _rm()
    candidates = [
        Candidate(symbol="1301", side="buy", entry_price=1000.0),
        Candidate(symbol="7203", side="buy", entry_price=2000.0),
    ]
    results = allocate_daily_entries(candidates, risk_manager=rm, account_equity=_EQUITY, **_KW)
    assert all(r.approved for r in results)
    assert rm.trades_today == 2


def test_cheapest_lot_cost_processed_first() -> None:
    # 予算48万円に対し、1株6000円(1単元60万)を先に処理すると2件目が入らない可能性がある。
    # 安い方(1000円)を先に処理する設計なので、まず1000円側が確実に承認される。
    rm = _rm()
    candidates = [
        Candidate(symbol="EXPENSIVE", side="buy", entry_price=6000.0),  # 1単元60万円
        Candidate(symbol="CHEAP", side="buy", entry_price=1000.0),      # 1単元10万円
    ]
    results = allocate_daily_entries(candidates, risk_manager=rm, account_equity=_EQUITY, **_KW)
    by_symbol = {r.candidate.symbol: r for r in results}
    assert by_symbol["CHEAP"].approved
    # 安い方が先に処理されたことをsharesで確認（cheapは予算いっぱいの4単元=400株のはず）
    assert by_symbol["CHEAP"].sizing.shares == 400


def test_daily_cap_rejects_third_candidate() -> None:
    # COMBO_V_MOM60_RISK.max_trades_per_day == 2
    rm = _rm()
    candidates = [
        Candidate(symbol="A", side="buy", entry_price=1000.0),
        Candidate(symbol="B", side="buy", entry_price=1000.0),
        Candidate(symbol="C", side="buy", entry_price=1000.0),
    ]
    results = allocate_daily_entries(candidates, risk_manager=rm, account_equity=_EQUITY, **_KW)
    approved = [r for r in results if r.approved]
    rejected = [r for r in results if not r.approved]
    assert len(approved) == 2
    assert len(rejected) == 1
    assert rejected[0].decision.reason == "max_trades_reached"


def test_duplicate_symbol_is_rejected() -> None:
    rm = _rm()
    rm.on_open("1301", notional=100_000.0)  # 既に建玉あり
    candidates = [Candidate(symbol="1301", side="buy", entry_price=1000.0)]
    results = allocate_daily_entries(candidates, risk_manager=rm, account_equity=_EQUITY, **_KW)
    assert not results[0].approved
    assert "重複" in results[0].decision.message


def test_lot_too_expensive_still_buys_one_lot() -> None:
    # 1単元が予算(48万)を超えても最低1単元は買う（size_position_fixed_fractionの仕様）
    rm = _rm()
    candidates = [Candidate(symbol="1301", side="buy", entry_price=6000.0)]  # 1単元60万円
    results = allocate_daily_entries(candidates, risk_manager=rm, account_equity=_EQUITY, **_KW)
    assert results[0].approved
    assert results[0].sizing.shares == 100


def test_lot_far_exceeding_equity_is_still_rejected() -> None:
    # size_position_fixed_fraction 自身の下限（lot_cost > account_equity）は
    # max_symbol_ratio=1.0 でも依然として効く（口座資金そのものより高い1単元は買えない）。
    rm = _rm()
    candidates = [Candidate(symbol="1301", side="buy", entry_price=20_000.0)]  # 1単元200万円
    results = allocate_daily_entries(candidates, risk_manager=rm, account_equity=_EQUITY, **_KW)
    assert not results[0].approved
    assert results[0].sizing.binding == "untradable"


def test_on_open_reflected_for_subsequent_candidates_same_batch() -> None:
    # 1件目の承認がrisk_managerに反映され、2件目の判定に影響することを確認
    rm = _rm(RiskParams(max_positions=1, max_trades_per_day=5, max_symbol_ratio=1.0))
    candidates = [
        Candidate(symbol="A", side="buy", entry_price=1000.0),
        Candidate(symbol="B", side="buy", entry_price=1000.0),
    ]
    results = allocate_daily_entries(candidates, risk_manager=rm, account_equity=_EQUITY, **_KW)
    assert results[0].approved
    assert not results[1].approved
    assert results[1].decision.reason == "max_positions_reached"
