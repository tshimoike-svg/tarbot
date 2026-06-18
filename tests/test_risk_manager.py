"""risk_manager.py のテスト（絶対原則4：これなしに変更しない）。

全発注が通る関門の各ゲートを固定する：
- 通常承認 / 不正注文
- サーキットブレーカー（当日損失上限）と、作動中でも決済は通ること
- 期待値ゲート（日跨ぎ持続）
- 1日の最大トレード回数・同時保有上限・1銘柄上限・1トレード最大リスク
- start_day 未呼び出しのガード
"""

from __future__ import annotations

from datetime import date

import pytest

from config.settings import RiskParams
from strategy.risk_manager import OrderRequest, RiskManager


def _rm(equity: float = 1_000_000, **param_kw: object) -> RiskManager:
    params = RiskParams(**param_kw) if param_kw else RiskParams()  # type: ignore[arg-type]
    rm = RiskManager(account_equity=equity, params=params)
    rm.start_day(date(2026, 1, 5))
    return rm


def _entry(symbol: str = "1301", shares: int = 100, price: float = 1000.0, **kw: object) -> OrderRequest:
    return OrderRequest(symbol=symbol, side="buy", shares=shares, price=price, is_entry=True, **kw)  # type: ignore[arg-type]


def _exit(symbol: str = "1301", shares: int = 100, price: float = 1000.0) -> OrderRequest:
    return OrderRequest(symbol=symbol, side="sell", shares=shares, price=price, is_entry=False)


# --- 基本 --------------------------------------------------------------------------
def test_normal_entry_approved() -> None:
    rm = _rm()
    d = rm.check_order(_entry())
    assert d.approved
    assert bool(d) is True
    assert d.reason == "ok"


def test_invalid_order_denied() -> None:
    rm = _rm()
    assert not rm.check_order(_entry(shares=0))
    assert rm.check_order(_entry(shares=0)).reason == "invalid_order"
    assert not rm.check_order(_entry(price=0))


def test_requires_start_day() -> None:
    rm = RiskManager(account_equity=1_000_000)
    with pytest.raises(RuntimeError):
        rm.check_order(_entry())


def test_account_equity_validation() -> None:
    with pytest.raises(ValueError):
        RiskManager(account_equity=0)


# --- サーキットブレーカー ----------------------------------------------------------
def test_circuit_breaker_halts_entries_but_allows_exits() -> None:
    rm = _rm(equity=1_000_000)  # 上限 3% = 30,000円
    rm.on_open("1301", notional=100_000)
    rm.on_close("1301", realized_pnl=-30_000)  # 上限到達
    assert rm.is_halted
    # 新規建ては拒否
    assert rm.check_order(_entry()).reason == "circuit_breaker"
    # 決済は作動中でも承認
    assert rm.check_order(_exit())


def test_circuit_breaker_resets_next_day() -> None:
    rm = _rm()
    rm.on_open("1301", notional=100_000)
    rm.on_close("1301", realized_pnl=-30_000)
    assert rm.is_halted
    rm.start_day(date(2026, 1, 6))
    assert not rm.is_halted
    assert rm.check_order(_entry())


# --- 期待値ゲート ------------------------------------------------------------------
def test_expectancy_gate_halts_entries_and_persists_across_days() -> None:
    rm = _rm()
    rm.set_recent_expectancy(-0.0001)  # コスト割れ
    assert rm.check_order(_entry()).reason == "expectancy_gate"
    # 日跨ぎでも持続
    rm.start_day(date(2026, 1, 6))
    assert rm.check_order(_entry()).reason == "expectancy_gate"
    # 回復したら解除
    rm.set_recent_expectancy(0.0002)
    assert rm.check_order(_entry())


# --- 回数・銘柄数・エクスポージャ・リスク -----------------------------------------
def test_max_trades_per_day() -> None:
    rm = _rm(max_trades_per_day=2)
    for i in range(2):
        assert rm.check_order(_entry(symbol=f"S{i}"))
        rm.on_open(f"S{i}", notional=10_000)
    assert rm.check_order(_entry(symbol="S2")).reason == "max_trades_reached"


def test_max_positions() -> None:
    rm = _rm(max_positions=2, max_symbol_ratio=1.0, max_trades_per_day=100)
    rm.on_open("A", notional=10_000)
    rm.on_open("B", notional=10_000)
    # 新しい3銘柄目は拒否
    assert rm.check_order(_entry(symbol="C")).reason == "max_positions_reached"
    # 既存銘柄への追加はエクスポージャ次第で可（銘柄数は増えない）
    assert rm.check_order(_entry(symbol="A", shares=1, price=100))


def test_symbol_exposure_exceeded() -> None:
    rm = _rm(equity=1_000_000, max_symbol_ratio=0.20)  # 上限 200,000円
    # 250株 × 1000円 = 250,000 > 200,000
    assert rm.check_order(_entry(shares=250, price=1000)).reason == "symbol_exposure_exceeded"
    # 既存建玉と合算で超過
    rm.on_open("1301", notional=150_000)
    assert rm.check_order(_entry(symbol="1301", shares=100, price=1000)).reason == "symbol_exposure_exceeded"


def test_risk_per_trade_exceeded() -> None:
    rm = _rm(equity=1_000_000, max_risk_per_trade=0.005)  # 上限 5,000円
    assert rm.check_order(_entry(risk_amount=5_001)).reason == "risk_per_trade_exceeded"
    assert rm.check_order(_entry(risk_amount=4_999))  # 範囲内は承認
    assert rm.check_order(_entry())  # risk_amount 未指定はスキップ


# --- 状態参照 ----------------------------------------------------------------------
def test_state_accessors() -> None:
    rm = _rm()
    rm.on_open("1301", notional=100_000)
    assert rm.trades_today == 1
    assert rm.open_positions == {"1301": 100_000}
    rm.on_close("1301", realized_pnl=500)
    assert rm.open_positions == {}
