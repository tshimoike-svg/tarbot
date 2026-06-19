"""ポジションサイジング（リスク基準で株数を決める）。

docs/trading_bot_design_v2.md §6 / CLAUDE.md 絶対原則2 に対応。
「1トレードで失う資金額」を固定する思想（固定%損切りではなく金額固定）に基づき、
損切り幅から逆算して株数を出す。算出結果（risk_amount）は `risk_manager` の
1トレード最大リスク・ゲートにそのまま渡せる。

株数は次の2つの上限の小さい方を、単元（既定100株）に切り捨てる：
  1. リスク上限：max_risk_per_trade × 資金 ÷ 損切り幅（1株あたり）
  2. エクスポージャ上限：max_symbol_ratio × 資金 ÷ 価格

損切り価格が無い（stop_price=None）場合はリスク上限を適用できないので、
エクスポージャ上限のみで決める（risk_amount は None）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from config.settings import RiskParams

__all__ = ["SizingResult", "size_position"]

Binding = Literal["risk", "exposure", "untradable"]


@dataclass(frozen=True)
class SizingResult:
    """サイジング結果。"""

    shares: int
    notional: float
    risk_amount: float | None  # 損切り到達時の想定損失額（円）。stop 無しなら None
    binding: Binding           # 何が株数を律速したか


def size_position(
    *,
    account_equity: float,
    entry_price: float,
    risk_params: RiskParams,
    stop_price: float | None = None,
    unit: int = 100,
) -> SizingResult:
    """リスク基準で発注株数を算出する。

    Args:
        account_equity: 口座資金（円、>0）。
        entry_price: 想定エントリー価格（円、>0）。
        risk_params: max_risk_per_trade / max_symbol_ratio を使う。
        stop_price: 保護ストップ価格。None ならエクスポージャ上限のみ。
        unit: 単元株数（既定 100）。株数はこの倍数に切り捨てる。

    Returns:
        SizingResult。取引可能株数が単元未満なら shares=0, binding="untradable"。
    """
    if account_equity <= 0:
        raise ValueError("account_equity は正")
    if entry_price <= 0:
        raise ValueError("entry_price は正")
    if unit < 1:
        raise ValueError("unit は 1 以上")

    # エクスポージャ上限（常に適用）
    max_notional = risk_params.max_symbol_ratio * account_equity
    shares_cap = max_notional / entry_price
    binding: Binding = "exposure"

    # リスク上限（損切り幅があるときのみ）
    stop_distance = abs(entry_price - stop_price) if stop_price is not None else 0.0
    if stop_distance > 0:
        risk_budget = risk_params.max_risk_per_trade * account_equity
        shares_by_risk = risk_budget / stop_distance
        if shares_by_risk < shares_cap:
            shares_cap = shares_by_risk
            binding = "risk"

    shares = int(shares_cap // unit) * unit
    if shares < unit:
        return SizingResult(shares=0, notional=0.0, risk_amount=None, binding="untradable")

    notional = shares * entry_price
    risk_amount = shares * stop_distance if stop_distance > 0 else None
    return SizingResult(shares=shares, notional=notional, risk_amount=risk_amount, binding=binding)
