"""複数戦略・複数銘柄を1つの資金プールで捌く割り当てロジック（2026-07-18決定）。

config_v（反転）+ mom_lb60_filtered（momentum）併用運用のために作った。信用3倍・
証拠金使用率80%・1銘柄配分20%（`position_sizer.size_position_fixed_fraction`）で
サイジングし、同日に複数シグナルが競合したら**1単元必要額が安い順**に処理する。

1日の新規建て上限（既定2件）・同時保有上限・1銘柄あたりの配分上限は
`RiskManager`（`config.settings.COMBO_V_MOM60_RISK`）が実際に判定する（絶対原則2：
発注可否の最終判断は必ず risk_manager を経由し、バイパスしない）。本モジュールは
「どの順で risk_manager に候補を提示するか」というサイジング・優先順位付けだけを担う。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from execution.fill_monitor import OrderSide
from strategy.position_sizer import SizingResult, size_position_fixed_fraction
from strategy.risk_manager import OrderRequest, RiskDecision, RiskManager

__all__ = ["Candidate", "AllocationDecision", "allocate_daily_entries"]


@dataclass(frozen=True)
class Candidate:
    """当日の新規建て候補シグナル。"""

    symbol: str
    side: OrderSide
    entry_price: float
    source: str = ""  # どの戦略/設定が出したシグナルか（ログ・分析用、判定には使わない）


@dataclass(frozen=True)
class AllocationDecision:
    """1候補に対する割り当て結果。"""

    candidate: Candidate
    sizing: SizingResult
    decision: RiskDecision

    @property
    def approved(self) -> bool:
        return self.decision.approved and self.sizing.shares >= 1


def allocate_daily_entries(
    candidates: Sequence[Candidate],
    *,
    risk_manager: RiskManager,
    account_equity: float,
    leverage: float,
    usage_rate: float,
    per_symbol_share: float,
    unit: int = 100,
) -> list[AllocationDecision]:
    """当日の候補を「1単元必要額が安い順」に risk_manager へ提示し、承認された分だけ建てる。

    承認された候補は `risk_manager.on_open()` をその場で呼ぶため、後続候補の
    銘柄集中度チェック・日次上限チェックに正しく反映される（呼び出し順が結果を左右する）。

    既に建玉のある銘柄は候補から除外する（1銘柄1ポジション。ピラミッディングなし）。
    """
    held_symbols = set(risk_manager.open_positions.keys())
    ordered = sorted(candidates, key=lambda c: c.entry_price * unit)

    results: list[AllocationDecision] = []
    for c in ordered:
        if c.symbol in held_symbols:
            results.append(
                AllocationDecision(
                    candidate=c,
                    sizing=SizingResult(shares=0, notional=0.0, risk_amount=None, binding="untradable"),
                    decision=RiskDecision(False, "invalid_order", "既に建玉のある銘柄（重複不可）"),
                )
            )
            continue

        sizing = size_position_fixed_fraction(
            account_equity=account_equity, entry_price=c.entry_price,
            leverage=leverage, usage_rate=usage_rate, per_symbol_share=per_symbol_share,
            unit=unit,
        )
        if sizing.shares < 1:
            results.append(
                AllocationDecision(
                    candidate=c, sizing=sizing,
                    decision=RiskDecision(False, "invalid_order", "サイジング結果が0株"),
                )
            )
            continue

        req = OrderRequest(
            symbol=c.symbol, side=c.side, shares=sizing.shares, price=c.entry_price,
            is_entry=True, risk_amount=None,
        )
        decision = risk_manager.check_order(req)
        if decision.approved:
            risk_manager.on_open(c.symbol, notional=sizing.notional)
            held_symbols.add(c.symbol)
        results.append(AllocationDecision(candidate=c, sizing=sizing, decision=decision))

    return results
