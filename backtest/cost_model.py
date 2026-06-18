"""往復コストモデル（期待値計算の心臓部）。

`config/costs.py` の `CostParams`（想定値）を受け取り、1往復トレードの
コストを「円」および「notional 比率」で算出する。

往復コスト ＝ スプレッド ＋ 滑り ＋ 手数料 ＋ 信用コスト（持ち越し時）＋ インパクト

設計対応（docs/trading_bot_design_v2.md §0, §7-2）：
- スプレッド：指値で half-spread を捕る/払うを `spread_capture_ratio` で表現。
- 滑り：片道ごとに保守マージンを計上（指値不約定→成行転換等を吸収）。
- 信用コスト：持ち越し日数 × 年率（当日返済 days=0 ならゼロ）。
- マーケットインパクト：薄い板を自注文が動かす分（出来高が分かるときのみ）。

⚠️ コスト“過小評価”は本番での確実な負けに直結する。係数は保守的側に倒すこと。
   金額は float で扱う（統計的見積もりのため）。実発注の価格丸めは別途
   呼値刻みで厳密化する（order_engine 側の責務。本モジュールは見積もり専用）。
"""

from __future__ import annotations

from dataclasses import dataclass

from config.costs import CostParams, Side, tick_size

__all__ = ["CostBreakdown", "CostModel"]


@dataclass(frozen=True)
class CostBreakdown:
    """1往復トレードのコスト内訳（すべて円）。

    spread は `spread_capture_ratio` 次第で負（＝スプレッドを捕って利得）に
    なり得る。他の項目は常に 0 以上。
    """

    spread: float
    slippage: float
    commission: float
    financing: float
    impact: float
    notional: float

    @property
    def total(self) -> float:
        """往復コスト合計（円）。"""
        return self.spread + self.slippage + self.commission + self.financing + self.impact

    @property
    def total_fraction(self) -> float:
        """notional に対するコスト比率（期待値ゲートで使う）。

        E[1トレード] のリターン（比率）と直接比較できる単位。
        """
        if self.notional <= 0:
            raise ValueError("notional が 0 以下のため比率を計算できません")
        return self.total / self.notional

    def as_dict(self) -> dict[str, float]:
        """構造化ログ用にフラットな dict を返す。"""
        return {
            "spread": self.spread,
            "slippage": self.slippage,
            "commission": self.commission,
            "financing": self.financing,
            "impact": self.impact,
            "notional": self.notional,
            "total": self.total,
            "total_fraction": self.total_fraction,
        }


class CostModel:
    """`CostParams` に基づき往復コストを算出する。

    Example:
        >>> from config.costs import get_cost_params
        >>> model = CostModel(get_cost_params("small"))
        >>> bd = model.round_trip_cost(price=1000.0, shares=100)
        >>> bd.total_fraction > 0
        True
    """

    def __init__(self, params: CostParams) -> None:
        self.params = params

    # --- 個別コスト要素 ----------------------------------------------------------
    def effective_spread_yen(self, price: float) -> float:
        """実効スプレッド（円／株）＝ spread_ticks × 呼値。"""
        return self.params.spread_ticks * tick_size(price)

    def round_trip_spread_cost_yen(self, price: float, shares: int) -> float:
        """往復のスプレッドコスト（円）。

        片道あたり half-spread を `(1 - 2*capture)` 倍だけ払う/受け取るとモデル化する：
          - capture=0.0 → 片道で half-spread を払う → 往復で full-spread（最保守）
          - capture=0.5 → 収支ゼロ
          - capture=1.0 → 片道で half-spread を受け取る → 往復で −full-spread（利得）
        したがって往復スプレッドコスト ＝ effective_spread × (1 − 2*capture) × shares。
        """
        capture = self.params.spread_capture_ratio
        return self.effective_spread_yen(price) * (1.0 - 2.0 * capture) * shares

    def round_trip_slippage_cost_yen(self, price: float, shares: int) -> float:
        """往復の滑りコスト（円）。常に 0 以上。

        片道 slippage_ticks_per_side × 呼値、往復で 2 倍。
        """
        per_side = self.params.slippage_ticks_per_side * tick_size(price)
        return 2.0 * per_side * shares

    def commission_yen(self, price: float, shares: int) -> float:
        """往復手数料（円）。SOR 無料前提なら既定 0。

        約定代金比例（往復＝エントリー＋イグジットの 2 約定）＋ 1約定固定 × 2。
        """
        notional = price * shares
        rate_part = self.params.commission_rate * notional * 2.0
        fixed_part = self.params.commission_per_trade_yen * 2.0
        return rate_part + fixed_part

    def financing_cost_yen(
        self, price: float, shares: int, holding_days: float, side: Side
    ) -> float:
        """信用コスト（円）。持ち越し日数に比例。当日返済（days=0）ならゼロ。

        long（信用買い）→ 金利、short（信用売り）→ 貸株料。
        """
        if holding_days <= 0:
            return 0.0
        notional = price * shares
        annual_rate = (
            self.params.margin_annual_rate
            if side == "long"
            else self.params.short_lending_annual_rate
        )
        return notional * annual_rate * holding_days / self.params.financing_days_per_year

    def market_impact_yen(self, price: float, shares: int, adv_shares: float | None) -> float:
        """マーケットインパクト（円）。出来高が分かるときのみ計上。

        平方根モデル：impact_fraction = coef × sqrt(shares / adv_shares)。
        エントリー・イグジットの両方で板を動かすため往復で 2 倍。
        adv_shares が None または 0 以下なら 0（情報がなければ計上しない）。
        """
        if adv_shares is None or adv_shares <= 0:
            return 0.0
        notional = price * shares
        participation = shares / adv_shares
        one_way_fraction = self.params.impact_coefficient * (participation**0.5)
        return notional * one_way_fraction * 2.0

    # --- 統合 --------------------------------------------------------------------
    def round_trip_cost(
        self,
        price: float,
        shares: int,
        *,
        holding_days: float = 0.0,
        side: Side = "long",
        adv_shares: float | None = None,
    ) -> CostBreakdown:
        """1往復トレードのコスト内訳を返す。

        Args:
            price: エントリー時の株価（円、>0）。見積もりは片側価格で近似する。
            shares: 株数（>0）。
            holding_days: 持ち越し日数（当日返済なら 0）。
            side: "long"（信用買い）/ "short"（信用売り）。
            adv_shares: 1日平均出来高（株）。None ならインパクト非計上。

        Returns:
            CostBreakdown（円ベース、total / total_fraction を持つ）。
        """
        if price <= 0:
            raise ValueError(f"price は正の値: {price}")
        if shares <= 0:
            raise ValueError(f"shares は正の値: {shares}")

        return CostBreakdown(
            spread=self.round_trip_spread_cost_yen(price, shares),
            slippage=self.round_trip_slippage_cost_yen(price, shares),
            commission=self.commission_yen(price, shares),
            financing=self.financing_cost_yen(price, shares, holding_days, side),
            impact=self.market_impact_yen(price, shares, adv_shares),
            notional=price * shares,
        )

    def round_trip_cost_fraction(
        self,
        price: float,
        shares: int,
        *,
        holding_days: float = 0.0,
        side: Side = "long",
        adv_shares: float | None = None,
    ) -> float:
        """往復コストを notional 比率で返す（期待値ゲート用の近道）。"""
        return self.round_trip_cost(
            price,
            shares,
            holding_days=holding_days,
            side=side,
            adv_shares=adv_shares,
        ).total_fraction
