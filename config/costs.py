"""コスト想定値の集約（往復コスト見積もりの中心）。

設計方針（docs/trading_bot_design_v2.md §0, §5-3, §13）：
- 手数料が無料化された今、頻回トレードで効く真のコストは
  「実効スプレッド ＋ 滑り ＋ 信用コスト（持ち越し時）」。
- これらを **保守的に** 見積もらないと、回数を増やすほど負ける。
- スプレッドは銘柄の流動性で大きく異なる（中小型は広い）ため、
  流動性ティアごとのプリセットを用意する。

このモジュールは「値（パラメータ）」だけを持ち、計算ロジックは
`backtest/cost_model.py` に置く（設定とロジックの分離）。

⚠️ ここに置く数値は **暫定の保守的プレースホルダ**。
   実データ（J-Quants の分足・気配、実約定ログ）でキャリブレーションする
   までは「真値」ではない。Phase 0 の評価はこの保守値で行い、後で実測へ更新する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

__all__ = [
    "TICK_SIZE_TABLE",
    "tick_size",
    "CostParams",
    "CONSERVATIVE_DEFAULT",
    "PRESETS",
    "get_cost_params",
    "Side",
    "LiquidityTier",
]

Side = Literal["long", "short"]
LiquidityTier = Literal["large", "mid", "small"]


# --- 東証 呼値（ティックサイズ）テーブル：TOPIX100 構成銘柄“以外”の通常株式 -----------
# (価格の上限〔この値を含む〕, 呼値) の昇順リスト。price <= upper の最初の段の呼値を使う。
# 出典：JPX 呼値の単位。TOPIX100 構成銘柄は別表（より細かい）だが、本ボットの主対象は
# 中小型株なのでこの通常表を既定とする。大型を扱う場合は別途対応。
TICK_SIZE_TABLE: tuple[tuple[float, float], ...] = (
    (3_000, 1),
    (5_000, 5),
    (30_000, 10),
    (50_000, 50),
    (300_000, 100),
    (500_000, 500),
    (3_000_000, 1_000),
    (5_000_000, 5_000),
    (30_000_000, 10_000),
    (50_000_000, 50_000),
    (math.inf, 100_000),
)


def tick_size(price: float) -> float:
    """株価に対応する呼値（円）を返す（TOPIX100 以外の通常表）。

    Args:
        price: 株価（円）。正の値であること。

    Returns:
        呼値（円）。

    Raises:
        ValueError: price が 0 以下のとき。
    """
    if price <= 0:
        raise ValueError(f"price は正の値である必要があります: {price}")
    for upper, tick in TICK_SIZE_TABLE:
        if price <= upper:
            return float(tick)
    # TICK_SIZE_TABLE の末尾が inf なので理論上ここには到達しない
    raise AssertionError("呼値テーブルが価格をカバーしていません")


@dataclass(frozen=True)
class CostParams:
    """往復コスト計算に必要なパラメータ一式。

    すべて「保守的に大きめ」に置くのが原則（楽観バイアス対策）。
    金額系は円、率系は年率（financing）・notional比率（impact）で表す。

    Attributes:
        spread_ticks: 実効スプレッド（best ask − best bid）を呼値の何倍と見るか。
            流動性の高い銘柄で 1.0、中小型では数倍に広がる前提。
        spread_capture_ratio: 指値で half-spread を「捕る」割合 [0,1]（片側あたり）。
            0.0 = 捕れず full-spread を払う（最保守）。
            0.5 = スプレッドに関して収支ゼロ。
            1.0 = 両側で half-spread を獲得（楽観・往復でスプレッド分の利得）。
            Phase 0 の合否判定は既定 0.0（保守）で行うこと。
        slippage_ticks_per_side: 片道あたりの滑り（呼値の何倍か）。
            指値の不約定→成行転換や、想定価格とのズレを吸収する保守マージン。
        commission_rate: 約定代金に対する手数料率。SOR 注文で無料の想定なので既定 0.0。
            ただし「SOR を選ばないと手数料が出るか」は未確定（§4）なのでパラメータ化。
        commission_per_trade_yen: 1約定あたり固定手数料（円）。既定 0.0。
        margin_annual_rate: 信用買いの金利（年率）。持ち越し時のみ効く。
        short_lending_annual_rate: 信用売りの貸株料（年率）。持ち越し時のみ効く。
        financing_days_per_year: 金利計算の年日数（通常 365）。
        impact_coefficient: マーケットインパクト係数（平方根モデル）。
            impact_fraction = coef * sqrt(発注株数 / 1日出来高)。中小型で効く。
            出来高情報がない場合はインパクトを計上しない。
    """

    spread_ticks: float
    slippage_ticks_per_side: float
    spread_capture_ratio: float = 0.0
    commission_rate: float = 0.0
    commission_per_trade_yen: float = 0.0
    margin_annual_rate: float = 0.028
    short_lending_annual_rate: float = 0.011
    financing_days_per_year: int = 365
    impact_coefficient: float = 0.1

    def __post_init__(self) -> None:
        if self.spread_ticks < 0:
            raise ValueError("spread_ticks は 0 以上")
        if self.slippage_ticks_per_side < 0:
            raise ValueError("slippage_ticks_per_side は 0 以上")
        if not 0.0 <= self.spread_capture_ratio <= 1.0:
            raise ValueError("spread_capture_ratio は [0,1]")
        if self.commission_rate < 0:
            raise ValueError("commission_rate は 0 以上")
        if self.commission_per_trade_yen < 0:
            raise ValueError("commission_per_trade_yen は 0 以上")
        if self.margin_annual_rate < 0 or self.short_lending_annual_rate < 0:
            raise ValueError("金利・貸株料は 0 以上")
        if self.financing_days_per_year <= 0:
            raise ValueError("financing_days_per_year は正")
        if self.impact_coefficient < 0:
            raise ValueError("impact_coefficient は 0 以上")


# --- 既定（最保守）。流動性ティア未指定のときのフォールバック --------------------------
# spread_capture_ratio=0.0（スプレッドは全額払う前提）で Phase 0 の合否を判定する。
CONSERVATIVE_DEFAULT = CostParams(
    spread_ticks=2.0,
    slippage_ticks_per_side=1.0,
    spread_capture_ratio=0.0,
)


# --- 流動性ティア別プリセット（暫定・要キャリブレーション） ----------------------------
# 中小型ほどスプレッド・滑りを広く見積もる。金利等は共通（CONSERVATIVE_DEFAULT 由来）。
PRESETS: dict[LiquidityTier, CostParams] = {
    "large": replace(CONSERVATIVE_DEFAULT, spread_ticks=1.0, slippage_ticks_per_side=0.5),
    "mid": replace(CONSERVATIVE_DEFAULT, spread_ticks=2.0, slippage_ticks_per_side=1.0),
    "small": replace(CONSERVATIVE_DEFAULT, spread_ticks=4.0, slippage_ticks_per_side=1.5),
}


def get_cost_params(tier: LiquidityTier | None = None) -> CostParams:
    """流動性ティアに対応するコストパラメータを返す。

    Args:
        tier: "large" / "mid" / "small"。None なら最保守の既定を返す。
    """
    if tier is None:
        return CONSERVATIVE_DEFAULT
    return PRESETS[tier]
