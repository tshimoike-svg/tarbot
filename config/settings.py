"""売買パラメータと実行フラグ（ハードコード禁止の集約先）。

CLAUDE.md 絶対原則：
- `DRY_RUN` は発注前に必ず確認する。**既定 True**。False へ切り替えるコードは書かない
  （本番切替は人間が環境変数等で明示的に行う前提）。
- しきい値・期間・倍率はここに集約し、戦略コードにハードコードしない。

このモジュールは「戦略パラメータ」を持つ。コスト想定は `config/costs.py`、
銘柄・流動性フィルタは `config/symbols.py`（未作成）に分ける。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = [
    "DRY_RUN",
    "SwingReversionParams",
    "DEFAULT_SWING_REVERSION",
    "SwingMomentumParams",
    "DEFAULT_SWING_MOMENTUM",
    "SwingCrossSectionParams",
    "DEFAULT_CROSS_SECTION",
    "RiskParams",
    "DEFAULT_RISK",
]


def _env_dry_run_default() -> bool:
    """DRY_RUN の既定値。環境変数 DRY_RUN=false のときのみ False。

    既定は安全側（True）。コード側で False を直書きしない（絶対原則1）。
    本番運用者が環境変数で明示的に解除する経路だけを用意する。
    """
    raw = os.getenv("DRY_RUN")
    if raw is None:
        return True
    return raw.strip().lower() not in {"false", "0", "no", "off"}


# 発注前に必ず参照するフラグ。既定 True（ドライラン）。
DRY_RUN: bool = _env_dry_run_default()


@dataclass(frozen=True)
class SwingReversionParams:
    """日足スイング平均回帰（押し目買い／戻り売り）のパラメータ。

    エントリー：終値の lookback 日 z スコアが ±entry_z を超えたら逆張り
      （z<=-entry_z かつ allow_long でロング、z>=+entry_z かつ allow_short でショート）。
    イグジット：移動平均（lookback）への回帰で利確／ATR×倍率で損切り／
      max_holding_days 到達でタイムストップ。

    Attributes:
        lookback: 移動平均・z スコアの期間（日）。
        entry_z: エントリー閾値（σ）。
        atr_length: ATR 期間（日）。
        atr_stop_mult: 損切り幅 = atr_stop_mult × ATR（オーバーナイト前提で日中より広め）。
        max_holding_days: 最大保有日数（タイムストップ）。
        allow_long / allow_short: 売買方向の許可。
    """

    lookback: int = 20
    entry_z: float = 2.0
    atr_length: int = 14
    atr_stop_mult: float = 2.0
    max_holding_days: int = 10
    allow_long: bool = True
    allow_short: bool = True
    # 市場レジームフィルタ：市場全体が特定レジームのときだけエントリーを許可
    enable_regime_filter: bool = False
    regime_ma_window: int = 60      # 市場指数の MA 期間
    regime_threshold: float = 0.03  # ±X% 以内をレンジと判定
    regime_filter_invert: bool = False  # True: トレンド期のみ入る（レンジ期を除外）
    # 季節フィルタ（デカンショ節効果）
    season_avoid_months: frozenset[int] = frozenset()  # 空 = 全月エントリー可
    # RSI フィルタ（過熱感確認）
    rsi_length: int = 14
    rsi_entry_max: float = 100.0  # ロング: RSI < この値でのみ入る（100 = 無効）
    rsi_entry_min: float = 0.0    # ショート: RSI > この値でのみ入る（0 = 無効）
    # 出来高フィルタ（投げ売り / 買われ過ぎ確認）
    volume_ratio_min: float = 0.0  # volume / vol_MA > この値でのみ入る（0 = 無効）
    volume_ma_length: int = 20

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback は 2 以上")
        if self.entry_z <= 0:
            raise ValueError("entry_z は正")
        if self.atr_length < 1:
            raise ValueError("atr_length は 1 以上")
        if self.atr_stop_mult <= 0:
            raise ValueError("atr_stop_mult は正")
        if self.max_holding_days < 1:
            raise ValueError("max_holding_days は 1 以上")
        if not (self.allow_long or self.allow_short):
            raise ValueError("allow_long / allow_short の少なくとも一方は True")
        for m in self.season_avoid_months:
            if not (1 <= m <= 12):
                raise ValueError(f"season_avoid_months の月は 1〜12: {m}")
        if not (0.0 <= self.rsi_entry_max <= 100.0):
            raise ValueError("rsi_entry_max は 0〜100")
        if not (0.0 <= self.rsi_entry_min <= 100.0):
            raise ValueError("rsi_entry_min は 0〜100")


DEFAULT_SWING_REVERSION = SwingReversionParams()


@dataclass(frozen=True)
class SwingMomentumParams:
    """日足スイング・モメンタム（ブレイクアウト）のパラメータ。

    エントリー：終値が直近 breakout_lookback 日の高値を上抜けでロング／安値を
      下抜けでショート（前日までの窓の極値と比較＝未来非参照）。
    イグジット：固定目標は置かず、ATR×倍率の損切り／max_holding_days のタイムストップ。

    Attributes:
        breakout_lookback: ブレイク判定の窓（日）。
        atr_length: ATR 期間（日）。
        atr_stop_mult: 損切り幅 = atr_stop_mult × ATR。
        max_holding_days: 最大保有日数（タイムストップ）。
        allow_long / allow_short: 売買方向の許可。
    """

    breakout_lookback: int = 20
    atr_length: int = 14
    atr_stop_mult: float = 2.0
    max_holding_days: int = 10
    allow_long: bool = True
    allow_short: bool = True

    def __post_init__(self) -> None:
        if self.breakout_lookback < 2:
            raise ValueError("breakout_lookback は 2 以上")
        if self.atr_length < 1:
            raise ValueError("atr_length は 1 以上")
        if self.atr_stop_mult <= 0:
            raise ValueError("atr_stop_mult は正")
        if self.max_holding_days < 1:
            raise ValueError("max_holding_days は 1 以上")
        if not (self.allow_long or self.allow_short):
            raise ValueError("allow_long / allow_short の少なくとも一方は True")


DEFAULT_SWING_MOMENTUM = SwingMomentumParams()


@dataclass(frozen=True)
class SwingCrossSectionParams:
    """日足クロスセクション平均回帰のパラメータ。

    各日にユニバース全銘柄のN日リターンをクロスセクションでz化し、
    ピア対比で大きく劣後した銘柄をロング・大きく優位した銘柄をショート。
    市場全体のβが消えるのでトレンド方向に依存しない（市場中立）。

    Attributes:
        return_lookback: クロスセクションランキングに使うN日リターンの窓（日）。
        entry_z: エントリー閾値（σ）。cs_z <= -entry_z でロング、>= +entry_z でショート。
        atr_length: ATR 期間（日）。
        atr_stop_mult: 損切り幅 = atr_stop_mult × ATR。
        max_holding_days: 最大保有日数（タイムストップ）。
        allow_long / allow_short: 売買方向の許可。
        min_universe_size: クロスセクション統計が信頼できる最低銘柄数。
    """

    return_lookback: int = 20
    entry_z: float = 1.5
    atr_length: int = 14
    atr_stop_mult: float = 2.0
    max_holding_days: int = 10
    allow_long: bool = True
    allow_short: bool = True
    min_universe_size: int = 10

    def __post_init__(self) -> None:
        if self.return_lookback < 2:
            raise ValueError("return_lookback は 2 以上")
        if self.entry_z <= 0:
            raise ValueError("entry_z は正")
        if self.atr_length < 1:
            raise ValueError("atr_length は 1 以上")
        if self.atr_stop_mult <= 0:
            raise ValueError("atr_stop_mult は正")
        if self.max_holding_days < 1:
            raise ValueError("max_holding_days は 1 以上")
        if not (self.allow_long or self.allow_short):
            raise ValueError("allow_long / allow_short の少なくとも一方は True")
        if self.min_universe_size < 2:
            raise ValueError("min_universe_size は 2 以上")


DEFAULT_CROSS_SECTION = SwingCrossSectionParams()


@dataclass(frozen=True)
class RiskParams:
    """リスク管理ルール（v2 §6 の再設計テーブル）。

    全発注は `strategy/risk_manager.py` の関門でこれらをチェックされる（絶対原則2）。
    比率はすべて口座資金に対する割合。

    Attributes:
        max_risk_per_trade: 1トレードで失ってよい最大額の比率（既定 0.5%）。
            固定%損切りではなく「1トレードで失う資金額」を固定する思想。
        max_symbol_ratio: 1銘柄あたり最大投資比率（既定 20%）。集中リスク制限。
        max_daily_loss: 1日の最大損失（サーキットブレーカー、既定 3%）。
            到達したら当日の新規建てを全停止する。
        max_trades_per_day: 1日の最大トレード回数（既定 30）。過剰トレード・暴走防止。
        max_positions: 同時保有最大銘柄数（既定 5）。
    """

    max_risk_per_trade: float = 0.005
    max_symbol_ratio: float = 0.20
    max_daily_loss: float = 0.03
    max_trades_per_day: int = 30
    max_positions: int = 5

    def __post_init__(self) -> None:
        for name in ("max_risk_per_trade", "max_symbol_ratio", "max_daily_loss"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} は (0, 1] の比率: {value}")
        if self.max_trades_per_day < 1:
            raise ValueError("max_trades_per_day は 1 以上")
        if self.max_positions < 1:
            raise ValueError("max_positions は 1 以上")


# 既定のリスク設定（初期案。実運用前に必ず自分で検証・調整すること）。
DEFAULT_RISK = RiskParams()
