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
from datetime import time

__all__ = ["DRY_RUN", "MeanReversionParams", "DEFAULT_MEAN_REVERSION"]


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
class MeanReversionParams:
    """日中平均回帰戦略のパラメータ（トラックA・主力）。

    エントリー：VWAP からの乖離 z スコアが ±entry_z を超えたら逆張りの指値。
      - z <= -entry_z かつ allow_long → ロング（行き過ぎ下落の反発を狙う）
      - z >= +entry_z かつ allow_short → ショート（行き過ぎ上昇の反落を狙う）
    イグジット：
      - 利確：価格が VWAP に回帰したら（VWAP タッチ）
      - 損切り：エントリー時 ATR × atr_stop_mult を逆行したら
      - 強制：force_close 時刻以降は新規建てせず、既存はクローズ（当日手仕舞い）

    Attributes:
        zscore_length: (close − VWAP) のローリング z スコア期間。
        entry_z: エントリー閾値（σ）。大きいほど“行き過ぎ”を厳しく要求＝回数減・質向上。
        atr_length: ATR 期間（損切り幅の基礎）。
        atr_stop_mult: 損切り幅 = atr_stop_mult × ATR。固定%でなくボラ適応にする（v2 §6）。
        allow_long: ロング側エントリーを許可。
        allow_short: ショート側（信用売り）エントリーを許可。
        force_close: この時刻（JST）以降は新規建てせず手仕舞い。大引け前を想定。
        use_typical_price_for_vwap: VWAP の価格に典型価格(H+L+C)/3 を使う。False なら終値。
    """

    zscore_length: int = 20
    entry_z: float = 2.0
    atr_length: int = 14
    atr_stop_mult: float = 1.5
    allow_long: bool = True
    allow_short: bool = True
    force_close: time = time(14, 55)
    use_typical_price_for_vwap: bool = True

    def __post_init__(self) -> None:
        if self.zscore_length < 2:
            raise ValueError("zscore_length は 2 以上")
        if self.entry_z <= 0:
            raise ValueError("entry_z は正")
        if self.atr_length < 1:
            raise ValueError("atr_length は 1 以上")
        if self.atr_stop_mult <= 0:
            raise ValueError("atr_stop_mult は正")
        if not (self.allow_long or self.allow_short):
            raise ValueError("allow_long / allow_short の少なくとも一方は True")


# 既定パラメータ（バックテストの初期仮説。最適化結果で上書きする）。
DEFAULT_MEAN_REVERSION = MeanReversionParams()
