"""swing_momentum.py のテスト（レジームフィルタ統合を中心に）。"""

from __future__ import annotations

import pandas as pd

from config.settings import SwingMomentumParams
from strategy.swing_momentum import compute_signals


def _breakout_df(n: int = 60) -> pd.DataFrame:
    """横ばい→終盤で急騰するダミー日足（ブレイクアウト・エントリーが発生する）。"""
    idx = pd.bdate_range("2024-01-02", periods=n)
    flat = [100.0] * (n - 5)
    breakout = [100.0 + (i + 1) * 3.0 for i in range(5)]  # 終盤5本で急騰
    closes = flat + breakout
    df = {
        "open": [c - 0.2 for c in closes],
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
    }
    return pd.DataFrame(df, index=idx, dtype="float64")


def _market_df(n: int, *, trending: bool) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=n)
    if trending:
        closes = [100.0 * (1.02**i) for i in range(n)]  # 一貫した強いトレンド
    else:
        closes = [100.0] * n  # 完全にレンジ（乖離ゼロ）
    return pd.DataFrame({"close": closes}, index=idx, dtype="float64")


def _params(**kw: object) -> SwingMomentumParams:
    base = dict(breakout_lookback=20, allow_long=True, allow_short=False)
    base.update(kw)
    return SwingMomentumParams(**base)  # type: ignore[arg-type]


# --- 既定はフィルタなし --------------------------------------------------------------
def test_no_regime_filter_by_default() -> None:
    df = _breakout_df()
    params = _params()  # enable_regime_filter は既定 False
    sig = compute_signals(df, params, market_df=_market_df(len(df), trending=False))
    assert (sig["entry"] == 1).any(), "デフォルトはフィルタなしでブレイクアウトが残るべき"


def test_regime_column_present_even_without_filter() -> None:
    df = _breakout_df()
    sig = compute_signals(df, _params())
    assert "regime" in sig.columns
    assert bool(sig["regime"].all())


# --- レジームフィルタ有効時 ----------------------------------------------------------
def test_regime_filter_blocks_entry_in_range_market() -> None:
    df = _breakout_df()
    params = _params(enable_regime_filter=True, regime_ma_window=10, regime_threshold=0.03, regime_filter_invert=True)
    sig = compute_signals(df, params, market_df=_market_df(len(df), trending=False))
    assert (sig["entry"] == 1).sum() == 0, "レンジ相場ではモメンタムのエントリーを抑制すべき"


def test_regime_filter_allows_entry_in_trending_market() -> None:
    df = _breakout_df()
    params = _params(enable_regime_filter=True, regime_ma_window=10, regime_threshold=0.03, regime_filter_invert=True)
    sig = compute_signals(df, params, market_df=_market_df(len(df), trending=True))
    assert (sig["entry"] == 1).any(), "トレンド相場ではブレイクアウトのエントリーを許可すべき"


def test_regime_filter_ignored_when_market_df_missing() -> None:
    df = _breakout_df()
    params = _params(enable_regime_filter=True)
    sig = compute_signals(df, params, market_df=None)
    assert (sig["entry"] == 1).any(), "market_df 未指定ならフィルタは適用されないべき"
