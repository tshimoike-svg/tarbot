"""strategy/swing_cross_section.py のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import SwingCrossSectionParams
from strategy.swing_cross_section import compute_cross_signals, generate_trades


# ---- ヘルパー ---------------------------------------------------------------

def _make_df(n: int = 100, seed: int = 0, drift: float = 0.0) -> pd.DataFrame:
    """シード固定のランダム日足OHLCV。drift > 0 で上昇トレンド。"""
    rng = np.random.default_rng(seed)
    close = 1000.0 * np.cumprod(1 + rng.normal(drift, 0.02, n))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1_000_000},
        index=index,
    )


def _make_universe(n_symbols: int = 20, n_bars: int = 100) -> dict[str, pd.DataFrame]:
    return {str(i): _make_df(n_bars, seed=i) for i in range(n_symbols)}


# ---- compute_cross_signals --------------------------------------------------

class TestComputeCrossSignals:
    def test_returns_dict_with_all_symbols(self) -> None:
        universe = _make_universe(15)
        params = SwingCrossSectionParams(return_lookback=10, entry_z=1.5, min_universe_size=5)
        signals = compute_cross_signals(universe, params)
        assert set(signals) == set(universe)

    def test_signal_values_are_minus1_0_plus1(self) -> None:
        universe = _make_universe(15)
        params = SwingCrossSectionParams(return_lookback=10, entry_z=1.5, min_universe_size=5)
        signals = compute_cross_signals(universe, params)
        for sym, s in signals.items():
            assert s.isin([-1, 0, 1]).all(), f"{sym} に -1/0/1 以外の値"

    def test_below_min_universe_size_returns_empty(self) -> None:
        universe = _make_universe(3)
        params = SwingCrossSectionParams(min_universe_size=10)
        signals = compute_cross_signals(universe, params)
        assert signals == {}

    def test_no_lookahead_shift1(self) -> None:
        """シグナルを1日早めても変わらないこと（shift(1)で先読みなし）を確認。"""
        universe = _make_universe(15, n_bars=80)
        params = SwingCrossSectionParams(return_lookback=10, entry_z=1.5, min_universe_size=5)
        signals_full = compute_cross_signals(universe, params)

        # 全銘柄を1日分だけ切り捨てた部分データで再計算
        universe_short = {sym: df.iloc[:-1] for sym, df in universe.items()}
        signals_short = compute_cross_signals(universe_short, params)

        for sym in universe:
            # 最終行を除いた部分は一致するはず（先読みがあれば変わる）
            common_idx = signals_short[sym].index
            pd.testing.assert_series_equal(
                signals_full[sym].reindex(common_idx),
                signals_short[sym],
                check_names=False,
            )

    def test_allow_long_false_no_positive_signals(self) -> None:
        universe = _make_universe(15)
        params = SwingCrossSectionParams(return_lookback=10, entry_z=1.5,
                                         allow_long=False, min_universe_size=5)
        signals = compute_cross_signals(universe, params)
        for s in signals.values():
            assert (s <= 0).all()

    def test_allow_short_false_no_negative_signals(self) -> None:
        universe = _make_universe(15)
        params = SwingCrossSectionParams(return_lookback=10, entry_z=1.5,
                                         allow_short=False, min_universe_size=5)
        signals = compute_cross_signals(universe, params)
        for s in signals.values():
            assert (s >= 0).all()

    def test_missing_required_column_skipped(self) -> None:
        universe = _make_universe(15)
        # 1銘柄の close 列を削除
        bad_sym = "0"
        universe[bad_sym] = universe[bad_sym].drop(columns=["close"])
        params = SwingCrossSectionParams(return_lookback=10, entry_z=1.5, min_universe_size=5)
        signals = compute_cross_signals(universe, params)
        assert bad_sym not in signals

    def test_high_entry_z_fewer_signals(self) -> None:
        universe = _make_universe(20)
        low_z = SwingCrossSectionParams(return_lookback=10, entry_z=0.5, min_universe_size=5)
        high_z = SwingCrossSectionParams(return_lookback=10, entry_z=3.0, min_universe_size=5)
        sig_low = compute_cross_signals(universe, low_z)
        sig_high = compute_cross_signals(universe, high_z)
        n_low = sum((s != 0).sum() for s in sig_low.values())
        n_high = sum((s != 0).sum() for s in sig_high.values())
        assert n_low > n_high, "閾値が低いほうがシグナルが多いはず"


# ---- generate_trades --------------------------------------------------------

class TestGenerateTrades:
    def test_returns_list(self) -> None:
        universe = _make_universe(15)
        params = SwingCrossSectionParams(return_lookback=10, entry_z=1.0, min_universe_size=5)
        trades = generate_trades(universe, params)
        assert isinstance(trades, list)

    def test_trades_have_required_fields(self) -> None:
        universe = _make_universe(15)
        params = SwingCrossSectionParams(return_lookback=10, entry_z=1.0, min_universe_size=5)
        trades = generate_trades(universe, params)
        if not trades:
            pytest.skip("トレードなし（パラメータ・データ長次第）")
        t = trades[0]
        assert hasattr(t, "entry_time")
        assert hasattr(t, "entry_price")
        assert hasattr(t, "stop_price")
        assert hasattr(t, "side")

    def test_empty_universe_returns_empty(self) -> None:
        trades = generate_trades({})
        assert trades == []

    def test_below_min_universe_returns_empty(self) -> None:
        universe = _make_universe(3)
        params = SwingCrossSectionParams(min_universe_size=10)
        trades = generate_trades(universe, params)
        assert trades == []

    def test_all_trades_after_lookback(self) -> None:
        universe = _make_universe(15, n_bars=60)
        params = SwingCrossSectionParams(return_lookback=20, entry_z=1.0, min_universe_size=5)
        trades = generate_trades(universe, params)
        # エントリーはlookback+1日目以降のみ（最初の20日はNaN）
        start = pd.Timestamp("2024-01-01") + pd.tseries.offsets.BusinessDay(20)
        for t in trades:
            assert t.entry_time >= start, f"ルックアヘッド疑い: {t.entry_time} < {start}"
