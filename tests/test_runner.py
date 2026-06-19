"""runner.py のテスト（オフラインのダミー日足を注入。ネットワーク・認証なし）。"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.cost_model import CostModel
from backtest.runner import (
    BacktestResult,
    compare_strategies,
    format_result,
    run_strategy,
)
from config.costs import CostParams


def _series() -> pd.DataFrame:
    # 25日フラット → 急落 → 回復 → 上昇トレンド（平均回帰とブレイクアウトの両方を誘発）
    closes = (
        [100.0] * 25
        + [90.0, 92.0, 94.0, 96.0, 98.0, 100.0]
        + [float(x) for x in range(101, 121)]
    )
    idx = pd.bdate_range("2026-01-01", periods=len(closes))
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
        },
        index=idx, dtype="float64",
    )


_DATA = {"AAA": _series(), "BBB": _series()}
_FREE = CostModel(CostParams(spread_ticks=0.0, slippage_ticks_per_side=0.0))


def _load(symbol: str) -> pd.DataFrame:
    return _DATA.get(symbol, pd.DataFrame())


def test_run_strategy_reversion() -> None:
    res = run_strategy(
        ["AAA", "BBB"], _load, strategy="swing_reversion", cost_model=_FREE,
        n_splits=2, min_trades=2,
    )
    assert isinstance(res, BacktestResult)
    assert res.strategy == "swing_reversion"
    assert res.n_symbols == 2
    assert res.evaluation.n_trades >= 2  # 2銘柄とも急落で反発エントリー
    assert isinstance(res.gate.passed, bool)
    assert "swing_reversion" in format_result(res)


def test_run_strategy_momentum() -> None:
    res = run_strategy(
        ["AAA", "BBB"], _load, strategy="swing_momentum", cost_model=_FREE,
        n_splits=2, min_trades=2,
    )
    assert res.evaluation.n_trades >= 1  # 上昇トレンドでブレイク


def test_compare_returns_both() -> None:
    out = compare_strategies(["AAA", "BBB"], _load, cost_model=_FREE, n_splits=2, min_trades=2)
    assert set(out) == {"swing_reversion", "swing_momentum"}
    assert all(isinstance(v, BacktestResult) for v in out.values())


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError):
        run_strategy(["AAA"], _load, strategy="nope", cost_model=_FREE)


def test_empty_symbol_skipped() -> None:
    res = run_strategy(["ZZZ"], _load, strategy="swing_reversion", cost_model=_FREE, min_trades=1)
    assert res.evaluation.is_empty
    assert "トレードなし" in format_result(res)


def test_gate_fails_on_too_few_trades() -> None:
    res = run_strategy(
        ["AAA"], _load, strategy="swing_reversion", cost_model=_FREE, min_trades=10_000
    )
    assert not res.gate.enough_trades
    assert not res.gate.passed
