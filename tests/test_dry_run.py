"""dry_run.py のテスト（スイング戦略をイベント再生。発注なし）。"""

from __future__ import annotations

from functools import partial

import pandas as pd
import pytest

from backtest.cost_model import CostModel
from config.costs import CostParams
from config.settings import SwingReversionParams
from data.storage import Storage
from execution.dry_run import DryRunHarness
from strategy import swing_reversion


def _dip_series(date_start: str = "2026-01-05", reps: int = 1) -> pd.DataFrame:
    """下振れ→回復を reps 回つなげた日足（平均回帰のロングが出るよう設計）。"""
    block = [100, 100, 100, 100, 100, 95, 98, 100, 100, 100]
    closes = block * reps
    idx = pd.bdate_range(date_start, periods=len(closes))
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
        },
        index=idx, dtype="float64",
    )


_P = SwingReversionParams(lookback=5, entry_z=1.5, atr_length=3, atr_stop_mult=3.0, max_holding_days=5)
_FREE = CostModel(CostParams(spread_ticks=0.0, slippage_ticks_per_side=0.0))


def _gen(params: SwingReversionParams = _P):  # type: ignore[no-untyped-def]
    return partial(swing_reversion.generate_trades, params=params)


def _harness(**kw: object) -> DryRunHarness:
    base: dict[str, object] = {
        "account_equity": 1_000_000.0,
        "cost_model": _FREE,
        "generate": _gen(),
    }
    base.update(kw)
    return DryRunHarness(**base)  # type: ignore[arg-type]


# --- 結線 --------------------------------------------------------------------------
def test_dry_run_executes_a_trade() -> None:
    rep = _harness().run(_dip_series(), symbol="1301")
    assert rep.n_signals >= 1
    assert rep.n_executed >= 1
    assert rep.fill_stats is not None
    assert rep.fill_stats.fill_rate == pytest.approx(1.0)  # 疑似約定は全量
    assert rep.evaluation is not None


def test_dry_run_persists_to_storage() -> None:
    db = Storage(":memory:")
    rep = _harness(storage=db).run(_dip_series(), symbol="1301")
    assert len(db.get_trades()) == rep.n_executed
    assert len(db.get_fills()) == rep.n_executed * 2  # entry/exit
    db.close()


def test_dry_run_size_too_small_rejected() -> None:
    # 資金が小さく単元(100株)すら買えない → size_too_small で拒否
    rep = _harness(account_equity=1_000.0).run(_dip_series(), symbol="1301")
    assert rep.n_executed == 0
    assert any(reason == "size_too_small" for _, reason in rep.rejected)


def test_dry_run_cost_reduces_net_return() -> None:
    costly = CostModel(CostParams(spread_ticks=2.0, slippage_ticks_per_side=1.0))
    rep = _harness(cost_model=costly).run(_dip_series(), symbol="1301")
    assert rep.evaluation is not None
    if rep.n_executed > 0:
        assert rep.evaluation.expectancy < rep.evaluation.gross_expectancy


def test_dry_run_no_signals_is_empty() -> None:
    flat = _dip_series().copy()
    for col in ("open", "high", "low", "close"):
        flat[col] = 100.0
    rep = _harness().run(flat, symbol="1301")
    assert rep.n_executed == 0
    assert rep.evaluation is not None
    assert rep.evaluation.is_empty


def test_dry_run_multi_block_runs() -> None:
    rep = _harness().run(_dip_series(reps=3), symbol="1301")
    assert rep.n_executed >= 1
    # 出口時刻は建て時刻以降
    for tr in rep.executed_trades:
        assert tr.entry_time <= tr.exit_time
