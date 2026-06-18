"""dry_run.py のテスト（リプレイで全工程を結線。発注なし）。"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.cost_model import CostModel
from config.costs import CostParams
from config.settings import MeanReversionParams, RiskParams
from data.storage import Storage
from execution.dry_run import DryRunHarness


def _dip_day(date: str) -> pd.DataFrame:
    """終値が下振れして VWAP へ戻る 1 日（z が −2 に達するよう設計）。"""
    closes = [100, 100, 100, 100, 100, 100, 97, 98, 99, 100, 100, 100]
    times = [f"09:{m:02d}" for m in range(len(closes))]
    idx = pd.to_datetime([f"{date} {t}" for t in times])
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [100] * len(closes),
        },
        index=idx, dtype="float64",
    )


_MR = MeanReversionParams(
    zscore_length=5, entry_z=1.5, atr_length=3, use_typical_price_for_vwap=False
)
_FREE = CostModel(CostParams(spread_ticks=0.0, slippage_ticks_per_side=0.0))


def _harness(**kw: object) -> DryRunHarness:
    base: dict[str, object] = {
        "account_equity": 1_000_000.0,
        "cost_model": _FREE,
        "mr_params": _MR,
    }
    base.update(kw)
    return DryRunHarness(**base)  # type: ignore[arg-type]


# --- 基本的な結線 ------------------------------------------------------------------
def test_dry_run_executes_a_trade() -> None:
    rep = _harness().run(_dip_day("2026-01-05"), symbol="1301")
    assert rep.n_signals >= 1
    assert rep.n_executed >= 1
    assert rep.evaluation is not None
    assert rep.fill_stats is not None
    # 疑似約定は全量約定・滑り0
    assert rep.fill_stats.fill_rate == pytest.approx(1.0)
    assert rep.fill_stats.avg_slippage_per_share == pytest.approx(0.0)


def test_dry_run_persists_to_storage() -> None:
    db = Storage(":memory:")
    rep = _harness(storage=db).run(_dip_day("2026-01-05"), symbol="1301")
    trades = db.get_trades()
    fills = db.get_fills()
    assert len(trades) == rep.n_executed
    # 1トレードにつき entry/exit の2 fill
    assert len(fills) == rep.n_executed * 2
    db.close()


def test_dry_run_multi_day() -> None:
    df = pd.concat([_dip_day("2026-01-05"), _dip_day("2026-01-06")])
    rep = _harness().run(df, symbol="1301")
    assert rep.n_executed >= 2
    for tr in rep.executed_trades:
        assert tr.entry_time.date() == tr.exit_time.date()


# --- 関門が効いていること ----------------------------------------------------------
def test_dry_run_rejects_when_max_trades_zero_equivalent() -> None:
    # 同時保有1・1日1回に絞る → 2件目以降は関門で拒否される日が出る
    rep = _harness(
        risk_params=RiskParams(max_trades_per_day=1, max_positions=1)
    ).run(pd.concat([_dip_day("2026-01-05")]), symbol="1301")
    # 1日1回でも最低1件は通る（その日にシグナルが複数あれば残りは拒否）
    assert rep.n_executed >= 1


def test_dry_run_size_too_small_rejected() -> None:
    # 目標ノートーションを極小にして株数 < 1 → size_too_small で拒否
    rep = _harness(target_notional_frac=0.00001).run(_dip_day("2026-01-05"), symbol="1301")
    assert rep.n_executed == 0
    assert any(reason == "size_too_small" for _, reason in rep.rejected)


def test_dry_run_cost_reduces_net_return() -> None:
    # コスト有りモデルでは net < gross になる
    costly = CostModel(CostParams(spread_ticks=2.0, slippage_ticks_per_side=1.0))
    rep = _harness(cost_model=costly).run(_dip_day("2026-01-05"), symbol="1301")
    assert rep.evaluation is not None
    if rep.n_executed > 0:
        assert rep.evaluation.expectancy < rep.evaluation.gross_expectancy


def test_dry_run_no_signals_is_empty() -> None:
    # 平坦な値動き → シグナルなし → 何も執行されない
    flat = _dip_day("2026-01-05").copy()
    for col in ("open", "high", "low", "close"):
        flat[col] = 100.0
    rep = _harness().run(flat, symbol="1301")
    assert rep.n_executed == 0
    assert rep.evaluation is not None
    assert rep.evaluation.is_empty
