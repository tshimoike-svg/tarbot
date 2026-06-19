"""mean_reversion.py / settings.py のテスト。

- 状態機械（_walk_session）の各分岐：ロング利確・損切り、ショート利確、強制クローズ、
  引け間際の新規建て抑止。
- generate_trades のエンドツーエンド（z スコアが実際に閾値を跨ぐよう設計した合成日足）。
- 因果性（compute_signals の truncation invariance）。
- 日跨ぎしない（VWAP リセットとトレードがセッション内に収まる）。
"""

from __future__ import annotations

from datetime import time

import pandas as pd
import pytest

from config.settings import MeanReversionParams
from strategy.mean_reversion import (
    Trade,
    _walk_session,
    compute_signals,
    generate_trades,
)


# --- settings: パラメータ検証 ------------------------------------------------------
def test_params_defaults() -> None:
    p = MeanReversionParams()
    assert p.entry_z == 2.0
    assert p.atr_stop_mult == 1.5
    assert p.force_close == time(14, 55)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"zscore_length": 1},
        {"entry_z": 0.0},
        {"atr_length": 0},
        {"atr_stop_mult": 0.0},
        {"allow_long": False, "allow_short": False},
    ],
)
def test_params_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        MeanReversionParams(**kwargs)


def test_dry_run_env_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    import config.settings as settings

    monkeypatch.delenv("DRY_RUN", raising=False)
    importlib.reload(settings)
    assert settings.DRY_RUN is True  # 既定は安全側

    monkeypatch.setenv("DRY_RUN", "false")
    importlib.reload(settings)
    assert settings.DRY_RUN is False

    monkeypatch.setenv("DRY_RUN", "true")
    importlib.reload(settings)
    assert settings.DRY_RUN is True
    # 後続テストへ副作用を残さないよう既定へ戻す
    monkeypatch.delenv("DRY_RUN", raising=False)
    importlib.reload(settings)


# --- _walk_session: 状態機械の分岐 -------------------------------------------------
def _session_frame(rows: list[dict], date: str = "2026-01-05") -> pd.DataFrame:
    idx = pd.to_datetime([f"{date} {r['t']}" for r in rows])
    return pd.DataFrame(
        {
            "high": [r["high"] for r in rows],
            "low": [r["low"] for r in rows],
            "close": [r["close"] for r in rows],
            "vwap": [r["vwap"] for r in rows],
            "zscore": [r["z"] for r in rows],
            "atr": [r["atr"] for r in rows],
        },
        index=idx,
        dtype="float64",
    )


_P = MeanReversionParams(
    zscore_length=2,
    entry_z=2.0,
    atr_length=2,
    atr_stop_mult=1.0,
    force_close=time(14, 55),
)


def test_walk_long_take_profit() -> None:
    g = _session_frame(
        [
            {"t": "09:00", "high": 100.5, "low": 99.5, "close": 100, "vwap": 100, "z": 0, "atr": 2},
            {"t": "09:01", "high": 97.5, "low": 96.5, "close": 97, "vwap": 100, "z": -2.5, "atr": 2},
            {"t": "09:02", "high": 100.5, "low": 99.5, "close": 100, "vwap": 99, "z": -1, "atr": 2},
            {"t": "09:03", "high": 101, "low": 100, "close": 100.5, "vwap": 99, "z": 0, "atr": 2},
        ]
    )
    trades = _walk_session(g, _P)
    assert len(trades) == 1
    tr = trades[0]
    assert tr.side == "long"
    assert tr.entry_price == pytest.approx(97.0)  # 09:01 終値で建て
    assert tr.exit_reason == "take_profit"
    assert tr.exit_price == pytest.approx(99.0)  # 09:02 の VWAP へ回帰
    assert tr.pnl_gross_per_share == pytest.approx(2.0)


def test_walk_sets_stop_price() -> None:
    g = _session_frame(
        [
            {"t": "09:00", "high": 100.5, "low": 99.5, "close": 100, "vwap": 100, "z": 0, "atr": 2},
            {"t": "09:01", "high": 97.5, "low": 96.5, "close": 97, "vwap": 100, "z": -2.5, "atr": 2},
            {"t": "09:02", "high": 100.5, "low": 99.5, "close": 100, "vwap": 99, "z": -1, "atr": 2},
            {"t": "09:03", "high": 101, "low": 100, "close": 100.5, "vwap": 99, "z": 0, "atr": 2},
        ]
    )
    [tr] = _walk_session(g, _P)
    # entry 97, atr 2, mult 1.0 → stop = 95
    assert tr.stop_price == pytest.approx(95.0)


def test_walk_long_stop() -> None:
    g = _session_frame(
        [
            {"t": "09:00", "high": 100.5, "low": 99.5, "close": 100, "vwap": 100, "z": 0, "atr": 2},
            {"t": "09:01", "high": 97.5, "low": 96.5, "close": 97, "vwap": 100, "z": -2.5, "atr": 2},
            {"t": "09:02", "high": 96, "low": 94, "close": 95, "vwap": 103, "z": -3, "atr": 2},
            {"t": "09:03", "high": 96, "low": 95, "close": 95.5, "vwap": 103, "z": -3, "atr": 2},
        ]
    )
    # entry 97, stop = 97 - 1.0*2 = 95。09:02 の low=94 <= 95 → 損切り（最終バーではない）
    trades = _walk_session(g, _P)
    assert len(trades) == 1
    assert trades[0].exit_reason == "stop"
    assert trades[0].exit_price == pytest.approx(95.0)
    assert trades[0].pnl_gross_per_share == pytest.approx(-2.0)


def test_walk_short_take_profit() -> None:
    g = _session_frame(
        [
            {"t": "09:00", "high": 100.5, "low": 99.5, "close": 100, "vwap": 100, "z": 0, "atr": 2},
            {"t": "09:01", "high": 103.5, "low": 102.5, "close": 103, "vwap": 100, "z": 2.5, "atr": 2},
            {"t": "09:02", "high": 101, "low": 98.5, "close": 99, "vwap": 100, "z": 1, "atr": 2},
            {"t": "09:03", "high": 100, "low": 99, "close": 99.5, "vwap": 100, "z": 0, "atr": 2},
        ]
    )
    # entry short 103, stop = 103 + 2 = 105。target = VWAP 100。09:02 low=98.5 <= 100 → 利確（最終バーではない）
    trades = _walk_session(g, _P)
    assert len(trades) == 1
    assert trades[0].side == "short"
    assert trades[0].exit_reason == "take_profit"
    assert trades[0].exit_price == pytest.approx(100.0)
    assert trades[0].pnl_gross_per_share == pytest.approx(3.0)


def test_walk_force_close() -> None:
    p = MeanReversionParams(zscore_length=2, entry_z=2.0, atr_length=2, force_close=time(9, 3))
    g = _session_frame(
        [
            {"t": "09:00", "high": 100.5, "low": 99.5, "close": 100, "vwap": 100, "z": 0, "atr": 2},
            {"t": "09:01", "high": 97.5, "low": 96.5, "close": 97, "vwap": 100, "z": -2.5, "atr": 2},
            {"t": "09:02", "high": 98, "low": 96.8, "close": 97.5, "vwap": 99.5, "z": -1, "atr": 2},
            {"t": "09:03", "high": 98, "low": 97.2, "close": 97.8, "vwap": 99, "z": 0, "atr": 2},
        ]
    )
    # 09:02 は stop(95)未達・target(99.5)未達で保有継続。09:03 は force_close 時刻 → 強制手仕舞い
    trades = _walk_session(g, p)
    assert len(trades) == 1
    assert trades[0].exit_reason == "force_close"
    assert trades[0].exit_price == pytest.approx(97.8)
    assert trades[0].exit_time.time() == time(9, 3)


def test_walk_no_entry_after_force_close() -> None:
    p = MeanReversionParams(zscore_length=2, entry_z=2.0, atr_length=2, force_close=time(9, 1))
    g = _session_frame(
        [
            {"t": "09:00", "high": 100.5, "low": 99.5, "close": 100, "vwap": 100, "z": 0, "atr": 2},
            {"t": "09:01", "high": 97.5, "low": 96.5, "close": 97, "vwap": 100, "z": -2.5, "atr": 2},
            {"t": "09:02", "high": 97.5, "low": 96.5, "close": 97, "vwap": 100, "z": -2.5, "atr": 2},
        ]
    )
    # 09:01 以降は引け間際扱いで新規建てしない
    assert _walk_session(g, p) == []


# --- generate_trades: エンドツーエンド（実指標で z が閾値を跨ぐ設計） ----------------
def _dip_day(date: str) -> pd.DataFrame:
    """終値が一時的に下振れして VWAP へ戻る 1 日分（z が −2 に達するよう設計）。"""
    closes = [100, 100, 100, 100, 100, 100, 97, 98, 99, 100, 100, 100]
    times = [f"09:{m:02d}" for m in range(len(closes))]
    idx = pd.to_datetime([f"{date} {t}" for t in times])
    return pd.DataFrame(
        {
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [100] * len(closes),
        },
        index=idx,
        dtype="float64",
    )


_E2E = MeanReversionParams(
    zscore_length=5,
    entry_z=1.5,
    atr_length=3,
    atr_stop_mult=1.5,
    use_typical_price_for_vwap=False,
    force_close=time(14, 55),
)


def test_compute_signals_fixture_crosses_threshold() -> None:
    df = _dip_day("2026-01-05")
    sig = compute_signals(df, _E2E)
    assert sig["zscore"].min() <= -1.5  # フィクスチャの妥当性（エントリーが発火する）
    assert set(sig.columns) == {"vwap", "zscore", "atr"}


def test_generate_trades_long_on_dip() -> None:
    df = _dip_day("2026-01-05")
    trades = generate_trades(df, _E2E)
    assert len(trades) >= 1
    first = trades[0]
    assert isinstance(first, Trade)
    assert first.side == "long"
    assert first.entry_price == pytest.approx(97.0)  # 下振れバーの終値で建て
    assert first.exit_reason in {"take_profit", "force_close"}


def test_generate_trades_flag_gates_direction() -> None:
    df = _dip_day("2026-01-05")
    # 下振れは long シグナル。long を禁じ short のみ許可 → 上振れが無いので 0 トレード
    params = MeanReversionParams(
        zscore_length=5,
        entry_z=1.5,
        atr_length=3,
        allow_long=False,
        allow_short=True,
        use_typical_price_for_vwap=False,
    )
    assert generate_trades(df, params) == []


def test_generate_trades_no_overnight_and_deterministic() -> None:
    df = pd.concat([_dip_day("2026-01-05"), _dip_day("2026-01-06")])
    trades = generate_trades(df, _E2E)
    assert len(trades) >= 2  # 各日少なくとも1つ
    for tr in trades:
        assert tr.entry_time <= tr.exit_time
        assert tr.entry_time.date() == tr.exit_time.date()  # 日跨ぎ無し
        assert tr.exit_reason in {"take_profit", "stop", "force_close"}
    # 決定論的
    assert [t.exit_price for t in generate_trades(df, _E2E)] == [t.exit_price for t in trades]


def test_generate_trades_requires_sorted_index() -> None:
    df = _dip_day("2026-01-05").iloc[::-1]  # 降順
    with pytest.raises(ValueError):
        generate_trades(df, _E2E)


def test_compute_signals_is_causal() -> None:
    df = _dip_day("2026-01-05")
    full = compute_signals(df, _E2E)
    for t in range(_E2E.zscore_length, len(df)):
        sub = compute_signals(df.iloc[: t + 1], _E2E)
        for col in ("vwap", "zscore", "atr"):
            a, b = sub.iloc[t][col], full.iloc[t][col]
            if pd.isna(a) and pd.isna(b):
                continue
            assert a == pytest.approx(b), f"{col} が t={t} で未来依存"
