"""yahoo_loader.py の正規化ロジックのテスト。

Phase 0 バックテストは J-Quants 調整後（分割調整・配当非調整）で実施したため、
ドライランの Yahoo ソースも methodology を一致させる必要がある。ここでは
ネットワークに依存しない純粋な正規化（列名・index・調整方式の取り違え防止）を固定する。
"""

from __future__ import annotations

import pandas as pd

from data.yahoo_loader import _normalize_yahoo, to_yahoo_ticker


def test_to_yahoo_ticker_adds_suffix() -> None:
    assert to_yahoo_ticker("6310") == "6310.T"
    assert to_yahoo_ticker("6310.T") == "6310.T"
    assert to_yahoo_ticker(" 7203 ") == "7203.T"


def _raw(multiindex: bool) -> pd.DataFrame:
    idx = pd.to_datetime(["2026-01-06", "2026-01-05"])  # わざと降順
    data = {
        "Open": [100.0, 99.0],
        "High": [101.0, 100.0],
        "Low": [98.0, 97.0],
        "Close": [100.5, 99.5],
        "Adj Close": [90.0, 89.0],  # 配当調整済み → 採用してはいけない
        "Volume": [1000, 1100],
    }
    df = pd.DataFrame(data, index=idx)
    if multiindex:
        df.columns = pd.MultiIndex.from_product([df.columns, ["6310.T"]])
    return df


def test_normalize_basic_columns_and_sort() -> None:
    out = _normalize_yahoo(_raw(multiindex=False))
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    # 昇順にソートされる
    assert list(out.index) == [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")]
    # Adj Close（配当調整）ではなく Close（分割調整のみ）を採用
    assert out.loc["2026-01-06", "close"] == 100.5


def test_normalize_handles_multiindex_columns() -> None:
    out = _normalize_yahoo(_raw(multiindex=True))
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.loc["2026-01-05", "close"] == 99.5


def test_normalize_index_is_normalized_naive() -> None:
    out = _normalize_yahoo(_raw(multiindex=False))
    assert out.index.tz is None
    assert (out.index == out.index.normalize()).all()


def test_normalize_empty_returns_empty_schema() -> None:
    out = _normalize_yahoo(pd.DataFrame())
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.empty


def test_normalize_missing_column_returns_empty() -> None:
    bad = pd.DataFrame({"Open": [1.0], "Close": [1.0]}, index=pd.to_datetime(["2026-01-05"]))
    out = _normalize_yahoo(bad)  # High/Low/Volume 欠損
    assert out.empty


def test_normalize_drops_nan_close() -> None:
    idx = pd.to_datetime(["2026-01-05", "2026-01-06"])
    raw = pd.DataFrame(
        {"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0],
         "Close": [1.0, float("nan")], "Volume": [10, 20]},
        index=idx,
    )
    out = _normalize_yahoo(raw)
    assert len(out) == 1
    assert out.index[0] == pd.Timestamp("2026-01-05")
