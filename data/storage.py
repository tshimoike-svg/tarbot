"""SQLite による永続化（実測の積み上げの土台）。

docs/trading_bot_design_v2.md §10（data/storage.py）に対応。純フォワード方針では、
ライブで集めた分足・確定トレード・約定実測を日々ためていく必要がある。本モジュールは
その保存先（標準ライブラリ sqlite3 のみ、追加依存なし）。

- bars   : 分足 OHLCV（(symbol, ts) で一意）
- trades : 確定トレード（コスト控除後の損益込み）
- fills  : 約定実測（約定率・滑り。fill_monitor の結果）

時刻は ISO 文字列（JST 前提）で保存する。`:memory:` を渡せばテスト用のインメモリDB。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from execution.fill_monitor import FillResult
    from strategy.mean_reversion import Trade

__all__ = ["Storage"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL,
    ts     TEXT NOT NULL,
    open   REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (symbol, ts)
);
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    entry_time  TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_time   TEXT NOT NULL,
    exit_price  REAL NOT NULL,
    exit_reason TEXT NOT NULL,
    gross_return REAL,
    cost         REAL,
    net_return   REAL
);
CREATE TABLE IF NOT EXISTS fills (
    order_id          TEXT PRIMARY KEY,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL,
    limit_price       REAL NOT NULL,
    reference_price   REAL NOT NULL,
    shares            INTEGER NOT NULL,
    status            TEXT NOT NULL,
    fill_price        REAL,
    filled_shares     INTEGER NOT NULL,
    placed_at         TEXT NOT NULL,
    filled_at         TEXT,
    slippage_per_share REAL,
    slippage_ticks     REAL
);
"""


class Storage:
    """分足・トレード・約定実測の保存先（SQLite ラッパ）。

    コンテキストマネージャとして使える：
        with Storage("data/db/forward.sqlite") as db:
            db.insert_bars("1301", df)
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def __enter__(self) -> Storage:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # --- bars --------------------------------------------------------------------
    def insert_bars(self, symbol: str, df: pd.DataFrame) -> int:
        """分足 OHLCV を保存（(symbol, ts) 重複は無視）。挿入試行件数を返す。

        df は DatetimeIndex（ts）と列 open/high/low/close/volume を持つこと。
        """
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"bars に必須列が不足: {missing}")
        rows: list[tuple[Any, ...]] = []
        for ts, r in df.iterrows():
            ts_any: Any = ts
            rows.append(
                (
                    symbol,
                    pd.Timestamp(ts_any).isoformat(),
                    _f(r["open"]), _f(r["high"]), _f(r["low"]), _f(r["close"]), _f(r["volume"]),
                )
            )
        self._conn.executemany(
            "INSERT OR IGNORE INTO bars (symbol, ts, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def get_bars(self, symbol: str) -> pd.DataFrame:
        """銘柄の分足を時刻昇順で返す（ts を DatetimeIndex に）。"""
        cur = self._conn.execute(
            "SELECT ts, open, high, low, close, volume FROM bars "
            "WHERE symbol = ? ORDER BY ts",
            (symbol,),
        )
        df = pd.DataFrame(cur.fetchall(), columns=["ts", "open", "high", "low", "close", "volume"])
        if df.empty:
            return df.set_index(pd.DatetimeIndex([], name="ts")).drop(columns=["ts"])
        df["ts"] = pd.to_datetime(df["ts"])
        return df.set_index("ts")

    # --- trades ------------------------------------------------------------------
    def insert_trade(
        self,
        symbol: str,
        trade: Trade,
        *,
        gross_return: float | None = None,
        cost: float | None = None,
        net_return: float | None = None,
    ) -> None:
        """確定トレードを保存（コスト控除後の損益は任意で付与）。"""
        self._conn.execute(
            "INSERT INTO trades (symbol, side, entry_time, entry_price, exit_time, exit_price, "
            "exit_reason, gross_return, cost, net_return) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                symbol,
                trade.side,
                trade.entry_time.isoformat(),
                float(trade.entry_price),
                trade.exit_time.isoformat(),
                float(trade.exit_price),
                trade.exit_reason,
                gross_return,
                cost,
                net_return,
            ),
        )
        self._conn.commit()

    def get_trades(self, symbol: str | None = None) -> pd.DataFrame:
        if symbol is None:
            cur = self._conn.execute("SELECT * FROM trades ORDER BY id")
        else:
            cur = self._conn.execute("SELECT * FROM trades WHERE symbol = ? ORDER BY id", (symbol,))
        return pd.DataFrame([dict(r) for r in cur.fetchall()])

    # --- fills -------------------------------------------------------------------
    def insert_fill(self, result: FillResult) -> None:
        """約定実測（fill_monitor の結果）を保存。"""
        intent = result.intent
        self._conn.execute(
            "INSERT OR REPLACE INTO fills (order_id, symbol, side, limit_price, reference_price, "
            "shares, status, fill_price, filled_shares, placed_at, filled_at, "
            "slippage_per_share, slippage_ticks) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                intent.order_id,
                intent.symbol,
                intent.side,
                float(intent.limit_price),
                float(intent.reference_price),
                int(intent.shares),
                result.status,
                None if result.fill_price is None else float(result.fill_price),
                int(result.filled_shares),
                intent.placed_at.isoformat(),
                None if result.filled_at is None else result.filled_at.isoformat(),
                result.slippage_per_share,
                result.slippage_ticks,
            ),
        )
        self._conn.commit()

    def get_fills(self) -> pd.DataFrame:
        cur = self._conn.execute("SELECT * FROM fills ORDER BY placed_at")
        return pd.DataFrame([dict(r) for r in cur.fetchall()])


def _f(value: object) -> float | None:
    """NaN/None を SQL の NULL に落とすための変換。"""
    if value is None:
        return None
    f = float(value)  # type: ignore[arg-type]
    return None if pd.isna(f) else f
