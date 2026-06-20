"""Phase 1 フォワードシグナルを SQLite に永続化するストア。

signal_date T のシグナル → entry_date T+1 の始値で建て → 出口条件まで追跡。
エントリー価格は T の終値で近似（翌日始値は不明のため）。
実際のギャップ分は保守的バイアスとして許容する（CLAUDE.md 絶対原則1: 発注なし）。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

__all__ = ["SignalStore", "ForwardSignal"]

ForwardSignalStatus = Literal["open", "closed"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS forward_signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT NOT NULL,
    signal_date   TEXT NOT NULL,
    side          TEXT NOT NULL DEFAULT 'long',
    entry_date    TEXT NOT NULL,
    entry_price   REAL NOT NULL,
    stop_price    REAL NOT NULL,
    target_price  REAL NOT NULL,
    max_exit_date TEXT NOT NULL,
    config_name   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'open',
    exit_date     TEXT,
    exit_price    REAL,
    exit_reason   TEXT,
    gross_return  REAL,
    net_return    REAL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_uniq ON forward_signals(symbol, signal_date, config_name);
CREATE INDEX IF NOT EXISTS idx_status ON forward_signals(status);
"""


@dataclass
class ForwardSignal:
    symbol: str
    signal_date: str          # T の日付 YYYY-MM-DD
    side: str                 # "long" / "short"
    entry_date: str           # T+1 の日付 YYYY-MM-DD
    entry_price: float        # T 終値近似（翌始値未知のため）
    stop_price: float         # entry_price - ATR × mult
    target_price: float       # T 時点の MA（利確目標）
    max_exit_date: str        # entry_date + max_holding_days YYYY-MM-DD
    config_name: str          # "config_iii" / "config_iv" / "config_v"


class SignalStore:
    """フォワードシグナルの CRUD（コンテキストマネージャ対応）。"""

    def __init__(self, path: str | Path = "data/db/forward_signals.sqlite") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def __enter__(self) -> SignalStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def insert_signal(self, sig: ForwardSignal) -> int:
        """シグナルを挿入する（重複は無視）。挿入行の id を返す。"""
        cur = self._conn.execute(
            """INSERT OR IGNORE INTO forward_signals
            (symbol, signal_date, side, entry_date, entry_price, stop_price,
             target_price, max_exit_date, config_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sig.symbol, sig.signal_date, sig.side, sig.entry_date,
                float(sig.entry_price), float(sig.stop_price),
                float(sig.target_price), sig.max_exit_date, sig.config_name,
            ),
        )
        self._conn.commit()
        return cur.lastrowid or 0

    def signal_exists(self, symbol: str, signal_date: str, config_name: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM forward_signals WHERE symbol=? AND signal_date=? AND config_name=?",
            (symbol, signal_date, config_name),
        )
        return cur.fetchone() is not None

    def get_open_signals(self) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM forward_signals WHERE status='open' ORDER BY signal_date, symbol"
        )
        return [dict(r) for r in cur.fetchall()]

    def close_signal(
        self,
        id_: int,
        *,
        exit_date: str,
        exit_price: float,
        exit_reason: str,
        gross_return: float,
        net_return: float,
    ) -> None:
        self._conn.execute(
            """UPDATE forward_signals SET
            status='closed', exit_date=?, exit_price=?, exit_reason=?,
            gross_return=?, net_return=? WHERE id=?""",
            (exit_date, float(exit_price), exit_reason,
             float(gross_return), float(net_return), id_),
        )
        self._conn.commit()

    def get_all(self) -> pd.DataFrame:
        cur = self._conn.execute("SELECT * FROM forward_signals ORDER BY id")
        return pd.DataFrame([dict(r) for r in cur.fetchall()])

    def signals_in_month(self, year_month: str) -> int:
        """YYYY-MM の月内シグナル件数（open / closed 合算）。"""
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM forward_signals WHERE signal_date LIKE ?",
            (f"{year_month}%",),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def summary(self) -> str:
        """オープン/クローズ件数と期待値を1行サマリで返す。"""
        df = self.get_all()
        if df.empty:
            return "シグナルなし"
        open_n = (df["status"] == "open").sum()
        closed = df[df["status"] == "closed"]
        if closed.empty or closed["net_return"].isna().all():
            return f"オープン {open_n} 件 / クローズ 0 件"
        avg = closed["net_return"].mean()
        wins = (closed["net_return"] > 0).sum()
        n = len(closed)
        return (
            f"オープン {open_n} 件 / クローズ {n} 件  "
            f"期待値 {avg*100:+.2f}%  勝率 {wins}/{n}={wins/n:.0%}"
        )
