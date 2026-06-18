"""kabu ステーション WebSocket 受信と分足集約（ライブ録り溜め）。

docs/trading_bot_design_v2.md §2, §10 / 純フォワード方針に対応。
口座開通後、kabu の PUSH 配信（ティック/板更新）を受けて**分足を組み立て、storage に
ためる**。実稼働分足を自前で蓄積するのが純フォワード検証のデータ源になる。

設計上の分離：
- `BarAggregator`：ティック列 → 分足 OHLCV。**純ロジック・完全テスト可能**。
- `KabuRecorder`：kabu PUSH メッセージを解釈して aggregator に流し、確定分足を storage へ。
  実際の WebSocket 接続（websocket-client 等）は本番（Windows）で `handle_message` に
  メッセージを供給する薄いトランスポートを被せる。ここでは接続は持たず、テストでは
  メッセージ列を直接流せる（API・ネットワーク非依存）。

kabu PUSH の想定フィールド（板/時価 push）：
  Symbol, CurrentPrice, CurrentPriceTime(ISO), TradingVolume(当日累計出来高)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from data.storage import Storage

__all__ = ["Bar", "BarAggregator", "KabuRecorder"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Bar:
    """確定した1本の分足。"""

    symbol: str
    ts: pd.Timestamp  # バケット（分）の開始時刻
    open: float
    high: float
    low: float
    close: float
    volume: float


class BarAggregator:
    """ティック（時刻・価格・出来高デルタ）を分足 OHLCV に集約する。

    新しい分に入ると直前の足を確定して emit する。`flush()` で残りを確定。
    因果的（過去のティックのみで現在足を作る）。
    """

    def __init__(
        self, *, interval: str = "1min", on_bar: Callable[[Bar], None] | None = None
    ) -> None:
        self._interval = interval
        self._on_bar = on_bar
        self._current: dict[str, dict[str, Any]] = {}

    def add_tick(
        self, symbol: str, ts: pd.Timestamp, price: float, volume_delta: float = 0.0
    ) -> None:
        if price <= 0:
            return  # 値が無い板更新等はスキップ
        bucket_ts = pd.Timestamp(ts).floor(self._interval)
        cur = self._current.get(symbol)
        if cur is None or bucket_ts > cur["ts"]:
            if cur is not None:
                self._emit(symbol, cur)
            cur = {
                "ts": bucket_ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": max(0.0, volume_delta),
            }
            self._current[symbol] = cur
        else:
            cur["high"] = max(cur["high"], price)
            cur["low"] = min(cur["low"], price)
            cur["close"] = price
            cur["volume"] += max(0.0, volume_delta)

    def flush(self, symbol: str | None = None) -> None:
        """未確定の足を確定して emit する。symbol 指定でその銘柄だけ。"""
        symbols = [symbol] if symbol is not None else list(self._current)
        for s in symbols:
            cur = self._current.pop(s, None)
            if cur is not None:
                self._emit(s, cur)

    def _emit(self, symbol: str, cur: dict[str, Any]) -> None:
        bar = Bar(
            symbol=symbol,
            ts=cur["ts"],
            open=cur["open"],
            high=cur["high"],
            low=cur["low"],
            close=cur["close"],
            volume=cur["volume"],
        )
        if self._on_bar is not None:
            self._on_bar(bar)


class KabuRecorder:
    """kabu PUSH メッセージを分足化して storage へ保存する。

    使い方（本番）：WebSocket の各メッセージを handle_message に渡し、最後に flush()。
    使い方（テスト）：run(messages) にメッセージ列を渡す。
    """

    def __init__(self, *, storage: Storage | None = None) -> None:
        self._storage = storage
        self._bars: list[Bar] = []
        self._last_cum_volume: dict[str, float] = {}
        self._aggregator = BarAggregator(on_bar=self._bars.append)

    def handle_message(self, msg: dict[str, Any]) -> None:
        """1つの kabu PUSH メッセージを処理。"""
        symbol = msg.get("Symbol")
        price = msg.get("CurrentPrice")
        ts_raw = msg.get("CurrentPriceTime")
        if symbol is None or price is None or ts_raw is None:
            return  # 必要フィールド欠落（板のみ更新等）はスキップ

        ts = pd.Timestamp(ts_raw)
        cum = msg.get("TradingVolume")
        volume_delta = 0.0
        if cum is not None:
            prev = self._last_cum_volume.get(symbol)
            # 当日累計のため、前回との差分が出来高。負（日跨ぎリセット等）は 0 に丸める。
            volume_delta = max(0.0, float(cum) - prev) if prev is not None else 0.0
            self._last_cum_volume[symbol] = float(cum)

        self._aggregator.add_tick(symbol, ts, float(price), volume_delta)

    def flush(self) -> None:
        """未確定足を確定し、溜まった分足を storage へ書き出す。"""
        self._aggregator.flush()
        if self._storage is not None and self._bars:
            for sym, group in _group_bars(self._bars).items():
                self._storage.insert_bars(sym, group)

    def run(self, messages: Iterable[dict[str, Any]]) -> list[Bar]:
        """メッセージ列を処理して確定分足のリストを返す（テスト・バッチ用）。"""
        for msg in messages:
            self.handle_message(msg)
        self.flush()
        return list(self._bars)

    @property
    def bars(self) -> list[Bar]:
        return list(self._bars)


def _group_bars(bars: list[Bar]) -> dict[str, pd.DataFrame]:
    """Bar リストを symbol ごとの OHLCV DataFrame に。"""
    out: dict[str, pd.DataFrame] = {}
    by_symbol: dict[str, list[Bar]] = {}
    for b in bars:
        by_symbol.setdefault(b.symbol, []).append(b)
    for sym, items in by_symbol.items():
        df = pd.DataFrame(
            {
                "open": [b.open for b in items],
                "high": [b.high for b in items],
                "low": [b.low for b in items],
                "close": [b.close for b in items],
                "volume": [b.volume for b in items],
            },
            index=pd.DatetimeIndex([b.ts for b in items], name="ts"),
        )
        out[sym] = df
    return out
