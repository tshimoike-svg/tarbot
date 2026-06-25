"""Phase 1 DryRun 用の Yahoo Finance 日次ローダー。

J-Quants 無料プランは日足が 12 週（約3ヶ月）遅延するため、当日終値を必要とする
ドライランには使えない。Yahoo Finance (yfinance) は日本株（`コード.T`）の当日終値を
0 日ラグで返すので、ドライランの日足ソースとして用いる。

**調整方式の整合性（重要）**
  Phase 0 バックテストは J-Quants の調整後（分割調整・配当は非調整）で実施した。
  yfinance の `auto_adjust=True`（= Adj Close）は分割＋配当を調整するため J-Quants と
  系列がズレる。よって本ローダーは `auto_adjust=False` の生 OHLC（＝分割調整のみ）を使う。
  実測（2025-12〜2026-03, 5 銘柄, 80 日）で J-Quants 調整後 close と Yahoo `Close` は
  max 0.00% / リターン相関 1.0000 で一致することを確認済み。

インターフェイスは data.daily_loader.build_daily_loader と同一（差し替え可能）。
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Callable

import pandas as pd

__all__ = ["build_yahoo_loader", "to_yahoo_ticker"]

logger = logging.getLogger(__name__)

_DEFAULT_CACHE = Path("data/db/yahoo_scan_cache")
_COLUMNS = ["open", "high", "low", "close", "volume"]


def to_yahoo_ticker(symbol: str) -> str:
    """銘柄コードを Yahoo の東証ティッカー（`コード.T`）へ変換する。"""
    s = str(symbol).strip()
    return s if s.endswith(".T") else f"{s}.T"


def _normalize_yahoo(raw: pd.DataFrame) -> pd.DataFrame:
    """yfinance.download の戻りを open/high/low/close/volume（Date index 昇順）へ正規化。

    `auto_adjust=False` の生 OHLCV（分割調整のみ）を使う。配当調整済みの Adj Close は
    J-Quants と methodology が異なるため採用しない。
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=_COLUMNS)

    df = raw.copy()
    # 単一ティッカーでも列が MultiIndex (field, ticker) になることがある → ticker 次元を除去
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(-1)

    rename = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    missing = [c for c in rename if c not in df.columns]
    if missing:
        logger.warning("Yahoo データに欠損列 %s（取得列=%s）", missing, list(df.columns))
        return pd.DataFrame(columns=_COLUMNS)

    out = df[list(rename)].rename(columns=rename)
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["close"])
    return out[_COLUMNS]


def build_yahoo_loader(
    *,
    from_date: str,
    to_date: str,
    cache_dir: str | Path = _DEFAULT_CACHE,
    **_ignored,
) -> Callable[[str], pd.DataFrame]:
    """差分更新キャッシュ付きの Yahoo 日足ローダーを返す。

    data.daily_loader.build_daily_loader と同一シグネチャ（min_interval 等の余分な
    引数は無視）なので、ドライランのソースをそのまま差し替えられる。

    Args:
        from_date: 取得開始日 YYYY-MM-DD（シグナル計算の窓分を確保）。
        to_date:   取得終了日 YYYY-MM-DD（通常は前日）。
        cache_dir: 銘柄ごとの pickle を保存するディレクトリ。

    Returns:
        Callable[[symbol], pd.DataFrame]: open/high/low/close/volume の DataFrame。
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    to_ts = pd.Timestamp(to_date)
    from_ts = pd.Timestamp(from_date)
    # yfinance の end は排他。to_date を含めるため翌日を渡す。
    fetch_end = (to_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    # stale 判定: キャッシュ最終日が to_date の 1 日前未満なら再取得（週末は許容）。
    staleness_threshold = to_ts - pd.Timedelta(days=1)

    def load(symbol: str) -> pd.DataFrame:
        pkl = cache_path / f"{symbol}.pkl"

        cached = pd.DataFrame()
        if pkl.exists():
            with pkl.open("rb") as f:
                cached = pickle.load(f)  # noqa: S301
            if not cached.empty and cached.index[-1] >= staleness_threshold:
                return cached[cached.index >= from_ts].copy()

        ticker = to_yahoo_ticker(symbol)
        try:
            import yfinance as yf  # noqa: PLC0415  # 遅延 import

            raw = yf.download(
                ticker, start=from_date, end=fetch_end,
                auto_adjust=False, progress=False, threads=False,
            )
            df = _normalize_yahoo(raw)
        except Exception as exc:  # ネットワーク/レート制限等
            logger.warning("銘柄 %s (%s) Yahoo 取得失敗: %s", symbol, ticker, exc)
            if not cached.empty:
                logger.info("  → 古いキャッシュを返す（最終: %s）", cached.index[-1].date())
                return cached[cached.index >= from_ts].copy()
            return pd.DataFrame(columns=_COLUMNS)

        if df.empty:
            if not cached.empty:
                return cached[cached.index >= from_ts].copy()
            return df

        if not cached.empty:
            df = pd.concat([cached, df]).groupby(level=0).last().sort_index()

        with pkl.open("wb") as f:
            pickle.dump(df, f)
        logger.debug("銘柄 %s Yahoo キャッシュ更新（最終: %s）", symbol, df.index[-1].date())

        return df[df.index >= from_ts].copy()

    return load
