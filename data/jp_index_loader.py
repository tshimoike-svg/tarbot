"""日経225 日次データをダウンロード・ディスクキャッシュするローダー。

yfinance (^N225) で取得し、pickle でキャッシュして再利用する。
strategy/swing_momentum.py の市場レジームフィルタ（enable_regime_filter）の
market_df として使う。us_loader.py（S&P500）と同一パターン。
返り値は close 列のみの DataFrame（DatetimeIndex、tz-naive）。
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd

__all__ = ["load_n225", "load_n225_fresh"]

logger = logging.getLogger(__name__)

_DEFAULT_CACHE = Path("data/db/jp_n225_cache")
_DAILY_CACHE_FILE = "n225_daily.pkl"


def load_n225(
    from_: str,
    to: str,
    cache_dir: str | Path = _DEFAULT_CACHE,
) -> pd.DataFrame:
    """日経225 日次 close データを返す。

    Args:
        from_: 開始日 YYYY-MM-DD（キャッシュキーに使用）。
        to:    終了日 YYYY-MM-DD（キャッシュキーに使用）。
        cache_dir: pickle キャッシュ保存先。

    Returns:
        DataFrame（columns=["close"], index=DatetimeIndex tz-naive, 単調増加）。
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    key = f"{from_}_{to}".replace("-", "")
    pkl = cache_path / f"n225_{key}.pkl"

    if pkl.exists():
        logger.debug("日経225 キャッシュから読み込み: %s", pkl)
        with pkl.open("rb") as f:
            return pickle.load(f)  # noqa: S301 - ローカル自前生成ファイルのみ

    try:
        import yfinance as yf  # 遅延 import（optional dev dependency）
    except ImportError as exc:
        raise ImportError("yfinance が必要です: uv add --dev yfinance") from exc

    logger.info("日経225 データを yfinance で取得中（%s〜%s）...", from_, to)
    raw = yf.download("^N225", start=from_, end=to, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError("日経225 データが空です（日付範囲や接続を確認）")

    close = raw["Close"].squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df = pd.DataFrame({"close": close.values}, index=idx)
    df = df.sort_index()

    with pkl.open("wb") as f:
        pickle.dump(df, f)
    logger.info("日経225 データ取得・保存完了: %d 日分", len(df))
    return df


def load_n225_fresh(
    lookback_days: int = 400,
    cache_dir: str | Path = _DEFAULT_CACHE,
) -> pd.DataFrame:
    """毎日の実行用: 最新データを自動追加して返す（差分更新）。

    load_n225 と異なり日付範囲固定のキャッシュキーを使わない。
    `_DAILY_CACHE_FILE` に蓄積し、stale なときだけ yfinance で差分取得する。
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    pkl = cache_path / _DAILY_CACHE_FILE

    today = pd.Timestamp.now().normalize()

    existing: pd.DataFrame | None = None
    if pkl.exists():
        with pkl.open("rb") as f:
            existing = pickle.load(f)  # noqa: S301
        if not existing.empty:
            last_date = existing.index[-1]
            if last_date >= today - pd.Timedelta(days=1):
                logger.debug("日経225 キャッシュ最新（最終: %s）。再取得不要", last_date.date())
                return existing

    fetch_from = (today - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    fetch_to = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance exclusive end

    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("yfinance が必要です: uv add --dev yfinance") from exc

    logger.info("日経225 データを更新中（%s〜）...", fetch_from)
    raw = yf.download("^N225", start=fetch_from, end=fetch_to, progress=False, auto_adjust=True)

    if raw.empty:
        if existing is not None and not existing.empty:
            logger.warning("日経225 新規データなし。キャッシュを返します")
            return existing
        raise ValueError("日経225 データが空です（日付範囲や接続を確認）")

    close = raw["Close"].squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    new_df = pd.DataFrame({"close": close.values}, index=idx).sort_index()

    merged = (
        pd.concat([existing, new_df]).groupby(level=0).last().sort_index()
        if existing is not None and not existing.empty
        else new_df
    )

    with pkl.open("wb") as f:
        pickle.dump(merged, f)
    logger.info("日経225 更新完了: %d 日分（最終: %s）", len(merged), merged.index[-1].date())
    return merged
