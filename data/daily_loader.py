"""Phase 1 DryRun 用の日次差分更新ローダー。

J-Quants 日足を銘柄ごとに pickle キャッシュし、stale なときだけ差分取得する。
Free プラン 5 req/min の制限があるため、初回以外は API 呼び出しを最小化する。

  初回(キャッシュなし): 全銘柄を取得（180 銘柄 × 13 秒 ≈ 39 分）
  2 回目以降(キャッシュ新鮮): キャッシュ返却（API 呼び出しなし）
  前日比: stale な銘柄だけ再取得（通常は 0 件）
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Callable

import pandas as pd

__all__ = ["build_daily_loader"]

logger = logging.getLogger(__name__)

_DEFAULT_CACHE = Path("data/db/daily_scan_cache")


def build_daily_loader(
    *,
    from_date: str,
    to_date: str,
    cache_dir: str | Path = _DEFAULT_CACHE,
    min_interval: float = 13.0,
) -> Callable[[str], pd.DataFrame]:
    """差分更新キャッシュ付きローダーを返す。

    Args:
        from_date: データ取得開始日 YYYY-MM-DD（シグナル計算に必要な窓分を確保）。
        to_date:   データ取得終了日 YYYY-MM-DD（通常は前日）。
        cache_dir: 銘柄ごとの pickle を保存するディレクトリ。
        min_interval: J-Quants API リクエスト間隔（秒）。Free=13, Light=1。

    Returns:
        Callable[[symbol], pd.DataFrame]: シグナル計算に使える OHLCV DataFrame。
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    today = pd.Timestamp.now().normalize()
    to_ts = pd.Timestamp(to_date)
    from_ts = pd.Timestamp(from_date)

    # stale 判定: キャッシュの最終日が to_date の 1 日前未満なら再取得
    # 週末対応: 金曜〜月曜は「3 日以上前」でも再取得しない（US と同様の考え方）
    staleness_threshold = to_ts - pd.Timedelta(days=1)

    _client: list = []  # 遅延初期化（認証エラーを初回まで先送り）

    def _client_instance():
        if not _client:
            from data.fetcher import JQuantsClient  # noqa: PLC0415
            _client.append(
                JQuantsClient(min_interval=min_interval, max_retries=8, retry_backoff=5.0)
            )
        return _client[0]

    def load(symbol: str) -> pd.DataFrame:
        pkl = cache_path / f"{symbol}.pkl"

        cached: pd.DataFrame = pd.DataFrame()
        if pkl.exists():
            with pkl.open("rb") as f:
                cached = pickle.load(f)  # noqa: S301
            if not cached.empty and cached.index[-1] >= staleness_threshold:
                return cached[cached.index >= from_ts].copy()

        # キャッシュが stale / 存在しない → J-Quants から取得
        try:
            df = _client_instance().get_daily_quotes(code=symbol, from_=from_date, to=to_date)
        except Exception as exc:
            logger.warning("銘柄 %s 取得失敗: %s", symbol, exc)
            if not cached.empty:
                logger.info("  → 古いキャッシュを返す（最終: %s）", cached.index[-1].date())
                return cached[cached.index >= from_ts].copy()
            return pd.DataFrame()

        if df.empty:
            return df

        # 既存キャッシュとマージ（J-Quants データを優先）
        if not cached.empty:
            df = pd.concat([cached, df]).groupby(level=0).last().sort_index()

        with pkl.open("wb") as f:
            pickle.dump(df, f)
        logger.debug("銘柄 %s キャッシュ更新（最終: %s）", symbol, df.index[-1].date())

        return df[df.index >= from_ts].copy()

    return load
