#!/usr/bin/env python3
"""config_v + mom_lb60_filtered 併用ロジックでフォワード追跡を作り直す（2026-07-18決定）。

それまで個別に記録していた config_iii/iv/v・mom_lb20/40/60 のフォワード追跡（絶対原則1・6
に反しない、発注なしのドライラン記録）を廃止し、決定した併用ロジック
（RiskManager経由・portfolio_allocator・size_position_fixed_fraction）だけで
forward_signals.sqlite を作り直す。バックテスト期間（2024-03-28）から現在まで一貫して
同じロジックで再生する。以降は daily_scan.py が同じロジックで追跡を継続する。

実行例:
    uv run python scripts/rebuild_combo_dry_run.py
    uv run python scripts/rebuild_combo_dry_run.py --dry  # DBを書き換えず件数だけ確認
"""

from __future__ import annotations

import argparse
import heapq
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from backtest.cost_model import CostModel
from config.costs import get_cost_params
from config.settings import COMBO_V_MOM60_RISK, SwingMomentumParams, SwingReversionParams
from data.jp_index_loader import load_n225
from data.signal_store import ForwardSignal, SignalStore
from data.us_loader import load_spx
from strategy.portfolio_allocator import Candidate, allocate_daily_entries
from strategy.risk_manager import RiskManager
from strategy.swing_momentum import generate_trades as gen_momentum
from strategy.swing_reversion import generate_trades as gen_reversion
from strategy.trade import Trade

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

_CACHE_DIR = _ROOT / "data/db/bars_cache_yahoo"
_INITIAL_EQUITY = 1_000_000.0
_LEVERAGE = 3.0
_USAGE_RATE = 0.8
_PER_SYMBOL_SHARE = 0.20
_CONFIG_NAME = "combo_v_mom60"
_UNIT = 100

_BASE_REV = dict(
    lookback=20, entry_z=2.0, atr_stop_mult=2.0, max_holding_days=10,
    allow_long=True, allow_short=False,
    us_t1_crash_threshold=-0.02, us_t1_soft_min=-0.005, us_t1_soft_max=0.0,
    us_t0_crash_threshold=-0.02, us_t0_soft_min=-0.005, us_t0_soft_max=0.0,
    us_t0_recovery_min=0.01,
)
CONFIG_V = SwingReversionParams(**_BASE_REV, rsi_entry_max=40.0)
MOM_LB60_FILTERED = SwingMomentumParams(
    breakout_lookback=60, atr_stop_mult=2.0, max_holding_days=10,
    allow_long=True, allow_short=False,
    enable_regime_filter=True, regime_ma_window=90, regime_threshold=0.05,
    regime_filter_invert=True,
)


def _load_bars(symbol: str) -> pd.DataFrame:
    matches = list(_CACHE_DIR.glob(f"{symbol}_*.pkl"))
    if not matches:
        return pd.DataFrame()
    return pd.read_pickle(matches[0])


@dataclass(order=True)
class _Scheduled:
    exit_time: pd.Timestamp
    seq: int
    trade: Trade = field(compare=False)
    symbol: str = field(compare=False)
    shares: int = field(compare=False)
    notional: float = field(compare=False)
    row_id: int = field(compare=False)


def _collect_tagged_trades(
    dfs: dict[str, pd.DataFrame], market_df: pd.DataFrame, us_df: pd.DataFrame
) -> list[tuple[str, Trade, bool]]:
    """全銘柄の config_v・mom_lb60_filtered トレードを集める。

    各銘柄データの最終バーで time_stop になったトレードは、実際には
    max_holding_days が経過しきっていない「打ち切りによる仮の決済」の可能性がある
    （walk_swing は手元データの最終バーで強制的に打ち切るため）。
    エントリー自体は実データに基づく本物のシグナルなので有効に使うが、
    決済（exit_time/exit_price/exit_reason）はまだ確定させず、DB上は
    「オープン中」のまま残す（provisional=True）。将来 daily_scan.py が
    新しいデータで正しく決済を判定する。
    """
    tagged: list[tuple[str, Trade, bool]] = []
    n_provisional = 0
    for symbol, df in dfs.items():
        if df.empty or len(df) < 70:
            continue
        last_bar = df.index[-1]

        rev_trades = gen_reversion(df, CONFIG_V, us_df=us_df)
        mom_trades = gen_momentum(df, MOM_LB60_FILTERED, market_df=market_df)

        for t in (*rev_trades, *mom_trades):
            provisional = t.exit_reason == "time_stop" and t.exit_time == last_bar
            if provisional:
                n_provisional += 1
            tagged.append((symbol, t, provisional))

    logger.info("収集トレード: %d 件（うちデータ末尾での仮決済 %d 件はオープン中扱い）", len(tagged), n_provisional)
    return tagged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="config_v + mom_lb60_filtered 併用ドライラン再構築")
    parser.add_argument("--dry", action="store_true", help="DBを書き換えず件数・最終評価額だけ表示する")
    parser.add_argument("--db", default=str(_ROOT / "data/db/forward_signals.sqlite"))
    args = parser.parse_args(argv)

    cache_files = sorted(_CACHE_DIR.glob("*.pkl"))
    symbols = sorted({p.stem.split("_")[0] for p in cache_files})
    dfs = {s: _load_bars(s) for s in symbols}
    dfs = {s: d for s, d in dfs.items() if not d.empty}
    logger.info("銘柄数: %d", len(dfs))

    # 期間は最頻の終了日（多くの銘柄が共有する最終日）を使う。一部銘柄だけデータが
    # 早く止まっている場合（例: 4384 Raksul が2026-05-28で停止）に引きずられない。
    end_counts: dict[pd.Timestamp, int] = {}
    for d in dfs.values():
        end_counts[d.index[-1]] = end_counts.get(d.index[-1], 0) + 1
    end_date = max(end_counts, key=lambda k: end_counts[k])
    for sym, d in dfs.items():
        if d.index[-1] < end_date:
            logger.warning("%s のデータが %s までしかありません（全体の終了日は %s）", sym, d.index[-1].date(), end_date.date())

    logger.info("日経225取得中...")
    n225 = load_n225("2024-01-01", (end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), cache_dir=str(_ROOT / "data/db/jp_n225_cache"))
    market_df = n225.rename(columns={n225.columns[0]: "close"})[["close"]] if "close" not in n225.columns else n225[["close"]]

    logger.info("S&P500取得中...")
    us_df = load_spx("2024-01-01", (end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), cache_dir=str(_ROOT / "data/db/us_spx_cache"))

    tagged = _collect_tagged_trades(dfs, market_df, us_df)
    tagged.sort(key=lambda st: st[1].entry_time)

    cost_model = CostModel(get_cost_params("mid"))
    rm = RiskManager(account_equity=_INITIAL_EQUITY, params=COMBO_V_MOM60_RISK)
    current_equity = _INITIAL_EQUITY

    if not args.dry:
        # 前回分（旧スキーマ含む）をテーブルごと削除して作り直す
        # （2026-07-18決定：併用ロジックのみにする。個別追跡は廃止）
        import sqlite3  # noqa: PLC0415

        db_path = Path(args.db)
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            conn.execute("DROP TABLE IF EXISTS forward_signals")
            conn.commit()
            conn.close()

    store = None if args.dry else SignalStore(args.db)

    close_heap: list[_Scheduled] = []
    idx = 0
    current_day = None
    seq = 0
    n_opened = 0
    n_closed = 0
    net_returns: list[float] = []

    def roll(day) -> None:
        nonlocal current_day
        if day != current_day:
            rm.start_day(day)
            current_day = day

    n_still_open = 0

    while idx < len(tagged) or close_heap:
        next_open_time = tagged[idx][1].entry_time if idx < len(tagged) else None
        next_close_time = close_heap[0].exit_time if close_heap else None

        if next_open_time is not None and (next_close_time is None or next_open_time <= next_close_time):
            today = next_open_time
        else:
            today = next_close_time

        roll(today.date())

        # 1. 今日が期日の決済をすべて処理
        while close_heap and close_heap[0].exit_time <= today:
            pos = heapq.heappop(close_heap)
            cost_frac = cost_model.round_trip_cost_fraction(
                price=pos.trade.entry_price, shares=1,
                holding_days=max(1, (pos.trade.exit_time.normalize() - pos.trade.entry_time.normalize()).days),
                side=pos.trade.side,
            )
            gross_frac = pos.trade.pnl_gross_per_share / pos.trade.entry_price
            net_frac = gross_frac - cost_frac
            realized_yen = pos.notional * net_frac
            current_equity += realized_yen
            rm.account_equity = current_equity
            rm.on_close(pos.symbol, realized_pnl=realized_yen)
            net_returns.append(net_frac)
            # 期待値ゲート（set_recent_expectancy）は意図的に呼ばない：
            # risk_manager の実装は「開始からの累積平均」であり、早期に数件不運が
            # 続くとマイナスに転じて新規建てが止まり、新規建てが止まると平均を
            # 動かすトレードも増えないため二度と回復しない（2026-07-18発見・要別途修正）。
            n_closed += 1
            if store is not None:
                store.close_signal(
                    pos.row_id,
                    exit_date=pos.trade.exit_time.strftime("%Y-%m-%d"),
                    exit_price=pos.trade.exit_price,
                    exit_reason=pos.trade.exit_reason,
                    gross_return=gross_frac,
                    net_return=net_frac,
                )

        # 2. 今日がエントリー日の候補をまとめて割り当てる
        if next_open_time is not None and today == next_open_time:
            todays: list[tuple[str, Trade, bool]] = []
            while idx < len(tagged) and tagged[idx][1].entry_time == today:
                todays.append(tagged[idx])
                idx += 1

            candidates = [
                Candidate(symbol=sym, side="buy", entry_price=t.entry_price, source=t.exit_reason)
                for sym, t, _prov in todays
            ]
            rm.account_equity = current_equity
            decisions = allocate_daily_entries(
                candidates, risk_manager=rm, account_equity=current_equity,
                leverage=_LEVERAGE, usage_rate=_USAGE_RATE, per_symbol_share=_PER_SYMBOL_SHARE,
                unit=_UNIT,
            )
            trade_by_symbol = {sym: (t, prov) for sym, t, prov in todays}
            for dec in decisions:
                if not dec.approved:
                    continue
                t, provisional = trade_by_symbol[dec.candidate.symbol]
                seq += 1
                n_opened += 1
                target_price = t.exit_price if t.exit_reason == "target" else float("inf")
                row_id = 0
                if store is not None:
                    row_id = store.insert_signal(
                        ForwardSignal(
                            symbol=dec.candidate.symbol,
                            signal_date=(t.entry_time - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                            side=t.side,
                            entry_date=t.entry_time.strftime("%Y-%m-%d"),
                            entry_price=t.entry_price,
                            stop_price=t.stop_price if t.stop_price is not None else t.entry_price,
                            target_price=target_price,
                            max_exit_date=t.exit_time.strftime("%Y-%m-%d"),
                            config_name=_CONFIG_NAME,
                            shares=dec.sizing.shares,
                        )
                    )
                if provisional:
                    # 決済はまだ確定させない（DB上は status='open' のまま残す）。
                    # close_heap には積まない＝このシミュレーション内では二度と処理しない。
                    n_still_open += 1
                    continue
                heapq.heappush(
                    close_heap,
                    _Scheduled(
                        exit_time=t.exit_time, seq=seq, trade=t, symbol=dec.candidate.symbol,
                        shares=dec.sizing.shares, notional=dec.sizing.notional, row_id=row_id,
                    ),
                )

    logger.info("完了: 新規建て %d 件 / 決済 %d 件 / オープン中 %d 件", n_opened, n_closed, n_still_open)
    logger.info("最終評価額: %.0f円（開始 %.0f円・騰落率 %+.1f%%）", current_equity, _INITIAL_EQUITY, (current_equity / _INITIAL_EQUITY - 1) * 100)

    if store is not None:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
