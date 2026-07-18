#!/usr/bin/env python3
"""Phase 1 ドライラン: 毎朝（平日 7:40 JST）実行の日次シグナルスキャン。

config_v（反転）+ mom_lb60_filtered（momentum）併用ロジック（2026-07-18決定）で
共有資金プールを運用する。信用3倍・証拠金使用率80%・1銘柄配分20%・1日の新規建て
上限2件・競合時は1単元必要額が安い方優先（strategy/portfolio_allocator.py）。

実行タイミング:
  7:40 AM JST — US 市場は既に閉場済み（夏時間で US 引け ~5:00 AM JST）。
                  J-Quants に前日（T）の日足が揃っている。東京寄り付き 9:00 前。
                  → T-1 / T0 US フィルタを両方適用してシグナルを確定できる。
                  （実行時刻の定義は .github/workflows/daily_scan.yml の cron）

処理フロー:
  1. S&P500・日経225 最新データ取得（差分更新）
  2. 全銘柄の日足取得（差分更新キャッシュ）
  3. オープンポジションの出口チェック（前日 OHLC で判定・実現損益を口座に反映）
  4. config_v・mom_lb60_filtered の新規シグナル検出（前日 T の終値ベース）
  5. risk_manager・portfolio_allocator で本日の新規建てを決定
  6. サマリ出力

注意:
  エントリー価格 = T の終値（近似）。実際の T+1 始値はギャップで異なる。
  発注は一切しない（CLAUDE.md 絶対原則1・6）。
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# プロジェクトルートを sys.path に追加（スクリプト単体実行用）
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from backtest.cost_model import CostModel
from config.costs import get_cost_params
from config.settings import COMBO_V_MOM60_RISK, SwingMomentumParams, SwingReversionParams
from config.symbols import SYMBOLS
from data.daily_loader import build_daily_loader
from data.jp_index_loader import load_n225_fresh
from data.signal_store import ForwardSignal, SignalStore
from data.us_loader import load_spx_fresh
from data.yahoo_loader import build_yahoo_loader
from notification.push_notifier import send_close_alert, send_signal_alert
from strategy.portfolio_allocator import Candidate, allocate_daily_entries
from strategy.risk_manager import RiskManager
from strategy.swing_momentum import compute_signals as compute_momentum_signals
from strategy.swing_reversion import compute_signals as compute_reversion_signals

logger = logging.getLogger(__name__)

# ── 併用運用の設定（2026-07-18決定。詳細: docs/results/combo_v_mom60_decision_2026-07-18.md）
_CONFIG_NAME = "combo_v_mom60"
_INITIAL_EQUITY = 1_000_000.0
_LEVERAGE = 3.0
_USAGE_RATE = 0.8
_PER_SYMBOL_SHARE = 0.20

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

# シグナル計算に十分な日数（lookback + ATR + RSI + 余裕）
_MIN_BARS = 70
# J-Quants 取得窓
_LOOKBACK_DAYS = 100


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Phase 1 日次シグナルスキャン（config_v + mom_lb60_filtered併用）")
    parser.add_argument(
        "--db", default=str(_ROOT / "data/db/forward_signals.sqlite"),
        help="シグナル DB パス",
    )
    parser.add_argument(
        "--source", choices=["yahoo", "jquants"], default="yahoo",
        help="日足ソース（既定: yahoo = 当日終値0日ラグ。jquants は無料だと3ヶ月遅延）",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="日足キャッシュ dir（未指定なら source ごとの既定）",
    )
    parser.add_argument(
        "--min-interval", type=float, default=13.0,
        help="J-Quants API レート制限（Free=13.0, Light=1.0）。yahoo では未使用",
    )
    parser.add_argument(
        "--dry", action="store_true",
        help="DB に書き込まず検出シグナルを表示だけする",
    )
    args = parser.parse_args(argv)

    today = date.today()
    # 週末はスキップ（Saturday=5, Sunday=6）—— --dry の場合は強制続行可
    if today.weekday() >= 5 and not args.dry:
        logger.info("週末のためスキャンをスキップ (%s)", today)
        return 0

    from_date = (today - timedelta(days=_LOOKBACK_DAYS)).isoformat()
    to_date = (today - timedelta(days=1)).isoformat()  # 前日まで（T = 昨日）

    # 1. S&P500 最新データ（config_v の US フィルタ用）
    logger.info("S&P500 データ確認中...")
    try:
        us_df = load_spx_fresh(lookback_days=_LOOKBACK_DAYS + 30)
        logger.info("S&P500 最終日: %s", us_df.index[-1].date())
    except Exception as exc:
        logger.error("S&P500 取得失敗: %s — US フィルタなしで続行", exc)
        us_df = None

    # 1b. 日経225 最新データ（mom_lb60_filtered の市場レジームフィルタ用）
    logger.info("日経225 データ確認中...")
    try:
        n225_df = load_n225_fresh(lookback_days=_LOOKBACK_DAYS + 100)
        logger.info("日経225 最終日: %s", n225_df.index[-1].date())
    except Exception as exc:
        logger.error("日経225 取得失敗: %s — レジームフィルタなしで続行", exc)
        n225_df = None

    # 2. 全銘柄リスト
    try:
        from config.symbols_cs import SYMBOLS_CS  # noqa: PLC0415
        all_symbols = list(dict.fromkeys(SYMBOLS + SYMBOLS_CS))
    except ImportError:
        all_symbols = list(SYMBOLS)
    logger.info("スキャン対象: %d 銘柄", len(all_symbols))

    # 3. 日足ローダー（差分更新キャッシュ）
    if args.source == "yahoo":
        cache_dir = args.cache_dir or str(_ROOT / "data/db/yahoo_scan_cache")
        logger.info("日足ソース: Yahoo Finance（当日終値・分割調整）")
        load_bars = build_yahoo_loader(from_date=from_date, to_date=to_date, cache_dir=cache_dir)
    else:
        cache_dir = args.cache_dir or str(_ROOT / "data/db/daily_scan_cache")
        logger.info("日足ソース: J-Quants（無料は3ヶ月遅延）")
        load_bars = build_daily_loader(
            from_date=from_date, to_date=to_date, cache_dir=cache_dir, min_interval=args.min_interval,
        )

    cost_model = CostModel(get_cost_params("mid"))

    with SignalStore(args.db) as store:
        # 4. 現在の資産・オープン建玉を復元
        current_equity = _reconstruct_equity(store)
        logger.info("現在の評価額（復元）: %.0f円", current_equity)

        open_sigs = store.get_open_signals()
        closed_records: list[dict] = []
        if open_sigs:
            logger.info("オープンシグナル %d 件の出口チェック中...", len(open_sigs))
            closed_count, closed_records, current_equity = _check_exits(
                open_sigs, store, load_bars, cost_model, today, current_equity, args.dry,
            )
            if closed_count:
                logger.info("  → %d 件クローズ  評価額: %.0f円", closed_count, current_equity)
                if not args.dry:
                    send_close_alert(closed_records)

        # 5. risk_manager を今日の状態で再構築（オープン建玉を復元）
        rm = RiskManager(account_equity=current_equity, params=COMBO_V_MOM60_RISK)
        rm.start_day(today)
        still_open = store.get_open_signals()
        for open_row in still_open:
            notional = float(open_row["shares"] or 0) * float(open_row["entry_price"])
            if notional > 0:
                rm._positions[open_row["symbol"]] = notional  # noqa: SLF001 — 状態復元（on_openは日次件数を誤って加算するため使わない）

        # 6. 新規シグナル検出（前日 T のデータ）→ 候補を集めて割り当て
        signal_date = to_date  # T = 昨日
        candidates_info = _detect_candidates(
            all_symbols, load_bars, us_df, n225_df, store, signal_date, today,
        )

        decisions = allocate_daily_entries(
            [c for c, _info in candidates_info],
            risk_manager=rm, account_equity=current_equity,
            leverage=_LEVERAGE, usage_rate=_USAGE_RATE, per_symbol_share=_PER_SYMBOL_SHARE,
        )
        info_by_symbol = {c.symbol: info for c, info in candidates_info}

        new_signals: list[ForwardSignal] = []
        for dec in decisions:
            if not dec.approved:
                logger.info("却下: %s reason=%s", dec.candidate.symbol, dec.decision.reason)
                continue
            info = info_by_symbol[dec.candidate.symbol]
            sig = ForwardSignal(
                symbol=dec.candidate.symbol,
                signal_date=signal_date,
                side="long",
                entry_date=today.isoformat(),
                entry_price=dec.candidate.entry_price,
                stop_price=info["stop_price"],
                target_price=info["target_price"],
                max_exit_date=(today + timedelta(days=info["max_holding_days"])).isoformat(),
                config_name=_CONFIG_NAME,
                shares=dec.sizing.shares,
            )
            if not args.dry:
                store.insert_signal(sig)
            new_signals.append(sig)

        # 7. LINE 通知（dry モードはスキップ）
        if not args.dry and new_signals:
            send_signal_alert(signal_date, {_CONFIG_NAME: new_signals})

        # 8. 月次 Professional プラン維持チェック（15日・25日）
        _check_monthly_plan_alert(store, today, dry=args.dry)

        # 9. 出力
        _print_report(new_signals, store, signal_date, today, current_equity)

    return 0


# ── 資産・状態の復元 ────────────────────────────────────────────────────────────────

def _reconstruct_equity(store: SignalStore) -> float:
    """クローズ済み combo_v_mom60 トレードの実現損益から現在の評価額を復元する。"""
    df = store.get_all()
    if df.empty:
        return _INITIAL_EQUITY
    closed = df[(df["status"] == "closed") & (df["config_name"] == _CONFIG_NAME)]
    if closed.empty:
        return _INITIAL_EQUITY
    realized = (closed["shares"].fillna(0) * closed["entry_price"] * closed["net_return"].fillna(0)).sum()
    return _INITIAL_EQUITY + float(realized)


# ── 月次プラン維持チェック ────────────────────────────────────────────────────────

def _check_monthly_plan_alert(store: SignalStore, today: date, *, dry: bool) -> None:
    """15日・25日に当月シグナルゼロなら Professionalプラン維持の警告を送る。

    kabuステーション Professional プランは月1回以上の取引維持が条件。
    US クラッシュが来ない月はシグナルが出ない可能性があるため手動取引を促す。
    """
    if today.day not in (15, 25):
        return
    ym = today.strftime("%Y-%m")
    n = store.signals_in_month(ym)
    if n > 0:
        logger.info("月次チェック: %s に %d 件のシグナルあり → プラン維持OK", ym, n)
        return

    msg = (
        f"今月（{ym}）はまだシグナルがありません。\n"
        "US 市場が穏やかな月はシグナルが出ない場合があります。\n"
        "kabuステーション Professional プランの月次取引要件（月1回以上）のため、\n"
        "手動での最小限の取引（現物少額など）を検討してください。"
    )
    logger.warning("月次チェック: %s シグナルゼロ — Professionalプラン維持要注意", ym)
    if not dry:
        from notification.push_notifier import send  # noqa: PLC0415
        send(title=f"⚠️ {ym} シグナルなし — プラン維持確認", body=msg, priority="high")


# ── シグナル検出 ────────────────────────────────────────────────────────────────

def _detect_candidates(
    symbols: list[str],
    load_bars,
    us_df: pd.DataFrame | None,
    n225_df: pd.DataFrame | None,
    store: SignalStore,
    signal_date: str,
    entry_date: date,
) -> list[tuple[Candidate, dict]]:
    """config_v・mom_lb60_filtered 両方の当日候補を集める（Candidate, 付随情報）。"""
    out: list[tuple[Candidate, dict]] = []

    for symbol in symbols:
        df = load_bars(symbol)
        if df.empty or len(df) < _MIN_BARS:
            continue

        # --- config_v（反転） ---
        rev = compute_reversion_signals(df, CONFIG_V, us_df=us_df)
        if not rev.empty and int(rev["entry"].iloc[-1]) == 1:
            actual_signal_date = rev.index[-1].strftime("%Y-%m-%d")
            if not store.signal_exists(symbol, actual_signal_date, _CONFIG_NAME):
                last_row = rev.iloc[-1]
                last_close = float(df["close"].iloc[-1])
                last_atr = float(last_row["atr"]) if not pd.isna(last_row["atr"]) else last_close * 0.02
                out.append((
                    Candidate(symbol=symbol, side="buy", entry_price=last_close, source="config_v"),
                    {
                        "stop_price": last_close - last_atr * CONFIG_V.atr_stop_mult,
                        "target_price": float(last_row["ma"]),
                        "max_holding_days": CONFIG_V.max_holding_days,
                    },
                ))
                continue  # 同一銘柄でmomentumも重複検出しない（1銘柄1シグナル/日）

        # --- mom_lb60_filtered（momentum） ---
        mom = compute_momentum_signals(df, MOM_LB60_FILTERED, market_df=n225_df)
        if not mom.empty and int(mom["entry"].iloc[-1]) == 1:
            actual_signal_date = mom.index[-1].strftime("%Y-%m-%d")
            if not store.signal_exists(symbol, actual_signal_date, _CONFIG_NAME):
                last_row = mom.iloc[-1]
                last_close = float(df["close"].iloc[-1])
                last_atr = float(last_row["atr"]) if not pd.isna(last_row["atr"]) else last_close * 0.02
                out.append((
                    Candidate(symbol=symbol, side="buy", entry_price=last_close, source="mom_lb60_filtered"),
                    {
                        "stop_price": last_close - last_atr * MOM_LB60_FILTERED.atr_stop_mult,
                        "target_price": float("inf"),
                        "max_holding_days": MOM_LB60_FILTERED.max_holding_days,
                    },
                ))

    return out


# ── 出口チェック ────────────────────────────────────────────────────────────────

def _check_exits(
    open_sigs: list[dict],
    store: SignalStore,
    load_bars,
    cost_model: CostModel,
    today: date,
    current_equity: float,
    dry: bool,
) -> tuple[int, list[dict], float]:
    """オープンシグナルの出口条件を昨日の OHLC で判定してクローズする（実現損益を評価額に反映）。"""
    closed = 0
    closed_records: list[dict] = []
    yesterday = (today - timedelta(days=1)).isoformat()

    for sig in open_sigs:
        symbol = sig["symbol"]
        entry_date_str = sig["entry_date"]
        entry_price = float(sig["entry_price"])
        stop_price = float(sig["stop_price"])
        target_price = float(sig["target_price"])
        max_exit_date = sig["max_exit_date"]
        shares = int(sig["shares"] or 0)

        df = load_bars(symbol)
        if df.empty:
            continue

        entry_ts = pd.Timestamp(entry_date_str)
        df_after = df[df.index >= entry_ts]
        if df_after.empty:
            continue

        for dt, row in df_after.iterrows():
            dt_str = dt.strftime("%Y-%m-%d")
            if dt_str > yesterday:
                break  # 本日以降はまだ確定していない

            exit_price: float | None = None
            exit_reason: str | None = None

            if row["low"] <= stop_price:
                exit_price = stop_price
                exit_reason = "stop"
            elif row["close"] >= target_price:
                exit_price = float(row["close"])
                exit_reason = "target"
            elif dt_str >= max_exit_date:
                exit_price = float(row["close"])
                exit_reason = "time_stop"

            if exit_price is not None:
                gross = (exit_price - entry_price) / entry_price
                holding = max(1, (dt - entry_ts).days)
                cost_frac = cost_model.round_trip_cost_fraction(
                    price=entry_price, shares=1, holding_days=holding, side="long",
                )
                net = gross - cost_frac
                realized_yen = shares * entry_price * net

                if not dry:
                    store.close_signal(
                        sig["id"], exit_date=dt_str, exit_price=exit_price,
                        exit_reason=exit_reason, gross_return=gross, net_return=net,
                    )
                current_equity += realized_yen
                logger.info(
                    "  クローズ: %s → %s @ %.0f  net %+.2f%%  実現損益 %+.0f円",
                    symbol, exit_reason, exit_price, net * 100, realized_yen,
                )
                closed_records.append({**sig, "exit_reason": exit_reason, "net_return": net})
                closed += 1
                break

    return closed, closed_records, current_equity


# ── レポート出力 ────────────────────────────────────────────────────────────────

def _print_report(
    new_signals: list[ForwardSignal],
    store: SignalStore,
    signal_date: str,
    entry_date: date,
    current_equity: float,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Phase 1 スキャン結果（{_CONFIG_NAME}）  シグナル日: {signal_date}  エントリー: {entry_date}")
    print(f"{'=' * 60}")
    print(f"\n評価額: {current_equity:,.0f}円（開始 {_INITIAL_EQUITY:,.0f}円・騰落率 {(current_equity/_INITIAL_EQUITY-1)*100:+.1f}%）")

    if new_signals:
        print(f"\n新規シグナル {len(new_signals)} 件:")
        for s in new_signals:
            print(
                f"  {s.symbol:6s}  買い {s.shares}株 @ 本日始値(≈{s.entry_price:,.0f}円)  "
                f"ストップ {s.stop_price:,.0f}  期限 {s.max_exit_date}"
            )
    else:
        print("\n新規シグナルなし")

    print("\n── 累計 ──")
    print(f"  {store.summary()}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
