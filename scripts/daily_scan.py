#!/usr/bin/env python3
"""Phase 1 ドライラン: 毎朝（平日 7:40 JST）実行の日次シグナルスキャン。

実行タイミング:
  7:40 AM JST — US 市場は既に閉場済み（夏時間で US 引け ~5:00 AM JST）。
                  J-Quants に前日（T）の日足が揃っている。東京寄り付き 9:00 前。
                  → T-1 / T0 US フィルタを両方適用してシグナルを確定できる。
                  （実行時刻の定義は .github/workflows/daily_scan.yml の cron）

処理フロー:
  1. S&P500 最新データ取得（差分更新）
  2. 全銘柄の日足取得（差分更新キャッシュ）
  3. オープンポジションの出口チェック（前日 OHLC で判定）
  4. 新規シグナル検出（前日 T の終値ベース）
  5. サマリ出力

注意:
  エントリー価格 = T の終値（近似）。実際の T+1 始値はギャップで異なる。
  発注は一切しない（CLAUDE.md 絶対原則1・6）。
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace as _dc_replace
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
from config.settings import SwingReversionParams
from config.symbols import SYMBOLS
from data.daily_loader import build_daily_loader
from data.signal_store import ForwardSignal, SignalStore
from data.us_loader import load_spx_fresh
from notification.push_notifier import send_close_alert, send_signal_alert
from strategy.swing_reversion import compute_signals

logger = logging.getLogger(__name__)

# ── Phase 1 設定 ────────────────────────────────────────────────────────────────
# バックテスト ③④⑤ に対応（④ = Phase 0 PASS: n=310, DD=13.3%, E=+7.37%/trade）
_BASE = dict(
    lookback=20, entry_z=2.0, atr_stop_mult=2.0, max_holding_days=10,
    allow_long=True, allow_short=False,
    us_t1_crash_threshold=-0.02, us_t1_soft_min=-0.005, us_t1_soft_max=0.0,
    us_t0_crash_threshold=-0.02, us_t0_soft_min=-0.005, us_t0_soft_max=0.0,
    us_t0_recovery_min=0.01,
)
PHASE1_CONFIGS: dict[str, SwingReversionParams] = {
    "config_iii": SwingReversionParams(**_BASE, rsi_entry_max=30.0),
    "config_iv":  SwingReversionParams(**_BASE, rsi_entry_max=35.0),  # 推奨
    "config_v":   SwingReversionParams(**_BASE, rsi_entry_max=40.0),
}

# シグナル計算に十分な日数（lookback + ATR + RSI + 余裕）
_MIN_BARS = 50
# J-Quants 取得窓
_LOOKBACK_DAYS = 90


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Phase 1 日次シグナルスキャン")
    parser.add_argument(
        "--config", choices=list(PHASE1_CONFIGS), default="config_v",
        help="追跡するパラメータ設定（既定: config_v = ⑤推奨メイン）",
    )
    parser.add_argument(
        "--all-configs", action="store_true",
        help="③④⑤ 全設定を同時追跡",
    )
    parser.add_argument(
        "--db", default=str(_ROOT / "data/db/forward_signals.sqlite"),
        help="シグナル DB パス",
    )
    parser.add_argument(
        "--cache-dir", default=str(_ROOT / "data/db/daily_scan_cache"),
        help="J-Quants 日足キャッシュ dir",
    )
    parser.add_argument(
        "--min-interval", type=float, default=13.0,
        help="J-Quants API レート制限（Free=13.0, Light=1.0）",
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

    configs_to_run = PHASE1_CONFIGS if args.all_configs else {args.config: PHASE1_CONFIGS[args.config]}

    from_date = (today - timedelta(days=_LOOKBACK_DAYS)).isoformat()
    to_date = (today - timedelta(days=1)).isoformat()  # 前日まで（T = 昨日）

    # 1. S&P500 最新データ
    logger.info("S&P500 データ確認中...")
    try:
        us_df = load_spx_fresh(lookback_days=_LOOKBACK_DAYS + 30)
        logger.info("S&P500 最終日: %s", us_df.index[-1].date())
    except Exception as exc:
        logger.error("S&P500 取得失敗: %s — US フィルタなしで続行", exc)
        us_df = None

    # 2. 全銘柄リスト
    try:
        from config.symbols_cs import SYMBOLS_CS  # noqa: PLC0415
        all_symbols = list(dict.fromkeys(SYMBOLS + SYMBOLS_CS))
    except ImportError:
        all_symbols = list(SYMBOLS)
    logger.info("スキャン対象: %d 銘柄", len(all_symbols))

    # 3. 日足ローダー（差分更新キャッシュ）
    load_bars = build_daily_loader(
        from_date=from_date, to_date=to_date,
        cache_dir=args.cache_dir, min_interval=args.min_interval,
    )

    cost_model = CostModel(get_cost_params("mid"))

    with SignalStore(args.db) as store:
        # 4. オープンポジションの出口チェック
        open_sigs = store.get_open_signals()
        closed_records: list[dict] = []
        if open_sigs:
            logger.info("オープンシグナル %d 件の出口チェック中...", len(open_sigs))
            closed_count, closed_records = _check_exits(open_sigs, store, load_bars, cost_model, today, args.dry)
            if closed_count:
                logger.info("  → %d 件クローズ", closed_count)
                if not args.dry:
                    send_close_alert(closed_records)

        # 5. 新規シグナル検出（前日 T のデータ）
        all_new: dict[str, list[ForwardSignal]] = {}
        signal_date = to_date  # T = 昨日

        for config_name, params in configs_to_run.items():
            new_sigs = _detect_new_signals(
                all_symbols, params, us_df, load_bars, store,
                signal_date, today, config_name, args.dry,
            )
            all_new[config_name] = new_sigs

        # 6. LINE 通知（dry モードはスキップ）
        if not args.dry:
            send_signal_alert(signal_date, {k: v for k, v in all_new.items()})

        # 7. 月次 Professional プラン維持チェック（15日・25日）
        _check_monthly_plan_alert(store, today, dry=args.dry)

        # 8. 出力
        _print_report(all_new, store, signal_date, today)

    return 0


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
        send(
            title=f"⚠️ {ym} シグナルなし — プラン維持確認",
            body=msg,
            priority="high",
        )


# ── シグナル検出 ────────────────────────────────────────────────────────────────

def _detect_new_signals(
    symbols: list[str],
    params: SwingReversionParams,
    us_df: pd.DataFrame | None,
    load_bars,
    store: SignalStore,
    signal_date: str,
    entry_date: date,
    config_name: str,
    dry: bool,
) -> list[ForwardSignal]:
    """全銘柄をスキャンし、signal_date にシグナルが出た銘柄を返す。"""
    new_signals: list[ForwardSignal] = []

    for symbol in symbols:
        df = load_bars(symbol)
        if df.empty or len(df) < _MIN_BARS:
            continue

        signals = compute_signals(df, params, us_df=us_df)
        if signals.empty:
            continue

        # 最終バー（= signal_date T）のシグナルを確認
        last_entry = int(signals["entry"].iloc[-1])
        if last_entry != 1:  # long only（config は allow_short=False）
            continue

        actual_signal_date = signals.index[-1].strftime("%Y-%m-%d")
        if store.signal_exists(symbol, actual_signal_date, config_name):
            continue

        last_row = signals.iloc[-1]
        last_close = float(df["close"].iloc[-1])
        last_atr = float(last_row["atr"]) if not pd.isna(last_row["atr"]) else last_close * 0.02
        target = float(last_row["ma"])
        stop = last_close - last_atr * params.atr_stop_mult
        max_exit = (entry_date + timedelta(days=params.max_holding_days)).isoformat()

        sig = ForwardSignal(
            symbol=symbol,
            signal_date=actual_signal_date,
            side="long",
            entry_date=entry_date.isoformat(),
            entry_price=last_close,      # 翌日始値の近似（保守的バイアス）
            stop_price=stop,
            target_price=target,
            max_exit_date=max_exit,
            config_name=config_name,
        )

        if not dry:
            store.insert_signal(sig)
        new_signals.append(sig)

    return new_signals


# ── 出口チェック ────────────────────────────────────────────────────────────────

def _check_exits(
    open_sigs: list[dict],
    store: SignalStore,
    load_bars,
    cost_model: CostModel,
    today: date,
    dry: bool,
) -> tuple[int, list[dict]]:
    """オープンシグナルの出口条件を昨日の OHLC で判定してクローズする。"""
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

        df = load_bars(symbol)
        if df.empty:
            continue

        entry_ts = pd.Timestamp(entry_date_str)
        df_after = df[df.index >= entry_ts]
        if df_after.empty:
            continue

        for i, (dt, row) in enumerate(df_after.iterrows()):
            dt_str = dt.strftime("%Y-%m-%d")
            if dt_str > yesterday:
                break  # 本日以降はまだ確定していない

            # 出口条件（longs のみ）
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
                    price=entry_price, shares=1,
                    holding_days=holding, side="long",
                )
                net = gross - cost_frac

                if not dry:
                    store.close_signal(
                        sig["id"],
                        exit_date=dt_str,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        gross_return=gross,
                        net_return=net,
                    )
                logger.info(
                    "  クローズ: %s(%s) → %s @ %.0f  net %+.2f%%",
                    symbol, config_label(sig["config_name"]), exit_reason, exit_price, net * 100,
                )
                closed_records.append({**sig, "exit_reason": exit_reason, "net_return": net})
                closed += 1
                break

    return closed, closed_records


# ── レポート出力 ────────────────────────────────────────────────────────────────

def config_label(name: str) -> str:
    labels = {"config_iii": "③rsi<30", "config_iv": "④rsi<35", "config_v": "⑤rsi<40"}
    return labels.get(name, name)


def _print_report(
    all_new: dict[str, list[ForwardSignal]],
    store: SignalStore,
    signal_date: str,
    entry_date: date,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Phase 1 スキャン結果  シグナル日: {signal_date}  エントリー: {entry_date}")
    print(f"{'=' * 60}")

    total_new = 0
    for config_name, sigs in all_new.items():
        label = config_label(config_name)
        if sigs:
            print(f"\n【{label}】新規シグナル {len(sigs)} 件:")
            for s in sigs:
                print(
                    f"  {s.symbol:6s}  買い @ 本日始値(≈{s.entry_price:,.0f}円)  "
                    f"目標 {s.target_price:,.0f}  ストップ {s.stop_price:,.0f}  "
                    f"期限 {s.max_exit_date}"
                )
            total_new += len(sigs)
        else:
            print(f"\n【{label}】新規シグナルなし")

    print(f"\n── 累計 ({', '.join(config_label(k) for k in all_new)}) ──")
    print(f"  {store.summary()}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
