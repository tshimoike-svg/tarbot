#!/usr/bin/env python3
"""Phase 1 週次パフォーマンスレポートを生成して LINE・メールで送信する。

GitHub Actions の weekly_report.yml から呼び出される（金曜 6:00 AM JST）。
手動実行:
    uv run python scripts/weekly_report.py
    uv run python scripts/weekly_report.py --stdout  # 送信せず画面に表示だけ
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from data.signal_store import SignalStore
from notification.email_reporter import send_report
from notification.line_notifier import send as line_send

logger = logging.getLogger(__name__)

_DB = _ROOT / "data/db/forward_signals.sqlite"

# バックテスト基準値（config ④）
_BT_EXPECTANCY = 0.0737   # +7.37% / トレード
_BT_WIN_RATE   = 0.686    # 推定勝率（バックテスト実績から）


def _build_report(as_of: date) -> str:
    """レポートテキストを生成する。"""
    week_start = as_of - timedelta(days=6)  # 月〜金

    if not _DB.exists():
        return f"【Phase1 週次レポート】{week_start} 〜 {as_of}\n\nシグナルDBが見つかりません。"

    with SignalStore(_DB) as store:
        df = store.get_all()

    if df.empty:
        return f"【Phase1 週次レポート】{week_start} 〜 {as_of}\n\nデータなし（まだスキャン未実行）。"

    # 今週のシグナル（signal_date が今週）
    df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.date
    week_new = df[(df["signal_date"] >= week_start) & (df["signal_date"] <= as_of)]

    # 今週クローズ
    df_closed = df[df["status"] == "closed"].copy()
    df_closed["exit_date"] = pd.to_datetime(df_closed["exit_date"]).dt.date
    week_closed = df_closed[
        (df_closed["exit_date"] >= week_start) & (df_closed["exit_date"] <= as_of)
    ]

    # 全体累計（config ④ メイン）
    iv_closed = df_closed[df_closed["config_name"] == "config_iv"]
    open_ = df[df["status"] == "open"]

    lines = [
        f"【Phase1 週次レポート】{week_start} 〜 {as_of}",
        "=" * 48,
        "",
        f"■ 今週のシグナル（{len(week_new)} 件）",
    ]

    for cfg in ("config_iii", "config_iv", "config_v"):
        label = {"config_iii": "③rsi<30", "config_iv": "④rsi<35", "config_v": "⑤rsi<40"}[cfg]
        n = len(week_new[week_new["config_name"] == cfg])
        if n:
            syms = ", ".join(week_new[week_new["config_name"] == cfg]["symbol"].tolist())
            lines.append(f"  {label}: {n}件  [{syms}]")
        else:
            lines.append(f"  {label}: 0件")

    lines += ["", f"■ 今週クローズ（{len(week_closed)} 件）"]
    if week_closed.empty:
        lines.append("  なし")
    else:
        reason_map = {"target": "目標", "stop": "ストップ", "time_stop": "タイム"}
        for _, r in week_closed.iterrows():
            net = float(r["net_return"]) * 100
            reason = reason_map.get(str(r.get("exit_reason", "")), str(r.get("exit_reason", "")))
            lines.append(f"  {r['symbol']}（{r['config_name'].replace('config_','④' if r['config_name']=='config_iv' else r['config_name'][-3:])}）"
                         f" {reason}  {'+' if net>0 else ''}{net:.1f}%")

        avg_wk = week_closed["net_return"].mean() * 100
        lines.append(f"  → 今週平均: {'+' if avg_wk>0 else ''}{avg_wk:.2f}%")

    lines += ["", "■ 累計パフォーマンス（④ rsi<35 メイン）"]
    if iv_closed.empty:
        lines.append("  クローズトレードなし")
    else:
        avg_e = iv_closed["net_return"].mean() * 100
        wr = (iv_closed["net_return"] > 0).mean()
        wins = int((iv_closed["net_return"] > 0).sum())
        total_ret = iv_closed["net_return"].sum() * 100
        eq = iv_closed["net_return"].cumsum()
        peak = eq.cummax()
        dd = float((peak - eq).max()) * 100

        lines += [
            f"  期待値: {'+' if avg_e>0 else ''}{avg_e:.2f}%  （目標: +{_BT_EXPECTANCY*100:.2f}%）",
            f"  勝率:   {wr:.0%}  {wins}勝/{len(iv_closed)}件",
            f"  累計:   {'+' if total_ret>0 else ''}{total_ret:.1f}%",
            f"  最大DD: {dd:.1f}%",
        ]

        # Phase 0 ゲート再確認
        gate_ok = avg_e > 0 and dd < 15.0
        lines.append(f"  Phase0ゲート: {'✓ PASS' if gate_ok else '✗ 要確認'}")

        if avg_e < _BT_EXPECTANCY * 100 * 0.5:
            lines.append("  ⚠ 期待値がバックテストの50%未満 → パラメータ再評価を検討")

    lines += [
        "",
        f"■ 現在のオープンポジション: {len(open_)} 件",
    ]
    if not open_.empty:
        for _, r in open_.iterrows():
            lines.append(f"  {r['symbol']}（{r['config_name']}）エントリー {r['entry_date']} / 期限 {r['max_exit_date']}")

    lines += ["", "=" * 48]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Phase 1 週次レポート送信")
    parser.add_argument("--stdout", action="store_true", help="送信せず画面に表示だけ")
    parser.add_argument("--date", default=date.today().isoformat(), help="基準日 YYYY-MM-DD")
    args = parser.parse_args(argv)

    as_of = date.fromisoformat(args.date)
    report = _build_report(as_of)

    print(report)

    if not args.stdout:
        subject = f"【Phase1 週次レポート】{as_of}"
        sent_mail = send_report(subject, report)
        sent_line = line_send(report[:1000])  # LINE は 1000 文字制限を考慮
        if not sent_mail and not sent_line:
            logger.warning("メール・LINE ともに送信されませんでした（認証情報を .env に設定してください）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
