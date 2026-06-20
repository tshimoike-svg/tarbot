"""プッシュ通知（ntfy.sh / Telegram）で即時通知を送る。

【ntfy.sh — アカウント不要・最も簡単】
  1. スマホに ntfy アプリをインストール（iOS / Android）
  2. アプリ内で「Subscribe to topic」→ 好きなトピック名を入力（例: my-trading-bot-abc123）
  3. .env に設定:
       NTFY_TOPIC=my-trading-bot-abc123
  ※ トピック名は推測されないよう 末尾にランダム文字列を付けること

【Telegram — アカウント要・より確実】
  1. Telegram で @BotFather に /newbot → トークン取得
  2. 作ったボットに一度メッセージを送る
  3. https://api.telegram.org/bot<TOKEN>/getUpdates で chat_id を確認
  4. .env に設定:
       TELEGRAM_BOT_TOKEN=xxx
       TELEGRAM_CHAT_ID=123456789

どちらも未設定の場合はスキップ（クラッシュしない）。
"""

from __future__ import annotations

import logging
import os

import requests

__all__ = ["send", "send_signal_alert", "send_close_alert"]

logger = logging.getLogger(__name__)

_NTFY_BASE = "https://ntfy.sh"
_TG_BASE = "https://api.telegram.org"


# ── 共通インターフェース ──────────────────────────────────────────────────────────

def send(title: str, body: str, *, priority: str = "default") -> bool:
    """タイトル＋本文を利用可能な全チャンネルへ送信する。"""
    sent = False
    if _send_ntfy(title, body, priority=priority):
        sent = True
    if _send_telegram(f"*{title}*\n{body}"):
        sent = True
    if not sent:
        logger.debug("プッシュ通知: 設定済みチャンネルなし（NTFY_TOPIC / TELEGRAM_BOT_TOKEN）")
    return sent


# ── ntfy.sh ──────────────────────────────────────────────────────────────────────

_NTFY_PRIORITY = {"default": 3, "high": 4, "max": 5, "low": 2, "min": 1}


def _send_ntfy(title: str, body: str, *, priority: str = "default") -> bool:
    topic = os.getenv("NTFY_TOPIC")
    if not topic:
        return False
    try:
        resp = requests.post(
            _NTFY_BASE,
            json={
                "topic": topic,
                "title": title,
                "message": body,
                "priority": _NTFY_PRIORITY.get(priority, 3),
                "tags": ["chart_increasing"],
            },
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info("ntfy 通知送信成功 (topic=%s)", topic)
            return True
        logger.warning("ntfy 通知失敗 status=%d: %s", resp.status_code, resp.text[:100])
        return False
    except Exception as exc:
        logger.warning("ntfy 通知エラー: %s", exc)
        return False


# ── Telegram ─────────────────────────────────────────────────────────────────────

def _send_telegram(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    try:
        resp = requests.post(
            f"{_TG_BASE}/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info("Telegram 通知送信成功")
            return True
        logger.warning("Telegram 通知失敗 status=%d: %s", resp.status_code, resp.text[:200])
        return False
    except Exception as exc:
        logger.warning("Telegram 通知エラー: %s", exc)
        return False


# ── Phase 1 専用メッセージ ────────────────────────────────────────────────────────

def send_signal_alert(signal_date: str, new_by_config: dict[str, list]) -> bool:
    total = sum(len(v) for v in new_by_config.values())
    if total == 0:
        return False

    labels = {"config_iii": "③rsi<30", "config_iv": "④rsi<35", "config_v": "⑤rsi<40"}
    lines: list[str] = []
    for cfg, sigs in new_by_config.items():
        if not sigs:
            continue
        label = labels.get(cfg, cfg)
        lines.append(f"{label}: {len(sigs)}件")
        for s in sigs[:5]:
            lines.append(f"  {s.symbol}  目標{s.target_price:,.0f} / 損切{s.stop_price:,.0f}")
        if len(sigs) > 5:
            lines.append(f"  ...他{len(sigs)-5}件")

    return send(
        title=f"Phase1 シグナル {signal_date} ({total}件)",
        body="\n".join(lines),
        priority="high",
    )


def send_close_alert(closed_today: list[dict]) -> bool:
    if not closed_today:
        return False
    reason_map = {"target": "目標達成", "stop": "ストップ", "time_stop": "タイム"}
    lines: list[str] = []
    for c in closed_today[:8]:
        net = float(c.get("net_return", 0)) * 100
        reason = reason_map.get(c.get("exit_reason", ""), c.get("exit_reason", ""))
        sign = "+" if net > 0 else ""
        lines.append(f"{c['symbol']} {reason}  {sign}{net:.1f}%")

    return send(
        title=f"Phase1 クローズ ({len(closed_today)}件)",
        body="\n".join(lines),
        priority="default",
    )
