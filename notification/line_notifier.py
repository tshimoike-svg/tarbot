"""LINE Messaging API でシグナル通知を送る。

必要な環境変数（.env に記載）:
  LINE_CHANNEL_ACCESS_TOKEN : チャンネルアクセストークン（Messaging API）
  LINE_USER_ID              : 送信先ユーザー ID（LINE Developers で確認）

未設定の場合はログ警告のみでクラッシュしない（フォールトトレラント）。
"""

from __future__ import annotations

import logging
import os

import requests

__all__ = ["send", "send_signal_alert", "send_close_alert"]

logger = logging.getLogger(__name__)

_PUSH_URL = "https://api.line.me/v2/bot/message/push"


def send(text: str) -> bool:
    """任意のテキストを LINE に送る。認証情報未設定時は False を返す。"""
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    user_id = os.getenv("LINE_USER_ID")
    if not token or not user_id:
        logger.debug("LINE 認証情報未設定（LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID）。通知をスキップ")
        return False
    try:
        resp = requests.post(
            _PUSH_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"to": user_id, "messages": [{"type": "text", "text": text}]},
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info("LINE 通知送信成功")
            return True
        logger.warning("LINE 通知失敗 status=%d: %s", resp.status_code, resp.text[:200])
        return False
    except Exception as exc:
        logger.warning("LINE 通知エラー: %s", exc)
        return False


def send_signal_alert(signal_date: str, new_by_config: dict[str, list]) -> bool:
    """新規シグナル検出時の通知メッセージを組み立てて送る。"""
    total = sum(len(v) for v in new_by_config.values())
    if total == 0:
        return False

    lines = [f"【Phase1 シグナル】{signal_date}", ""]
    labels = {"config_iii": "③rsi<30", "config_iv": "④rsi<35(推奨)", "config_v": "⑤rsi<40"}
    for cfg, sigs in new_by_config.items():
        if not sigs:
            continue
        label = labels.get(cfg, cfg)
        lines.append(f"{label}: {len(sigs)} 件")
        for s in sigs[:5]:  # 最大5件まで
            lines.append(f"  {s.symbol} 目標{s.target_price:,.0f} / 損切{s.stop_price:,.0f}")
        if len(sigs) > 5:
            lines.append(f"  ...他 {len(sigs)-5} 件")

    return send("\n".join(lines))


def send_close_alert(closed_today: list[dict]) -> bool:
    """当日クローズしたポジションの通知。"""
    if not closed_today:
        return False
    labels = {"config_iii": "③", "config_iv": "④", "config_v": "⑤"}
    lines = [f"【Phase1 クローズ】{len(closed_today)} 件", ""]
    for c in closed_today[:8]:
        sign = "+" if c["net_return"] > 0 else ""
        reason_map = {"target": "目標達成", "stop": "ストップ", "time_stop": "タイム"}
        reason = reason_map.get(c.get("exit_reason", ""), c.get("exit_reason", ""))
        cfg = labels.get(c.get("config_name", ""), "")
        lines.append(f"  {c['symbol']}{cfg} {reason} {sign}{c['net_return']*100:.1f}%")
    return send("\n".join(lines))
