"""SMTP メールで週次レポートを送る。

必要な環境変数（.env に記載）:
  SMTP_HOST      : SMTP サーバー（省略時 smtp.gmail.com）
  SMTP_PORT      : ポート番号（省略時 587）
  SMTP_USER      : 送信元メールアドレス（Gmail の場合はアカウント）
  SMTP_PASS      : Gmail はアプリパスワード（通常パスワード不可）
                   取得: Google アカウント → セキュリティ → 2段階認証 → アプリパスワード
  REPORT_TO_EMAIL: 送信先（省略時は SMTP_USER と同じ）

認証情報未設定時はログ警告のみでクラッシュしない。
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

__all__ = ["send_report"]

logger = logging.getLogger(__name__)


def send_report(subject: str, body: str) -> bool:
    """テキスト形式のレポートメールを送る。認証情報未設定時は False を返す。"""
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    to_addr = os.getenv("REPORT_TO_EMAIL") or smtp_user

    if not smtp_user or not smtp_pass:
        logger.debug("SMTP 認証情報未設定（SMTP_USER / SMTP_PASS）。メール送信をスキップ")
        return False

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_addr
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(smtp_user, [to_addr], msg.as_string())
        logger.info("週次レポートメール送信成功 → %s", to_addr)
        return True
    except Exception as exc:
        logger.warning("メール送信失敗: %s", exc)
        return False
