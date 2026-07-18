# Windows引き継ぎドキュメント（2026-07-18作成）

kabuステーションAPI（本番/検証とも疎通確認済み）はWindows上のkabuステーション起動が
前提のため、以降の実発注関連の開発はWindows側のClaude Codeに引き継ぐ。
戦略・バックテスト・ダッシュボード等の開発は引き続きMac側でも進めてよい
（**git経由でどちらからでも同じリポジトリを触る**。ファイルを手動コピーする必要はない）。

## 現在の状態（要約）

- **Phase 0（エッジ検証）完了**：⑤config_v（lb=20, z≥2.0, RSI<40, US T-1/T0フィルタ）で
  E+5.65%/WR74%/n=419のバックテスト結果。ただし直近OOS（2026-04〜05）でE+0.81%まで減衰
  （詳細: `docs/results/oos_eval_2026-06-25.md`）。**Phase2移行はまだ正式ゲート未クリア**。
- **Phase 1（ドライラン）稼働中**：`scripts/daily_scan.py`がGitHub Actionsで平日朝
  自動実行し、`data/db/forward_signals.sqlite`にシグナル/クローズを蓄積中（実発注なし）。
- **ダッシュボード公開済み**：https://tarbot-n7kcur5g6wo8ctrrb7mbab.streamlit.app/
  （Streamlit Community Cloud、GitHub push検知で自動更新）。
- **証券口座**：三菱UFJ eスマート証券（旧auカブコム証券）で口座開設・信用取引・
  kabuステーションAPI利用が全て完了。本番/検証ともAPIパスワード設定済み、
  `scripts/kabu_ping.py`での疎通確認までは実施予定（Windows側でこれから）。

## 絶対原則（CLAUDE.mdより・Windows側でも厳守）

1. `DRY_RUN`既定はTrue。Falseへ切り替えるコードを勝手に書かない。
2. 全発注は`risk_manager.py`を経由する。バイパス実装を作らない。
3. 期待値がコスト控除後でプラスでない戦略は稼働対象にしない。
4. `order_engine.py`/`risk_manager.py`/`cost_model.py`はテストなしで変更しない。
5. 板情報・歩み値をシグナルの主軸にしない。
6. **本物の発注を伴う操作は人間が実行する**。Claudeはコードを書くが、実際のAPI発注
   ボタンを押す・実行するのは常にユーザー自身。

## Windows側のセットアップ手順

```powershell
# 1. uv インストール（未導入なら）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. リポジトリを clone（Macで作業中のものと同じGitHubリポジトリ）
git clone https://github.com/tshimoike-svg/tarbot.git
cd tarbot

# 3. 依存関係インストール
uv sync

# 4. .env を作成（.env は git管理外なので、Mac版からコピーではなく新規に値を入れる）
copy .env.example .env
```

`.env`に設定する値：

| 変数 | 用途 | 備考 |
|---|---|---|
| `KABU_API_PASSWORD_PROD` | kabuステーションAPI 本番用パスワード | システム設定「APIタブ」で登録した値 |
| `KABU_API_PASSWORD_DEMO` | kabuステーションAPI 検証用パスワード | 同上（本番とは別パスワード） |
| `JQUANTS_API_KEY` | J-Quants（過去データ・研究用） | ローカルでバックテストを回す場合のみ必要 |
| `DRY_RUN` | 発注フラグ | **設定しない（既定Trueのまま）**。コメントアウトを外さない |

GitHub Actions側のシークレット（NTFY_TOPIC等）はリポジトリ設定側にあり、Windows側の
`.env`には不要（daily_scanはGitHub Actions上で動いておりMac/Windowsどちらのローカル
環境にも依存しない）。

## 起動確認（Windows側でまずやること）

1. kabuステーションを起動・ログイン
2. 疎通確認：
   ```powershell
   uv run python scripts/kabu_ping.py --env demo   # 検証環境（固定値応答・安全）
   uv run python scripts/kabu_ping.py --env prod   # 本番環境（実残高を読むだけ、発注なし）
   ```
3. 両方成功すれば、トークン発行・残高照会・板情報取得の一連の疎通がOK。

## Windows Claude Codeへの最初の指示（コピペ用）

```
このリポジトリ（tarbot）はMac側のセッションから引き継いだ日本株スイングトレードボット。
CLAUDE.md（絶対原則・アーキテクチャ）と docs/windows-handoff.md を読んで現状を把握して。

直近やりたいこと:
1. scripts/kabu_ping.py --env demo / --env prod で疎通確認（kabuステーション起動済み・
   APIパスワードは.envに設定済み）。エラーが出たら原因を調査して。
2. 疎通確認が取れたら、次にexecution/order_engine.py（未実装）の設計を一緒に考えたい。
   ただしCLAUDE.md絶対原則の通り、実発注は一切書かない/実行しない前提で、まずは
   risk_manager.py 経由の設計だけ相談させて。
```

## 今後の役割分担（提案）

- **Mac**：戦略検証・バックテスト・ダッシュボード・フォワードOOSの分析（今まで通り）
- **Windows**：kabuステーションAPI連携・`order_engine.py`実装・実際のAPI接続確認
- 同期は常に`git push`/`git pull`。作業前に必ず`git pull`してから始める（コンフリクト防止）。
