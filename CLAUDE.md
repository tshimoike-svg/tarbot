# CLAUDE.md — Trading Bot Project

> Claude Code が毎セッション参照する常時コンテキスト。
> 設計の全体像は `docs/trading_bot_design_v2.md`（v2設計書）を参照。本ファイルはその要約＋実装規約。

---

## プロジェクト概要

kabuステーションAPI を使った日本株（信用取引）デイトレードボット。
**速度では戦わない。** 容量制約ニッチ（機関が無視する中小型株の数秒〜数分の歪み）での、
規律ある頻回トレードにより、**コスト控除後でプラスの期待値**を積み上げることを狙う。

現状フェーズ：**Phase 0**（口座開通待ち。J-Quants過去データでエッジの有無を白黒つける段階）

---

## 絶対原則（違反する変更は提案しない）

1. **DRY_RUN ファースト**：`config/settings.py` の `DRY_RUN` フラグを発注前に必ず確認する。既定値は `True`。`False` へ切り替えるコードを勝手に書かない。
2. **全発注は `risk_manager.py` を通す**：成行・指値・取消を問わず、発注系の処理は例外なくリスク管理の関門を経由する。バイパスする実装を作らない。
3. **期待値ゲート**：「1トレードの期待値がコスト控除後でプラス」でない戦略は稼働対象にしない。バックテスト評価は**必ずコスト控除後**で行う。
4. **テストなしで触らない**：`order_engine.py` / `risk_manager.py` / `cost_model.py` は対応するテストを伴わずに変更しない。
5. **板情報・歩み値をシグナルの主軸にしない**：バックテスト不能なため。使う場合も L1 の補助フィルタに限定し、検証はドライランのみ。
6. **本物の発注を伴う操作は人間に確認を求める**：API接続・本番発注・資金に関わる不可逆操作は、コードを書いても実行は人間が行う前提で進める。

---

## よく使うコマンド（想定規約）

パッケージマネージャは **uv**（Python 3.11+ も uv が管理）。`pip` は使わない。

```bash
# 環境構築（uv 未導入なら: curl -LsSf https://astral.sh/uv/install.sh | sh）
uv sync                                       # base + dev グループを導入（Python 3.11 は uv が自動取得）
uv sync --extra backtest                      # vectorbt 等（optional・numpy との整合に注意）
uv sync --extra dashboard                     # streamlit / plotly（Phase 1〜）

# テスト（コマンド頭に uv run を付ける）
uv run pytest                                 # 全テスト
uv run pytest tests/test_risk_manager.py -v   # リスク管理だけ
uv run pytest tests/test_cost_model.py -v     # コストモデルだけ

# バックテスト（Phase 0 の主役）
uv run python -m backtest.runner --strategy mean_reversion --symbols config/symbols.py

# ダッシュボード（Phase 1〜）
uv run streamlit run dashboard/app.py

# ドライラン（Phase 1〜、発注なし）
uv run python -m execution.dry_run

# 依存追加 / リント・型チェック
uv add <package>                              # base へ追加（uv add --dev <pkg> で dev グループ）
uv run ruff check . && uv run mypy .
```

> 依存は `pyproject.toml` で管理し、`uv.lock` で再現性を固定する（両方コミットする）。
> 指標（VWAP/ATR/BB）は numpy 2.x で壊れる `pandas-ta` を避け、`strategy/indicators.py` に自前実装する方針。

---

## コード規約

- Python 3.11 以上。型ヒントを付ける。
- 設定値・しきい値・パラメータはコードにハードコードせず `config/` に集約する。
  - 売買パラメータ → `config/settings.py`
  - 監視銘柄・流動性フィルタ → `config/symbols.py`
  - コスト想定（スプレッド/滑り/信用コスト）→ `config/costs.py`
- 金額・株価は浮動小数点の丸め誤差に注意（必要なら `Decimal`）。
- 時刻は JST 固定、取引時間（前場・後場・寄り・引け）の境界を明示的に扱う。
- ログは構造化して残す（戦略名・シグナル・約定/不約定・滑り・コスト）。あとで期待値検証に使う。
- 外部API（kabu / J-Quants）呼び出しはリトライ・タイムアウト・レート制限を実装する。

---

## アーキテクチャ（どこに何があるか）

```
config/      設定・パラメータ・コスト想定
data/        データ取得（J-Quants過去データ / kabu WebSocket）・DB保存
strategy/    指標・シグナル生成・リスク管理
  mean_reversion.py  ← 【主力・トラックA】日中平均回帰（バックテスト可能）
  event_reaction.py  ← 【トラックB】TDnetイベント反応（ドライラン専用）
  risk_manager.py    ← 全発注が通る関門
execution/   執行（指値中心）・約定率/滑り計測・ドライラン
backtest/    バックテスト実行・コストモデル・バイアスチェック付き評価
notification/ LINE Messaging API・メール通知
dashboard/   Streamlit + Plotly
tests/       各モジュールのテスト
```

検証は2トラックに分離する：
- **トラックA（バックテスト可能）**：平均回帰・寄り後パターン・相対強弱 → OHLCVで過去検証
- **トラックB（バックテスト不能）**：イベント反応・L1フィルタ → ドライランでのみ検証

---

## ドメイン制約（環境固有・忘れやすい）

- kabuステーションAPI はローカルホスト経由（`http://localhost:18080/kabusapi`）。
- **kabuステーション（Windows GUIアプリ）が起動・ログイン中でないとAPIは使えない**。本番VPSも Windows 必須。
- 手数料無料は **SOR注文選択が条件**（2026年5月18日以降）。APIからSORが指定できるか・選ばないと手数料が出るかは要確認事項（未確定）。
- **Professionalプランはほぼ毎月の取引で維持**。切れるとAPIが停止する。月次で適用状況を監視する処理を入れる。
- 信用の金利・貸株料は手数料無料後も残る。当日返済前提でも持ち越し時は効く。
- 通知は **LINE Messaging API**（LINE Notify は終了済み）またはメール。

---

## 現在のタスク：Phase 0（最優先・口座不要）

**ゴール**：J-Quants の過去データで日中平均回帰戦略を1つ実装し、
**コスト控除後・ウォークフォワードで明確な正の期待値が出るか**を検証する。
ここを越えられないなら後続（ML・イベント・実取引）は全て無意味。最優先で白黒つける。

### 着手順序
1. `config/costs.py` と `backtest/cost_model.py` を**先に**作る
   - 往復コスト ＝ 実効スプレッド ＋ 滑り ＋ 信用コスト（持ち越し時）を保守的に見積もる
   - スプレッドは銘柄ごとに異なる前提でモデル化（中小型は広い）
2. `data/fetcher.py` で J-Quants の OHLCV（分足）と**上場廃止銘柄情報**を取得
3. `strategy/indicators.py` に VWAP・ATR・ボリンジャーバンドを実装
4. `strategy/mean_reversion.py` を実装（初期仮説）
   - エントリー：VWAP（または移動平均）からの行き過ぎ（例：-Nσ）で逆張りの指値
   - イグジット：VWAP回帰で利確 ／ ATRベースの損切り（例：1.5×ATR）
   - 当日中に強制クローズ
   - パラメータは `config/settings.py` に置き、ハードコードしない
5. `backtest/runner.py` + `backtest/evaluator.py` で検証

### evaluator.py の必須チェック（§7のバイアス対策）
- [ ] ルックアヘッド・バイアスがないか（シグナル時点で未来値を見ていない）
- [ ] サバイバーシップ・バイアス対策（廃止・併合・分割銘柄を含む）
- [ ] **コスト控除後**の損益で評価しているか
- [ ] 指値の不約定・成行の滑りをモデル化しているか
- [ ] マーケットインパクト（自注文が薄い板を動かす）を考慮しているか
- [ ] イン/アウトオブサンプルを分離しているか
- [ ] トレード数が統計的に十分か（目安：数百回以上）

### Phase 0 の完了条件（Definition of Done）
```
- コスト控除後 E[1トレード] > 0
- ウォークフォワードの各区間で安定（特定期間依存でない）
- 最大ドローダウン < 15%
- 統計的に十分なトレード数
→ 満たせば Phase 1（ドライラン）へ。満たせなければ戦略を見直すか撤退する。
```

> Phase 0 を越える前に ML / LLM / 強化学習 には着手しない（§9「凍結と解凍条件」）。

---

## やらないこと（スコープ外・凍結）

- 板情報・歩み値を主軸にした低レイテンシ戦略（インフラ上勝てない）
- 機械学習・LLM感情分析・強化学習（Phase 0 のエッジ確認まで凍結）
- 米国株の自動発注（APIの機能制限あり。国内株を優先）

---

## 重要な前提（開発者の心構え）

- 大半の個人アルゴ・デイトレードはコストに負ける。本プロジェクトは「エッジがあれば抽出し、なければ安く早く見切る」ためのもの。利益は保証されない。
- 頻回トレードの最大の敵は**スプレッドと滑り**。期待値計算でここを甘く見ると本番で必ず負ける。
- バックテストの良成績は楽観バイアスの塊になりやすい。常に保守的に見積もる。
