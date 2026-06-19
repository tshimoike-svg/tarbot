# CLAUDE.md — Trading Bot Project

> Claude Code が毎セッション参照する常時コンテキスト。
> 設計の全体像は `docs/trading_bot_design_v3.md`（v3設計書・現行）を参照。本ファイルはその要約＋実装規約。
> v3 で時間軸を「日中（分足）」から「数日〜2週間のスイング（日足）」へ転換した。

---

## プロジェクト概要

kabuステーションAPI を使った日本株（信用取引）**スイングトレード**ボット。
**速度では戦わない。** 容量制約ニッチ（機関が無視する中小型株の歪み）を、
**数日〜最大2週間保有**の規律あるトレードで取り、**コスト控除後でプラスの期待値**を狙う。
日足で完結するため、分足・ライブ常時接続は不要（J-Quants 日足で過去検証できる）。

現状フェーズ：**Phase 0**（J-Quants日足で日足スイング戦略のエッジを白黒つける。口座不要）

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

# バックテスト（Phase 0 の主役。strategy は swing_reversion / swing_momentum）
uv run python -m backtest.runner --strategy swing_reversion --symbols config/symbols.py

# ダッシュボード（Phase 1〜）
uv run streamlit run dashboard/app.py

# ドライラン（Phase 1〜、発注なし）
uv run python -m execution.dry_run

# 依存追加 / リント・型チェック
uv add <package>                              # base へ追加（uv add --dev <pkg> で dev グループ）
uv run ruff check . && uv run mypy .
```

> 依存は `pyproject.toml` で管理し、`uv.lock` で再現性を固定する（両方コミットする）。
> 指標（移動平均/ATR/zスコア等）は numpy 2.x で壊れる `pandas-ta` を避け、`strategy/indicators.py` に自前実装する方針。

---

## コード規約

- Python 3.11 以上。型ヒントを付ける。
- 設定値・しきい値・パラメータはコードにハードコードせず `config/` に集約する。
  - 売買パラメータ → `config/settings.py`
  - 監視銘柄・流動性フィルタ → `config/symbols.py`
  - コスト想定（スプレッド/滑り/信用コスト）→ `config/costs.py`
- 金額・株価は浮動小数点の丸め誤差に注意（必要なら `Decimal`）。
- 時刻は JST 固定。日足の日付境界・約定は翌営業日の寄りを基準にする。
- ログは構造化して残す（戦略名・シグナル・約定/不約定・滑り・コスト）。あとで期待値検証に使う。
- 外部API（kabu / J-Quants）呼び出しはリトライ・タイムアウト・レート制限を実装する。

---

## アーキテクチャ（どこに何があるか）

```
config/      設定・パラメータ・コスト想定
data/        データ取得（J-Quants日足 / kabu WebSocket）・DB保存
strategy/    指標・シグナル生成・リスク管理
  trade.py           ← 共有 Trade 型
  indicators.py      ← 移動平均/ATR/zスコア等（自前実装）
  swing.py           ← 日足スイング共通エンジン（保有・出口の状態機械）
  swing_reversion.py ← 【トラックA】日足平均回帰（押し目買い/戻り売り）
  swing_momentum.py  ← 【トラックA】日足ブレイクアウト（モメンタム）
  position_sizer.py  ← リスク基準のサイジング
  risk_manager.py    ← 全発注が通る関門
  event_reaction.py  ← 【トラックB・未】TDnetイベント反応（ドライラン専用）
execution/   fill_monitor.py（約定率/滑り実測）・dry_run.py（イベント駆動）・order_engine.py(未)
backtest/    cost_model.py・evaluator.py・runner.py(未)
notification/ LINE Messaging API・メール通知（未）
dashboard/   Streamlit + Plotly（未）
tests/       各モジュールのテスト
```

検証は2トラックに分離する：
- **トラックA（バックテスト可能）**：日足スイング平均回帰・モメンタム → 日足OHLCVで過去検証
- **トラックB（バックテスト不能）**：イベント反応・L1フィルタ → ドライランでのみ検証

---

## ドメイン制約（環境固有・忘れやすい）

- kabuステーションAPI はローカルホスト経由（`http://localhost:18080/kabusapi`）。
- **kabuステーション（Windows GUIアプリ）が起動・ログイン中でないとAPIは使えない**。本番VPSも Windows 必須。
- 手数料無料は **SOR注文選択が条件**（2026年5月18日以降）。APIからSORが指定できるか・選ばないと手数料が出るかは要確認事項（未確定）。
- **Professionalプランはほぼ毎月の取引で維持**。切れるとAPIが停止する。月次で適用状況を監視する処理を入れる。
- 信用の金利・貸株料は手数料無料後も残る。**スイングは数日〜2週間持ち越すので必ず効く**（cost_model が holding_days で計上）。オーバーナイトのギャップリスクにも注意。
- 通知は **LINE Messaging API**（LINE Notify は終了済み）またはメール。

---

## 現在のタスク：Phase 0（最優先・口座不要）

**ゴール**：J-Quants の**日足**で日足スイング戦略（平均回帰・モメンタム）を検証し、
**コスト控除後・ウォークフォワードで明確な正の期待値が出るか**を白黒つける。
ここを越えられないなら後続（ML・イベント・実取引）は全て無意味。最優先で見切る。

### 進捗（実装済み）
- `config/costs.py` / `backtest/cost_model.py`（往復コスト＝スプレッド＋滑り＋**保有日数ぶんの金利**）
- `data/fetcher.py`（J-Quants 日足・調整後価格・廃止銘柄込みユニバース）
- `strategy/indicators.py`（移動平均/ATR/zスコア等・自前・因果性テスト済み）
- `strategy/swing_reversion.py` / `swing_momentum.py` ＋ 共通エンジン `swing.py`
- `strategy/risk_manager.py`（関門）/ `position_sizer.py`（リスク基準サイジング）
- `backtest/evaluator.py`（コスト控除後・ウォークフォワード・Phase0ゲート判定）
- `execution/fill_monitor.py`・`dry_run.py`（イベント駆動・発注なし）/ `data/storage.py`

### 残作業
1. `backtest/runner.py`：複数銘柄・期間を束ね、両戦略を回して比較する司令塔
2. 実データ投入：`.env` に J-Quants 認証を置き、無料日足で**本物のバックテスト**を実行
3. 両戦略のコスト控除後の期待値・ウォークフォワード安定性を比較して go/no-go

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

- 大半の個人アルゴはコストとギャップに負ける。本プロジェクトは「エッジがあれば抽出し、なければ安く早く見切る」ためのもの。利益は保証されない。
- スイングの敵は**持ち越し金利＋オーバーナイトギャップ**（日中の最大の敵だったスプレッド・滑りは相対的に軽くなるが、依然コストとして必ず計上する）。期待値計算でここを甘く見ると本番で必ず負ける。
- バックテストの良成績は楽観バイアスの塊になりやすい。常に保守的に見積もる。
