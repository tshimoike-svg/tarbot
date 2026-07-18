# Trading Bot

kabuステーションAPI を使った日本株（信用取引）**スイングトレード**ボット。
**速度では戦わない。** 容量制約ニッチを数日〜最大2週間保有で取り、コスト控除後にプラスの期待値を狙う。
日足で完結するため分足・ライブ常時接続は不要。

- 設計方針：[docs/trading_bot_design_v3.md](docs/trading_bot_design_v3.md)（現行。v3で日足スイングへ転換）
- 実装規約・常時コンテキスト：[CLAUDE.md](CLAUDE.md)
- 現在フェーズ：**Phase 0完了・Phase 1（ドライラン）稼働中**（フォワードで正式ゲート判定待ち。詳細は[CLAUDE.md](CLAUDE.md)・[docs/windows-handoff.md](docs/windows-handoff.md)）

## セットアップ

パッケージマネージャは [uv](https://docs.astral.sh/uv/) を使用（Python 3.11+ も uv が管理）。

```bash
# uv 未導入なら
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存導入（dev グループ含む。Python 3.11 は uv が自動取得）
uv sync

# テスト
uv run pytest

# バックテスト（Phase 0。strategy は swing_reversion / swing_momentum）
uv run python -m backtest.runner --strategy swing_reversion --symbols config/symbols.py
```

> `vectorbt`（backtest）・`streamlit`/`plotly`（dashboard）は optional グループ。
> numpy 等とのバージョン整合に注意が必要なため base から分離している。導入は
> `uv sync --extra backtest` / `uv sync --extra dashboard`。

## Phase 0 バックテスト（実データ）

無料の J-Quants 日足で、日足スイング2戦略のエッジを**コスト控除後**で検証できる。

```bash
# 1) J-Quants に無料登録（https://jpx-jquants.com/）。分足は不要・日足は無料プランで可
# 2) ダッシュボードで APIキーを発行し .env に置く（v2 は x-api-key 認証。.env は git 管理外）
cp .env.example .env        # JQUANTS_API_KEY=発行したキー を記入

# 3) 実行（両戦略を比較）
uv run python -m backtest.runner --compare --from 2024-06-01 --to 2026-06-01
# 単一戦略・銘柄指定も可
uv run python -m backtest.runner --strategy swing_reversion --symbols 7203,6758
```

`--symbols` 省略時は `config/symbols.py` の暫定ユニバースを使う（実運用前に流動性で絞った
銘柄群へ差し替えること）。出力はトレード数・**コスト控除後の期待値**（持ち越し金利込み）・
勝率/PF・最大DD・ウォークフォワード・Phase0 ゲート判定。

## 安全原則（詳細は CLAUDE.md）

- `config/settings.py` の `DRY_RUN` 既定は `True`。本番発注の不可逆操作は人間が実行する。
- 全発注は `strategy/risk_manager.py` を通す。
- バックテスト評価は必ずコスト控除後で行う。
