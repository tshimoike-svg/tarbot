# Trading Bot

kabuステーションAPI を使った日本株（信用取引）**スイングトレード**ボット。
**速度では戦わない。** 容量制約ニッチを数日〜最大2週間保有で取り、コスト控除後にプラスの期待値を狙う。
日足で完結するため分足・ライブ常時接続は不要。

- 設計方針：[docs/trading_bot_design_v3.md](docs/trading_bot_design_v3.md)（現行。v3で日足スイングへ転換）
- 実装規約・常時コンテキスト：[CLAUDE.md](CLAUDE.md)
- 現在フェーズ：**Phase 0**（J-Quants 日足で日足スイング戦略のエッジを検証。口座不要）

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

## 安全原則（詳細は CLAUDE.md）

- `config/settings.py` の `DRY_RUN` 既定は `True`。本番発注の不可逆操作は人間が実行する。
- 全発注は `strategy/risk_manager.py` を通す。
- バックテスト評価は必ずコスト控除後で行う。
