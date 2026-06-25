#!/usr/bin/env python3
"""バックテスト2年分の個別トレードを CSV に書き出す。

ダッシュボードの「バックテストシミュレーション」モードで読み込む。

実行:
    uv run python scripts/simulate_leverage.py
    uv run python scripts/simulate_leverage.py --configs iii iv v
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from backtest.cost_model import CostModel
from backtest.evaluator import compute_trade_results
from config.costs import get_cost_params
from config.settings import SwingReversionParams
from config.symbols import SYMBOLS
from data.us_loader import load_spx
from strategy.swing_reversion import generate_trades

logger = logging.getLogger(__name__)

_CACHE_DIR = _ROOT / "data/db/bars_cache"
_OUT_CSV   = _ROOT / "data/db/bt_trades.csv"

_BASE = dict(
    lookback=20, entry_z=2.0, atr_stop_mult=2.0, max_holding_days=10,
    allow_long=True, allow_short=False,
    us_t1_crash_threshold=-0.02, us_t1_soft_min=-0.005, us_t1_soft_max=0.0,
    us_t0_crash_threshold=-0.02, us_t0_soft_min=-0.005, us_t0_soft_max=0.0,
    us_t0_recovery_min=0.01,
)

# ショート専用ベース設定
# 設計思想: T-1 US 強い（+0.5%+）→ 日本株が US 追い風で上昇→過熱
#           T0 US 平坦/弱（≤0%）→ 追い風消滅 → 日本株が平均回帰しやすい
_SHORT_BASE = dict(
    lookback=20, atr_stop_mult=2.5, max_holding_days=7,  # ショートは短め保有・広めストップ
    allow_long=False, allow_short=True,
    us_t1_short_strength_min=0.005,  # T-1 US >= +0.5% でのみショート
    us_t0_short_weakness_max=0.0,    # T0 US <= 0% でのみショート
)

# Tier 2 補助設定 — メイン（priority=1）がスロットを埋めきれない日に入る
# 設計思想:
#   メインが入れない理由は主に 2 つ:
#     a) US フィルタ不通過（US が「クラッシュでも軟調でも回復でもない」平常日）
#     b) z スコアが -2.0 未満でない（穏やかな押し目）
#   補助 A: US フィルタを外し z 閾値を 1.5 に下げ → 両方のギャップを補完
#   補助 B: ルックバック 30 日 → 20 日 MA と異なるタイミングでシグナル
_TIER2_BASE = dict(
    lookback=20, entry_z=1.5, atr_stop_mult=1.5, max_holding_days=7,
    allow_long=True, allow_short=False,
    rsi_entry_max=45.0,
    # US フィルタなし（省略 = nan = 全日エントリー可）
)
_TIER2_LB30 = dict(
    lookback=30, entry_z=2.0, atr_stop_mult=2.0, max_holding_days=10,
    allow_long=True, allow_short=False,
    rsi_entry_max=40.0,
    # US フィルタなし
)

CONFIGS: dict[str, SwingReversionParams] = {
    # Tier 1 ロング（メイン・priority=1）
    "config_iii": SwingReversionParams(**_BASE, rsi_entry_max=30.0),
    "config_iv":  SwingReversionParams(**_BASE, rsi_entry_max=35.0),
    "config_v":   SwingReversionParams(**_BASE, rsi_entry_max=40.0),
    # Tier 1 ショート（priority=1）
    "config_sa":  SwingReversionParams(**_SHORT_BASE, entry_z=2.5, rsi_entry_min=70.0),
    "config_sb":  SwingReversionParams(**_SHORT_BASE, entry_z=2.0, rsi_entry_min=65.0),
    "config_sc":  SwingReversionParams(**_SHORT_BASE, entry_z=3.0, rsi_entry_min=75.0),
    "config_sd":  SwingReversionParams(
        lookback=20, entry_z=2.5, atr_stop_mult=2.5, max_holding_days=7,
        allow_long=False, allow_short=True, rsi_entry_min=70.0,
    ),
    # Tier 2 補助ロング（priority=2 — 空きスロット埋め）
    "config_t2a": SwingReversionParams(**_TIER2_BASE),          # z≥1.5, RSI<45, USフィルタなし（実験的）
    "config_t2b": SwingReversionParams(**_TIER2_LB30),          # lb=30, z≥2.0, RSI<40, USフィルタなし
    "config_t2c": SwingReversionParams(**{**_BASE, "lookback": 30, "rsi_entry_max": 40.0}),  # lb=30, USフィルタあり（T1同品質・別タイミング）
}

# スロット優先度（1=メイン・高確度、2=補助・低確度）
# portfolio_simulation では priority=1 が先にスロットを獲得し、残りを priority=2 が埋める
_PRIORITY: dict[str, int] = {
    "config_iii": 1, "config_iv": 1, "config_v": 1,
    "config_sa": 1, "config_sb": 1, "config_sc": 1, "config_sd": 1,
    "config_t2a": 2, "config_t2b": 2, "config_t2c": 2,
}


def _find_cache(symbol: str, cache_dir: Path = _CACHE_DIR) -> Path | None:
    """銘柄コードに対応するキャッシュファイルを探す（期間は問わない）。"""
    candidates = sorted(cache_dir.glob(f"{symbol}_*.pkl"), reverse=True)
    return candidates[0] if candidates else None


def _load_cached(symbol: str, cache_dir: Path = _CACHE_DIR) -> pd.DataFrame:
    pkl = _find_cache(symbol, cache_dir)
    if pkl is None:
        return pd.DataFrame()
    with pkl.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="バックテスト2年シミュレーション → CSV出力")
    parser.add_argument(
        "--configs", nargs="+",
        choices=["iii", "iv", "v", "sa", "sb", "sc", "sd", "t2a", "t2b", "t2c"],
        default=["iii", "iv", "v", "sa", "sb", "sc", "sd", "t2a", "t2b", "t2c"],
        help="対象設定（省略時: 全設定）",
    )
    parser.add_argument(
        "--expand", action="store_true",
        help="symbols_300.py を追加して拡張ユニバース（420銘柄）で実行",
    )
    parser.add_argument(
        "--cache-dir", default=str(_CACHE_DIR),
        help="日足キャッシュ dir（既定: bars_cache = J-Quants 2年）",
    )
    parser.add_argument(
        "--out", default=str(_OUT_CSV),
        help="トレード CSV 出力先",
    )
    args = parser.parse_args(argv)
    cache_dir = Path(args.cache_dir)
    out_csv = Path(args.out)

    selected = {f"config_{k}": CONFIGS[f"config_{k}"] for k in args.configs}

    # シンボルリスト（既定: SYMBOLS + SYMBOLS_CS の 180 銘柄、--expand で +symbols_300）
    raw: list[str] = [str(s) for s in SYMBOLS]
    try:
        from config.symbols_cs import SYMBOLS_CS  # noqa: PLC0415
        raw += [str(s) for s in SYMBOLS_CS]
    except ImportError:
        pass
    if args.expand:
        try:
            from config.symbols_300 import SYMBOLS as SYMBOLS_300  # noqa: PLC0415
            raw += [str(s) for s in SYMBOLS_300]
        except ImportError:
            logger.warning("config/symbols_300.py が見つかりません。--expand を無視します")
    # 重複除去・キャッシュ確認
    seen: set[str] = set()
    all_symbols = []
    for s in raw:
        if s not in seen and _find_cache(s, cache_dir) is not None:
            seen.add(s)
            all_symbols.append(s)
    logger.info("対象銘柄: %d 件%s", len(all_symbols), "（拡張420）" if args.expand else "")

    # US データ（キャッシュ or yfinance）
    us_df: pd.DataFrame | None = None
    try:
        from data.us_loader import load_spx_fresh  # noqa: PLC0415
        us_df = load_spx_fresh(lookback_days=800)
        logger.info("S&P500 取得: %d 行", len(us_df))
    except Exception as exc:
        logger.warning("S&P500 取得失敗: %s — US フィルタなしで続行", exc)

    cost_model = CostModel(get_cost_params("mid"))

    rows: list[dict] = []

    _labels = {
        "config_iii": "③rsi<30(L,T1)",
        "config_iv":  "④rsi<35(L,T1)",
        "config_v":   "⑤rsi<40(L,T1)",
        "config_sa":  "SA_short(z2.5,rsi>70)",
        "config_sb":  "SB_short(z2.0,rsi>65)",
        "config_sc":  "SC_short(z3.0,rsi>75)",
        "config_sd":  "SD_short(z2.5,noUS)",
        "config_t2a": "補助A(z1.5,rsi<45,noUS)",
        "config_t2b": "補助B(lb30,z2.0,noUS)",
        "config_t2c": "補助C(lb30,z2.0,rsi<40,USあり)",
    }

    for config_name, params in selected.items():
        label = _labels.get(config_name, config_name)
        logger.info("=== %s (%s) ===", config_name, label)
        n_trades = 0

        for symbol in all_symbols:
            df = _load_cached(symbol, cache_dir)
            if df.empty or len(df) < 50:
                continue

            trades = generate_trades(df, params, us_df=us_df)
            if not trades:
                continue

            results = compute_trade_results(trades, cost_model)
            for tr in results:
                t = tr.trade
                rows.append({
                    "config_name":  config_name,
                    "symbol":       symbol,
                    "signal_date":  (t.entry_time - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    "entry_date":   t.entry_time.strftime("%Y-%m-%d"),
                    "exit_date":    t.exit_time.strftime("%Y-%m-%d"),
                    "entry_price":  t.entry_price,
                    "exit_price":   t.exit_price,
                    "stop_price":   t.stop_price or 0.0,
                    "exit_reason":  t.exit_reason,
                    "gross_return": tr.gross_return,
                    "net_return":   tr.net_return,
                    "holding_days": tr.holding_days,
                    "status":       "closed",
                    "side":         t.side,
                    "priority":     _PRIORITY.get(config_name, 1),
                })
            n_trades += len(results)

        logger.info("  → %d トレード", n_trades)

    if not rows:
        logger.error("トレードが1件も生成されませんでした。キャッシュを確認してください。")
        return 1

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    logger.info("保存完了: %s  (%d 行)", out_csv, len(out))

    print("\n=== サマリ ===")
    for cfg, g in out.groupby("config_name"):
        label = _labels.get(cfg, cfg)
        e = g["net_return"].mean() * 100
        wr = (g["net_return"] > 0).mean()
        avg_hold = g["holding_days"].mean()
        print(f"  {label}: n={len(g)}  E={e:+.2f}%  WR={wr:.0%}  hold={avg_hold:.1f}d")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
