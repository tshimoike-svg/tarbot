#!/usr/bin/env python3
"""内需セクター絞り込み + Vol+Z / クロスセクション戦略のバックテスト。

⑤（US クラッシュ日）がシグナルを出さない「平常日」にもエッジがあるかを検証する。

エントリー条件:
  Vol+Z : 時系列 z ≤ −1.5 + 出来高 > 20日平均の 1.5倍。US フィルタなし。
           個別株の需給崩れ（決算・大口売り）を狙う。
  CS    : ピア対比の N日リターンが −1.5σ 以下（同日の内需銘柄内で最も下げた株）。
           出口はその銘柄の MA(20) 到達・ATR 損切・タイムストップ。

実行:
    uv run python scripts/run_cs_vol_backtest.py

オプション:
    --no-fetch-sector   sector_map.csv がなくても全銘柄で続行（デバッグ用）
    --save-csv          トレード結果を data/db/cs_vol_trades.csv に保存
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
from config.settings import SwingCrossSectionParams, SwingReversionParams
from config.symbols import SYMBOLS
from strategy.indicators import atr as compute_atr
from strategy.swing import walk_swing
from strategy.swing_cross_section import compute_cross_signals
from strategy.swing_reversion import generate_trades as gen_reversion_trades

logger = logging.getLogger(__name__)

_CACHE_DIR = _ROOT / "data/db/bars_cache"
_SECTOR_MAP = _ROOT / "data/db/sector_map.csv"
_BT_TRADES   = _ROOT / "data/db/bt_trades.csv"
_OUT_CSV     = _ROOT / "data/db/cs_vol_trades.csv"

# ─── 内需セクター（J-Quants Sector33CodeName の実際の文字列）───────────────────
# US 相関が低く、会社固有の材料（決算・需給・規制）で動きやすいセクター。
# 除外: 電気機器・輸送用機器・機械・化学・非鉄金属・鉄鋼・精密機器・海運業・石油石炭
# 注: J-Quants は中黒に ｢･｣(U+FF65) を使う（「・」(U+30FB) ではない）。
_DOMESTIC_SECTOR_NAMES = frozenset([
    "食料品",
    "医薬品",
    "電気･ガス業",      # J-Quants 表記（U+FF65）
    "建設業",
    "小売業",
    "不動産業",
    "陸運業",
    "サービス業",
    "情報･通信業",      # J-Quants 表記（U+FF65）
    "卸売業",           # 商社系は外需も含むが内需寄りを多く含む
    "その他",
    "その他製品",
    "空運業",
])

# ─── Vol+Z 設定（US フィルタなし・出来高条件あり）───────────────────────────
# z ≤ −1.5 はマクロクラッシュより小さい押し目 → 個別材料に絞るため出来高で補完。
_VOL_Z_CONFIG = SwingReversionParams(
    lookback=20,
    entry_z=1.5,
    atr_stop_mult=2.0,
    max_holding_days=7,
    allow_long=True,
    allow_short=False,
    rsi_entry_max=45.0,
    volume_ratio_min=1.5,   # 20日平均の 1.5 倍以上の出来高でのみ入る
    # US フィルタは全て nan（= 無効 = 全日エントリー可）
)

# ─── クロスセクション設定 ───────────────────────────────────────────────────
# 同日の内需銘柄内でのピア対比外れ値を検出。出口は時系列 MA(20)。
_CS_CONFIG = SwingCrossSectionParams(
    return_lookback=5,    # 5日リターンのクロスセクション z
    entry_z=1.5,
    atr_length=14,
    atr_stop_mult=2.0,
    max_holding_days=7,
    allow_long=True,
    allow_short=False,
)
_CS_MA_WINDOW = 20  # CS エントリー後の利確目標: 銘柄自身の MA(20) 回帰


# ─── ユーティリティ ──────────────────────────────────────────────────────────

def _find_cache(symbol: str) -> Path | None:
    candidates = sorted(_CACHE_DIR.glob(f"{symbol}_*.pkl"), reverse=True)
    return candidates[0] if candidates else None


def _load_cached(symbol: str) -> pd.DataFrame:
    pkl = _find_cache(symbol)
    if pkl is None:
        return pd.DataFrame()
    with pkl.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def _load_all_symbols() -> list[str]:
    raw: list[str] = [str(s) for s in SYMBOLS]
    for mod_name, attr in [("config.symbols_cs", "SYMBOLS_CS")]:
        try:
            import importlib
            m = importlib.import_module(mod_name)
            raw += [str(s) for s in getattr(m, attr)]
        except (ImportError, AttributeError):
            pass
    return list(dict.fromkeys(raw))


def _domestic_symbols(all_syms: list[str], *, no_fetch: bool) -> list[str]:
    """内需セクターの銘柄コードを返す。sector_map.csv がなければ全銘柄を返す。"""
    if not _SECTOR_MAP.exists():
        if no_fetch:
            logger.warning("sector_map.csv なし。全銘柄 (%d 件) で続行", len(all_syms))
            return all_syms
        # セクターマップを自動取得
        logger.info("sector_map.csv がありません。J-Quants から取得します...")
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, str(_ROOT / "scripts/fetch_sector_map.py")],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                logger.warning("セクターマップ取得失敗:\n%s", result.stderr)
                return all_syms
        except Exception as e:
            logger.warning("セクターマップ取得失敗: %s", e)
            return all_syms

    smap = pd.read_csv(_SECTOR_MAP)
    if "sector33_name" not in smap.columns:
        logger.warning("sector33_name 列なし。全銘柄で続行")
        return all_syms

    domestic = (
        smap[smap["sector33_name"].isin(_DOMESTIC_SECTOR_NAMES)]["symbol"]
        .astype(str)
        .tolist()
    )
    # 対象ユニバース内にある銘柄のみ
    sym_set = set(all_syms)
    domestic = [s for s in domestic if s in sym_set]
    logger.info(
        "内需セクター: %d 件 / 全 %d 件 (対象銘柄 %d 件中)",
        len(domestic), len(smap), len(all_syms),
    )
    return domestic


def _crash_signal_dates() -> frozenset[str]:
    """⑤ (config_v) のシグナル日 = US クラッシュ日を返す。"""
    if not _BT_TRADES.exists():
        return frozenset()
    df = pd.read_csv(_BT_TRADES)
    v = df[df["config_name"] == "config_v"]
    return frozenset(v["signal_date"].unique())


def _signal_date(entry_time: pd.Timestamp) -> str:
    """entry_time から signal_date を計算（bt_trades.csv と同じ方式）。"""
    return (entry_time - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _print_summary(
    rows: list[dict],
    crash_dates: frozenset[str],
    label: str,
) -> None:
    if not rows:
        print(f"\n{label}: トレードなし")
        return
    df = pd.DataFrame(rows)
    n = len(df)
    e = df["net_return"].mean() * 100
    wr = (df["net_return"] > 0).mean()

    crash = df[df["signal_date"].isin(crash_dates)]
    noncr = df[~df["signal_date"].isin(crash_dates)]

    exit_counts = df["exit_reason"].value_counts()

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"  全体:          n={n:4d}  E={e:+.2f}%  WR={wr:.0%}  hold={df['holding_days'].mean():.1f}d")
    print(f"  出口内訳:      {dict(exit_counts)}")
    if len(crash):
        ec = crash["net_return"].mean() * 100
        wrc = (crash["net_return"] > 0).mean()
        print(f"  US クラッシュ日: n={len(crash):4d}  E={ec:+.2f}%  WR={wrc:.0%}")
    if len(noncr):
        en = noncr["net_return"].mean() * 100
        wrn = (noncr["net_return"] > 0).mean()
        ex_nc = noncr["exit_reason"].value_counts()
        print(f"  平常日 ★:     n={len(noncr):4d}  E={en:+.2f}%  WR={wrn:.0%}  hold={noncr['holding_days'].mean():.1f}d")
        print(f"    出口内訳:    {dict(ex_nc)}")
    if len(crash) and len(noncr):
        print(f"  → 平常日の割合: {len(noncr)/n:.0%}")


# ─── メイン ─────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="CS / Vol+Z 内需バックテスト")
    parser.add_argument("--no-fetch-sector", action="store_true",
                        help="sector_map.csv なしで全銘柄を使う")
    parser.add_argument("--save-csv", action="store_true",
                        help="トレードを data/db/cs_vol_trades.csv に保存")
    args = parser.parse_args(argv)

    all_syms = _load_all_symbols()
    dom_syms = _domestic_symbols(all_syms, no_fetch=args.no_fetch_sector)
    crash_dates = _crash_signal_dates()
    logger.info("⑤ クラッシュ日: %d 日", len(crash_dates))

    # 価格データをロード
    dfs: dict[str, pd.DataFrame] = {}
    for sym in dom_syms:
        df = _load_cached(sym)
        if not df.empty and len(df) >= 50:
            dfs[sym] = df
    logger.info("ロード完了: %d 銘柄（内需セクター）", len(dfs))

    if not dfs:
        logger.error("価格データなし。キャッシュを確認してください。")
        return 1

    # US データ（Vol+Z には不要だが spx_fresh があれば将来の拡張に備え保持）
    us_df: pd.DataFrame | None = None
    try:
        from data.us_loader import load_spx_fresh  # noqa: PLC0415
        us_df = load_spx_fresh(lookback_days=800)
        logger.info("S&P500: %d 行", len(us_df))
    except Exception as exc:
        logger.info("S&P500 取得スキップ: %s", exc)

    cost_model = CostModel(get_cost_params("mid"))
    all_rows: list[dict] = []

    # ─── 1. Vol+Z バックテスト ──────────────────────────────────────────────
    logger.info("=== Vol+Z バックテスト ===")
    vol_rows: list[dict] = []
    for sym, df in dfs.items():
        trades = gen_reversion_trades(df, _VOL_Z_CONFIG)   # US フィルタなし
        if not trades:
            continue
        for tr in compute_trade_results(trades, cost_model):
            t = tr.trade
            row = {
                "strategy":    "vol_z",
                "symbol":      sym,
                "signal_date": _signal_date(t.entry_time),
                "entry_date":  t.entry_time.strftime("%Y-%m-%d"),
                "exit_date":   t.exit_time.strftime("%Y-%m-%d"),
                "net_return":  tr.net_return,
                "holding_days": tr.holding_days,
                "exit_reason": t.exit_reason,
            }
            vol_rows.append(row)
            all_rows.append(row)

    # ─── 2. クロスセクション バックテスト ──────────────────────────────────
    logger.info("=== クロスセクション バックテスト (%d 銘柄) ===", len(dfs))
    cs_signals = compute_cross_signals(dfs, _CS_CONFIG)
    cs_rows: list[dict] = []
    for sym, entry in cs_signals.items():
        df = dfs[sym]
        atr_s  = compute_atr(df["high"], df["low"], df["close"], length=_CS_CONFIG.atr_length)
        ma_tgt = df["close"].rolling(_CS_MA_WINDOW, min_periods=_CS_MA_WINDOW // 2).mean()

        trades = walk_swing(
            df,
            entries=entry,
            atr=atr_s,
            target=ma_tgt,          # MA(20) 到達で利確（CS の exit target として流用）
            atr_stop_mult=_CS_CONFIG.atr_stop_mult,
            max_holding_days=_CS_CONFIG.max_holding_days,
        )
        if not trades:
            continue
        for tr in compute_trade_results(trades, cost_model):
            t = tr.trade
            row = {
                "strategy":    "cs",
                "symbol":      sym,
                "signal_date": _signal_date(t.entry_time),
                "entry_date":  t.entry_time.strftime("%Y-%m-%d"),
                "exit_date":   t.exit_time.strftime("%Y-%m-%d"),
                "net_return":  tr.net_return,
                "holding_days": tr.holding_days,
                "exit_reason": t.exit_reason,
            }
            cs_rows.append(row)
            all_rows.append(row)

    # ─── サマリ ─────────────────────────────────────────────────────────────
    _print_summary(vol_rows,  crash_dates, "Vol+Z（内需・出来高スパイク・US フィルタなし）")
    _print_summary(cs_rows,   crash_dates, "クロスセクション（内需・ピア外れ値・MA(20) 目標）")

    # 参照: ⑤ 単独の成績（比較用）
    if _BT_TRADES.exists():
        ref = pd.read_csv(_BT_TRADES)
        v = ref[ref["config_name"] == "config_v"]
        if len(v):
            print(f"\n{'='*60}")
            print("参照: ⑤（config_v 全銘柄・US フィルタあり）")
            print(f"{'='*60}")
            print(f"  n={len(v)}  E={v['net_return'].mean()*100:+.2f}%  WR={(v['net_return']>0).mean():.0%}")

    if args.save_csv and all_rows:
        out_df = pd.DataFrame(all_rows)
        out_df.to_csv(_OUT_CSV, index=False)
        logger.info("保存: %s (%d 行)", _OUT_CSV, len(out_df))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
