"""株ユニバースを J-Quants から構築して config/symbols*.py を更新するスクリプト。

戦略：
  1. get_listed_info() で全上場銘柄を取得
  2. 市場区分・規模区分・除外セクターでフィルタ（APIコール1回）
  3. get_daily_quotes(date_=recent) で一日分の株価・出来高を一括取得（APIコール最小）
  4. 価格帯・売買代金（出来高×価格）でフィルタ → 上位 N 銘柄を保存

実行例：
    # 中小型（従来）: TOPIX Core30/Large70 除外
    uv run python scripts/build_universe.py

    # 大型含む（クロスセクション向け）: 規模フィルタなし・価格上限拡張
    uv run python scripts/build_universe.py --include-large --max-symbols 120 \\
        --max-adv 5_000_000_000_000 --max-price 50000 --output config/symbols_cs.py

    # 結果だけ確認（ファイル書き換えなし）
    uv run python scripts/build_universe.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加（scripts/ から実行したとき用）
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---- フィルタ定数 -----------------------------------------------------------

# 除外する規模区分（大型株：機関投資家が主役で容量制約ニッチなし）
# ScaleCat 実値: "TOPIX Core30" / "TOPIX Large70" / "TOPIX Mid400" / "TOPIX Small 1" / "TOPIX Small 2" / "-"
_EXCLUDE_SCALE = {"TOPIX Core30", "TOPIX Large70"}

# 除外する市場区分コード
_EXCLUDE_MARKET: set[str] = set()  # 全市場を含める（Standard/Growth も対象）

# 除外するセクター（33業種コード or 業種名に含む文字列）
# 銀行・証券・保険・その他金融 → スプレッドが構造的に異なる
# 不動産投資信託（REIT）→ 値動きが別物
_EXCLUDE_SECTOR_SUBSTRINGS = [
    "銀行",
    "証券",
    "保険",
    "その他金融",
    "不動産投資",
]

# 価格帯（100株単位で現実的に売買できる範囲）
_MIN_PRICE = 300.0   # 円
_DEFAULT_MAX_PRICE = 5_000.0   # 円（中小型デフォルト）
_LARGE_MAX_PRICE = 50_000.0    # 円（--include-large 時）

# 最低売買代金（ADV yen。流動性の最低ライン）
_DEFAULT_MIN_ADV = 30_000_000  # 3千万円/日

# 最大売買代金（上限：大型株を除く。TOPIX100除外だけでは漏れるので念のため）
_DEFAULT_MAX_ADV = 5_000_000_000       # 50億円/日（中小型デフォルト）
_LARGE_MAX_ADV = 5_000_000_000_000    # 5兆円/日（--include-large 時は実質無制限）


def _should_exclude_sector(name: str) -> bool:
    if not isinstance(name, str):
        return False
    return any(s in name for s in _EXCLUDE_SECTOR_SUBSTRINGS)


def fetch_universe(
    *,
    reference_date: str,
    min_adv: float,
    max_adv: float,
    max_price: float,
    max_symbols: int,
    include_large: bool = False,
    sort_by: str = "median",   # "median": 中央値ADV近傍、"adv-desc": 最流動性降順
    min_interval: float = 1.0,
) -> list[str]:
    """J-Quants からユニバースを構築して銘柄コードリストを返す。"""
    from data.fetcher import JQuantsClient

    client = JQuantsClient(min_interval=min_interval, max_retries=5, retry_backoff=2.0)

    # --- Step 1: 銘柄マスタ取得 ------------------------------------------------
    logger.info("銘柄マスタを取得中...")
    master = client.get_listed_info(date_=reference_date)
    logger.info("上場銘柄数: %d", len(master))

    # 列名を確認してフィルタ
    logger.debug("列名: %s", list(master.columns))

    # 規模区分フィルタ（v2列名: ScaleCat）
    scale_col = next((c for c in ["ScaleCat", "ScaleCategory"] if c in master.columns), None)
    if scale_col and not include_large:
        before = len(master)
        master = master[~master[scale_col].isin(_EXCLUDE_SCALE)]
        logger.info("TOPIX100除外後: %d → %d", before, len(master))
    elif include_large:
        logger.info("--include-large: 規模フィルタをスキップ（大型株含む）")
    else:
        logger.warning("ScaleCategory/ScaleCat 列が見つからない。規模フィルタをスキップ")

    # セクターフィルタ（v2列名: S33Nm / S17Nm）
    sector_col = next(
        (c for c in ["S33Nm", "S17Nm", "Sector33CodeName", "Sector17CodeName"]
         if c in master.columns),
        None,
    )
    if sector_col:
        before = len(master)
        master = master[~master[sector_col].apply(_should_exclude_sector)]
        logger.info("金融等セクター除外後: %d → %d", before, len(master))
    else:
        logger.warning("セクター名列が見つからない。セクターフィルタをスキップ")

    # 信用取引可能銘柄だけに絞る（Mrgn列: 1=信用買のみ, 2=貸借, 0=不可）
    mrgn_col = next((c for c in ["Mrgn", "MarginCode"] if c in master.columns), None)
    if mrgn_col:
        before = len(master)
        master = master[master[mrgn_col].astype(str).isin(["1", "2"])]
        logger.info("信用取引可能銘柄のみ: %d → %d", before, len(master))

    # コード列の取得（v2: Code が5桁 "13010" 形式）
    code_col = next((c for c in ["Code", "code"] if c in master.columns), None)
    if code_col is None:
        raise RuntimeError("銘柄コード列（Code）が見つかりません")
    # 5桁コード → 先頭4桁に正規化（例: "13010" → "1301"）
    codes_in_master: set[str] = set(
        str(c)[:4] if len(str(c)) >= 5 else str(c) for c in master[code_col]
    )
    logger.info("フィルタ後の候補銘柄数: %d", len(codes_in_master))

    # --- Step 2: 指定日の株価・出来高を一括取得 --------------------------------
    logger.info("基準日 %s の株価データを取得中...", reference_date)
    day_bars = client.get_daily_quotes(date_=reference_date)
    if day_bars.empty:
        raise RuntimeError(f"日付 {reference_date} の株価データが空です")
    logger.info("取得行数: %d", len(day_bars))

    # コード正規化（4桁または5桁、末尾の "0" を除くかどうかは API によって異なる）
    if "code" in day_bars.columns:
        # 日足レスポンスのコードも5桁形式の場合あり → 先頭4桁に統一
        day_bars["_code4"] = day_bars["code"].apply(
            lambda c: str(c)[:4] if len(str(c)) >= 5 else str(c)
        )
    else:
        raise RuntimeError("日足データにコード列がない")

    # 銘柄マスタとの突合
    day_bars = day_bars[day_bars["_code4"].isin(codes_in_master)].copy()
    logger.info("マスタ突合後: %d 行", len(day_bars))

    # --- Step 3: 価格・売買代金フィルタ -----------------------------------------
    day_bars["adv_yen"] = day_bars["close"] * day_bars["volume"]

    filtered = day_bars[
        (day_bars["close"] >= _MIN_PRICE)
        & (day_bars["close"] <= max_price)
        & (day_bars["adv_yen"] >= min_adv)
        & (day_bars["adv_yen"] <= max_adv)
    ].copy()
    logger.info(
        "価格[%g-%g円]・ADV[%gM-%gM円]フィルタ後: %d 銘柄",
        _MIN_PRICE, max_price,
        min_adv / 1e6, max_adv / 1e6,
        len(filtered),
    )

    # 純数字4桁コードのみ（アルファベット入り = 新規上場で2年分データが無い可能性が高い）
    before = len(filtered)
    filtered = filtered[filtered["_code4"].str.match(r"^\d{4}$")]
    if len(filtered) < before:
        logger.info("数字4桁コードのみ: %d → %d", before, len(filtered))

    # ソート
    if sort_by == "adv-desc":
        # 最流動性降順（大型株含むユニバース向け：実際に大型株を含める）
        filtered = filtered.sort_values("adv_yen", ascending=False)
        logger.info("ソート: ADV 降順（最流動性）")
    else:
        # 中央値ADV近傍（中小型ニッチ向けデフォルト）
        median_adv = filtered["adv_yen"].median()
        filtered["_adv_dist"] = (filtered["adv_yen"] - median_adv).abs()
        filtered = filtered.sort_values("_adv_dist")
        logger.info("ソート: 中央値ADV近傍")

    # 上位 max_symbols を取得
    result_codes = filtered["_code4"].head(max_symbols).tolist()
    logger.info("最終ユニバース: %d 銘柄", len(result_codes))
    return result_codes


def write_symbols_py(codes: list[str], path: Path, *, include_large: bool = False) -> None:
    """config/symbols*.py を生成する。"""
    if include_large:
        description = "流動性フィルタ（価格帯・売買代金）で絞った全規模ユニバース（大型株含む）。"
        regen = "再生成: uv run python scripts/build_universe.py --include-large --max-symbols 120 --max-adv 5_000_000_000_000 --max-price 50000 --output config/symbols_cs.py"
        varname = "SYMBOLS_CS"
    else:
        description = "流動性フィルタ（価格帯・売買代金）と規模区分（TOPIX100除外）で絞った中小型中心。"
        regen = "再生成: uv run python scripts/build_universe.py"
        varname = "SYMBOLS"
    lines = [
        '"""監視銘柄ユニバース（scripts/build_universe.py で自動生成）。',
        "",
        description,
        regen,
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f'__all__ = ["{varname}"]',
        "",
        f"{varname}: list[str] = [",
    ]
    for code in codes:
        lines.append(f'    "{code}",')
    lines += ["]", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("%s を更新しました（%d 銘柄）", path, len(codes))


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="J-Quants ユニバース構築")
    parser.add_argument("--max-symbols", type=int, default=60, help="最大銘柄数（既定 60）")
    parser.add_argument("--min-adv", type=float, default=_DEFAULT_MIN_ADV,
                        help=f"最低売買代金/日（既定 {_DEFAULT_MIN_ADV:,.0f}円）")
    parser.add_argument("--max-adv", type=float, default=None,
                        help="最大売買代金/日（既定: 中小型=50億、--include-large=5兆）")
    parser.add_argument("--max-price", type=float, default=None,
                        help="最大株価（既定: 中小型=5000円、--include-large=50000円）")
    parser.add_argument("--date", default="2026-03-27",
                        help="基準日（価格・出来高の参照日）")
    parser.add_argument("--include-large", action="store_true",
                        help="TOPIX Core30/Large70 を含める（クロスセクション戦略向け）")
    parser.add_argument("--sort", choices=["median", "adv-desc"], default=None,
                        help="ソート方法（既定: 中小型=median, --include-large=adv-desc）")
    parser.add_argument("--output", default=None,
                        help="出力ファイルパス（既定: 中小型=config/symbols.py, 大型含む=config/symbols_cs.py）")
    parser.add_argument("--dry-run", action="store_true",
                        help="ファイルを書き換えず結果だけ表示")
    args = parser.parse_args()

    if not os.getenv("JQUANTS_API_KEY"):
        print("JQUANTS_API_KEY が未設定です。.env に設定してください。")
        return 2

    # デフォルト値を include-large に応じて切り替え
    max_adv = args.max_adv if args.max_adv is not None else (
        _LARGE_MAX_ADV if args.include_large else _DEFAULT_MAX_ADV
    )
    max_price = args.max_price if args.max_price is not None else (
        _LARGE_MAX_PRICE if args.include_large else _DEFAULT_MAX_PRICE
    )

    sort_by = args.sort if args.sort is not None else ("adv-desc" if args.include_large else "median")
    codes = fetch_universe(
        reference_date=args.date,
        min_adv=args.min_adv,
        max_adv=max_adv,
        max_price=max_price,
        max_symbols=args.max_symbols,
        include_large=args.include_large,
        sort_by=sort_by,
    )

    print(f"\n=== ユニバース ({len(codes)} 銘柄) ===")
    print(", ".join(codes))

    if not args.dry_run:
        if args.output:
            symbols_path = Path(args.output)
        elif args.include_large:
            symbols_path = _ROOT / "config" / "symbols_cs.py"
        else:
            symbols_path = _ROOT / "config" / "symbols.py"
        write_symbols_py(codes, symbols_path, include_large=args.include_large)
    else:
        print("\n[dry-run] ファイルは変更しません")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
