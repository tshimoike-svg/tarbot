"""中小型株ユニバースを J-Quants から構築して config/symbols.py を更新するスクリプト。

戦略：
  1. get_listed_info() で全上場銘柄を取得
  2. 市場区分・規模区分・除外セクターでフィルタ（APIコール1回）
  3. get_daily_quotes(date_=recent) で一日分の株価・出来高を一括取得（APIコール最小）
  4. 価格帯・売買代金（出来高×価格）でフィルタ → 上位 N 銘柄を保存

目標：スイング平均回帰向けの「機関が入りにくい中小型・十分な流動性」のニッチ

実行例：
    uv run python scripts/build_universe.py
    uv run python scripts/build_universe.py --max-symbols 80 --min-adv 50_000_000
    uv run python scripts/build_universe.py --dry-run  # config/symbols.py を書き換えない
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
_EXCLUDE_MARKET = set()  # 全市場を含める（Standard/Growth も対象）

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
_MAX_PRICE = 5_000.0  # 円

# 最低売買代金（ADV yen。流動性の最低ライン）
_DEFAULT_MIN_ADV = 30_000_000  # 3千万円/日

# 最大売買代金（上限：大型株を除く。TOPIX100除外だけでは漏れるので念のため）
_DEFAULT_MAX_ADV = 5_000_000_000  # 50億円/日


def _should_exclude_sector(name: str) -> bool:
    if not isinstance(name, str):
        return False
    return any(s in name for s in _EXCLUDE_SECTOR_SUBSTRINGS)


def fetch_universe(
    *,
    reference_date: str,
    min_adv: float,
    max_adv: float,
    max_symbols: int,
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
    if scale_col:
        before = len(master)
        master = master[~master[scale_col].isin(_EXCLUDE_SCALE)]
        logger.info("TOPIX100除外後: %d → %d", before, len(master))
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
        & (day_bars["close"] <= _MAX_PRICE)
        & (day_bars["adv_yen"] >= min_adv)
        & (day_bars["adv_yen"] <= max_adv)
    ].copy()
    logger.info(
        "価格[%g-%g円]・ADV[%gM-%gM円]フィルタ後: %d 銘柄",
        _MIN_PRICE, _MAX_PRICE,
        min_adv / 1e6, max_adv / 1e6,
        len(filtered),
    )

    # 純数字4桁コードのみ（アルファベット入り = 新規上場で2年分データが無い可能性が高い）
    before = len(filtered)
    filtered = filtered[filtered["_code4"].str.match(r"^\d{4}$")]
    if len(filtered) < before:
        logger.info("数字4桁コードのみ: %d → %d", before, len(filtered))

    # 売買代金の中央値に近い順にソート（上限・下限の外れ値を避ける）
    median_adv = filtered["adv_yen"].median()
    filtered["_adv_dist"] = (filtered["adv_yen"] - median_adv).abs()
    filtered = filtered.sort_values("_adv_dist")

    # 上位 max_symbols を取得
    result_codes = filtered["_code4"].head(max_symbols).tolist()
    logger.info("最終ユニバース: %d 銘柄", len(result_codes))
    return result_codes


def write_symbols_py(codes: list[str], path: Path) -> None:
    """config/symbols.py を生成する。"""
    lines = [
        '"""監視銘柄ユニバース（scripts/build_universe.py で自動生成）。',
        "",
        "流動性フィルタ（価格帯・売買代金）と規模区分（TOPIX100除外）で絞った中小型中心。",
        "再生成: uv run python scripts/build_universe.py",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        '__all__ = ["SYMBOLS"]',
        "",
        "SYMBOLS: list[str] = [",
    ]
    for code in codes:
        lines.append(f'    "{code}",')
    lines += ["]", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("config/symbols.py を更新しました（%d 銘柄）", len(codes))


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="J-Quants ユニバース構築")
    parser.add_argument("--max-symbols", type=int, default=60, help="最大銘柄数（既定 60）")
    parser.add_argument("--min-adv", type=float, default=_DEFAULT_MIN_ADV,
                        help=f"最低売買代金/日（既定 {_DEFAULT_MIN_ADV:,.0f}円）")
    parser.add_argument("--max-adv", type=float, default=_DEFAULT_MAX_ADV,
                        help=f"最大売買代金/日（既定 {_DEFAULT_MAX_ADV:,.0f}円）")
    parser.add_argument("--date", default="2026-03-27",
                        help="基準日（価格・出来高の参照日）")
    parser.add_argument("--dry-run", action="store_true",
                        help="config/symbols.py を書き換えず結果だけ表示")
    args = parser.parse_args()

    if not os.getenv("JQUANTS_API_KEY"):
        print("JQUANTS_API_KEY が未設定です。.env に設定してください。")
        return 2

    codes = fetch_universe(
        reference_date=args.date,
        min_adv=args.min_adv,
        max_adv=args.max_adv,
        max_symbols=args.max_symbols,
    )

    print(f"\n=== ユニバース ({len(codes)} 銘柄) ===")
    print(", ".join(codes))

    if not args.dry_run:
        symbols_path = _ROOT / "config" / "symbols.py"
        write_symbols_py(codes, symbols_path)
    else:
        print("\n[dry-run] config/symbols.py は変更しません")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
