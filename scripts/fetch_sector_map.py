#!/usr/bin/env python3
"""J-Quants 銘柄マスターからセクター情報を取得し data/db/sector_map.csv に保存。

実行:
    uv run python scripts/fetch_sector_map.py
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from config.symbols import SYMBOLS
from data.fetcher import JQuantsClient

logger = logging.getLogger(__name__)
_OUT = _ROOT / "data/db/sector_map.csv"


def _all_symbols() -> list[str]:
    raw: list[str] = [str(s) for s in SYMBOLS]
    for mod_name, attr in [("config.symbols_cs", "SYMBOLS_CS")]:
        try:
            m = importlib.import_module(mod_name)
            raw += [str(s) for s in getattr(m, attr)]
        except (ImportError, AttributeError):
            pass
    return list(dict.fromkeys(raw))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    symbols = _all_symbols()
    logger.info("対象銘柄: %d 件", len(symbols))

    client = JQuantsClient()

    logger.info("J-Quants /equities/master を取得中...")
    info_df = client.get_listed_info()
    logger.info("取得レコード: %d 件", len(info_df))

    cols = info_df.columns.tolist()
    logger.debug("列: %s", cols)

    # 4桁コードに変換（J-Quants は 5桁: 末尾"0" の場合が多い）
    code_col = next((c for c in ("Code", "code") if c in cols), None)
    if code_col is None:
        logger.error("Code 列が見つかりません。列: %s", cols)
        return 1
    info_df["symbol"] = info_df[code_col].astype(str).str[:4]

    # セクターコード列を探す（長名・短縮名の両方に対応）
    _S33_CODE_CANDIDATES = ("Sector33Code", "sector33Code", "S33")
    _S33_NAME_CANDIDATES = ("Sector33CodeName", "sector33CodeName", "S33Nm")
    s33_code_col = next((c for c in _S33_CODE_CANDIDATES if c in cols), None)
    s33_name_col = next((c for c in _S33_NAME_CANDIDATES if c in cols), None)

    if s33_code_col is None:
        logger.error("Sector33Code 列が見つかりません。列: %s", cols)
        return 1

    keep = ["symbol", s33_code_col]
    if s33_name_col:
        keep.append(s33_name_col)

    sym_set = set(symbols)
    filtered = (
        info_df[info_df["symbol"].isin(sym_set)][keep]
        .copy()
        .drop_duplicates(subset=["symbol"])
        .rename(columns={s33_code_col: "sector33_code"})
    )
    if s33_name_col:
        filtered = filtered.rename(columns={s33_name_col: "sector33_name"})

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(_OUT, index=False)
    logger.info("保存: %s (%d 件)", _OUT, len(filtered))

    if "sector33_name" in filtered.columns:
        print("\n=== セクター分布（対象銘柄） ===")
        for name, grp in filtered.groupby("sector33_name"):
            print(f"  {name:30s}: {len(grp):3d} 銘柄")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
