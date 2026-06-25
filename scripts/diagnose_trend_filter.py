#!/usr/bin/env python3
"""当てこみゼロの診断：反転トレードを「エントリー判定時点(signal_date=T)のトレンド」で割る。

`close > SMA(win)` を「上トレンドの押し目」、それ以外を「下降トレンドの押し目(落ちるナイフ)」とし、
全期間と OOS(>2026-03-27) の両方で E/WR を比べる。トレンドフィルタが「この窓だけ」でなく
複数レジームで効くか（＝原理的に正しいか）を、パラメータを弄らず確認するのが目的。

入力は simulate_leverage.py の出力 CSV（signal_date / net_return 等）と Yahoo bars。
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

_CACHE = _ROOT / "data/db/bars_cache_yahoo"
_CSV = _ROOT / "data/db/bt_trades_yahoo.csv"
_SPLIT = pd.Timestamp("2026-03-27")
_WINS = [100, 200]


def _close(sym: str) -> pd.Series | None:
    cands = sorted(_CACHE.glob(f"{sym}_*.pkl"))
    if not cands:
        return None
    with cands[0].open("rb") as f:
        df = pickle.load(f)  # noqa: S301
    return df["close"]


def _stats(g: pd.DataFrame) -> str:
    if len(g) == 0:
        return "n=   0"
    return (f"n={len(g):4} E={g.net_return.mean()*100:+5.2f}% "
            f"WR={(g.net_return > 0).mean()*100:3.0f}%")


def main() -> int:
    tr = pd.read_csv(_CSV, parse_dates=["signal_date", "entry_date"])
    tr["symbol"] = tr.symbol.astype(str)

    # 銘柄ごとに close と SMA を前計算 → signal_date 時点の up/down を判定
    sma: dict[str, pd.DataFrame] = {}
    for sym in tr.symbol.unique():
        c = _close(sym)
        if c is None:
            continue
        d = pd.DataFrame({"close": c})
        for w in _WINS:
            d[f"sma{w}"] = c.rolling(w).mean()
        sma[sym] = d

    def regime(row, win: int) -> str | None:
        d = sma.get(row.symbol)
        if d is None:
            return None
        idx = d.index[d.index <= row.signal_date]
        if len(idx) == 0:
            return None
        r = d.loc[idx[-1]]
        if pd.isna(r[f"sma{win}"]):
            return None
        return "up" if r["close"] > r[f"sma{win}"] else "down"

    for win in _WINS:
        tr[f"reg{win}"] = tr.apply(lambda r, w=win: regime(r, w), axis=1)

    configs = ["config_v", "config_iv", "config_t2c"]
    for win in _WINS:
        print(f"\n========== トレンドフィルタ: close vs SMA({win}) ==========")
        col = f"reg{win}"
        for cfg in configs:
            g = tr[(tr.config_name == cfg) & tr[col].notna()]
            ins, oos = g[g.entry_date <= _SPLIT], g[g.entry_date > _SPLIT]
            print(f"\n[{cfg}]  （SMA判定可能 {len(g)} / 全 {len(tr[tr.config_name==cfg])} トレード）")
            for name, seg in [("全期間", g), ("  イン", ins), ("  OOS ", oos)]:
                up, dn = seg[seg[col] == "up"], seg[seg[col] == "down"]
                share = f"up比率={len(up)/len(seg)*100:3.0f}%" if len(seg) else ""
                print(f"  {name}: up[{_stats(up)}]  down[{_stats(dn)}]  {share}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
