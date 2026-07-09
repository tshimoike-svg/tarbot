"""Phase 1 ドライラン監視ダッシュボード（Streamlit）。

起動:
    uv sync --extra dashboard
    uv run streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.signal_store import SignalStore

_DB              = Path(__file__).parent.parent / "data/db/forward_signals.sqlite"
_BT_CSV          = Path(__file__).parent.parent / "data/db/bt_trades.csv"
_INITIAL_CAPITAL = 1_000_000

# ── ドライラン固定パラメータ ─────────────────────────────────────────────────────
_LOT_SIZE        = 100        # 単元株（TSE標準）
_PER_POS_TARGET  = 500_000    # 1銘柄の買付目安（¥500K）
_LEVERAGE_LIMIT  = 2.5        # 買付総額 / 資本 の上限

st.set_page_config(page_title="Phase 1 監視", page_icon="📈", layout="wide")


# ── ヘルパー ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _load_live() -> pd.DataFrame:
    if not _DB.exists():
        return pd.DataFrame()
    with SignalStore(_DB) as s:
        return s.get_all()


@st.cache_data(ttl=3600)
def _load_bt() -> pd.DataFrame:
    if not _BT_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(_BT_CSV)


def _calc_lots(entry_price: float) -> tuple[int, float]:
    """1ロット≥¥500K → 1ロット、未満 → ¥500K以内で最大ロット。
    Returns (lots, investment_yen)
    """
    lot_cost = entry_price * _LOT_SIZE
    if lot_cost >= _PER_POS_TARGET:
        return 1, lot_cost
    lots = int(_PER_POS_TARGET // lot_cost)
    return lots, lots * lot_cost


_CONFIG_ABBR  = {"config_iii": "③", "config_iv": "④", "config_v": "⑤", "config_t2c": "補C"}
_CONFIG_ORDER = list(_CONFIG_ABBR.keys())


def _dedup_by_stock(df: pd.DataFrame) -> pd.DataFrame:
    """同銘柄・同エントリー日の複数コンフィグ行を1行に集約。
    entry_price / exit_price / net_return は同一銘柄・同日で同値なので first。
    config_name はトリガーしたコンフィグを「③/④/⑤」形式で結合。
    """
    if df.empty:
        return df.copy()

    def _cfgs(s: pd.Series) -> str:
        configs = s.tolist()
        ordered = [c for c in _CONFIG_ORDER if c in configs]
        rest    = [c for c in configs if c not in _CONFIG_ORDER]
        return "/".join(_CONFIG_ABBR.get(c, c) for c in ordered + rest)

    agg: dict = {"config_name": _cfgs}
    for col in ["signal_date", "entry_price", "stop_price", "target_price",
                "max_exit_date", "side", "status",
                "exit_date", "exit_price", "exit_reason", "gross_return", "net_return"]:
        if col in df.columns:
            agg[col] = "first"

    return (
        df.groupby(["symbol", "entry_date"], sort=False)
          .agg(agg)
          .reset_index()
          .sort_values("entry_date")
    )


def _equity_from_lots(closed_df: pd.DataFrame, initial: float = _INITIAL_CAPITAL) -> np.ndarray:
    """ロット基準・順次複利でエクイティを計算（dedup済みを前提）。"""
    if closed_df.empty:
        return np.array([initial])
    equity = initial
    result = []
    for _, row in closed_df.sort_values("exit_date").iterrows():
        _, investment = _calc_lots(float(row["entry_price"]))
        equity += investment * float(row["net_return"])
        result.append(equity)
    return np.array(result)


def _simulate_fills(df: pd.DataFrame, cap_limit: float) -> pd.DataFrame:
    """時系列で買付枠（cap_limit）制約を適用し、各建玉が実約定したか(filled)を判定。

    ルール（ユーザー確定）:
      - 同日エントリーは 1ロット必要額の安い順に詰める（枠を使い切る）
      - 残枠に収まらない建玉は見送り（filled=False）
      - クローズで枠が空いても、過去に見送った建玉は買い直さない（追わない）

    dedup 済み（銘柄×エントリー日で1行）を前提。investment 列を付与して返す。
    """
    d = df.copy().reset_index(drop=True)
    d["_entry"]     = pd.to_datetime(d["entry_date"])
    d["_exit"]      = pd.to_datetime(d["exit_date"]) if "exit_date" in d.columns else pd.NaT
    d["investment"] = d["entry_price"].apply(lambda p: _calc_lots(float(p))[1])
    d["filled"]     = False

    exit_days = [x for x in d["_exit"].tolist() if pd.notna(x)]
    all_days  = sorted(set(d["_entry"].tolist() + exit_days))

    open_pos: list[dict] = []  # {"exit": Timestamp|NaT, "invest": float}
    used = 0.0
    for day in all_days:
        # クローズを先に処理して枠を戻す
        still = []
        for p in open_pos:
            if pd.notna(p["exit"]) and p["exit"] <= day:
                used -= p["invest"]
            else:
                still.append(p)
        open_pos = still
        # その日のエントリーを 1ロット額の安い順に枠へ詰める
        todays = d[d["_entry"] == day].sort_values("investment")
        for idx in todays.index:
            inv = float(d.at[idx, "investment"])
            if used + inv <= cap_limit + 1e-6:
                used += inv
                open_pos.append({"exit": d.at[idx, "_exit"], "invest": inv})
                d.at[idx, "filled"] = True

    return d.drop(columns=["_entry", "_exit"])


def _portfolio_simulation(
    trades_df: pd.DataFrame,
    position_ratio: float,
    leverage: float,
    max_positions: int,
    initial: float = _INITIAL_CAPITAL,
) -> tuple[np.ndarray, np.ndarray]:
    """資本プール共有の並列シミュレーション（バックテストモード用）。"""
    df = trades_df.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])

    sort_cols  = ["entry_date", "priority"] if "priority" in df.columns else ["entry_date"]
    has_symbol = "symbol" in df.columns
    df = df.sort_values(sort_cols).reset_index(drop=True)

    cash       = initial
    open_pos: list[dict] = []
    equity_log: list[tuple[pd.Timestamp, float]] = []
    all_events = sorted(set(df["entry_date"].tolist() + df["exit_date"].tolist()))
    entry_idx  = 0

    for event_date in all_events:
        still_open = []
        for pos in open_pos:
            if pos["exit_date"] <= event_date:
                cash += pos["margin"] * (1.0 + pos["net_return"] * leverage)
            else:
                still_open.append(pos)
        open_pos = still_open

        while entry_idx < len(df):
            row = df.iloc[entry_idx]
            if row["entry_date"] > event_date:
                break
            entry_idx += 1
            if len(open_pos) >= max_positions:
                continue
            if has_symbol and row["symbol"] in {p["symbol"] for p in open_pos}:
                continue
            total_equity = cash + sum(p["margin"] for p in open_pos)
            margin = total_equity * position_ratio
            if margin > cash:
                margin = cash
            if margin < initial * 0.001:
                continue
            cash -= margin
            pos_entry: dict = {
                "exit_date": row["exit_date"], "margin": margin, "net_return": row["net_return"],
            }
            if has_symbol:
                pos_entry["symbol"] = row["symbol"]
            open_pos.append(pos_entry)

        equity_log.append((event_date, cash + sum(p["margin"] for p in open_pos)))

    for pos in open_pos:
        cash += pos["margin"] * (1.0 + pos["net_return"] * leverage)
    equity_log.append((all_events[-1] if all_events else pd.Timestamp.now(), cash))

    return (
        np.array([e[1] for e in equity_log]),
        np.array([e[0] for e in equity_log], dtype="datetime64[ns]"),
    )


def _max_dd_pct(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(((peak - equity) / peak).max()) * 100


def _kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    return win_rate - (1 - win_rate) / (avg_win / avg_loss)


# ── ヘッダー ────────────────────────────────────────────────────────────────────

col_hd1, col_src, col_hd2 = st.columns([3, 2, 1])
with col_hd1:
    st.caption(f"最終更新: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
with col_src:
    data_source = st.radio(
        "データソース", ["ドライラン実績", "バックテスト2年（シミュレーション）"],
        horizontal=True, label_visibility="collapsed",
    )
with col_hd2:
    if st.button("🔄 更新", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

is_bt = data_source == "バックテスト2年（シミュレーション）"

# ── バックテストモード専用コントロール ─────────────────────────────────────────

CONFIG_LABELS = {
    "config_iii":  "③ rsi<30（高精度・少トレード）",
    "config_iv":   "④ rsi<35",
    "config_v":    "⑤ rsi<40（推奨メイン）",
    "config_t2c":  "補助C（lb=30, z≥2.0, RSI<40, US フィルタあり）",
    "main+aux":    "⑤メイン + 補助C（lb=30 空きスロット埋め）",
    "all":         "全設定合算",
}

if is_bt:
    st.info("過去2年のバックテスト個別トレードを使ったシミュレーションです。レバレッジ・銘柄数を調整してください。", icon="📊")
    col_cfg, col_ratio, col_lev = st.columns([2, 3, 2])
    with col_cfg:
        cfg_sel = st.selectbox(
            "設定", list(CONFIG_LABELS.keys()), index=2,
            format_func=lambda k: CONFIG_LABELS[k],
        )
    with col_ratio:
        ratio_pct = st.slider(
            "1銘柄あたり証拠金比率", min_value=5, max_value=33,
            value=20, step=1, format="%d%%",
        )
        position_ratio = ratio_pct / 100.0
        max_pos = max(1, int(1.0 / position_ratio))
        st.caption(f"→ 自動: **{max_pos}銘柄**（{ratio_pct}% × {max_pos} = {ratio_pct*max_pos}%）")
    with col_lev:
        leverage = st.slider(
            "信用倍率", min_value=1.0, max_value=3.0,
            value=1.0, step=0.1, format="%.1fx",
        )
else:
    # ドライランは固定値（スライダーなし）
    cfg_sel        = "all"
    position_ratio = 0.20
    max_pos        = 5
    leverage       = 1.0

# ── データ読み込み & フィルタ ────────────────────────────────────────────────────

df = _load_bt() if is_bt else _load_live()

if df.empty:
    st.markdown(
        f"""
        <div style="background:#1e1e2e;border-radius:16px;padding:32px 40px;margin:16px 0;text-align:center">
          <div style="color:#aaa;font-size:14px;margin-bottom:8px">現在の評価額（ドライラン）</div>
          <div style="color:#fff;font-size:60px;font-weight:700;letter-spacing:-2px">
            ¥{_INITIAL_CAPITAL:,.0f}
          </div>
          <div style="color:#666;font-size:13px;margin-top:8px">シグナル待ち</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("シグナルがまだありません。daily_scan.py が実行されると表示されます。")
    st.stop()

# バックテスト: コンフィグフィルタ。ドライラン: フィルタなし
if is_bt:
    if cfg_sel == "all":
        _df = df.copy()
    elif cfg_sel == "main+aux":
        _df = df[df["config_name"].isin(["config_v", "config_t2c"])].copy()
    else:
        _df = df[df["config_name"] == cfg_sel].copy()
else:
    _df = df.copy()

if is_bt:
    closed      = _df[_df["status"] == "closed"].sort_values("exit_date").copy()
    open_       = _df[_df["status"] == "open"].copy()
    skipped_open = pd.DataFrame()
else:
    # 同銘柄・同日の複数コンフィグを1建玉に集約 → 時系列で買付枠制約を適用
    all_dedup = _dedup_by_stock(_df)
    all_dedup = _simulate_fills(all_dedup, _INITIAL_CAPITAL * _LEVERAGE_LIMIT)
    filled    = all_dedup[all_dedup["filled"]]
    closed    = filled[filled["status"] == "closed"].sort_values("exit_date").copy()
    open_     = filled[filled["status"] == "open"].copy()
    # 買付枠オーバーで見送った、現在オープン相当の建玉
    skipped_open = all_dedup[(~all_dedup["filled"]) & (all_dedup["status"] == "open")].copy()

# ── 複利計算 ────────────────────────────────────────────────────────────────────

if not closed.empty:
    ret = closed["net_return"].values

    if is_bt:
        eq_base,  bt_dates = _portfolio_simulation(closed, position_ratio, 1.0, max_pos)
        eq_lev,   _        = _portfolio_simulation(closed, position_ratio, leverage, max_pos)
        eq_ref20, _        = _portfolio_simulation(closed, 0.20, 1.0, 5)
    else:
        eq_base  = _equity_from_lots(closed)
        eq_lev   = eq_base
        eq_ref20 = eq_base
        bt_dates = None

    cur_equity = float(eq_base[-1])
    gain       = cur_equity - _INITIAL_CAPITAL
    pct        = (cur_equity / _INITIAL_CAPITAL - 1) * 100
    dd         = _max_dd_pct(eq_base)

    # BT用追加指標
    cur_lev  = float(eq_lev[-1])
    gain_lev = cur_lev - _INITIAL_CAPITAL
    pct_lev  = (cur_lev / _INITIAL_CAPITAL - 1) * 100
    dd_lev   = _max_dd_pct(eq_lev)

    wins     = int((ret > 0).sum())
    wr       = float((ret > 0).mean())
    avg_win  = float(ret[ret > 0].mean()) if (ret > 0).any() else 0.0
    avg_loss = float(abs(ret[ret < 0].mean())) if (ret < 0).any() else 0.0
    avg_e    = float(ret.mean()) * 100
    kelly_f  = _kelly(wr, avg_win, avg_loss)
    kelly_lev = kelly_f / position_ratio if is_bt and position_ratio > 0 else 0.0
else:
    cur_equity = cur_lev = float(_INITIAL_CAPITAL)
    gain = gain_lev = pct = pct_lev = dd = dd_lev = 0.0
    wins = 0; wr = avg_win = avg_loss = avg_e = kelly_f = kelly_lev = 0.0
    ret = np.array([])
    eq_base = eq_lev = eq_ref20 = np.array([_INITIAL_CAPITAL])
    bt_dates = None

# ── 評価額カード ────────────────────────────────────────────────────────────────

sign = "+" if gain >= 0 else ""

if not is_bt:
    # ドライランモード: 実ロット基準の評価額 + 露出状況
    open_exposure = sum(_calc_lots(float(r["entry_price"]))[1] for _, r in open_.iterrows())
    cap_limit     = _INITIAL_CAPITAL * _LEVERAGE_LIMIT
    exp_used_pct  = open_exposure / cap_limit * 100 if cap_limit > 0 else 0.0

    st.markdown(
        f"""
        <div style="background:#1e1e2e;border-radius:16px;padding:32px 40px;
                    margin:8px 0 20px;text-align:center">
          <div style="color:#aaa;font-size:14px;margin-bottom:8px">
            実現評価額（ドライラン・ロット基準）
          </div>
          <div style="color:#fff;font-size:60px;font-weight:700;letter-spacing:-2px;line-height:1.1">
            ¥{cur_equity:,.0f}
          </div>
          <div style="color:{'#4ade80' if gain>=0 else '#f87171'};
                      font-size:22px;font-weight:600;margin-top:10px">
            {sign}¥{gain:,.0f}（{sign}{pct:.2f}%）
          </div>
          <div style="color:#666;font-size:13px;margin-top:6px">
            オープン露出: ¥{open_exposure:,.0f}　/　上限 ¥{cap_limit:,.0f}（{exp_used_pct:.0f}%使用）
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"1銘柄: 1ロット≥¥{_PER_POS_TARGET:,.0f}→1ロット / 未満→¥{_PER_POS_TARGET:,.0f}以内で最大ロット　"
        f"買付上限: 資本×{_LEVERAGE_LIMIT}倍（¥{cap_limit:,.0f}）　単元 {_LOT_SIZE}株"
    )
else:
    # バックテストモード: 従来の比較カード
    dd_warn        = dd_lev >= 15.0
    lev_col        = "#f87171" if dd_warn else ("#4ade80" if gain_lev > 0 else "#888")
    sign_b         = "+" if gain >= 0 else ""
    sign_l         = "+" if gain_lev >= 0 else ""
    margin_dep     = ratio_pct * max_pos
    exposure_pct_v = margin_dep * leverage

    if leverage == 1.0 and position_ratio == 0.20:
        st.markdown(
            f"""
            <div style="background:#1e1e2e;border-radius:16px;padding:32px 40px;
                        margin:8px 0 20px;text-align:center">
              <div style="color:#aaa;font-size:14px;margin-bottom:8px">
                現在の評価額（{position_ratio*100:.0f}%×{max_pos}銘柄・1.0x）
              </div>
              <div style="color:#fff;font-size:60px;font-weight:700;letter-spacing:-2px;line-height:1.1">
                ¥{cur_equity:,.0f}
              </div>
              <div style="color:{'#4ade80' if gain>=0 else '#f87171'};
                          font-size:22px;font-weight:600;margin-top:10px">
                {sign_b}¥{gain:,.0f}（{sign_b}{pct:.2f}%）
              </div>
              <div style="color:#666;font-size:13px;margin-top:6px">
                開始: ¥{_INITIAL_CAPITAL:,.0f}　最大投入: {margin_dep:.0f}%　複利計算
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        col_l, col_r = st.columns(2)
        with col_l:
            ref_val  = float(eq_ref20[-1])
            ref_gain = ref_val - _INITIAL_CAPITAL
            ref_pct  = (ref_val / _INITIAL_CAPITAL - 1) * 100
            ref_dd   = _max_dd_pct(eq_ref20)
            ref_sign = "+" if ref_gain >= 0 else ""
            st.markdown(
                f"""
                <div style="background:#1e1e2e;border-radius:16px;padding:24px 28px;
                            margin:8px 0 20px;text-align:center">
                  <div style="color:#aaa;font-size:13px;margin-bottom:6px">基準（20%×5銘柄・1.0x）</div>
                  <div style="color:#fff;font-size:40px;font-weight:700;letter-spacing:-1px">
                    ¥{ref_val:,.0f}
                  </div>
                  <div style="color:{'#4ade80' if ref_gain>=0 else '#f87171'};
                              font-size:18px;font-weight:600;margin-top:8px">
                    {ref_sign}¥{ref_gain:,.0f}（{ref_sign}{ref_pct:.2f}%）
                  </div>
                  <div style="color:#555;font-size:12px;margin-top:4px">
                    DD: {ref_dd:.1f}%　証拠金投入: 100%
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_r:
            dd_badge = " ⚠ DD超過" if dd_warn else ""
            st.markdown(
                f"""
                <div style="background:#1e1e2e;border-radius:16px;padding:24px 28px;
                            margin:8px 0 20px;text-align:center;
                            border:2px solid {lev_col}">
                  <div style="color:#aaa;font-size:13px;margin-bottom:6px">
                    試算（{ratio_pct}%×{max_pos}銘柄・{leverage:.1f}x）
                  </div>
                  <div style="color:#fff;font-size:40px;font-weight:700;letter-spacing:-1px">
                    ¥{cur_lev:,.0f}
                  </div>
                  <div style="color:{lev_col};font-size:18px;font-weight:600;margin-top:8px">
                    {sign_l}¥{gain_lev:,.0f}（{sign_l}{pct_lev:.2f}%）
                  </div>
                  <div style="color:{'#f87171' if dd_warn else '#555'};font-size:12px;margin-top:4px">
                    DD: {dd_lev:.1f}%　証拠金: {margin_dep}%　露出: {exposure_pct_v:.0f}%{dd_badge}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── KPI ─────────────────────────────────────────────────────────────────────────

if not is_bt:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("期待値 / トレード", f"{avg_e:+.2f}%")
    c2.metric("勝率", f"{wr:.0%}", f"{wins}勝 / {len(closed)}件")
    c3.metric("最大DD", f"{dd:.1f}%", delta_color="inverse")
    c4.metric("オープン銘柄数", len(open_))
else:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("期待値 / トレード", f"{avg_e:+.2f}%")
    c2.metric("勝率", f"{wr:.0%}", f"{wins}勝 / {len(closed)}件")
    c3.metric("DD（基準）", f"{dd:.1f}%", delta_color="inverse")
    c4.metric("DD（試算）", f"{dd_lev:.1f}%",
              "⚠ 15%超" if dd_lev >= 15 else "15%以内", delta_color="inverse")
    c5.metric("Kelly 推奨倍率", f"{kelly_lev:.1f}x" if kelly_lev > 0 else "—")
    c6.metric("オープン", len(open_), f"最大{max_pos}銘柄")
    if not closed.empty:
        total_dep = ratio_pct * max_pos
        st.caption(
            f"証拠金: {position_ratio*100:.0f}%/銘柄 × {max_pos}銘柄 = {total_dep:.0f}%投入　"
            f"露出: {total_dep * leverage:.0f}%　損益増幅: {leverage:.1f}x"
        )

# 2025年4月集中リスク警告（BTのみ）
if is_bt and not closed.empty:
    _dates = pd.to_datetime(closed["entry_date"]) if "entry_date" in closed.columns else pd.Series(dtype="datetime64[ns]")
    if not _dates.empty:
        _apr25_n = int(((_dates.dt.year == 2025) & (_dates.dt.month == 4)).sum())
        _total_n = len(closed)
        if _total_n > 0 and _apr25_n / _total_n >= 0.20:
            st.warning(
                f"⚠ 2025年4月（関税ショック）にトレードの **{_apr25_n/_total_n*100:.0f}%** が集中（{_apr25_n}/{_total_n}件）。"
                "1イベント依存の可能性があります。",
                icon="⚠️",
            )

st.divider()

# ── タブ ────────────────────────────────────────────────────────────────────────

tab_open, tab_closed, tab_chart = st.tabs(["オープン", "クローズ履歴", "エクイティ曲線"])

with tab_open:
    if is_bt:
        if open_.empty:
            st.info("現在オープンポジションはありません。")
        else:
            st.dataframe(
                open_[["symbol", "config_name", "entry_date", "entry_price",
                       "stop_price", "target_price", "max_exit_date"]],
                use_container_width=True, hide_index=True,
            )
    else:
        cap_limit = _INITIAL_CAPITAL * _LEVERAGE_LIMIT
        if open_.empty and skipped_open.empty:
            st.info("現在オープンポジションはありません。")
        else:
            if not open_.empty:
                lots_data = [_calc_lots(float(r["entry_price"])) for _, r in open_.iterrows()]
                total_exp = sum(r[1] for r in lots_data)
                remaining = cap_limit - total_exp

                d = open_[["symbol", "config_name", "signal_date", "entry_date",
                           "entry_price", "stop_price", "target_price", "max_exit_date"]].copy()
                d["ロット"]      = [r[0] for r in lots_data]
                d["実投資額"]    = [int(r[1]) for r in lots_data]
                d["目標利益(¥)"] = (
                    d["ロット"] * _LOT_SIZE * (d["target_price"] - d["entry_price"])
                ).round(0).astype(int)
                d = d.rename(columns={
                    "symbol": "銘柄", "config_name": "設定", "signal_date": "シグナル日",
                    "entry_date": "エントリー日", "entry_price": "単価",
                    "stop_price": "ストップ", "target_price": "目標", "max_exit_date": "期限",
                })
                st.caption(
                    f"買付総額: ¥{total_exp:,.0f}　残枠: ¥{remaining:,.0f}　"
                    f"（上限 ¥{cap_limit:,.0f} = 資本×{_LEVERAGE_LIMIT}倍）"
                )
                st.dataframe(
                    d[["銘柄", "設定", "シグナル日", "エントリー日", "単価", "ストップ", "目標", "期限",
                       "ロット", "実投資額", "目標利益(¥)"]],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("枠内で約定したオープンポジションはありません。")

            if not skipped_open.empty:
                st.warning(
                    f"⚠️ 買付枠（上限 ¥{cap_limit:,.0f}）が埋まり、以下 {len(skipped_open)} 銘柄は"
                    "見送り（このまま追わない）"
                )
                sd = skipped_open[["symbol", "config_name", "entry_date",
                                   "entry_price", "target_price"]].copy()
                sd["1ロット必要額"] = [int(_calc_lots(float(p))[1]) for p in sd["entry_price"]]
                sd = sd.rename(columns={
                    "symbol": "銘柄", "config_name": "設定", "entry_date": "エントリー日",
                    "entry_price": "単価", "target_price": "目標",
                })
                st.dataframe(
                    sd[["銘柄", "設定", "エントリー日", "単価", "目標", "1ロット必要額"]],
                    use_container_width=True, hide_index=True,
                )

with tab_closed:
    if closed.empty:
        st.info("クローズしたトレードはまだありません。")
    elif not is_bt:
        d = closed[["symbol", "config_name", "signal_date", "exit_date",
                    "entry_price", "exit_price", "exit_reason",
                    "gross_return", "net_return"]].copy()
        lots_data     = [_calc_lots(float(p)) for p in d["entry_price"]]
        d["ロット"]   = [r[0] for r in lots_data]
        d["実投資額"] = [int(r[1]) for r in lots_data]
        d["実損益(¥)"] = (
            d["ロット"] * _LOT_SIZE * (d["exit_price"] - d["entry_price"])
        ).round(0).astype(int)
        d["gross_return"] = (d["gross_return"] * 100).round(2)
        d["net_return"]   = (d["net_return"]   * 100).round(2)
        d["exit_reason"]  = d["exit_reason"].map(
            {"target": "目標達成", "stop": "ストップ", "time_stop": "タイム"}
        ).fillna(d["exit_reason"])
        d = d.rename(columns={
            "symbol": "銘柄", "config_name": "設定", "signal_date": "シグナル日",
            "exit_date": "クローズ日", "entry_price": "単価", "exit_price": "出口",
            "exit_reason": "理由", "gross_return": "グロス%", "net_return": "ネット%",
        })

        def _clr(v: float) -> str:
            return "color:green" if v > 0 else ("color:red" if v < 0 else "")

        st.dataframe(
            d[["銘柄", "設定", "シグナル日", "クローズ日", "単価", "出口", "理由",
               "グロス%", "ネット%", "ロット", "実投資額", "実損益(¥)"]].style.map(
                _clr, subset=["ネット%", "実損益(¥)"]
            ),
            use_container_width=True, hide_index=True,
        )
        st.caption("出口理由の内訳")
        st.bar_chart(d["理由"].value_counts())
    else:
        d = closed[["symbol", "config_name", "signal_date", "exit_date",
                    "entry_price", "exit_price", "exit_reason",
                    "gross_return", "net_return"]].copy()
        d["gross_return"] = (d["gross_return"] * 100).round(2)
        d["net_return"]   = (d["net_return"]   * 100).round(2)
        d["config_name"]  = d["config_name"].map(CONFIG_LABELS).fillna(d["config_name"])
        d["exit_reason"]  = d["exit_reason"].map(
            {"target": "目標達成", "stop": "ストップ", "time_stop": "タイム"}
        ).fillna(d["exit_reason"])
        d = d.rename(columns={
            "symbol": "銘柄", "config_name": "設定", "signal_date": "シグナル日",
            "exit_date": "クローズ日", "entry_price": "エントリー", "exit_price": "出口",
            "exit_reason": "理由", "gross_return": "グロス%", "net_return": "ネット%",
        })

        def _clr_bt(v: float) -> str:
            return "color:green" if v > 0 else ("color:red" if v < 0 else "")

        st.dataframe(
            d.style.map(_clr_bt, subset=["ネット%"]),
            use_container_width=True, hide_index=True,
        )
        st.caption("出口理由の内訳")
        st.bar_chart(d["理由"].value_counts())

with tab_chart:
    if closed.empty:
        st.info("クローズトレードが蓄積されるとエクイティ曲線が表示されます。")
    else:
        if is_bt and bt_dates is not None:
            x_axis = bt_dates.astype("datetime64[ms]").tolist()
            xlabel = "日付"
        else:
            x_axis = list(range(1, len(eq_base) + 1))
            xlabel = "トレード番号（銘柄単位）"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_axis, y=eq_base.tolist(), mode="lines",
            name="ロット基準" if not is_bt else "基準（20%×1.0x×5銘柄）",
            line=dict(color="#378add", width=2),
        ))

        if is_bt and (position_ratio != 0.20 or leverage != 1.0 or max_pos != 5):
            fig.add_trace(go.Scatter(
                x=x_axis, y=eq_lev.tolist(), mode="lines",
                name=f"試算（{position_ratio*100:.0f}%×{leverage:.1f}x×{max_pos}銘柄）",
                line=dict(color="#f59e0b", width=2, dash="dash"),
            ))

        dd_abs = np.maximum.accumulate(eq_base) - eq_base
        fig.add_trace(go.Scatter(
            x=x_axis, y=(eq_base - dd_abs).tolist(), mode="lines",
            name="DD",
            line=dict(color="#e24b4a", width=1, dash="dot"),
            fill="tozeroy", fillcolor="rgba(226,75,74,0.06)",
        ))

        fig.add_hline(y=_INITIAL_CAPITAL * 0.85, line_dash="dash",
                      line_color="#f87171", line_width=1,
                      annotation_text="DD 15% 限界", annotation_position="bottom right")
        fig.add_hline(y=_INITIAL_CAPITAL, line_dash="dash", line_color="#444", line_width=1)

        fig.update_layout(
            height=420, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title=xlabel,
            yaxis=dict(tickprefix="¥", tickformat=",", title="評価額（円）"),
        )
        st.plotly_chart(fig, use_container_width=True)

        if is_bt and kelly_f > 0:
            safe = leverage <= kelly_lev
            st.info(
                f"**Kelly 推奨倍率: {kelly_lev:.1f}x**　"
                f"（勝率 {wr:.0%} / 平均利益 {avg_win*100:.2f}% / 平均損失 {avg_loss*100:.2f}%）\n"
                f"現在の {leverage:.1f}x は {'✓ 安全圏です。' if safe else f'⚠ Kelly超過（推奨: {kelly_lev:.1f}x 以下）。'}"
            )
