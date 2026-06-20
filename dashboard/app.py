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

_DB     = Path(__file__).parent.parent / "data/db/forward_signals.sqlite"
_BT_CSV = Path(__file__).parent.parent / "data/db/bt_trades.csv"
_INITIAL_CAPITAL = 1_000_000

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


def _compound_equity(returns: np.ndarray, effective_ratio: float,
                     initial: float = _INITIAL_CAPITAL) -> np.ndarray:
    """順次複利（ドライラン実績用: トレードが1本ずつ順番にクローズする前提）。"""
    eq = np.empty(len(returns))
    cur = initial
    for i, r in enumerate(returns):
        cur *= (1.0 + r * effective_ratio)
        eq[i] = cur
    return eq


def _portfolio_simulation(
    trades_df: pd.DataFrame,
    position_ratio: float,
    leverage: float,
    max_positions: int,
    initial: float = _INITIAL_CAPITAL,
) -> tuple[np.ndarray, np.ndarray]:
    """資本プール共有の並列シミュレーション。

    信用取引モデル:
      - 証拠金(現金拘束) = total_equity × position_ratio
      - 損益の増幅      = net_return × leverage
      - 証拠金投入上限  = 100%（position_ratio × max_positions ≈ 100%）
      → レバレッジを上げても現金の拘束は変わらず、損益だけ倍になる

    priority 列が存在する場合:
      - priority=1（メイン）が同日に priority=2（補助）より先にスロットを獲得する
      - 同一銘柄の重複エントリーを防ぐ（Tier1/Tier2 の二重乗りを排除）
    """
    df = trades_df.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])

    # priority 列があれば優先度昇順（1→2）、なければ entry_date のみ
    sort_cols = ["entry_date", "priority"] if "priority" in df.columns else ["entry_date"]
    has_symbol = "symbol" in df.columns
    df = df.sort_values(sort_cols).reset_index(drop=True)

    cash = initial
    open_pos: list[dict] = []  # {exit_date, margin, net_return, symbol?}
    equity_log: list[tuple[pd.Timestamp, float]] = []
    all_events = sorted(set(df["entry_date"].tolist() + df["exit_date"].tolist()))
    entry_idx = 0

    for event_date in all_events:
        # 決済: 証拠金 + 損益（leverage 倍）を返還
        still_open = []
        for pos in open_pos:
            if pos["exit_date"] <= event_date:
                cash += pos["margin"] * (1.0 + pos["net_return"] * leverage)
            else:
                still_open.append(pos)
        open_pos = still_open

        # エントリー: 証拠金 = total_equity × position_ratio
        while entry_idx < len(df):
            row = df.iloc[entry_idx]
            if row["entry_date"] > event_date:
                break
            entry_idx += 1
            if len(open_pos) >= max_positions:
                continue
            # 同一銘柄の重複エントリーを防ぐ（Tier2 が Tier1 と同銘柄に乗り直すのを排除）
            if has_symbol:
                open_symbols = {p["symbol"] for p in open_pos}
                if row["symbol"] in open_symbols:
                    continue
            total_equity = cash + sum(p["margin"] for p in open_pos)
            margin = total_equity * position_ratio
            if margin > cash:
                margin = cash
            if margin < initial * 0.001:
                continue
            cash -= margin
            pos_entry: dict = {
                "exit_date":  row["exit_date"],
                "margin":     margin,
                "net_return": row["net_return"],
            }
            if has_symbol:
                pos_entry["symbol"] = row["symbol"]
            open_pos.append(pos_entry)

        total = cash + sum(p["margin"] for p in open_pos)
        equity_log.append((event_date, total))

    # 残ポジション強制決済
    for pos in open_pos:
        cash += pos["margin"] * (1.0 + pos["net_return"] * leverage)
    equity_log.append((all_events[-1] if all_events else pd.Timestamp.now(), cash))

    dates  = np.array([e[0] for e in equity_log], dtype="datetime64[ns]")
    equity = np.array([e[1] for e in equity_log])
    return equity, dates


def _max_dd_pct(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return float(dd.max()) * 100


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
if is_bt:
    st.info("過去2年のバックテスト個別トレードを使ったシミュレーションです。レバレッジ・銘柄数を調整してください。", icon="📊")

# ── フィルタ & シミュレーション設定 ─────────────────────────────────────────────

CONFIG_LABELS = {
    "config_iii":  "③ rsi<30（高精度・少トレード）",
    "config_iv":   "④ rsi<35",
    "config_v":    "⑤ rsi<40（推奨メイン）",
    "config_t2c":  "補助C（lb=30, z≥2.0, RSI<40, US フィルタあり）",
    "main+aux":    "⑤メイン + 補助C（lb=30 空きスロット埋め）",
    "all":         "全設定合算",
}

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
        help="1ポジションで資産の何%を証拠金として使うか。銘柄数は自動計算（比率×銘柄数≈100%）",
    )
    position_ratio = ratio_pct / 100.0
    # 銘柄数は比率から自動導出（積が100%に近くなるよう連動）
    max_pos = max(1, int(1.0 / position_ratio))
    st.caption(f"→ 自動: **{max_pos}銘柄**（{ratio_pct}% × {max_pos} = {ratio_pct*max_pos}%）")
with col_lev:
    leverage = st.slider(
        "信用倍率", min_value=1.0, max_value=3.0,
        value=1.0, step=0.1, format="%.1fx",
        help="損益を何倍に増幅するか。証拠金は変わらず損益だけ拡大（信用買いのモデル）",
    )

# 証拠金投入率 ≈ 100%、エクスポージャーは leverage 倍
margin_deployment = ratio_pct * max_pos   # ≈ 100% (証拠金拘束率)
exposure_pct      = margin_deployment * leverage
total_deployment  = margin_deployment     # 証拠金ベースの投入率（カード表示用）
effective_ratio   = position_ratio * leverage  # ドライランモード順次複利用

df = _load_bt() if is_bt else _load_live()

# ── データなし ───────────────────────────────────────────────────────────────────

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

if cfg_sel == "all":
    _df = df.copy()
elif cfg_sel == "main+aux":
    _df = df[df["config_name"].isin(["config_v", "config_t2c"])].copy()
else:
    _df = df[df["config_name"] == cfg_sel].copy()
closed = _df[_df["status"] == "closed"].sort_values("exit_date").copy()
open_  = _df[_df["status"] == "open"].copy()

# ── 複利計算 ────────────────────────────────────────────────────────────────────

if not closed.empty:
    ret = closed["net_return"].values

    if is_bt:
        # バックテストモード: 資本プール並列シミュレーション
        eq_base, bt_dates = _portfolio_simulation(closed, position_ratio, 1.0, max_pos)
        eq_lev,  _        = _portfolio_simulation(closed, position_ratio, leverage, max_pos)
        # 基準（20%×1x×5銘柄）も常時計算
        eq_ref20, _       = _portfolio_simulation(closed, 0.20, 1.0, 5)
    else:
        # ドライランモード: 順次複利
        eq_base = _compound_equity(ret, position_ratio)
        eq_lev  = _compound_equity(ret, effective_ratio)
        eq_ref20 = _compound_equity(ret, 0.20)
        bt_dates = None

    cur_base  = float(eq_base[-1])
    cur_lev   = float(eq_lev[-1])
    gain_base = cur_base - _INITIAL_CAPITAL
    gain_lev  = cur_lev  - _INITIAL_CAPITAL
    pct_base  = (cur_base / _INITIAL_CAPITAL - 1) * 100
    pct_lev   = (cur_lev  / _INITIAL_CAPITAL - 1) * 100
    dd_base   = _max_dd_pct(eq_base)
    dd_lev    = _max_dd_pct(eq_lev)

    wins     = int((ret > 0).sum())
    wr       = float((ret > 0).mean())
    avg_win  = float(ret[ret > 0].mean()) if (ret > 0).any() else 0.0
    avg_loss = float(abs(ret[ret < 0].mean())) if (ret < 0).any() else 0.0
    avg_e    = float(ret.mean()) * 100
    kelly_f  = _kelly(wr, avg_win, avg_loss)
    kelly_lev = kelly_f / position_ratio if position_ratio > 0 else 0.0
else:
    cur_base = cur_lev = float(_INITIAL_CAPITAL)
    gain_base = gain_lev = pct_base = pct_lev = 0.0
    dd_base = dd_lev = 0.0
    wins = 0; wr = avg_win = avg_loss = avg_e = kelly_f = kelly_lev = 0.0
    ret = np.array([])
    eq_base = eq_lev = eq_ref20 = np.array([_INITIAL_CAPITAL])
    bt_dates = None

# ── 評価額カード ────────────────────────────────────────────────────────────────

dd_warn  = dd_lev >= 15.0
sign_b   = "+" if gain_base >= 0 else ""
sign_l   = "+" if gain_lev  >= 0 else ""
lev_col  = "#f87171" if dd_warn else ("#4ade80" if gain_lev > 0 else "#888")

if leverage == 1.0 and position_ratio == 0.20:
    # デフォルト設定: シンプル1カラム
    st.markdown(
        f"""
        <div style="background:#1e1e2e;border-radius:16px;padding:32px 40px;
                    margin:8px 0 20px;text-align:center">
          <div style="color:#aaa;font-size:14px;margin-bottom:8px">
            現在の評価額（{position_ratio*100:.0f}%×{max_pos}銘柄・1.0x）
          </div>
          <div style="color:#fff;font-size:60px;font-weight:700;letter-spacing:-2px;line-height:1.1">
            ¥{cur_base:,.0f}
          </div>
          <div style="color:{'#4ade80' if gain_base>=0 else '#f87171'};
                      font-size:22px;font-weight:600;margin-top:10px">
            {sign_b}¥{gain_base:,.0f}（{sign_b}{pct_base:.2f}%）
          </div>
          <div style="color:#666;font-size:13px;margin-top:6px">
            開始: ¥{_INITIAL_CAPITAL:,.0f}　最大投入: {total_deployment:.0f}%　複利計算
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # 設定変更時: 2カラム比較
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
              <div style="color:#aaa;font-size:13px;margin-bottom:6px">
                基準（20%×5銘柄・1.0x）
              </div>
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
        dd_badge = ' ⚠ DD超過' if dd_warn else ""
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
                DD: {dd_lev:.1f}%　証拠金: {margin_deployment}%　エクスポージャー: {exposure_pct:.0f}%{dd_badge}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── KPI ─────────────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("期待値 / トレード", f"{avg_e:+.2f}%")
c2.metric("勝率", f"{wr:.0%}", f"{wins}勝/{len(closed)}件")
c3.metric("DD（基準）",  f"{dd_base:.1f}%", delta_color="inverse")
c4.metric(f"DD（試算）", f"{dd_lev:.1f}%",
          "⚠ 15%超" if dd_warn else "15%以内", delta_color="inverse")
kelly_str = f"{kelly_lev:.1f}x" if kelly_lev > 0 else "—"
c5.metric("Kelly 推奨倍率", kelly_str,
          help="実測勝率・損益比から算出。この倍率を超えると長期的に破産リスクが高まる")
c6.metric("オープン", len(open_), f"最大{max_pos}銘柄")

# 設定インサイト
if not closed.empty:
    pos_tip = (
        f"証拠金: {position_ratio*100:.0f}%/銘柄 × {max_pos}銘柄 = {total_deployment:.0f}%投入　"
        f"エクスポージャー: {exposure_pct:.0f}%　損益増幅: {leverage:.1f}x"
    )
    if kelly_lev > 0:
        safe = leverage <= kelly_lev
        pos_tip += f"　Kelly推奨: {kelly_lev:.1f}x（現在{'安全圏' if safe else '⚠ 超過'}）"
    st.caption(pos_tip)

# バックテストモード: イベント集中リスク警告
if is_bt and not closed.empty:
    _dates = pd.to_datetime(closed["entry_date"]) if "entry_date" in closed.columns else pd.Series(dtype="datetime64[ns]")
    if not _dates.empty:
        _apr25 = (_dates.dt.year == 2025) & (_dates.dt.month == 4)
        _apr25_n = int(_apr25.sum())
        _total_n = len(closed)
        _apr25_pct = _apr25_n / _total_n * 100 if _total_n > 0 else 0
        if _apr25_pct >= 20:
            st.warning(
                f"⚠ 2025年4月（関税ショック急落→急回復）にトレードの **{_apr25_pct:.0f}%** が集中しています（{_apr25_n}/{_total_n}件）。"
                f"この1イベントを除いた実力値はより低くなります。",
                icon="⚠️",
            )

st.divider()

# ── タブ ────────────────────────────────────────────────────────────────────────

tab_open, tab_closed, tab_chart = st.tabs(["オープン", "クローズ履歴", "エクイティ曲線"])

with tab_open:
    if open_.empty:
        st.info("現在オープンポジションはありません。")
    else:
        d = open_[["symbol","config_name","signal_date","entry_date",
                   "entry_price","stop_price","target_price","max_exit_date"]].copy()
        d["config_name"] = d["config_name"].map(CONFIG_LABELS).fillna(d["config_name"])
        d.columns = ["銘柄","設定","シグナル日","エントリー日","エントリー価格","ストップ","目標","期限"]
        st.dataframe(d, use_container_width=True, hide_index=True)

with tab_closed:
    if closed.empty:
        st.info("クローズしたトレードはまだありません。")
    else:
        d = closed[["symbol","config_name","signal_date","exit_date",
                    "entry_price","exit_price","exit_reason","gross_return","net_return"]].copy()
        d["gross_return"] = (d["gross_return"] * 100).round(2)
        d["net_return"]   = (d["net_return"]   * 100).round(2)
        d["config_name"]  = d["config_name"].map(CONFIG_LABELS).fillna(d["config_name"])
        d["exit_reason"]  = d["exit_reason"].map(
            {"target":"目標達成","stop":"ストップ","time_stop":"タイム"}
        ).fillna(d["exit_reason"])
        d.columns = ["銘柄","設定","シグナル日","クローズ日","エントリー","出口価格","理由","グロス%","ネット%"]

        def _color(v: float) -> str:
            return "color:green" if v > 0 else ("color:red" if v < 0 else "")

        st.dataframe(d.style.map(_color, subset=["ネット%"]), use_container_width=True, hide_index=True)
        st.caption("出口理由の内訳")
        st.bar_chart(d["理由"].value_counts())

with tab_chart:
    if closed.empty:
        st.info("クローズトレードが蓄積されるとエクイティ曲線が表示されます。")
    else:
        # X軸: BTは日付、ドライランはトレード番号
        if is_bt and bt_dates is not None:
            x_base = bt_dates.astype("datetime64[ms]").tolist()
            x_ref  = x_base
            xlabel = "日付"
        else:
            x_base = list(range(1, len(eq_base) + 1))
            x_ref  = list(range(1, len(eq_ref20) + 1))
            xlabel = "トレード番号"

        fig = go.Figure()

        # 基準ライン（20%×1x×5銘柄）
        fig.add_trace(go.Scatter(
            x=x_ref, y=eq_ref20.tolist(), mode="lines",
            name="基準（20%×1.0x×5銘柄）",
            line=dict(color="#378add", width=2),
        ))

        # 試算ライン（設定変更時）
        if position_ratio != 0.20 or leverage != 1.0 or max_pos != 5:
            fig.add_trace(go.Scatter(
                x=x_base, y=eq_lev.tolist(), mode="lines",
                name=f"試算（{position_ratio*100:.0f}%×{leverage:.1f}x×{max_pos}銘柄）",
                line=dict(color="#f59e0b", width=2, dash="dash"),
            ))

        # ドローダウン（基準）
        dd_abs = np.maximum.accumulate(eq_ref20) - eq_ref20
        fig.add_trace(go.Scatter(
            x=x_ref, y=(eq_ref20 - dd_abs).tolist(), mode="lines",
            name="DD（基準）",
            line=dict(color="#e24b4a", width=1, dash="dot"),
            fill="tozeroy", fillcolor="rgba(226,75,74,0.06)",
        ))

        # DD 15% 限界ライン
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

        # Kelly 解説
        if kelly_f > 0:
            safe = leverage <= kelly_lev
            st.info(
                f"**Kelly 推奨倍率: {kelly_lev:.1f}x**  "
                f"（勝率 {wr:.0%} / 平均利益 {avg_win*100:.2f}% / 平均損失 {avg_loss*100:.2f}%）  \n"
                f"現在の {leverage:.1f}x は {'✓ 安全圏です。' if safe else f'⚠ Kelly超過です（推奨: {kelly_lev:.1f}x 以下）。'}  \n"
                f"銘柄数を {max_pos} → {max_pos*2} に増やすと、同じリターンでDDを約 {100/max_pos:.0f}% 低減できます（分散効果）。"
            )
