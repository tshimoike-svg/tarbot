"""Phase 1 ドライラン監視ダッシュボード（Streamlit）。

起動:
    uv sync --extra dashboard
    uv run streamlit run dashboard/app.py

data/db/forward_signals.sqlite を読み、5分ごとに自動更新する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.signal_store import SignalStore

_DB = Path(__file__).parent.parent / "data/db/forward_signals.sqlite"

st.set_page_config(page_title="Phase 1 監視", page_icon="📈", layout="wide")

# ── データ取得（5分キャッシュ）─────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _load() -> pd.DataFrame:
    if not _DB.exists():
        return pd.DataFrame()
    with SignalStore(_DB) as s:
        return s.get_all()


df = _load()

# ── ヘッダー ────────────────────────────────────────────────────────────────────

st.title("Phase 1 ドライラン監視")
col_hd1, col_hd2 = st.columns([3, 1])
with col_hd1:
    st.caption(
        f"DB: `{_DB}`  |  "
        f"最終更新: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
    )
with col_hd2:
    if st.button("🔄 更新", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── 設定フィルタ ────────────────────────────────────────────────────────────────

CONFIG_LABELS = {
    "config_iii": "③ rsi<30",
    "config_iv":  "④ rsi<35（推奨）",
    "config_v":   "⑤ rsi<40",
    "all":        "全設定合算",
}
cfg_sel = st.selectbox("設定", list(CONFIG_LABELS.keys()), index=1, format_func=lambda k: CONFIG_LABELS[k])

if df.empty:
    st.info("シグナルがまだありません。daily_scan.py が実行されると表示されます。")
    st.stop()

_df = df if cfg_sel == "all" else df[df["config_name"] == cfg_sel]
closed = _df[_df["status"] == "closed"].copy()
open_ = _df[_df["status"] == "open"].copy()

# ── KPI カード ──────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)

avg_e = closed["net_return"].mean() * 100 if not closed.empty else 0.0
wr = float((closed["net_return"] > 0).mean()) if not closed.empty else 0.0
total_ret = closed["net_return"].sum() * 100 if not closed.empty else 0.0
wins = int((closed["net_return"] > 0).sum()) if not closed.empty else 0

# 最大ドローダウン（口座資金 0.5% リスク基準）
if not closed.empty:
    eq = closed["net_return"].cumsum()
    peak = eq.cummax()
    dd = float((peak - eq).max()) * 100
else:
    dd = 0.0

c1.metric("期待値", f"{avg_e:+.2f}%", help="コスト控除後 / トレード")
c2.metric("勝率", f"{wr:.0%}", f"{wins}勝/{len(closed)}件")
c3.metric("累計リターン", f"{total_ret:+.1f}%")
c4.metric("最大DD", f"{dd:.1f}%", delta_color="inverse")
c5.metric("オープン", len(open_))

# バックテスト基準との比較（設定 ④ の場合のみ）
if cfg_sel == "config_iv" and not closed.empty:
    bt_e = 7.37  # バックテスト期待値
    delta_e = avg_e - bt_e
    st.caption(
        f"バックテスト比（④ 期待値 +{bt_e:.2f}%）: "
        f"{'上回り' if delta_e >= 0 else '下回り'} {delta_e:+.2f}%  "
        f"（{'良好' if delta_e >= -2 else '要確認'}）"
    )

st.divider()

# ── タブ ────────────────────────────────────────────────────────────────────────

tab_open, tab_closed, tab_chart = st.tabs(["オープン", "クローズ履歴", "エクイティ曲線"])

# ── オープンポジション ───────────────────────────────────────────────────────────

with tab_open:
    if open_.empty:
        st.info("現在オープンポジションはありません。")
    else:
        display_open = open_[[
            "symbol", "config_name", "signal_date", "entry_date",
            "entry_price", "stop_price", "target_price", "max_exit_date",
        ]].copy()
        display_open["config_name"] = display_open["config_name"].map(CONFIG_LABELS).fillna(display_open["config_name"])
        display_open.columns = ["銘柄", "設定", "シグナル日", "エントリー日", "エントリー価格", "ストップ", "目標", "期限"]
        st.dataframe(display_open, use_container_width=True, hide_index=True)

# ── クローズ履歴 ────────────────────────────────────────────────────────────────

with tab_closed:
    if closed.empty:
        st.info("クローズしたトレードはまだありません。")
    else:
        display_closed = closed[[
            "symbol", "config_name", "signal_date", "exit_date",
            "entry_price", "exit_price", "exit_reason", "gross_return", "net_return",
        ]].copy()
        display_closed["gross_return"] = (display_closed["gross_return"] * 100).round(2)
        display_closed["net_return"] = (display_closed["net_return"] * 100).round(2)
        display_closed["config_name"] = display_closed["config_name"].map(CONFIG_LABELS).fillna(display_closed["config_name"])
        reason_map = {"target": "目標達成", "stop": "ストップ", "time_stop": "タイム"}
        display_closed["exit_reason"] = display_closed["exit_reason"].map(reason_map).fillna(display_closed["exit_reason"])
        display_closed.columns = ["銘柄", "設定", "シグナル日", "クローズ日", "エントリー", "出口価格", "理由", "グロス%", "ネット%"]

        def _color(val: float) -> str:
            return "color: green" if val > 0 else ("color: red" if val < 0 else "")

        st.dataframe(
            display_closed.style.map(_color, subset=["ネット%"]),
            use_container_width=True, hide_index=True,
        )

        # 出口理由の内訳
        st.caption("出口理由の内訳")
        reason_counts = display_closed["理由"].value_counts()
        st.bar_chart(reason_counts)

# ── エクイティ曲線 ──────────────────────────────────────────────────────────────

with tab_chart:
    if closed.empty:
        st.info("クローズトレードが蓄積されるとエクイティ曲線が表示されます。")
    else:
        eq_series = closed.sort_values("exit_date")["net_return"].cumsum() * 100
        peak = eq_series.cummax()
        dd_series = eq_series - peak

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(eq_series) + 1)),
            y=eq_series.round(2).tolist(),
            mode="lines+markers",
            name="累計リターン %",
            line=dict(color="#378add", width=2),
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=list(range(1, len(dd_series) + 1)),
            y=dd_series.round(2).tolist(),
            mode="lines",
            name="ドローダウン %",
            line=dict(color="#e24b4a", width=1.5, dash="dot"),
            fill="tozeroy",
            fillcolor="rgba(226,75,74,0.08)",
        ))
        # バックテスト期待値の参考線（設定 ④）
        if cfg_sel == "config_iv":
            n = len(closed)
            bt_line = [7.37 * i / 100 * 100 for i in range(1, n + 1)]  # 1トレードあたり +7.37%
            fig.add_trace(go.Scatter(
                x=list(range(1, n + 1)), y=[round(v, 2) for v in bt_line],
                mode="lines", name="バックテスト期待線（参考）",
                line=dict(color="#aaa", width=1, dash="dot"),
            ))

        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="トレード番号",
            yaxis_title="累計 %",
            yaxis=dict(ticksuffix="%"),
        )
        st.plotly_chart(fig, use_container_width=True)
