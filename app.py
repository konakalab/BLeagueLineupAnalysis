import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 1. ページ全体の基本設定
st.set_page_config(page_title="B-League Analytics Dash", layout="wide")

# 2. データの読み込み処理
@st.cache_data
def load_all_data():
    # 基本テーブルの読み込み
    df_t = pd.read_csv('table_team.csv')
    df_p = pd.read_csv('table_players.csv')
    df_l = pd.read_csv('table_lineups.csv')
    
    # 全てのデータフレームの列名の空白を削除し、ID系を数値型に統一
    for df in [df_t, df_p, df_l]:
        df.columns = df.columns.str.strip()
        if 'TeamID' in df.columns:
            df['TeamID'] = pd.to_numeric(df['TeamID'], errors='coerce')
        if 'PlayerID' in df.columns:
            df['PlayerID'] = pd.to_numeric(df['PlayerID'], errors='coerce')

    period_str = "期間データなし"
    try:
        df_res = pd.read_csv('table_BLeagueResult_2025.csv')
        df_res.columns = df_res.columns.str.strip()
        df_res['Date'] = pd.to_datetime(df_res['Date'])
        finished = df_res.dropna(subset=['HomeScore', 'AwayScore'])
        if not finished.empty:
            s_dt = finished['Date'].min().strftime('%Y/%m/%d')
            e_dt = finished['Date'].max().strftime('%Y/%m/%d')
            period_str = f"{s_dt} から {e_dt}"
    except:
        pass
        
    return df_t, df_p, df_l, period_str

df_team, df_player, df_lineup, analysis_period = load_all_data()

# 3. サイドバーのフィルター
st.sidebar.header("検索フィルター")
sel_league = st.sidebar.selectbox("リーグ選択", df_team['League'].unique())
teams_in_league = df_team[df_team['League'] == sel_league]
sel_team_name = st.sidebar.selectbox("チーム選択", teams_in_league['Team'].unique())

team_row = df_team[df_team['Team'] == sel_team_name].iloc[0]
target_team_id = team_row['TeamID']

# 4. メイン画面
st.title(f"🏀 {sel_team_name} 分析ダッシュボード")
st.info(f"📅 分析対象期間：{analysis_period} まで")

tab1, tab2 = st.tabs(["選手分析", "ラインナップ分析"])

# --- タブ1: 選手分析 ---
with tab1:
    st.subheader("選手別 評価値分布 (全体比較)")
    df_all = df_player.copy()
    df_all['TotalApps'] = df_all['OFFApps'] + df_all['DEFApps']
    df_all['MarkerSize'] = np.sqrt(df_all['TotalApps'] + 1)
    df_all['is_selected'] = df_all['TeamID'] == target_team_id
    df_all['DisplayGroup'] = df_all['is_selected'].map({True: sel_team_name, False: 'その他'})
    df_all['Label'] = df_all.apply(lambda r: str(r['PlayerNo']) if r['is_selected'] else "", axis=1)
    
    df_all = df_all.sort_values('is_selected')

    fig = px.scatter(
        df_all, x='HensatiOFF', y='HensatiDEF', color='DisplayGroup', size='MarkerSize',
        text='Label', hover_name='PlayerNameJ',
        hover_data={'HensatiOFF': ':.1f', 'HensatiDEF': ':.1f', 'TotalApps': True, 'DisplayGroup': False, 'MarkerSize': False, 'Label': False},
        color_discrete_map={sel_team_name: '#EF553B', 'その他': '#E5ECF6'},
        labels={'HensatiOFF': '攻撃評価', 'HensatiDEF': '守備評価'},
        opacity=df_all['is_selected'].map({True: 1.0, False: 0.4})
    )
    fig.update_layout(xaxis=dict(range=[-30, 30], scaleanchor="y", scaleratio=1), yaxis=dict(range=[-30, 30]), width=800, height=800)
    fig.update_traces(textposition='top center')
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig, width="stretch") # 新しい仕様に対応
    
    st.write(f"### {sel_team_name} 選手一覧")
    disp_p = df_all[df_all['is_selected']].copy()
    if not disp_p.empty:
        st.dataframe(disp_p[['PlayerNo', 'PlayerNameJ', 'TotalApps', 'HensatiOFF', 'HensatiDEF']].sort_values('TotalApps', ascending=False), width="stretch", hide_index=True)

# --- タブ2: ラインナップ分析 ---
with tab2:
    st.subheader("ラインナップ別 評価値分布 (全体比較)")
    df_all_l = df_lineup.copy()
    df_all_l['TotalApps_L'] = df_all_l['OFFApps'] + df_all_l['DEFApps']
    df_all_l['MarkerSize'] = np.sqrt(df_all_l['TotalApps_L'] + 1)
    
    team_players = df_player[df_player['TeamID'] == target_team_id].sort_values('PlayerNo')
    p_options = ["指定なし"] + [f"{r['PlayerNo']} {r['PlayerNameJ']}" for _, r in team_players.iterrows()]
    sel_p_lineup = st.selectbox("強調表示する選手を選択", p_options)
    
    target_p_id = None
    if sel_p_lineup != "指定なし":
        sel_p_no = int(sel_p_lineup.split()[0])
        target_p_id = team_players[team_players['PlayerNo'] == sel_p_no]['PlayerID'].values[0]

    def classify(row):
        if target_p_id:
            ids = [row[f'Lineup_{i}'] for i in range(1, 6)]
            if target_p_id in ids: return "★注目選手含む"
        if row['TeamID'] == target_team_id: return sel_team_name
        return "その他"

    df_all_l['DisplayGroup'] = df_all_l.apply(classify, axis=1)
    df_all_l = df_all_l.sort_values('DisplayGroup', key=lambda x: x.map({"その他":0, sel_team_name:1, "★注目選手含む":2}))

    fig_l = px.scatter(
        df_all_l, x='HensatiOFF', y='HensatiDEF', color='DisplayGroup', size='MarkerSize',
        color_discrete_map={sel_team_name: '#EF553B', "★注目選手含む": '#19D3F3', 'その他': '#E5ECF6'},
        labels={'HensatiOFF': '攻撃評価', 'HensatiDEF': '守備評価'},
        opacity=df_all_l['DisplayGroup'].map(lambda x: 1.0 if x != "その他" else 0.3)
    )
    fig_l.update_layout(xaxis=dict(range=[-30, 30], scaleanchor="y", scaleratio=1), yaxis=dict(range=[-30, 30]), width=800, height=800)
    fig_l.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_l.add_vline(x=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_l, width="stretch") # 新しい仕様に対応

    st.write(f"### {sel_team_name} ラインナップ一覧")
    disp_l = df_all_l[df_all_l['TeamID'] == target_team_id].copy()
    if not disp_l.empty:
        st.dataframe(disp_l[['OFFApps', 'DEFApps', 'RatingOFF', 'HensatiOFF', 'HensatiDEF']].sort_values('OFFApps', ascending=False), width="stretch", hide_index=True)
