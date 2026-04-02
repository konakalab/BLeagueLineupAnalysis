import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 1. ページ全体の基本設定
st.set_page_config(page_title="B-League Analytics Dash", layout="wide")

# 2. データの読み込み処理（最小限の読み込みのみキャッシュ）
@st.cache_data
def load_raw_csv():
    df_t = pd.read_csv('table_team.csv')
    df_p = pd.read_csv('table_players.csv')
    df_l = pd.read_csv('table_lineups.csv')
    
    # 全データフレームの列名クリーンアップとIDの数値化
    for df in [df_t, df_p, df_l]:
        df.columns = df.columns.str.strip()
        for col in ['TeamID', 'PlayerID', 'Lineup_1', 'Lineup_2', 'Lineup_3', 'Lineup_4', 'Lineup_5']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 期間情報の取得
    period_str = "データ期間不明"
    try:
        df_res = pd.read_csv('table_BLeagueResult_2025.csv')
        df_res.columns = df_res.columns.str.strip()
        df_res['Date'] = pd.to_datetime(df_res['Date'])
        finished = df_res.dropna(subset=['HomeScore', 'AwayScore'])
        if not finished.empty:
            period_str = f"{finished['Date'].min().strftime('%Y/%m/%d')} 〜 {finished['Date'].max().strftime('%Y/%m/%d')}"
    except:
        pass
        
    return df_t, df_p, df_l, period_str

df_team, df_player, df_lineup, analysis_period = load_raw_csv()

# 3. サイドバーのフィルター（ここが変更されると下全体が再実行されます）
st.sidebar.header("検索フィルター")
list_league = df_team['League'].unique()
sel_league = st.sidebar.selectbox("リーグ選択", list_league)

# 選択リーグに所属するチームのみ抽出
teams_in_league = df_team[df_team['League'] == sel_league].sort_values('Team')
sel_team_name = st.sidebar.selectbox("チーム選択", teams_in_league['Team'].unique())

# 選択チーム情報の確定
team_row = teams_in_league[teams_in_league['Team'] == sel_team_name].iloc[0]
target_team_id = int(team_row['TeamID'])

# 4. メイン画面
st.title(f"🏀 {sel_team_name} 分析ダッシュボード")
st.info(f"📅 分析対象期間：{analysis_period}")

tab1, tab2 = st.tabs(["選手分析", "ラインナップ分析"])

# --- タブ1: 選手分析 ---
with tab1:
    st.subheader("選手別 評価値分布")
    
    df_all_p = df_player.copy()
    # フィルタリングを確実にするため型を合わせる
    df_all_p['is_selected'] = df_all_p['TeamID'] == target_team_id
    
    # 描画用データの作成
    df_all_p['TotalApps'] = df_all_p['OFFApps'] + df_all_p['DEFApps']
    df_all_p['MarkerSize'] = np.sqrt(df_all_p['TotalApps'] + 1)
    df_all_p['DisplayGroup'] = df_all_p['is_selected'].map({True: sel_team_name, False: 'その他'})
    df_all_p['Label'] = df_all_p.apply(lambda r: str(int(r['PlayerNo'])) if r['is_selected'] and not pd.isna(r['PlayerNo']) else "", axis=1)
    
    # 選択チームを上に重ねるためにソート
    df_all_p = df_all_p.sort_values('is_selected')

    fig = px.scatter(
        df_all_p, x='HensatiOFF', y='HensatiDEF', color='DisplayGroup', size='MarkerSize',
        text='Label', hover_name='PlayerNameJ',
        hover_data={'HensatiOFF': ':.1f', 'HensatiDEF': ':.1f', 'TotalApps': True, 'DisplayGroup': False, 'MarkerSize': False, 'Label': False},
        color_discrete_map={sel_team_name: '#EF553B', 'その他': '#E5ECF6'},
        labels={'HensatiOFF': '攻撃評価', 'HensatiDEF': '守備評価'},
        opacity=df_all_p['is_selected'].map({True: 1.0, False: 0.4})
    )
    fig.update_layout(xaxis=dict(range=[-30, 30], scaleanchor="y", scaleratio=1), yaxis=dict(range=[-30, 30]), width=700, height=700)
    fig.update_traces(textposition='top center')
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig, width="stretch")

    # 一覧表
    df_tp = df_all_p[df_all_p['is_selected']].copy()
    if not df_tp.empty:
        st.write("### 選手一覧")
        st.dataframe(df_tp[['PlayerNo', 'PlayerNameJ', 'TotalApps', 'HensatiOFF', 'HensatiDEF']].sort_values('TotalApps', ascending=False), width="stretch", hide_index=True)

# --- タブ2: ラインナップ分析 ---
with tab2:
    st.subheader("ラインナップ別 評価値分布")
    
    df_all_l = df_lineup.copy()
    df_all_l['TotalApps_L'] = df_all_l['OFFApps'] + df_all_l['DEFApps']
    df_all_l['MarkerSize'] = np.sqrt(df_all_l['TotalApps_L'] + 1)
    
    # チーム所属選手の強調フィルター
    team_players = df_player[df_player['TeamID'] == target_team_id].sort_values('PlayerNo')
    p_options = ["指定なし"] + [f"{int(r['PlayerNo'])} {r['PlayerNameJ']}" for _, r in team_players.iterrows()]
    sel_p_lineup = st.selectbox("強調表示する選手を選択", p_options)
    
    target_p_id = None
    if sel_p_lineup != "指定なし":
        sel_p_no = int(sel_p_lineup.split()[0])
        target_p_id = team_players[team_players['PlayerNo'] == sel_p_no]['PlayerID'].values[0]

    def classify_l(row):
        if target_p_id:
            if target_p_id in [row[f'Lineup_{i}'] for i in range(1, 6)]: return "★注目選手含む"
        if row['TeamID'] == target_team_id: return sel_team_name
        return "その他"

    df_all_l['DisplayGroup'] = df_all_l.apply(classify_l, axis=1)
    df_all_l = df_all_l.sort_values('DisplayGroup', key=lambda x: x.map({"その他":0, sel_team_name:1, "★注目選手含む":2}))

    fig_l = px.scatter(
        df_all_l, x='HensatiOFF', y='HensatiDEF', color='DisplayGroup', size='MarkerSize',
        color_discrete_map={sel_team_name: '#EF553B', "★注目選手含む": '#19D3F3', 'その他': '#E5ECF6'},
        labels={'HensatiOFF': '攻撃評価', 'HensatiDEF': '守備評価'},
        opacity=df_all_l['DisplayGroup'].map(lambda x: 1.0 if x != "その他" else 0.3)
    )
    fig_l.update_layout(xaxis=dict(range=[-30, 30], scaleanchor="y", scaleratio=1), yaxis=dict(range=[-30, 30]), width=700, height=700)
    fig_l.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_l.add_vline(x=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_l, width="stretch")

    # テーブル
    disp_l = df_all_l[df_all_l['TeamID'] == target_team_id].copy()
    if not disp_l.empty:
        st.write("### ラインナップ一覧")
        st.dataframe(disp_l[['OFFApps', 'DEFApps', 'RatingOFF', 'HensatiOFF', 'HensatiDEF']].sort_values('OFFApps', ascending=False), width="stretch", hide_index=True)
