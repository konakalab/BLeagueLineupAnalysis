import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 1. ページ全体の基本設定
st.set_page_config(page_title="B-League Analytics Dash", layout="wide")

# 2. データの読み込み処理
@st.cache_data
def load_raw_csv():
    df_t = pd.read_csv('table_team.csv')
    df_p = pd.read_csv('table_players.csv')
    df_l = pd.read_csv('table_lineups.csv')
    
    for df in [df_t, df_p, df_l]:
        df.columns = [str(c).strip().replace('\n', '') for c in df.columns]
        # orderを含む主要ID系を数値型に変換
        cols_to_fix = ['TeamID', 'PlayerID', 'Lineup_1', 'Lineup_2', 'Lineup_3', 'Lineup_4', 'Lineup_5', 'Order']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
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

# 3. サイドバーのフィルター
st.sidebar.header("検索フィルター")

# リーグの選択（CSVの登場順を維持）
list_league = df_team['League'].unique()
sel_league = st.sidebar.selectbox("リーグ選択", list_league)

# --- 修正ポイント：チームを order カラムでソート ---
# 選択されたリーグのチームのみを抽出し、orderカラムで昇順ソート
teams_in_league = df_team[df_team['League'] == sel_league].copy()

if 'order' in teams_in_league.columns:
    # orderでソート。orderが同じ場合はTeamID順にする
    teams_sorted = teams_in_league.sort_values(by=['Order'])
else:
    # orderがない場合はTeamID順
    teams_sorted = teams_in_league.sort_values(by='TeamID')

# ソート済みのデータフレームからチーム名のリストを作成（これで表示順が固定される）
list_teams = teams_sorted['Team'].tolist()
sel_team_name = st.sidebar.selectbox("チーム選択", list_teams)

# 選択されたチームの情報を取得
team_row = teams_sorted[teams_sorted['Team'] == sel_team_name].iloc[0]
target_team_id = int(team_row['TeamID'])

# 4. メイン画面
st.title(f"🏀 {sel_team_name} 分析ダッシュボード")
st.info(f"📅 分析対象期間：{analysis_period}")

tab1, tab2 = st.tabs(["選手分析", "ラインナップ分析"])

# --- タブ1: 選手分析 ---
with tab1:
    st.subheader("選手別 評価値分布")
    df_all_p = df_player.copy()
    df_all_p['is_selected'] = df_all_p['TeamID'] == target_team_id
    df_all_p['TotalApps'] = df_all_p['OFFApps'] + df_all_p['DEFApps']
    df_all_p['MarkerSize'] = np.sqrt(df_all_p['TotalApps'] + 1)
    df_all_p['DisplayGroup'] = df_all_p['is_selected'].map({True: sel_team_name, False: 'その他'})
    
    def make_label(r):
        if r['is_selected'] and not pd.isna(r['PlayerNo']):
            return str(int(r['PlayerNo']))
        return ""
    df_all_p['Label'] = df_all_p.apply(make_label, axis=1)
    df_all_p = df_all_p.sort_values('is_selected')

    fig_p = px.scatter(
        df_all_p, x='HensatiOFF', y='HensatiDEF', color='DisplayGroup', size='MarkerSize',
        text='Label', hover_name='PlayerNameJ',
        hover_data={'HensatiOFF': ':.1f', 'HensatiDEF': ':.1f', 'TotalApps': True, 'DisplayGroup': False, 'MarkerSize': False, 'Label': False},
        color_discrete_map={sel_team_name: '#EF553B', 'その他': '#E5ECF6'},
        labels={'HensatiOFF': '攻撃評価', 'HensatiDEF': '守備評価'},
        opacity=df_all_p['is_selected'].map({True: 1.0, False: 0.4})
    )
    fig_p.update_layout(xaxis=dict(range=[-30, 30], scaleanchor="y", scaleratio=1), yaxis=dict(range=[-30, 30]), width=700, height=700)
    fig_p.update_traces(textposition='top center')
    fig_p.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_p.add_vline(x=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_p, width="stretch")

    df_tp = df_all_p[df_all_p['is_selected']].copy()
    if not df_tp.empty:
        st.write(f"### {sel_team_name} 選手一覧")
        df_tp['PlayerNo'] = df_tp['PlayerNo'].fillna(0).astype(int)
        output_p = df_tp[['PlayerNo', 'PlayerNameJ', 'TotalApps', 'HensatiOFF', 'HensatiDEF']].sort_values('TotalApps', ascending=False)
        output_p.columns = ['背番号', '選手名', '合計プレイ数', '攻撃評価', '守備評価']
        st.dataframe(output_p, width="stretch", hide_index=True)

# --- タブ2: ラインナップ分析 ---
with tab2:
    st.subheader("ラインナップ別 評価値分布")
    df_all_l = df_lineup.copy()
    df_all_l['TotalApps_L'] = df_all_l['OFFApps'] + df_all_l['DEFApps']
    df_all_l['MarkerSize'] = np.sqrt(df_all_l['TotalApps_L'] + 1)
    
    team_players = df_player[df_player['TeamID'] == target_team_id].sort_values(['PlayerNo', 'PlayerID'])
    p_options = ["指定なし"] + [f"{int(r['PlayerNo'])} {r['PlayerNameJ']}" for _, r in team_players.iterrows()]
    sel_p_lineup = st.selectbox("強調表示する選手を選択", p_options)
    
    target_p_id = None
    if sel_p_lineup != "指定なし":
        sel_p_no = int(sel_p_lineup.split()[0])
        matched_p = team_players[team_players['PlayerNo'] == sel_p_no]
        if not matched_p.empty:
            target_p_id = int(matched_p.iloc[0]['PlayerID'])

    def classify_l(row):
        if target_p_id:
            l_ids = [int(row[f'Lineup_{i}']) for i in range(1, 6)]
            if target_p_id in l_ids: return "★注目選手含む"
        if int(row['TeamID']) == target_team_id: return sel_team_name
        return "その他"

    df_all_l['DisplayGroup'] = df_all_l.apply(classify_l, axis=1)
    df_all_l = df_all_l.sort_values('DisplayGroup', key=lambda x: x.map({"その他":0, sel_team_name:1, "★注目選手含む":2}))

    def get_unit_names(row):
        names = []
        for i in range(1, 6):
            pid = int(row[f'Lineup_{i}'])
            p_match = df_player[df_player['PlayerID'] == pid]
            names.append(p_match.iloc[0]['PlayerNameJ'] if not p_match.empty else "Unknown")
        return " / ".join(names)

    fig_l = px.scatter(
        df_all_l, x='HensatiOFF', y='HensatiDEF', color='DisplayGroup', size='MarkerSize',
        hover_name=df_all_l.apply(get_unit_names, axis=1),
        color_discrete_map={sel_team_name: '#EF553B', "★注目選手含む": '#19D3F3', 'その他': '#E5ECF6'},
        labels={'HensatiOFF': '攻撃評価', 'HensatiDEF': '守備評価'},
        opacity=df_all_l['DisplayGroup'].map(lambda x: 1.0 if x != "その他" else 0.3)
    )
    fig_l.update_layout(xaxis=dict(range=[-30, 30], scaleanchor="y", scaleratio=1), yaxis=dict(range=[-30, 30]), width=700, height=700)
    fig_l.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_l.add_vline(x=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_l, width="stretch")

    disp_l = df_all_l[df_all_l['TeamID'] == target_team_id].copy()
    if not disp_l.empty:
        st.write(f"### {sel_team_name} ラインナップ一覧")
        disp_l['ユニット構成'] = disp_l.apply(get_unit_names, axis=1)
        output_l = disp_l[['ユニット構成', 'TotalApps_L', 'RatingOFF', 'HensatiOFF', 'HensatiDEF']].sort_values('TotalApps_L', ascending=False)
        output_l.columns = ['ユニット構成', '合計プレイ数', 'Rating', '攻撃評価', '守備評価']
        st.dataframe(output_l, width="stretch", hide_index=True)
