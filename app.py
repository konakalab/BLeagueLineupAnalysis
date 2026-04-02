import streamlit as st
import pandas as pd
import plotly.express as px

# ページ設定
st.set_page_config(page_title="B-League Analytics Dash", layout="wide")

# データの読み込み
@st.cache_data
def load_data():
    df_team = pd.read_csv('table_team.csv')
    df_player = pd.read_csv('table_players.csv')
    df_lineup = pd.read_csv('table_lineups.csv')
    return df_team, df_player, df_lineup

df_team, df_player, df_lineup = load_data()

# サイドバー：フィルタリング
st.sidebar.header("Filter")
selected_league = st.sidebar.selectbox("リーグ選択", df_team['League'].unique())
teams_in_league = df_team[df_team['League'] == selected_league]
selected_team_name = st.sidebar.selectbox("チーム選択", teams_in_league['Team'].unique())

# チーム情報の取得
team_info = df_team[df_team['Team'] == selected_team_name].iloc[0]
team_id = team_info['TeamID']

# メイン表示
st.title(f" {selected_team_name} 分析ダッシュボード")
st.write(f"{selected_league} / {team_info['Division']}地区")

tab1, tab2 = st.tabs(["選手分析", "ラインナップ分析"])

with tab1:
    st.subheader("選手別 Rating (OFF vs DEF)")
    team_players = df_player[df_player['TeamID'] == team_id].copy()
    if not team_players.empty:
        fig = px.scatter(
            team_players, x='RatingOFF', y='RatingDEF',
            text='PlayerNameJ', color='HensatiOFF',
            hover_data=['PlayerNo', 'HensatiOFF', 'HensatiDEF'],
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(team_players[['PlayerNo', 'PlayerNameJ', 'RatingOFF', 'RatingDEF']].sort_values('RatingOFF', ascending=False))

with tab2:
    st.subheader("ラインナップ (Rating順)")
    team_lineups = df_lineup[df_lineup['TeamID'] == team_id].copy()
    if not team_lineups.empty:
        top_lineups = team_lineups.sort_values('RatingOFF', ascending=False).head(5)
        def get_names(row):
            names = []
            for i in range(1, 6):
                p_id = row[f'Lineup_{i}']
                p_name = df_player[df_player['PlayerID'] == p_id]['PlayerNameJ'].values
                names.append(p_name[0] if len(p_name) > 0 else f"ID:{p_id}")
            return " / ".join(names)
        top_lineups['選手構成'] = top_lineups.apply(get_names, axis=1)
        st.table(top_lineups[['選手構成', 'OFFApps', 'RatingOFF', 'HensatiOFF']])
