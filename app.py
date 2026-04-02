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
    
    # 2025年の試合結果データのみを読み込む
    try:
        df_results = pd.read_csv('BLeagueResult03.xlsx - 2025.csv')
        # 日付型に変換
        df_results['Date'] = pd.to_datetime(df_results['Date'])
        # スコア(HomeScore)が記入されている行だけを「分析対象」とする
        target_games = df_results.dropna(subset=['HomeScore', 'AwayScore'])
        
        if not target_games.empty:
            start_date = target_games['Date'].min().strftime('%Y/%m/%d')
            end_date = target_games['Date'].max().strftime('%Y/%m/%d')
            analysis_period = f"{start_date} から {end_date}"
        else:
            analysis_period = "データ準備中"
    except FileNotFoundError:
        analysis_period = "試合結果ファイルが見つかりません"
    
    return df_team, df_player, df_lineup, analysis_period

df_team, df_player, df_lineup, analysis_period = load_data()

# サイドバー：フィルタリング
st.sidebar.header("Filter")
selected_league = st.sidebar.selectbox("リーグ選択", df_team['League'].unique())
teams_in_league = df_team[df_team['League'] == selected_league]
selected_team_name = st.sidebar.selectbox("チーム選択", teams_in_league['Team'].unique())

# チーム情報の取得
team_info = df_team[df_team['Team'] == selected_team_name].iloc[0]
team_id = team_info['TeamID']

# メイン表示
st.title(f"🏀 {selected_team_name} 分析ダッシュボード")
st.write(f"{selected_league} / {team_info['Division']}地区")

# 分析対象期間の表示
st.info(f"📅 分析対象：{analysis_period} まで")

tab1, tab2 = st.tabs(["選手分析", "ラインナップ分析"])

with tab1:
    st.subheader("選手別 偏差値分析 (攻撃 vs 守備)")
    team_players = df_player[df_player['TeamID'] == team_id].copy()
    
    if not team_players.empty:
        team_players['TotalApps'] = team_players['OFFApps'] + team_players['DEFApps']
        
        fig = px.scatter(
            team_players, x='HensatiOFF', y='HensatiDEF',
            text='PlayerNo', color='TotalApps',
            labels={'HensatiOFF': '攻撃評価', 'HensatiDEF': '守備評価', 'TotalApps': '合計出場数'},
            hover_name='PlayerNameJ',
            hover_data={'PlayerNo': True, 'HensatiOFF': ':.1f', 'HensatiDEF': ':.1f', 'TotalApps': True},
            color_continuous_scale='Viridis',
        )

        max_val = max(team_players['HensatiOFF'].max(), team_players['HensatiDEF'].max(), 60)
        min_val = min(team_players['HensatiOFF'].min(), team_players['HensatiDEF'].min(), 40)
        axis_range = [min_val - 2, max_val + 2]
        
        fig.update_layout(
            xaxis=dict(range=axis_range, scaleanchor="y", scaleratio=1),
            yaxis=dict(range=axis_range),
            width=700, height=700
        )
        fig.add_hline(y=50, line_dash="dot", line_color="gray")
        fig.add_vline(x=50, line_dash="dot", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 選手分析表
        st.write("### 選手スタッツ一覧 (合計出場数順)")
        display_df = team_players[['PlayerNo', 'PlayerNameJ', 'TotalApps', 'OFFApps', 'DEFApps', 'HensatiOFF', 'HensatiDEF']].copy()
        display_df['HensatiOFF'] = display_df['HensatiOFF'].round(1)
        display_df['HensatiDEF'] = display_df['HensatiDEF'].round(1)
        display_df.columns = ['背番号', '選手名', '合計プレイ数', '攻撃プレイ数', '守備プレイ数', '攻撃偏差値', '守備偏差値']
        display_df = display_df.sort_values('合計プレイ数', ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.warning("選手データが見つかりませんでした。")

with tab2:
    st.subheader("最強ラインナップ (Rating順)")
    team_lineups = df_lineup[df_lineup['TeamID'] == team_id].copy()
    if not team_lineups.empty:
        # Rating順にソート
        top_lineups = team_lineups.sort_values('RatingOFF', ascending=False).head(10)
        
        def get_names(row):
            names = []
            for i in range(1, 6):
                p_id = row[f'Lineup_{i}']
                p_name_list = df_player[df_player['PlayerID'] == p_id]['PlayerNameJ'].values
                names.append(p_name_list[0] if len(p_name_list) > 0 else f"ID:{p_id}")
            return " / ".join(names)
        
        top_lineups['選手構成'] = top_lineups.apply(get_names, axis=1)
        
        # 表示用整理
        show_lineups = top_lineups[['選手構成', 'OFFApps', 'RatingOFF', 'HensatiOFF']].copy()
        show_lineups.columns = ['ユニット構成', '攻撃プレイ数', 'Rating', '偏差値']
        st.table(show_lineups)
    else:
        st.info("ラインナップデータがありません。")
