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
st.title(f"🏀 {selected_team_name} 分析ダッシュボード")
st.write(f"{selected_league} / {team_info['Division']}地区")

tab1, tab2 = st.tabs(["選手分析", "ラインナップ分析"])

with tab1:
    st.subheader("選手別 偏差値分析 (攻撃 vs 守備)")
    
    # チーム所属選手の抽出
    team_players = df_player[df_player['TeamID'] == team_id].copy()
    
    if not team_players.empty:
        # 修正ポイント：マーカーの色を OFFApps + DEFApps の合計に連動
        team_players['TotalApps'] = team_players['OFFApps'] + team_players['DEFApps']
        
        # 散布図の作成
        fig = px.scatter(
            team_players, 
            x='HensatiOFF', 
            y='HensatiDEF',
            text='PlayerNameJ',
            color='TotalApps',  # 合計プレイ数に連動
            labels={
                'HensatiOFF': '攻撃評価 (偏差値)',
                'HensatiDEF': '守備評価 (偏差値)',
                'TotalApps': '合計プレイ数'
            },
            hover_data=['PlayerNo', 'OFFApps', 'DEFApps'],
            color_continuous_scale='Viridis', # 出場数が見やすい色合い
            title=f"{selected_team_name} 選手スタッツ分布"
        )

        # 修正ポイント：縦軸と横軸のスケールを合わせる
        # 偏差値データなので、40〜60（あるいは30〜70）など、中心を50にして範囲を固定すると比較しやすい
        axis_range = [
            min(team_players['HensatiOFF'].min(), team_players['HensatiDEF'].min()) - 5,
            max(team_players['HensatiOFF'].max(), team_players['HensatiDEF'].max()) + 5
        ]
        
        fig.update_layout(
            xaxis=dict(range=axis_range, scaleanchor="y", scaleratio=1),
            yaxis=dict(range=axis_range),
            width=800,
            height=800
        )
        
        fig.update_traces(textposition='top center')

        # 中央線（偏差値50）を描画して分かりやすくする
        fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="平均")
        fig.add_vline(x=50, line_dash="dot", line_color="gray", annotation_text="平均")

        st.plotly_chart(fig, use_container_width=True)
        
        # スタッツ表の表示
        st.dataframe(
            team_players[['PlayerNo', 'PlayerNameJ', 'HensatiOFF', 'HensatiDEF', 'TotalApps']]
            .sort_values('HensatiOFF', ascending=False)
        )
    else:
        st.warning("選手データが見つかりませんでした。")

with tab2:
    # ラインナップ分析（以前のロジックを維持）
    st.subheader("最強ラインナップ (Rating順)")
    team_lineups = df_lineup[df_lineup['TeamID'] == team_id].copy()
    if not team_lineups.empty:
        top_lineups = team_lineups.sort_values('HensatiOFF', ascending=False).head(5)
        def get_names(row):
            names = []
            for i in range(1, 6):
                p_id = row[f'Lineup_{i}']
                p_name = df_player[df_player['PlayerID'] == p_id]['PlayerNameJ'].values
                names.append(p_name[0] if len(p_name) > 0 else f"ID:{p_id}")
            return " / ".join(names)
        top_lineups['選手構成'] = top_lineups.apply(get_names, axis=1)
        st.table(top_lineups[['選手構成', 'OFFApps', 'RatingOFF', 'HensatiOFF']])
