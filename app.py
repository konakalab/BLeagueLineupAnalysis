import streamlit as st
import pandas as pd
import plotly.express as px

# 1. ページ全体の基本設定
st.set_page_config(page_title="B-League Analytics Dash", layout="wide")

# 2. データの読み込み処理
@st.cache_data
def load_all_data():
    # 基本テーブルの読み込み
    df_t = pd.read_csv('table_team.csv')
    df_p = pd.read_csv('table_players.csv')
    df_l = pd.read_csv('table_lineups.csv')
    
    # 新しいファイル名で読み込み
    period_str = "期間データなし"
    try:
        df_res = pd.read_csv('table_BLeagueResult_2025.csv')
        df_res['Date'] = pd.to_datetime(df_res['Date'])
        # スコアがある試合のみ抽出
        finished = df_res.dropna(subset=['HomeScore', 'AwayScore'])
        if not finished.empty:
            s_dt = finished['Date'].min().strftime('%Y/%m/%d')
            e_dt = finished['Date'].max().strftime('%Y/%m/%d')
            period_str = f"{s_dt} から {e_dt}"
        else:
            period_str = "完了した試合データがありません"
    except Exception as e:
        period_str = "ファイル読み込みエラー"
        
    return df_t, df_p, df_l, period_str

# データの取得
df_team, df_player, df_lineup, analysis_period = load_all_data()

# 3. サイドバーのフィルター
st.sidebar.header("検索フィルター")
list_league = df_team['League'].unique()
sel_league = st.sidebar.selectbox("リーグ選択", list_league)

teams_in_league = df_team[df_team['League'] == sel_league]
sel_team_name = st.sidebar.selectbox("チーム選択", teams_in_league['Team'].unique())

# 選択されたチームのIDを取得
team_row = df_team[df_team['Team'] == sel_team_name].iloc[0]
target_team_id = team_row['TeamID']

# 4. メイン画面のタイトル表示
st.title(f"🏀 {sel_team_name} 分析ダッシュボード")
st.write(f"{sel_league} / {team_row['Division']}地区")
st.info(f"📅 分析対象期間：{analysis_period} まで")

# 5. タブの作成
tab1, tab2 = st.tabs(["選手分析", "ラインナップ分析"])

# --- タブ1: 選手分析 ---
with tab1:
    st.subheader("選手別 偏差値分布 (攻撃評価 vs 守備評価)")
    
    df_tp = df_player[df_player['TeamID'] == target_team_id].copy()
    
    if not df_tp.empty:
        df_tp['TotalApps'] = df_tp['OFFApps'] + df_tp['DEFApps']
        
        fig = px.scatter(
            df_tp, 
            x='HensatiOFF', 
            y='HensatiDEF',
            text='PlayerNo',
            color='TotalApps',
            labels={
                'HensatiOFF': '攻撃評価 (偏差値)',
                'HensatiDEF': '守備評価 (偏差値)',
                'TotalApps': '合計出場数'
            },
            hover_name='PlayerNameJ',
            color_continuous_scale='Viridis'
        )

        all_vals = pd.concat([df_tp['HensatiOFF'], df_tp['HensatiDEF']])
        ax_min = min(all_vals.min(), 40) - 2
        ax_max = max(all_vals.max(), 60
