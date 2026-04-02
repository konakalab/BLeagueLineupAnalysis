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
                'HensatiOFF': '攻撃評価',
                'HensatiDEF': '守備評価',
                'TotalApps': '合計出場数'
            },
            hover_name='PlayerNameJ',
            color_continuous_scale='Viridis'
        )

        all_vals = pd.concat([df_tp['HensatiOFF'], df_tp['HensatiDEF']])
        ax_min = min(all_vals.min(), 40) - 2
        ax_max = max(all_vals.max(), 60) + 2
        
        fig.update_layout(
            xaxis=dict(range=[ax_min, ax_max], scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[ax_min, ax_max]),
            width=700, height=700
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.add_vline(x=0, line_dash="dot", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("### 選手スタッツ一覧 (出場数順)")
        disp_df = df_tp[['PlayerNo', 'PlayerNameJ', 'TotalApps', 'OFFApps', 'DEFApps', 'HensatiOFF', 'HensatiDEF']].copy()
        disp_df['HensatiOFF'] = disp_df['HensatiOFF'].round(1)
        disp_df['HensatiDEF'] = disp_df['HensatiDEF'].round(1)
        disp_df.columns = ['背番号', '選手名', '合計プレイ数', '攻撃プレイ数', '守備プレイ数', '攻撃偏差値', '守備偏差値']
        
        disp_df = disp_df.sort_values('合計プレイ数', ascending=False)
        st.dataframe(disp_df, use_container_width=True, hide_index=True)
    else:
        st.warning("選手データが見つかりませんでした。")

# --- タブ2: ラインナップ分析 ---
with tab2:
    st.subheader("最強ラインナップ (偏差値順)")
    df_tl = df_lineup[df_lineup['TeamID'] == target_team_id].copy()
    
    if not df_tl.empty:
        top_10 = df_tl.sort_values('HensatiOFF', ascending=False).head(10)
        
        def get_p_names(row):
            names = []
            for i in range(1, 6):
                p_id = row[f'Lineup_{i}']
                found = df_player[df_player['PlayerID'] == p_id]['PlayerNameJ'].values
                names.append(found[0] if len(found) > 0 else f"ID:{p_id}")
            return " / ".join(names)
        
        top_10['ユニット構成'] = top_10.apply(get_p_names, axis=1)
        
        res_lineup = top_10[['ユニット構成', 'OFFApps', 'RatingOFF', 'HensatiOFF']].copy()
        res_lineup.columns = ['ユニット構成', '攻撃プレイ数', 'Rating', '偏差値']
        
        st.table(res_lineup)
    else:
        st.info("ラインナップデータがありません。")
