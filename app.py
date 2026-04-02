import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 1. ページ全体の基本設定
st.set_page_config(page_title="B-League Analytics Dash", layout="wide")

# 2. データの読み込み処理
@st.cache_data
def load_all_data():
    df_t = pd.read_csv('table_team.csv')
    df_p = pd.read_csv('table_players.csv')
    df_l = pd.read_csv('table_lineups.csv')
    
    period_str = "期間データなし"
    try:
        df_res = pd.read_csv('table_BLeagueResult_2025.csv')
        df_res['Date'] = pd.to_datetime(df_res['Date'])
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

df_team, df_player, df_lineup, analysis_period = load_all_data()

# 3. サイドバーのフィルター
st.sidebar.header("検索フィルター")
list_league = df_team['League'].unique()
sel_league = st.sidebar.selectbox("リーグ選択", list_league)

teams_in_league = df_team[df_team['League'] == sel_league]
sel_team_name = st.sidebar.selectbox("チーム選択", teams_in_league['Team'].unique())

team_row = df_team[df_team['Team'] == sel_team_name].iloc[0]
target_team_id = team_row['TeamID']

# 4. メイン画面
st.title(f"🏀 {sel_team_name} 分析ダッシュボード")
st.write(f"{sel_league} / {team_row['Division']}地区")
st.info(f"📅 分析対象期間：{analysis_period} まで")

tab1, tab2 = st.tabs(["選手分析", "ラインナップ分析"])

# --- タブ1: 選手分析 ---
with tab1:
    st.subheader("選手別 偏差値分布 (全体比較)")
    
    # 全選手データの準備
    df_all = df_player.copy()
    df_all['TotalApps'] = df_all['OFFApps'] + df_all['DEFApps']
    # 半径を平方根に比例させるための計算（サイズ調整用）
    df_all['MarkerSize'] = np.sqrt(df_all['TotalApps'] + 1)
    
    # 選択チーム判定フラグ
    df_all['is_selected'] = df_all['TeamID'] == target_team_id
    df_all['DisplayGroup'] = df_all['is_selected'].map({True: sel_team_name, False: 'その他'})
    
    # 描画順：その他を先に描いて、選択チームを上に重ねる
    df_all = df_all.sort_values('is_selected')

    # 散布図作成
    fig = px.scatter(
        df_all, 
        x='HensatiOFF', 
        y='HensatiDEF',
        color='DisplayGroup',
        size='MarkerSize',
        text=df_all.apply(lambda r: str(r['PlayerNo']) if r['is_selected'] else "", axis=1),
        hover_name='PlayerNameJ',
        hover_data={'HensatiOFF': ':.1f', 'HensatiDEF': ':.1f', 'TotalApps': True, 'DisplayGroup': False, 'MarkerSize': False},
        color_discrete_map={sel_team_name: '#EF553B', 'その他': '#E5ECF6'}, # 選択チームを赤、その他を薄いグレーに
        labels={'HensatiOFF': '攻撃評価', 'HensatiDEF': '守備評価', 'TotalApps': '合計出場数'},
        opacity=df_all['is_selected'].map({True: 1.0, False: 0.4}) # その他を半透明に
    )

    # 軸の設定（-30から30）と点線
    fig.update_layout(
        xaxis=dict(range=[-30, 30], title="攻撃評価 (偏差値)", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-30, 30], title="守備評価 (偏差値)"),
        width=800, height=800,
        legend_title_text="表示グループ",
        showlegend=True
    )
    fig.update_traces(textposition='top center')
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 選手一覧表（選択チームのみ）
    st.write(f"### {sel_team_name} 選手スタッツ一覧")
    df_tp = df_all[df_all['is_selected']].copy()
    disp_df = df_tp[['PlayerNo', 'PlayerNameJ', 'TotalApps', 'OFFApps', 'DEFApps', 'HensatiOFF', 'HensatiDEF']].copy()
    disp_df['HensatiOFF'] = disp_df['HensatiOFF'].round(1)
    disp_df['HensatiDEF'] = disp_df['HensatiDEF'].round(1)
    disp_df.columns = ['背番号', '選手名', '合計プレイ数', '攻撃プレイ数', '守備プレイ数', '攻撃偏差値', '守備偏差値']
    
    st.dataframe(disp_df.sort_values('合計プレイ数', ascending=False), use_container_width=True, hide_index=True)

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
