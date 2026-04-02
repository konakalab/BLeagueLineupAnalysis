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
    
    # 列名の前後にある余計な空白を削除（KeyError対策）
    df_t.columns = df_t.columns.str.strip()
    df_p.columns = df_p.columns.str.strip()
    df_l.columns = df_l.columns.str.strip()
    
    # 2025年の試合結果から期間を計算
    period_str = "期間データなし"
    try:
        df_res = pd.read_csv('table_BLeagueResult_2025.csv')
        df_res.columns = df_res.columns.str.strip()
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
    st.subheader("選手別 評価値分布 (全体比較)")
    
    # 全選手データの準備
    df_all = df_player.copy()
    df_all['TotalApps'] = df_all['OFFApps'] + df_all['DEFApps']
    df_all['MarkerSize'] = np.sqrt(df_all['TotalApps'] + 1)
    
    df_all['is_selected'] = df_all['TeamID'] == target_team_id
    df_all['DisplayGroup'] = df_all['is_selected'].map({True: sel_team_name, False: 'その他'})
    
    # グラフ表示用ラベル
    df_all['Label'] = df_all.apply(lambda r: str(r['PlayerNo']) if r['is_selected'] else "", axis=1)
    
    # 描画順
    df_all = df_all.sort_values('is_selected')

    # 散布図作成
    fig = px.scatter(
        df_all, 
        x='HensatiOFF', y='HensatiDEF',
        color='DisplayGroup', size='MarkerSize',
        text='Label', hover_name='PlayerNameJ',
        hover_data={'HensatiOFF': ':.1f', 'HensatiDEF': ':.1f', 'TotalApps': True, 'PlayerNo': True, 'Label': False, 'DisplayGroup': False, 'MarkerSize': False},
        color_discrete_map={sel_team_name: '#EF553B', 'その他': '#E5ECF6'},
        labels={'HensatiOFF': '攻撃評価', 'HensatiDEF': '守備評価', 'TotalApps': '合計出場数', 'PlayerNo': '背番号'},
        opacity=df_all['is_selected'].map({True: 1.0, False: 0.4})
    )

    fig.update_layout(
        xaxis=dict(range=[-30, 30], title="攻撃評価", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-30, 30], title="守備評価"),
        width=800, height=800
    )
    fig.update_traces(textposition='top center')
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 選手一覧表
    st.write(f"### {sel_team_name} 選手スタッツ一覧")
    df_tp = df_all[df_all['is_selected']].copy()
    disp_df = df_tp[['PlayerNo', 'PlayerNameJ', 'TotalApps', 'OFFApps', 'DEFApps', 'HensatiOFF', 'HensatiDEF']].copy()
    disp_df.columns = ['背番号', '選手名', '合計プレイ数', '攻撃プレイ数', '守備プレイ数', '攻撃評価', '守備評価']
    st.dataframe(disp_df.sort_values('合計プレイ数', ascending=False), use_container_width=True, hide_index=True)

# --- タブ2: ラインナップ分析 ---
with tab2:
    st.subheader("ラインナップ別 評価値分布 (全体比較)")
    
    df_all_l = df_lineup.copy()
    # 合計プレイ数の計算
    df_all_l['TotalApps_L'] = df_all_l['OFFApps'] + df_all_l['DEFApps']
    df_all_l['MarkerSize'] = np.sqrt(df_all_l['TotalApps_L'] + 1)
    
    # 強調選手選択
    team_players = df_player[df_player['TeamID'] == target_team_id].sort_values('PlayerNo')
    p_options = ["指定なし"] + [f"{r['PlayerNo']} {r['PlayerNameJ']}" for _, r in team_players.iterrows()]
    sel_p_lineup = st.selectbox("特定の選手が含まれるユニットを強調表示", p_options)
    
    target_p_id = None
    if sel_p_lineup != "指定なし":
        sel_p_no = int(sel_p_lineup.split()[0])
        target_p_id = team_players[team_players['PlayerNo'] == sel_p_no]['PlayerID'].values[0]

    def classify_lineup(row):
        if target_p_id:
            ids = [row[f'Lineup_{i}'] for i in range(1, 6)]
            if target_p_id in ids: return "★注目選手含む"
        if row['TeamID'] == target_team_id: return sel_team_name
        return "その他"

    df_all_l['DisplayGroup'] = df_all_l.apply(classify_lineup, axis=1)
    df_all_l['z'] = df_all_l['DisplayGroup'].map({"その他": 0, sel_team_name: 1, "★注目選手含む": 2})
    df_all_l = df_all_l.sort_values('z')

    def get_unit_names(row):
        names = []
        for i in range(1, 6):
            pid = row[f'Lineup_{i}']
            pname = df_player[df_player['PlayerID'] == pid]['PlayerNameJ'].values
            names.append(pname[0] if len(pname) > 0 else "Unknown")
        return " / ".join(names)

    # 散布図作成
    fig_l = px.scatter(
        df_all_l, x='HensatiOFF', y='HensatiDEF',
        color='DisplayGroup', size='MarkerSize',
        hover_name=df_all_l.apply(get_unit_names, axis=1),
        hover_data={'HensatiOFF': ':.1f', 'HensatiDEF': ':.1f', 'TotalApps_L': True, 'DisplayGroup': False, 'MarkerSize': False, 'z': False},
        color_discrete_map={sel_team_name: '#EF553B', "★注目選手含む": '#19D3F3', 'その他': '#E5ECF6'},
        labels={'HensatiOFF': '攻撃評価', 'HensatiDEF': '守備評価', 'TotalApps_L': '合計プレイ数'},
        opacity=df_all_l['DisplayGroup'].map(lambda x: 1.0 if x != "その他" else 0.3)
    )

    fig_l.update_layout(
        xaxis=dict(range=[-30, 30], title="攻撃評価", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-30, 30], title="守備評価"),
        width=800, height=800
    )
    fig_l.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_l.add_vline(x=0, line_dash="dot", line_color="gray")

    st.plotly_chart(fig_l, use_container_width=True)

    # テーブル表示
    st.write(f"### {sel_team_name} ラインナップ・スタッツ")
    df_table = df_all_l[df_all_l['TeamID'] == target_team_id].copy()
    if not df_table.empty:
        df_table['ユニット構成'] = df_table.apply(get_unit_names, axis=1)
        sort_col = 'z' if target_p_id else 'TotalApps_L'
        df_table = df_table.sort_values([sort_col, 'TotalApps_L'], ascending=False)
        output_table = df_table[['ユニット構成', 'TotalApps_L', 'RatingOFF', 'HensatiOFF', 'HensatiDEF']].copy()
        output_table.columns = ['ユニット構成', '合計プレイ数', 'Rating', '攻撃評価', '守備評価']
        st.dataframe(output_table, use_container_width=True, hide_index=True)
