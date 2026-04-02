import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 1. ページ全体の基本設定
st.set_page_config(page_title="B-League Analytics Dash", layout="wide")

# 2. データの読み込み処理（キャッシュを使用して高速化）
@st.cache_data
def load_raw_csv():
    # 各CSVの読み込み
    df_t = pd.read_csv('table_team.csv')
    df_p = pd.read_csv('table_players.csv')
    df_l = pd.read_csv('table_lineups.csv')
    
    # 全データフレームの列名クリーンアップ（空白や改行を削除）
    for df in [df_t, df_p, df_l]:
        df.columns = [str(c).strip().replace('\n', '') for c in df.columns]
        # ID系・順序系を確実に数値（Int）にする
        cols_to_fix = ['TeamID', 'PlayerID', 'Lineup_1', 'Lineup_2', 'Lineup_3', 'Lineup_4', 'Lineup_5', 'order']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
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

# 3. サイドバーのフィルター
st.sidebar.header("検索フィルター")

# リーグの選択（CSVの登場順を維持）
list_league = df_team['League'].unique()
sel_league = st.sidebar.selectbox("リーグ選択", list_league)

# 選択リーグ内のチームを抽出し、'order' カラムがあればその順にソート
teams_in_league = df_team[df_team['League'] == sel_league].copy()
if 'order' in teams_in_league.columns:
    teams_in_league = teams_in_league.sort_values('order')
else:
    # orderがない場合はTeamID順
    teams_in_league = teams_in_league.sort_values('TeamID')

list_teams = teams_in_league['Team'].tolist()
sel_team_name = st.sidebar.selectbox("チーム選択", list_teams)

# ターゲットチームIDの特定
team_row = teams_in_league[teams_in_league['Team'] == sel_team_name].iloc[0]
target_team_id = int(team_row['TeamID'])

# 4. メイン画面のヘッダー
st.title(f"🏀 {sel_team_name} 分析ダッシュボード")
st.info(f"📅 分析対象期間：{analysis_period}")

tab1, tab2 = st.tabs(["選手分析", "ラインナップ分析"])

# --- タブ1: 選手分析 ---
with tab1:
    st.subheader("選手別 評価値分布 (全体比較)")
    
    df_all_p = df_player.copy()
    df_all_p['is_selected'] = df_all_p['TeamID'] == target_team_id
    
    # 合計出場数とバブルサイズ（平方根）
    df_all_p['TotalApps'] = df_all_p['OFFApps'] + df_all_p['DEFApps']
    df_all_p['MarkerSize'] = np.sqrt(df_all_p['TotalApps'] + 1)
    
    # グループ分けとラベル
    df_all_p['DisplayGroup'] = df_all_p['is_selected'].map({True: sel_team_name, False: 'その他'})
    df_all_p['Label'] = df_all_p.apply(lambda r: str(int(r['PlayerNo'])) if r['is_selected'] and not pd.isna(r['PlayerNo']) else "", axis=1)
    
    # 描画順：その他を先に、自チームを後に（上に重ねる）
    df_all_p = df_all_p.sort_values('is_selected')

    fig_p = px.scatter(
        df_all_p, x='HensatiOFF', y='HensatiDEF', color='DisplayGroup', size='MarkerSize',
        text='Label', hover_name='PlayerNameJ',
        hover_data={'HensatiOFF': ':.1f', 'HensatiDEF': ':.1f', 'TotalApps': True, 'DisplayGroup': False, 'MarkerSize': False, 'Label': False},
        color_discrete_map={sel_team_name: '#EF553B', 'その他': '#E5ECF6'},
        labels={'HensatiOFF': '攻撃評価', 'HensatiDEF': '守備評価', 'TotalApps': '合計プレイ数'},
        opacity=df_all_p['is_selected'].map({True: 1.0, False: 0.4})
    )
    fig_p.update_layout(
        xaxis=dict(range=[-30, 30], title="攻撃評価", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-30, 30], title="守備評価"),
        width=700, height=700,
        legend_title_text="表示グループ"
    )
    fig_p.update_traces(textposition='top center')
    fig_p.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_p.add_vline(x=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_p, width="stretch")

    # 選手一覧テーブル
    df_tp = df_all_p[df_all_p['is_selected']].copy()
    if not df_tp.empty:
        st.write(f"### {sel_team_name} 選手一覧")
        df_tp['PlayerNo'] = df_tp['PlayerNo'].fillna(0).astype(int)
        # 数値の丸め
        df_tp['HensatiOFF'] = df_tp['HensatiOFF'].round(1)
        df_tp['HensatiDEF'] = df_tp['HensatiDEF'].round(1)
        
        output_p = df_tp[['PlayerNo', 'PlayerNameJ', 'TotalApps', 'HensatiOFF', 'HensatiDEF']].sort_values('TotalApps', ascending=False)
        output_p.columns = ['背番号', '選手名', '合計プレイ数', '攻撃評価', '守備評価']
        st.dataframe(output_p, width="stretch", hide_index=True)

# --- タブ2: ラインナップ分析 ---
with tab2:
    st.subheader("ラインナップ別 評価値分布 (全体比較)")
    
    df_all_l = df_lineup.copy()
    # 合計プレイ数の算出
    df_all_l['TotalApps_L'] = df_all_l['OFFApps'] + df_all_l['DEFApps']
    df_all_l['MarkerSize'] = np.sqrt(df_all_l['TotalApps_L'] + 1)
    
    # 強調表示する選手の選択肢（背番号順）
    team_players = df_player[df_player['TeamID'] == target_team_id].sort_values(['PlayerNo', 'PlayerID'])
    p_options = ["指定なし"] + [f"{int(r['PlayerNo'])} {r['PlayerNameJ']}" for _, r in team_players.iterrows()]
    sel_p_lineup = st.selectbox("強調表示する選手を選択", p_options)
    
    target_p_id = None
    if sel_p_lineup != "指定なし":
        sel_p_no = int(sel_p_lineup.split()[0])
        matched_p = team_players[team_players['PlayerNo'] == sel_p_no]
        if not matched_p.empty:
            target_p_id = int(matched_p.iloc[0]['PlayerID'])

    # グループ判定ロジック
    def classify_l(row):
        if target_p_id:
            # 5人のPlayerIDのいずれかと一致するか
            lineup_ids = [int(row[f'Lineup_{i}']) for i in range(1, 6)]
            if target_p_id in lineup_ids: return "★注目選手含む"
        if int(row['TeamID']) == target_team_id: return sel_team_name
        return "その他"

    df_all_l['DisplayGroup'] = df_all_l.apply(classify_l, axis=1)
    # 描画順：その他(0) -> チーム(1) -> 注目選手(2)
    df_all_l = df_all_l.sort_values('DisplayGroup', key=lambda x: x.map({"その他":0, sel_team_name:1, "★注目選手含む":2}))

    # ユニット名を生成する関数（Tooltip用）
    def get_unit_names(row):
        names = []
        for i in range(1, 6):
            pid = int(row[f'Lineup_{i}'])
            p_match = df_player[df_player['PlayerID'] == pid]
            names.append(p_match.iloc[0]['PlayerNameJ'] if not p_match.empty else f"ID:{pid}")
        return " / ".join(names)

    fig_l = px.scatter(
        df_all_l, x='HensatiOFF', y='HensatiDEF', color='DisplayGroup', size='MarkerSize',
        hover_name=df_all_l.apply(get_unit_names, axis=1),
        hover_data={'HensatiOFF': ':.1f', 'HensatiDEF': ':.1f', 'TotalApps_L': True, 'DisplayGroup': False, 'MarkerSize': False},
        color_discrete_map={sel_team_name: '#EF553B', "★注目選手含む": '#19D3F3', 'その他': '#E5ECF6'},
        labels={'HensatiOFF': '攻撃評価', 'HensatiDEF': '守備評価', 'TotalApps_L': '合計プレイ数'},
        opacity=df_all_l['DisplayGroup'].map(lambda x: 1.0 if x != "その他" else 0.3)
    )
    fig_l.update_layout(
        xaxis=dict(range=[-30, 30], title="攻撃評価", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-30, 30], title="守備評価"),
        width=700, height=700,
        legend_title_text="表示グループ"
    )
    fig_l.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_l.add_vline(x=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_l, width="stretch")

    # ラインナップ一覧テーブル
    disp_l_full = df_all_l[df_all_l['TeamID'] == target_team_id].copy()
    if not disp_l_full.empty:
        st.write(f"### {sel_team_name} ラインナップ一覧")
        disp_l_full['ユニット構成'] = disp_l_full.apply(get_unit_names, axis=1)
        # 数値の丸め
        disp_l_full['RatingOFF'] = disp_l_full['RatingOFF'].round(1)
        disp_l_full['HensatiOFF'] = disp_l_full['HensatiOFF'].round(1)
        disp_l_full['HensatiDEF'] = disp_l_full['HensatiDEF'].round(1)
        
        output_l = disp_l_full[['ユニット構成', 'TotalApps_L', 'RatingOFF', 'HensatiOFF', 'HensatiDEF']].sort_values('TotalApps_L', ascending=False)
        output_l.columns = ['ユニット構成', '合計プレイ数', 'Rating', '攻撃評価', '守備評価']
        st.dataframe(output_l, width="stretch", hide_index=True)
