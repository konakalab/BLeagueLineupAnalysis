import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 1. ページ全体の基本設定
st.set_page_config(page_title="B-League Analytics Dash", layout="wide")

# 2. データの読み込みと前処理
@st.cache_data
def load_all_data():
    df_t = pd.read_csv('table_team.csv')
    df_p = pd.read_csv('table_players.csv')
    df_l = pd.read_csv('table_lineups.csv')
    
    for df in [df_t, df_p, df_l]:
        # 全列名の空白削除 ＋ すべて小文字に変換して統一（大文字Order対策）
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # 数値型への変換（列名を小文字で指定）
        numeric_cols = [
            'teamid', 'playerid', 'order', 'playerno',
            'lineup_1', 'lineup_2', 'lineup_3', 'lineup_4', 'lineup_5',
            'offapps', 'defapps', 'hensatioff', 'hensatidef', 'ratingoff', 'ratingdef'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    period_str = "データ期間不明"
    try:
        df_res = pd.read_csv('table_BLeagueResult_2025.csv')
        df_res.columns = df_res.columns.str.strip().str.lower()
        df_res['date'] = pd.to_datetime(df_res['date'])
        finished = df_res.dropna(subset=['homescore', 'awayscore'])
        if not finished.empty:
            period_str = f"{finished['date'].min().strftime('%Y/%m/%d')} 〜 {finished['date'].max().strftime('%Y/%m/%d')}"
    except:
        pass
        
    return df_t, df_p, df_l, period_str

df_team, df_player, df_lineup, analysis_period = load_all_data()

# 3. サイドバーのフィルター
st.sidebar.header("検索フィルター")

# リーグ選択 (小文字 league で指定)
list_league = list(dict.fromkeys(df_team['league']))
sel_league = st.sidebar.selectbox("リーグ選択", list_league)

# チーム選択 (小文字 order で厳密にソート)
teams_in_league = df_team[df_team['league'] == sel_league].copy()
# orderがあればそれを使用、なければteamid順
sort_col = 'order' if 'order' in teams_in_league.columns else 'teamid'
teams_sorted = teams_in_league.sort_values(by=sort_col, ascending=True)

list_teams = teams_sorted['team'].tolist()
sel_team_name = st.sidebar.selectbox("チーム選択", list_teams)

# 選択されたチームのID取得
target_team_id = int(teams_sorted[teams_sorted['team'] == sel_team_name]['teamid'].iloc[0])

# 4. メイン画面
st.title(f"🏀 {sel_team_name} 分析ダッシュボード")
st.info(f"📅 分析対象期間：{analysis_period}")

tab1, tab2 = st.tabs(["選手分析", "ラインナップ分析"])

# --- タブ1: 選手分析 ---
with tab1:
    st.subheader("選手別 評価値分布 (全体比較)")
    df_all_p = df_player.copy()
    df_all_p['is_selected'] = (df_all_p['teamid'] == target_team_id)
    df_all_p['totalapps'] = df_all_p['offapps'] + df_all_p['defapps']
    df_all_p['markersize'] = np.sqrt(df_all_p['totalapps'] + 1)
    df_all_p['displaygroup'] = df_all_p['is_selected'].map({True: sel_team_name, False: 'その他'})
    
    df_all_p['label'] = df_all_p.apply(
        lambda r: str(int(r['playerno'])) if r['is_selected'] and r['playerno'] != 0 else "", axis=1
    )
    df_all_p = df_all_p.sort_values('is_selected')

    fig_p = px.scatter(
        df_all_p, x='hensatioff', y='hensatidef', color='displaygroup', size='markersize',
        text='label', hover_name='playernamej',
        hover_data={'hensatioff': ':.1f', 'hensatidef': ':.1f', 'totalapps': True, 'displaygroup': False, 'markersize': False, 'label': False},
        color_discrete_map={sel_team_name: '#EF553B', 'その他': '#E5ECF6'},
        labels={'hensatioff': '攻撃評価', 'hensatidef': '守備評価'},
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
        df_tp['背番号'] = df_tp['playerno'].astype(int)
        output_p = df_tp[['背番号', 'playernamej', 'totalapps', 'hensatioff', 'hensatidef']].sort_values('totalapps', ascending=False)
        output_p.columns = ['背番号', '選手名', '合計プレイ数', '攻撃評価', '守備評価']
        st.dataframe(output_p.style.format({'攻撃評価': '{:.1f}', '守備評価': '{:.1f}'}), width="stretch", hide_index=True)

# --- タブ2: ラインナップ分析 ---
with tab2:
    st.subheader("ラインナップ別 評価値分布 (全体比較)")
    df_all_l = df_lineup.copy()
    df_all_l['totalapps_l'] = df_all_l['offapps'] + df_all_l['defapps']
    df_all_l['markersize'] = np.sqrt(df_all_l['totalapps_l'] + 1)
    
    team_p = df_player[df_player['teamid'] == target_team_id].sort_values('playerno')
    p_options = ["指定なし"] + [f"{int(r['playerno'])} {r['playernamej']}" for _, r in team_p.iterrows()]
    sel_p = st.selectbox("強調表示する選手を選択", p_options)
    
    target_p_id = None
    if sel_p != "指定なし":
        sel_p_no = int(sel_p.split()[0])
        target_p_id = int(team_p[team_p['playerno'] == sel_p_no]['playerid'].iloc[0])

    def classify_lineup(row):
        if target_p_id:
            l_ids = [int(row[f'lineup_{i}']) for i in range(1, 6)]
            if target_p_id in l_ids: return "★注目選手含む"
        if int(row['teamid']) == target_team_id: return sel_team_name
        return "その他"

    df_all_l['displaygroup'] = df_all_l.apply(classify_lineup, axis=1)
    df_all_l = df_all_l.sort_values('displaygroup', key=lambda x: x.map({"その他":0, sel_team_name:1, "★注目選手含む":2}))

    def get_unit_names(row):
        names = []
        for i in range(1, 6):
            pid = int(row[f'lineup_{i}'])
            match = df_player[df_player['playerid'] == pid]
            names.append(match.iloc[0]['playernamej'] if not match.empty else f"ID:{pid}")
        return " / ".join(names)

    fig_l = px.scatter(
        df_all_l, x='hensatioff', y='hensatidef', color='displaygroup', size='markersize',
        hover_name=df_all_l.apply(get_unit_names, axis=1),
        color_discrete_map={sel_team_name: '#EF553B', "★注目選手含む": '#19D3F3', 'その他': '#E5ECF6'},
        labels={'hensatioff': '攻撃評価', 'hensatidef': '守備評価'},
        opacity=df_all_l['displaygroup'].map(lambda x: 1.0 if x != "その他" else 0.3)
    )
    fig_l.update_layout(xaxis=dict(range=[-30, 30], scaleanchor="y", scaleratio=1), yaxis=dict(range=[-30, 30]), width=700, height=700)
    fig_l.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_l.add_vline(x=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_l, width="stretch")

    disp_l = df_all_l[df_all_l['teamid'] == target_team_id].copy()
    if not disp_l.empty:
        st.write(f"### {sel_team_name} ラインナップ一覧")
        disp_l['ユニット構成'] = disp_l.apply(get_unit_names, axis=1)
        output_l = disp_l[['ユニット構成', 'totalapps_l', 'ratingoff', 'hensatioff', 'hensatidef']].sort_values('totalapps_l', ascending=False)
        output_l.columns = ['ユニット構成', '合計プレイ数', 'Rating', '攻撃評価', '守備評価']
        st.dataframe(output_l.style.format({'Rating': '{:.1f}', '攻撃評価': '{:.1f}', '守備評価': '{:.1f}'}), width="stretch", hide_index=True)
