import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 1. ページ全体の基本設定
st.set_page_config(page_title="B-League Analytics Dash", layout="wide")

# 2. データの読み込みと前処理
@st.cache_data
def load_all_data():
    # 各CSVの読み込み
    df_t = pd.read_csv('table_team.csv')
    df_p = pd.read_csv('table_players.csv')
    df_l = pd.read_csv('table_lineups.csv')
    
    # 列名の前後にある空白を削除（大文字・小文字は維持）
    for df in [df_t, df_p, df_l]:
        df.columns = [str(c).strip() for c in df.columns]
        
        # --- 重要：数値型への変換 ---
        # 'Order' カラムが文字列（"1", "2"）だと 1, 10, 2... と並んでしまうため、
        # 確実に整数型(int)に変換します。
        numeric_cols = [
            'TeamID', 'PlayerID', 'Order', 'PlayerNo',
            'Lineup_1', 'Lineup_2', 'Lineup_3', 'Lineup_4', 'Lineup_5',
            'OFFApps', 'DEFApps', 'HensatiOFF', 'HensatiDEF', 'RatingOFF', 'RatingDEF'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # 偏差値などは小数点を維持するため float に戻す（念のため）
        float_cols = ['HensatiOFF', 'HensatiDEF', 'RatingOFF', 'RatingDEF']
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # 試合結果から分析期間を取得
    period_str = "データ期間不明"
    try:
        df_res = pd.read_csv('table_BLeagueResult_2025.csv')
        df_res.columns = [str(c).strip() for c in df_res.columns]
        df_res['Date'] = pd.to_datetime(df_res['Date'])
        finished = df_res.dropna(subset=['HomeScore', 'AwayScore'])
        if not finished.empty:
            period_str = f"{finished['Date'].min().strftime('%Y/%m/%d')} 〜 {finished['Date'].max().strftime('%Y/%m/%d')}"
    except:
        pass
        
    return df_t, df_p, df_l, period_str

df_team, df_player, df_lineup, analysis_period = load_all_data()

# 3. サイドバーのフィルター
st.sidebar.header("検索フィルター")

# リーグ選択 (CSVの登場順を維持)
list_league = list(dict.fromkeys(df_team['League']))
sel_league = st.sidebar.selectbox("リーグ選択", list_league)

# --- チーム名の表示順を 'Order' カラムで厳密に制御 ---
# 1. 選択されたリーグのチームだけを抽出
teams_in_league = df_team[df_team['League'] == sel_league].copy()

# 2. 'Order' カラムで昇順ソート（数値としてソートされるので 1, 2, 3... の順になります）
if 'Order' in teams_in_league.columns:
    teams_sorted = teams_in_league.sort_values(by='Order', ascending=True)
else:
    teams_sorted = teams_in_league.sort_values(by='TeamID', ascending=True)

# 3. ソートされた順序のまま「チーム名」をリスト化
list_teams = teams_sorted['Team'].tolist()

# 4. セレクトボックスを表示（これで文字順ではなく Order 順になります）
sel_team_name = st.sidebar.selectbox("チーム選択", list_teams)

# 選択されたチームのIDを取得
target_team_id = int(teams_sorted[teams_sorted['Team'] == sel_team_name]['TeamID'].iloc[0])

# 4. メインタイトル
st.title(f"🏀 {sel_team_name} 分析サイト")
st.info(f"📅 分析対象期間：{analysis_period}")

tab1, tab2 = st.tabs(["選手分析", "ラインナップ分析"])

# --- タブ1: 選手分析 ---
with tab1:
    st.subheader("選手別 評価値分布 (全体比較)")
    
    df_all_p = df_player.copy()
    df_all_p['is_selected'] = (df_all_p['TeamID'] == target_team_id)
    
    # プレイ合計数とバブルサイズ
    df_all_p['TotalApps'] = df_all_p['OFFApps'] + df_all_p['DEFApps']
    df_all_p['MarkerSize'] = np.sqrt(df_all_p['TotalApps'] + 1)
    df_all_p['DisplayGroup'] = df_all_p['is_selected'].map({True: sel_team_name, False: 'その他'})
    
    # ラベル表示
    df_all_p['Label'] = df_all_p.apply(
        lambda r: str(int(r['PlayerNo'])) if r['is_selected'] and r['PlayerNo'] != 0 else "", axis=1
    )
    
    # 描画順の制御（自チームを上に）
    df_all_p = df_all_p.sort_values('is_selected')

    fig_p = px.scatter(
        df_all_p, x='HensatiOFF', y='HensatiDEF', color='DisplayGroup', size='MarkerSize',
        text='Label', hover_name='PlayerNameJ',
        hover_data={'HensatiOFF': ':.1f', 'HensatiDEF': ':.1f', 'TotalApps': True, 'DisplayGroup': False, 'MarkerSize': False, 'Label': False},
        color_discrete_map={sel_team_name: '#EF553B', 'その他': '#E5ECF6'},
        labels={'HensatiOFF': '攻撃評価', 'HensatiDEF': '守備評価', 'TotalApps': '合計プレイ数'},
        opacity=df_all_p['is_selected'].map({True: 1.0, False: 0.4})
    )
    fig_p.update_layout(xaxis=dict(range=[-30, 30], scaleanchor="y", scaleratio=1), yaxis=dict(range=[-30, 30]), width=700, height=700)
    fig_p.update_traces(textposition='top center')
    fig_p.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_p.add_vline(x=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_p, width="stretch")

    # テーブル表示
    df_tp = df_all_p[df_all_p['is_selected']].copy()
    if not df_tp.empty:
        st.write(f"### {sel_team_name} 選手一覧")
        df_tp['背番号'] = df_tp['PlayerNo'].astype(int)
        output_p = df_tp[['背番号', 'PlayerNameJ', 'TotalApps', 'HensatiOFF', 'HensatiDEF']].sort_values('TotalApps', ascending=False)
        output_p.columns = ['背番号', '選手名', '合計プレイ数', '攻撃評価', '守備評価']
        st.dataframe(output_p.style.format({'攻撃評価': '{:.1f}', '守備評価': '{:.1f}'}), width="stretch", hide_index=True)

# --- タブ2: ラインナップ分析 ---
with tab2:
    st.subheader("ラインナップ別 評価値分布 (全体比較)")
    
    df_all_l = df_lineup.copy()
    df_all_l['TotalApps_L'] = df_all_l['OFFApps'] + df_all_l['DEFApps']
    df_all_l['MarkerSize'] = np.sqrt(df_all_l['TotalApps_L'] + 1)
    
    # 注目選手選択肢
    team_p = df_player[df_player['TeamID'] == target_team_id].sort_values('PlayerNo')
    p_options = ["指定なし"] + [f"{int(r['PlayerNo'])} {r['PlayerNameJ']}" for _, r in team_p.iterrows()]
    sel_p = st.selectbox("強調表示する選手を選択", p_options)
    
    target_p_id = None
    if sel_p != "指定なし":
        sel_p_no = int(sel_p.split()[0])
        target_p_id = int(team_p[team_p['PlayerNo'] == sel_p_no]['PlayerID'].iloc[0])

    def classify_lineup(row):
        if target_p_id:
            l_ids = [int(row[f'Lineup_{i}']) for i in range(1, 6)]
            if target_p_id in l_ids: return "★注目選手含む"
        if int(row['TeamID']) == target_team_id: return sel_team_name
        return "その他"

    df_all_l['DisplayGroup'] = df_all_l.apply(classify_lineup, axis=1)
    df_all_l = df_all_l.sort_values('DisplayGroup', key=lambda x: x.map({"その他":0, sel_team_name:1, "★注目選手含む":2}))

    def get_unit_names(row):
        names = []
        for i in range(1, 6):
            pid = int(row[f'Lineup_{i}'])
            match = df_player[df_player['PlayerID'] == pid]
            names.append(match.iloc[0]['PlayerNameJ'] if not match.empty else f"ID:{pid}")
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

    # テーブル表示
    disp_l = df_all_l[df_all_l['TeamID'] == target_team_id].copy()
    if not disp_l.empty:
        st.write(f"### {sel_team_name} ラインナップ一覧")
        disp_l['ユニット構成'] = disp_l.apply(get_unit_names, axis=1)
        output_l = disp_l[['ユニット構成', 'TotalApps_L', 'RatingOFF', 'HensatiOFF', 'HensatiDEF']].sort_values('TotalApps_L', ascending=False)
        output_l.columns = ['ユニット構成', '合計プレイ数', 'Rating', '攻撃評価', '守備評価']
        st.dataframe(output_l.style.format({'Rating': '{:.1f}', '攻撃評価': '{:.1f}', '守備評価': '{:.1f}'}), width="stretch", hide_index=True)
