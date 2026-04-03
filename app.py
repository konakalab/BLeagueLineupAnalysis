import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 1. ページ全体の基本設定
st.set_page_config(page_title="B-League Analytics Dash", layout="wide")

# --- 2. データの読み込みと前処理（高速化版） ---
@st.cache_data(ttl=3600)
def load_all_data():
    df_t = pd.read_csv('table_team.csv')
    df_p = pd.read_csv('table_players.csv')
    df_l = pd.read_csv('table_lineups.csv')
    
    # 前処理（列名のトリミングと数値変換）
    for df in [df_t, df_p, df_l]:
        df.columns = [str(c).strip() for c in df.columns]
        num_cols = ['TeamID', 'PlayerID', 'Order', 'PlayerNo', 'Lineup_1', 'Lineup_2', 'Lineup_3', 'Lineup_4', 'Lineup_5', 'OFFApps', 'DEFApps']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        for col in ['HensatiOFF', 'HensatiDEF', 'RatingOFF']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).round(1)

    # IDから名前と背番号を引く辞書
    p_dict = dict(zip(df_p['PlayerID'], df_p['PlayerNameJ']))
    p_no_dict = dict(zip(df_p['PlayerID'], df_p['PlayerNo']))

    # ユニット名生成（背番号順にソートして結合）
    def get_sorted_unit_names(row):
        p_ids = [int(row[f'Lineup_{i}']) for i in range(1, 6) if pd.notna(row[f'Lineup_{i}']) and int(row[f'Lineup_{i}']) != 0]
        p_info = sorted([(p_no_dict.get(pid, 999), p_dict.get(pid, "??")) for pid in p_ids], key=lambda x: x[0])
        return " / ".join([x[1] for x in p_info])

    df_l['UnitNames'] = df_l.apply(get_sorted_unit_names, axis=1)
    df_l['LineupSet'] = df_l.apply(lambda r: {int(r[f'Lineup_{i}']) for i in range(1, 6)}, axis=1)
    df_l['TotalApps_L'] = df_l['OFFApps'] + df_l['DEFApps']

    # 期間取得
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
list_league = list(dict.fromkeys(df_team['League']))
sel_league = st.sidebar.selectbox("リーグ選択", list_league)

teams_in_league = df_team[df_team['League'] == sel_league].copy()
teams_sorted = teams_in_league.sort_values(by='Order' if 'Order' in teams_in_league.columns else 'TeamID')
list_teams = ["リーグ全体"] + teams_sorted['Team'].tolist()
sel_team_name = st.sidebar.selectbox("チーム選択", list_teams)

target_team_id = None if sel_team_name == "リーグ全体" else int(teams_sorted[teams_sorted['Team'] == sel_team_name]['TeamID'].iloc[0])

# 4. メインタイトル
st.title(f"🏀 Bリーグ分析：{sel_team_name}")
st.info(f"📅 分析対象期間：{analysis_period}")

tab1, tab2, tab3 = st.tabs(["選手分析", "ラインナップ分析", "評価方法の概要"])

# --- タブ1: 選手分析 ---
with tab1:
    df_all_p = df_player.copy()
    df_all_p['TotalApps'] = df_all_p['OFFApps'] + df_all_p['DEFApps']
    is_league_mode = (target_team_id is None)

    if is_league_mode:
        st.subheader(f"リーグ全体 選手評価分布 ({sel_league})")
        df_all_p['DisplayGroup'] = sel_league
        df_all_p['is_selected'] = True
        color_map = {sel_league: '#636EFA'}
    else:
        st.subheader(f"選手別 評価値分布 ({sel_team_name})")
        df_all_p['is_selected'] = (df_all_p['TeamID'] == target_team_id)
        df_all_p['DisplayGroup'] = df_all_p['is_selected'].map({True: sel_team_name, False: 'その他'})
        color_map = {sel_team_name: '#EF553B', 'その他': '#E5ECF6'}
        df_all_p = df_all_p.sort_values('is_selected')

    # グラフ描画
    fig_p = px.scatter(
        df_all_p, x='HensatiOFF', y='HensatiDEF', color='DisplayGroup',
        size=np.sqrt(df_all_p['TotalApps'] + 1), hover_name='PlayerNameJ',
        hover_data={'TotalApps': True, 'HensatiOFF': ':.1f', 'HensatiDEF': ':.1f', 'DisplayGroup': False},
        color_discrete_map=color_map, opacity=0.4 if is_league_mode else None
    )
    fig_p.update_layout(xaxis=dict(range=[-30, 30], title="攻撃評価"), yaxis=dict(range=[-30, 30], title="守備評価", scaleanchor="x"), height=700, plot_bgcolor='white')
    fig_p.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_p.add_vline(x=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_p, use_container_width=True)

    # 選手テーブル
    output_p = df_all_p[df_all_p['is_selected']].copy()
    output_p['公式サイト'] = "https://www.bleague.jp/roster_detail/?PlayerID=" + output_p['PlayerID'].astype(str)
    output_p['貢献量'] = (output_p['HensatiOFF'] + output_p['HensatiDEF']) * output_p['TotalApps']
    
    # リーグ全体時はチーム名を表示
    cols = (['TeamID', 'PlayerNo', 'PlayerNameJ', '公式サイト', 'TotalApps', '貢献量', 'HensatiOFF', 'HensatiDEF'] 
            if is_league_mode else ['PlayerNo', 'PlayerNameJ', '公式サイト', 'TotalApps', '貢献量', 'HensatiOFF', 'HensatiDEF'])
    
    res_p = output_p[cols].sort_values('TotalApps', ascending=False)
    if is_league_mode:
        team_dict = dict(zip(df_team['TeamID'], df_team['Team']))
        res_p['TeamID'] = res_p['TeamID'].map(team_dict)
        res_p.rename(columns={'TeamID': 'チーム'}, inplace=True)
    
    res_p.columns = [c.replace('PlayerNo','背番号').replace('PlayerNameJ','選手名').replace('TotalApps','合計プレイ数').replace('HensatiOFF','攻撃評価').replace('HensatiDEF','守備評価') for c in res_p.columns]

    st.dataframe(res_p, use_container_width=True, hide_index=True, column_config={"公式サイト": st.column_config.LinkColumn("公式", display_text="↗")})

# --- タブ2: ラインナップ分析 ---
with tab2:
    n_league_lineups = 50
    df_plot = df_lineup[['TeamID', 'HensatiOFF', 'HensatiDEF', 'TotalApps_L', 'UnitNames', 'LineupSet']].copy()
    is_league_mode = (target_team_id is None)

    # 注目選手選択（特定チーム時）
    target_p_id = None
    if not is_league_mode:
        team_p = df_player[df_player['TeamID'] == target_team_id].sort_values('PlayerNo')
        p_options = ["指定なし"] + [f"{int(r['PlayerNo'])} {r['PlayerNameJ']}" for _, r in team_p.iterrows()]
        sel_p = st.selectbox("強調表示する選手を選択", p_options)
        if sel_p != "指定なし":
            target_p_id = int(team_p[team_p['PlayerNameJ'] == sel_p.split(" ", 1)[1]]['PlayerID'].iloc[0])

    # グループ分け
    if is_league_mode:
        top_idx = df_plot.sort_values('TotalApps_L', ascending=False).head(n_league_lineups).index
        df_plot['Group'] = df_plot.index.isin(top_idx).map({True: f"上位{n_league_lineups}件", False: "その他"})
        configs = [{"name": "その他", "color": "#E5ECF6", "opacity": 0.1}, {"name": f"上位{n_league_lineups}件", "color": "#636EFA", "opacity": 0.8}]
    else:
        def get_g(r):
            if target_p_id and target_p_id in r['LineupSet']: return "注目選手"
            return sel_team_name if r['TeamID'] == target_team_id else "その他"
        df_plot['Group'] = df_plot.apply(get_g, axis=1)
        configs = [{"name": "その他", "color": "#E5ECF6", "opacity": 0.15}, {"name": sel_team_name, "color": "#EF553B", "opacity": 0.4}, {"name": "注目選手", "color": "#19D3F3", "opacity": 0.8}]

    # グラフ
    fig_l = go.Figure()
    for cfg in configs:
        sub = df_plot[df_plot['Group'] == cfg["name"]]
        if sub.empty: continue
        fig_l.add_trace(go.Scattergl(
            x=sub['HensatiOFF'], y=sub['HensatiDEF'], mode='markers', name=cfg["name"], text=sub['UnitNames'],
            marker=dict(size=np.sqrt(sub['TotalApps_L']+1)*1.5, color=cfg["color"], opacity=cfg["opacity"], line=dict(width=0.5, color='white'))
        ))
    fig_l.update_layout(xaxis=dict(range=[-30, 30], title="攻撃評価"), yaxis=dict(range=[-30, 30], title="守備評価", scaleanchor="x"), height=700, plot_bgcolor='white')
    st.plotly_chart(fig_l, use_container_width=True)

    # 詳細表
    st.write("### ラインナップ詳細")
    if is_league_mode:
        out_l = df_plot[df_plot['Group'] != "その他"].copy()
        team_dict = dict(zip(df_team['TeamID'], df_team['Team']))
        out_l['チーム'] = out_l['TeamID'].map(team_dict)
        out_l = out_l[['チーム', 'UnitNames', 'TotalApps_L', 'HensatiOFF', 'HensatiDEF']]
        st.info(f"💡 プレイ数上位 {n_league_lineups} 件を表示しています。")
    else:
        out_l = df_plot[df_plot['Group'] != "その他"][['UnitNames', 'TotalApps_L', 'HensatiOFF', 'HensatiDEF']]
    
    out_l.columns = [c.replace('UnitNames','ユニット構成').replace('TotalApps_L','プレイ数').replace('HensatiOFF','攻撃評価').replace('HensatiDEF','守備評価') for c in out_l.columns]
    st.dataframe(out_l.sort_values('プレイ数', ascending=False), use_container_width=True, hide_index=True)

# --- タブ3: 算出方法 ---
with tab3:
    st.header("評価値の算出方法について")
    st.markdown("""
    本分析サイトで使用している指標の定義と算出方法は以下の通りです。
    ### 1. 評価値の定義
    リーグ全体の平均を **0**，標準偏差を **10**として算出しています．
    ### 2. ラインナップデータの集計
    5人の組み合わせごとの勝率（得点/失点を反映）を算出し、評価値に変換しています。
    """)
    st.info("※ 本データは公式統計を元に独自に算出したものであり、B.LEAGUE公式の指標ではありません")
