import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # ← これが抜けていたためエラーになっています
import numpy as np

# 1. ページ全体の基本設定
st.set_page_config(page_title="B-League Analytics Dash", layout="wide")

# --- 2. データの読み込みと前処理（高速化版） ---
@st.cache_data(ttl=3600)
def load_all_data():
    df_t = pd.read_csv('table_team.csv')
    df_p = pd.read_csv('table_players.csv')
    df_l = pd.read_csv('table_lineups.csv')
    
    # 前処理
    for df in [df_t, df_p, df_l]:
        df.columns = [str(c).strip() for c in df.columns]
        num_cols = ['TeamID', 'PlayerID', 'Order', 'PlayerNo', 'Lineup_1', 'Lineup_2', 'Lineup_3', 'Lineup_4', 'Lineup_5', 'OFFApps', 'DEFApps']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        for col in ['HensatiOFF', 'HensatiDEF', 'RatingOFF']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).round(1)

    p_dict = dict(zip(df_p['PlayerID'], df_p['PlayerNameJ']))
    df_l['UnitNames'] = df_l.apply(lambda r: " / ".join([p_dict.get(int(r[f'Lineup_{i}']), "??") for i in range(1,6)]), axis=1)
    df_l['LineupSet'] = df_l.apply(lambda r: {int(r[f'Lineup_{i}']) for i in range(1, 6)}, axis=1)
    df_l['TotalApps_L'] = df_l['OFFApps'] + df_l['DEFApps']

    # --- 期間取得のロジックを復活 ---
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
        
    # 4つの値を返す
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

with tab2:
    st.subheader("ラインナップ別 評価値分布")
    
    # --- 1. 注目選手の選択処理 ---
    team_p = df_player[df_player['TeamID'] == target_team_id].sort_values('PlayerNo')
    p_options = ["指定なし"] + [f"{int(r['PlayerNo'])} {r['PlayerNameJ']}" for _, r in team_p.iterrows()]
    sel_p = st.selectbox("強調表示する選手を選択", p_options)
    
    target_p_id = None
    if sel_p != "指定なし":
        # 選択された文字列から PlayerID を取得
        sel_no = int(sel_p.split()[0])
        target_p_id = int(team_p[team_p['PlayerNo'] == sel_no]['PlayerID'].iloc[0])

    # 判定用の一時データフレーム作成
    df_plot = df_lineup[['TeamID', 'HensatiOFF', 'HensatiDEF', 'TotalApps_L', 'UnitNames', 'LineupSet']].copy()
    
    # 表示グループの判定関数
    def get_group(row):
        if target_p_id and target_p_id in row['LineupSet']:
            return "注目選手"
        return sel_team_name if row['TeamID'] == target_team_id else "その他"

    df_plot['DisplayGroup'] = df_plot.apply(get_group, axis=1)
    
    # --- 2. Plotly グラフ作成 ---
    fig_l = go.Figure()
    
    # 透明度の調整
    plot_configs = [
        {"name": "その他", "color": "#E5ECF6", "opacity": 0.15},
        {"name": sel_team_name, "color": "#EF553B", "opacity": 0.4},
        {"name": "注目選手", "color": "#19D3F3", "opacity": 0.6}
    ]

    for cfg in plot_configs:
        sub = df_plot[df_plot['DisplayGroup'] == cfg["name"]]
        if sub.empty: continue
        
        # 判定ロジック（名称を「★注目選手含む」に合わせました）
        if cfg["name"] == "その他":
            hover_setting = "skip"
        elif target_p_id is not None:
            hover_setting = "all" if cfg["name"] == "★注目選手含む" else "skip"
        else:
            hover_setting = "all"

        fig_l.add_trace(go.Scattergl(
            x=sub['HensatiOFF'],
            y=sub['HensatiDEF'],
            mode='markers',
            name=cfg["name"],
            text=sub['UnitNames'],
            customdata=sub['TotalApps_L'], 
            hoverinfo=hover_setting,
            marker=dict(
                size=np.sqrt(sub['TotalApps_L'] + 1) * 1.2, 
                color=cfg["color"],
                opacity=cfg["opacity"],
                line=dict(width=0.5, color='white') if cfg["name"] != "その他" else None
            ),
            # 【追加】ツールチップの外観設定（テンプレートの内容は変えずに見栄えを調整）
            hoverlabel=dict(
                bgcolor="rgba(255, 255, 255, 0.9)", # 背景を白（少し透過）にして文字を読みやすく
                bordercolor="gray",                # 枠線をつけてマーカーとの境界を明示
                font_size=12,
                namelength=0                       # 横のラベル（トレース名）を消してコンパクトに
            ),
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "合計プレイ数: %{customdata}回，" + 
                "攻撃評価: %{x}，" +
                "守備評価: %{y}<extra></extra>"
            ) if hover_setting == "all" else None
        ))

    # --- グラフ全体のレイアウト設定 ---
    fig_l.update_layout(
        # 【重要】最も近い1点だけに反応させることで、重なりによる混雑を回避
        hovermode='closest', 
        # ホバーラベルがグラフの端で切れないように調整
        hoverdistance=100,
        xaxis=dict(range=[-30, 30], title="攻撃評価", gridcolor='lightgray'),
        yaxis=dict(range=[-30, 30], title="守備評価", gridcolor='lightgray', scaleanchor="x", scaleratio=1),
        height=700, margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='white'
    )
    
    # ツールチップの表示位置を最適化する設定
    fig_l.update_traces(hoveron='points')
    
    fig_l.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_l.add_vline(x=0, line_dash="dot", line_color="gray")
    
    st.plotly_chart(fig_l, use_container_width=True)
    
    # --- 3. ラインナップ詳細表の表示 ---
    st.write(f"### {sel_team_name} ラインナップ詳細")
    
    # フィルタリング処理（チームIDと選手IDで確実に抽出）
    if target_p_id:
        df_table = df_plot[
            (df_plot['TeamID'] == target_team_id) & 
            (df_plot['LineupSet'].apply(lambda x: target_p_id in x))
        ].copy()
    else:
        df_table = df_plot[df_plot['TeamID'] == target_team_id].copy()

    if not df_table.empty:
        # データの整形とカラム名の統一
        output_l = df_table[['UnitNames', 'TotalApps_L', 'HensatiOFF', 'HensatiDEF']].sort_values('TotalApps_L', ascending=False)
        output_l.columns = ['ラインナップ構成', '合計プレイ数', '攻撃評価', '守備評価']
        
        # 表を幅いっぱいに収める設定
        st.dataframe(
            output_l.style.format({'攻撃評価': '{:.1f}', '守備評価': '{:.1f}'}),
            use_container_width=True,
            hide_index=True,
            column_config={
                "ラインナップ構成": st.column_config.TextColumn("ラインナップ構成", width="large"),
                "合計プレイ数": st.column_config.NumberColumn("合計プレイ数", width="small"),
                "攻撃評価": st.column_config.NumberColumn("攻撃評価", width="small"),
                "守備評価": st.column_config.NumberColumn("守備評価", width="small"),
            }
        )
    else:
        st.info("該当するラインナップデータが見つかりませんでした。")
