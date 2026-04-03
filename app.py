import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 1. ページ全体の基本設定
st.set_page_config(page_title="B-League Analytics Dash", layout="wide")

# --- 2. データの読み込みと前処理 ---
@st.cache_data(ttl=3600)
def load_all_data():
    df_t = pd.read_csv('table_team.csv')
    df_p = pd.read_csv('table_players.csv')
    df_l = pd.read_csv('table_lineups.csv')
    # ショットデータの読み込み
    df_s = pd.read_csv('table_shotpos.csv')
    
    # 前処理
    for df in [df_t, df_p, df_l, df_s]:
        df.columns = [str(c).strip() for c in df.columns]
        
    num_cols = ['TeamID', 'PlayerID', 'Order', 'PlayerNo', 'OFFApps', 'DEFApps']
    for col in num_cols:
        if col in df_p.columns:
            df_p[col] = pd.to_numeric(df_p[col], errors='coerce').fillna(0).astype(int)

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
        
    return df_t, df_p, df_l, df_s, period_str

df_team, df_player, df_lineup, df_shot, analysis_period = load_all_data()

# --- コート描画用関数 ---
def draw_shot_chart(player_shots, player_name):
    # 成功・失敗のラベル
    player_shots = player_shots.copy()
    player_shots['Result'] = player_shots['ShotPoints'].apply(lambda x: '成功 (Made)' if x > 0 else '失敗 (Missed)')
    
    fig = px.scatter(
        player_shots,
        x='RelativeShotX',
        y='RelativeShotY',
        color='Result',
        symbol='Result',
        color_discrete_map={'成功 (Made)': '#EF553B', '失敗 (Missed)': '#636EFA'},
        symbol_sequence=['circle', 'x'],
        hover_data={'PlayText': True, 'RelativeShotX': False, 'RelativeShotY': False},
        title=f"🏀 {player_name} ショットチャート"
    )

    # --- コート図のライン描画 (FIBA規格) ---
    # 制限区域 (ペイント)
    fig.add_shape(type="rect", x0=-2.45, y0=0, x1=2.45, y1=5.8, line=dict(color="lightgray", width=2), layer="below")
    # フリースローサークル
    fig.add_shape(type="circle", x0=-1.8, y0=4.0, x1=1.8, y1=7.6, line=dict(color="lightgray", width=2), layer="below")
    # ゴール・バックボード
    fig.add_shape(type="line", x0=-0.9, y0=1.2, x1=0.9, y1=1.2, line=dict(color="black", width=3)) # ボード
    fig.add_shape(type="circle", x0=-0.22, y0=1.375, x1=0.22, y1=1.815, line=dict(color="orange", width=2)) # リング
    # 3Pライン
    fig.add_shape(type="path", 
                  path="M -6.6 0 L -6.6 2.8 A 6.75 6.75 0 0 1 6.6 2.8 L 6.6 0", 
                  line=dict(color="lightgray", width=2), layer="below")

    fig.update_layout(
        width=700, height=600,
        xaxis=dict(range=[-8, 8], title="", showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-1, 14], title="", showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig

# 3. サイドバー
st.sidebar.header("検索フィルター")
list_league = list(dict.fromkeys(df_team['League']))
sel_league = st.sidebar.selectbox("リーグ選択", list_league)
teams_in_league = df_team[df_team['League'] == sel_league].copy()
teams_sorted = teams_in_league.sort_values(by='Order', ascending=True) if 'Order' in teams_in_league.columns else teams_in_league.sort_values(by='TeamID', ascending=True)
list_teams = ["リーグ全体"] + teams_sorted['Team'].tolist()
sel_team_name = st.sidebar.selectbox("チーム選択", list_teams)
target_team_id = None if sel_team_name == "リーグ全体" else int(teams_sorted[teams_sorted['Team'] == sel_team_name]['TeamID'].iloc[0])

# 4. メイン
st.title(f"🏀 Bリーグ選手評価：{sel_team_name} ")
st.info(f"📅 分析対象期間：{analysis_period}")

tab1, tab2, tab3 = st.tabs(["選手分析", "ラインナップ分析","評価方法の概要"])

# --- タブ1: 選手分析 ---
with tab1:
    df_all_p = df_player.copy()
    df_all_p['TotalApps'] = df_all_p['OFFApps'] + df_all_p['DEFApps']
    
    # チームが選択されている場合、ショットチャート用の選手選択を出す
    selected_player_id = None
    if target_team_id:
        team_players = df_all_p[df_all_p['TeamID'] == target_team_id].sort_values('PlayerNo')
        p_list = ["(チャートを表示する選手を選択)"] + [f"{int(r['PlayerNo'])} {r['PlayerNameJ']}" for _, r in team_players.iterrows()]
        sel_p_chart = st.selectbox("ショットチャート表示", p_list)
        if sel_p_chart != "(チャートを表示する選手を選択)":
            p_name_only = sel_p_chart.split(" ", 1)[1]
            selected_player_id = int(team_players[team_players['PlayerNameJ'] == p_name_only]['PlayerID'].iloc[0])

    # 散布図描画
    is_league_mode = (target_team_id is None)
    df_all_p['MarkerSize'] = np.sqrt(df_all_p['TotalApps'] + 1)
    if is_league_mode:
        df_all_p['DisplayGroup'], df_all_p['is_selected'], df_all_p['Label'] = sel_league, True, ""
        color_map, opacity_val = {sel_league: '#636EFA'}, 0.3
    else:
        df_all_p['is_selected'] = (df_all_p['TeamID'] == target_team_id)
        df_all_p['DisplayGroup'] = df_all_p['is_selected'].map({True: sel_team_name, False: 'その他'})
        df_all_p['Label'] = df_all_p.apply(lambda r: str(int(r['PlayerNo'])) if r['is_selected'] and r['PlayerNo'] != 0 else "", axis=1)
        color_map = {sel_team_name: '#EF553B', 'その他': '#E5ECF6'}
        df_all_p = df_all_p.sort_values('is_selected')
        opacity_val = 0.3

    fig_p = px.scatter(
        df_all_p, x='HensatiOFF', y='HensatiDEF', color='DisplayGroup', size='MarkerSize', text='Label', hover_name='PlayerNameJ',
        color_discrete_map=color_map, height=600, labels={'HensatiOFF': '攻撃評価', 'HensatiDEF': '守備評価'},
        opacity=opacity_val if is_league_mode else None
    )
    fig_p.add_hline(y=0, line_dash="dot", line_color="gray"); fig_p.add_vline(x=0, line_dash="dot", line_color="gray")
    fig_p.update_layout(plot_bgcolor='white', xaxis=dict(range=[-30, 30]), yaxis=dict(range=[-30, 30], scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig_p, use_container_width=True)

    # ショットチャートの表示
    if selected_player_id:
        p_shots = df_shot[df_shot['PlayerID'] == selected_player_id]
        if not p_shots.empty:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(draw_shot_chart(p_shots, sel_p_chart), use_container_width=True)
            with col2:
                st.write("### シュート統計")
                total = len(p_shots)
                made = len(p_shots[p_shots['ShotPoints'] > 0])
                fg_pct = (made / total * 100) if total > 0 else 0
                st.metric("総シュート数", f"{total} 本")
                st.metric("成功数", f"{made} 本")
                st.metric("成功率 (FG%)", f"{fg_pct:.1f} %")
        else:
            st.warning("この選手のショットデータが見つかりません。")

    # --- テーブル表示 ---
    st.write(f"### {sel_team_name} 選手データ一覧")
    
    # 選択されたチーム（またはリーグ全体）の選手のみを抽出
    output_p = df_all_p[df_all_p['is_selected']].copy()
    
    if not output_p.empty:
        # 必要な列の計算
        output_p['総合評価'] = (output_p['HensatiOFF'] + output_p['HensatiDEF']) / 2
        output_p['貢献量'] = (output_p['HensatiOFF'] + output_p['HensatiDEF']) * output_p['TotalApps']
        output_p['公式サイト'] = "https://www.bleague.jp/roster_detail/?PlayerID=" + output_p['PlayerID'].astype(str)
        
        # 表示する列のリストを動的に作成
        if is_league_mode:
            team_dict = dict(zip(df_team['TeamID'], df_team['Team']))
            output_p['チーム'] = output_p['TeamID'].map(team_dict)
            cols = ['チーム', 'PlayerNo', 'PlayerNameJ', '公式サイト', 'TotalApps', '貢献量', '総合評価', 'HensatiOFF', 'HensatiDEF']
        else:
            cols = ['PlayerNo', 'PlayerNameJ', '公式サイト', 'TotalApps', '貢献量', '総合評価', 'HensatiOFF', 'HensatiDEF']
        
        # リネーム辞書
        rename_dict = {
            'PlayerNo': '背番号',
            'PlayerNameJ': '選手名',
            'TotalApps': '合計プレイ数',
            'HensatiOFF': '攻撃評価',
            'HensatiDEF': '守備評価'
        }
        
        # 存在する列だけを抽出してリネーム
        res_p = output_p[cols].rename(columns=rename_dict).sort_values('合計プレイ数', ascending=False)

        st.dataframe(
            res_p.style.format({
                '合計プレイ数': '{:d}', 
                '貢献量': '{:,.0f}', 
                '攻撃評価': '{:.1f}', 
                '守備評価': '{:.1f}', 
                '総合評価': '{:.1f}'
            }), 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "公式サイト": st.column_config.LinkColumn("公式", display_text="↗", width="small"),
                "背番号": st.column_config.NumberColumn(width="small"),
                "総合評価": st.column_config.NumberColumn(help="(攻撃評価 + 守備評価) / 2")
            }
        )
    else:
        st.info("表示できる選手データがありません。")

# --- タブ2: ラインナップ分析 ---
with tab2:
    n_league_lineups = 50
    df_plot = df_lineup[['TeamID', 'HensatiOFF', 'HensatiDEF', 'TotalApps_L', 'UnitNames', 'LineupSet']].copy()
    is_league_mode = (target_team_id is None)

    if not is_league_mode:
        st.subheader(f"ラインナップ別 評価値分布 ({sel_team_name})")
        team_p = df_player[df_player['TeamID'] == target_team_id].sort_values('PlayerNo')
        p_options = ["指定なし"] + [f"{int(r['PlayerNo'])} {r['PlayerNameJ']}" for _, r in team_p.iterrows()]
        sel_p = st.selectbox("強調表示する選手を選択", p_options)
        
        # 選択された選手のIDを取得
        target_p_id = int(team_p[team_p['PlayerNameJ'] == sel_p.split(" ", 1)[1]]['PlayerID'].iloc[0]) if sel_p != "指定なし" else None
    else:
        st.subheader(f"リーグ全体 ラインナップ評価分布 ({sel_league})")
        target_p_id = None

    # グループ分けロジック
    if is_league_mode:
        top_indices = df_plot.sort_values('TotalApps_L', ascending=False).head(n_league_lineups).index
        df_plot['is_top'] = df_plot.index.isin(top_indices)
        df_plot['DisplayGroup'] = df_plot['is_top'].map({True: f"上位{n_league_lineups}件", False: "その他"})
        plot_configs = [{"name": "その他", "color": "#E5ECF6", "opacity": 0.1}, {"name": f"上位{n_league_lineups}件", "color": "#636EFA", "opacity": 0.4}]
    else:
        def get_group_team(row):
            if target_p_id and target_p_id in row['LineupSet']: return "注目選手"
            return sel_team_name if row['TeamID'] == target_team_id else "その他"
        df_plot['DisplayGroup'] = df_plot.apply(get_group_team, axis=1)
        plot_configs = [{"name": "その他", "color": "#E5ECF6", "opacity": 0.15}, {"name": sel_team_name, "color": "#EF553B", "opacity": 0.4}, {"name": "注目選手", "color": "#19D3F3", "opacity": 0.8}]

    fig_l = go.Figure()
    for cfg in plot_configs:
        sub = df_plot[df_plot['DisplayGroup'] == cfg["name"]]
        if sub.empty: continue
        fig_l.add_trace(go.Scattergl(
            x=sub['HensatiOFF'], y=sub['HensatiDEF'], mode='markers', name=cfg["name"], text=sub['UnitNames'], customdata=sub['TotalApps_L'],
            marker=dict(size=np.sqrt(sub['TotalApps_L'] + 1) * 1.5, color=cfg["color"], opacity=cfg["opacity"], line=dict(width=0.5, color='white') if cfg["name"] != "その他" else None),
            hovertemplate="<b>%{text}</b><br>プレイ数: %{customdata}回<br>攻: %{x} / 守: %{y}<extra></extra>" if cfg["name"] != "その他" else None
        ))

    # --- ラインナップ分析タブ内のグラフ描画セクション ---
    if not is_league_mode and target_p_id:
        p_stats = df_player[df_player['PlayerID'] == target_p_id]
        if not p_stats.empty:
            p_off = p_stats['HensatiOFF'].iloc[0]
            p_def = p_stats['HensatiDEF'].iloc[0]
            
            # 縦線（攻撃偏差値）
            fig_l.add_vline(
                x=p_off, 
                line_dash="dash", 
                line_color="#19D3F3", 
                line_width=1.5, 
                annotation_text=f"攻: {p_off:+.1f}", # ← ここを修正
                annotation_position="top right"
            )
            # 横線（守備偏差値）
            fig_l.add_hline(
                y=p_def, 
                line_dash="dash", 
                line_color="#19D3F3", 
                line_width=1.5, 
                annotation_text=f"守: {p_def:+.1f}", # ← ここを修正
                annotation_position="bottom right"
            )

    fig_l.update_layout(
        title={'text': f"<b>{sel_team_name}</b> ラインナップ分析<br><span style='font-size:12px; color:gray;'>期間: {analysis_period}</span>", 'x': 0.5, 'y': 0.98, 'xanchor': 'center', 'yanchor': 'top'},
        margin=dict(l=20, r=20, t=110, b=100), xaxis=dict(range=[-30, 30], title="攻撃評価"), yaxis=dict(range=[-30, 30], title="守備評価", scaleanchor="x", scaleratio=1),
        height=750, plot_bgcolor='white', hovermode='closest', legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
    )
    fig_l.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_l.add_vline(x=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_l, use_container_width=True)

    # --- ラインナップ詳細表（総合評価の追加と列順変更） ---
    st.write(f"### {sel_team_name} ラインナップ詳細")
    if is_league_mode:
        df_table = df_plot[df_plot['is_top']].copy()
        team_dict = dict(zip(df_team['TeamID'], df_team['Team']))
        df_table['チーム'] = df_table['TeamID'].map(team_dict)
        # 総合評価（平均：(攻+守)/2）
        df_table['総合評価'] = (df_table['HensatiOFF'] + df_table['HensatiDEF']) / 2
        # 列順：総合 -> 攻撃 -> 守備
        output_l = df_table[['チーム', 'UnitNames', 'TotalApps_L', '総合評価', 'HensatiOFF', 'HensatiDEF']]
        output_l.columns = ['チーム', 'ラインナップ', '合計プレイ数', '総合評価', '攻撃評価', '守備評価']
    else:
        # 特定チームモード：target_p_id があれば絞り込む
        mask = df_plot['TeamID'] == target_team_id
        if target_p_id:
            mask = mask & (df_plot['LineupSet'].apply(lambda x: target_p_id in x))
        
        df_table = df_plot[mask].copy()
        # 総合評価（平均：(攻+守)/2）
        df_table['総合評価'] = (df_table['HensatiOFF'] + df_table['HensatiDEF']) / 2
        # 列順：総合 -> 攻撃 -> 守備
        output_l = df_table[['UnitNames', 'TotalApps_L', '総合評価', 'HensatiOFF', 'HensatiDEF']]
        output_l.columns = ['ラインナップ', '合計プレイ数', '総合評価', '攻撃評価', '守備評価']

    if not output_l.empty:
        st.dataframe(
            output_l.sort_values('合計プレイ数', ascending=False).style.format({'攻撃評価': '{:.1f}', '守備評価': '{:.1f}', '総合評価': '{:.1f}'}),
            use_container_width=True, hide_index=True,
            column_config={"総合評価": st.column_config.NumberColumn(help="(攻撃評価 + 守備評価) / 2")}
        )
    else:
        st.info("該当するデータがありません。")

# --- タブ3: 算出方法 ---
with tab3:
    st.header("評価値の算出方法について")
    st.markdown("""
    本分析サイトで使用している指標の定義と算出方法は以下の通りです。

    ### 1. 評価値の定義
    グラフの軸となっている **「攻撃評価」「守備評価」** は、リーグ全体の平均を **0**，標準偏差を **10**として算出しています．
    * **総合評価**: 攻撃評価と守備評価の平均値（(攻撃＋守備)/2）です．
    * **選手評価**: 後述する「ラインナップ評価」で，その選手を含むラインナップのプレイ数重み付平均です．

    ### 2. ラインナップデータの集計
    * 同時にコートに立っている5人の組み合わせを1つの「ラインナップ」として集計しています。
    * **合計プレイ数**: その5人の組み合わせが合計で何回起用されたか（攻撃/守備の合計）を示します。
    * 1回のプレイで，攻撃側は得点すること，守備側は失点しないことが **小さな勝利**であるとみなし，その勝率を評価式で評価しました。
    """)
    st.info("※ 本データは公式統計を元にkonakalabが独自に算出したものであり、B.LEAGUE公式の指標ではありません")
