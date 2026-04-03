import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 1. ページ全体の基本設定
st.set_page_config(page_title="B-League Analytics Dash", layout="wide")

# --- 2. データの読み込みと前処理（ショットデータ追加版） ---
@st.cache_data(ttl=3600)
def load_all_data():
    df_t = pd.read_csv('table_team.csv')
    df_p = pd.read_csv('table_players.csv')
    df_l = pd.read_csv('table_lineups.csv')
    # 1. ショット位置データを読み込み対象に追加
    df_s = pd.read_csv('table_shotpos.csv')
    
    # 前処理（数値変換など）
    # 2. ループ対象に df_s を追加して、列名の空白削除などを一括で行う
    for df in [df_t, df_p, df_l, df_s]:
        df.columns = [str(c).strip() for c in df.columns]
        
        # 数値型の列を安全に変換
        num_cols = ['TeamID', 'PlayerID', 'Order', 'PlayerNo', 'Lineup_1', 'Lineup_2', 'Lineup_3', 'Lineup_4', 'Lineup_5', 'OFFApps', 'DEFApps', 'ShotPoints']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # 小数点型の列を丸める
        for col in ['HensatiOFF', 'HensatiDEF', 'RatingOFF', 'RelativeShotX', 'RelativeShotY']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # --- IDから「名前」を引く辞書等の作成 (既存ロジック維持) ---
    p_dict = dict(zip(df_p['PlayerID'], df_p['PlayerNameJ']))
    p_no_dict = dict(zip(df_p['PlayerID'], df_p['PlayerNo']))

    def get_sorted_unit_names(row):
        p_ids = []
        for i in range(1, 6):
            val = row[f'Lineup_{i}']
            if pd.notna(val) and int(val) != 0:
                p_ids.append(int(val))
        p_info = []
        for pid in p_ids:
            no = p_no_dict.get(pid, 999) 
            name = p_dict.get(pid, "??")
            p_info.append((no, name))
        p_info.sort(key=lambda x: x[0])
        return " / ".join([x[1] for x in p_info])

    df_l['UnitNames'] = df_l.apply(get_sorted_unit_names, axis=1)
    df_l['LineupSet'] = df_l.apply(lambda r: {int(r[f'Lineup_{i}']) for i in range(1, 6)}, axis=1)
    df_l['TotalApps_L'] = df_l['OFFApps'] + df_l['DEFApps']

    # --- 期間取得のロジック ---
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
        
    # 3. 戻り値に df_s を追加
    return df_t, df_p, df_l, df_s, period_str
    
def draw_shot_chart(player_shots, player_name):
    """
    MATLABコードの規格と向きを完全に再現したショットチャート
    """
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

    line_color = "#333333"
    
    # --- FIBA / MATLAB 規格ライン ---
    
    # 1. バックボード (x=1.2, y=[-0.9, 0.9])
    fig.add_shape(type="line", x0=1.2, y0=-0.9, x1=1.2, y1=0.9, line=dict(color="black", width=3))
    
    # 2. リング (中心x=1.575, 半径0.225)
    fig.add_shape(type="circle", x0=1.575-0.225, y0=-0.225, x1=1.575+0.225, y1=0.225, line=dict(color="orange", width=2))
    
    # 3. 制限区域 (x=[0, 5.8], y=[-4.9/2, 4.9/2])
    fig.add_shape(type="rect", x0=0, y0=-2.45, x1=5.8, y1=2.45, line=dict(color=line_color, width=1), layer="below")
    
    # 4. フリースローサークル (中心x=5.8, 半径1.8)
    fig.add_shape(type="circle", x0=5.8-1.8, y0=-1.8, x1=5.8+1.8, y1=1.8, line=dict(color=line_color, width=1, dash="dot"), layer="below")
    
    # 5. 3ポイントライン (MATLABロジックの厳密な再現)
    # 直線部: y=±6.6m (7.5-0.9), x=[0, 2.99]
    # 円弧部: 中心(1.575, 0), 半径6.75m
    # PlotlyのArcコマンド(A)を使用: 半径x 半径y 角度 フラグ フラグ 終点x 終点y
    three_point_path = (
        "M 0 -6.6 "                   # 下側サイドライン接点から開始
        "L 2.99 -6.6 "                # 直線部
        "A 6.75 6.75 0 0 1 2.99 6.6 " # 円弧 (終点x=2.99, 終点y=6.6)
        "L 0 6.6"                     # 上側直線部
    )
    fig.add_shape(type="path", path=three_point_path, line=dict(color=line_color, width=2), layer="below")

    # 6. コート外枠とセンターライン (x=14)
    fig.add_shape(type="rect", x0=0, y0=-7.5, x1=14, y1=7.5, line=dict(color=line_color, width=2), layer="below")
    fig.add_shape(type="line", x0=14, y0=-7.5, x1=14, y1=7.5, line=dict(color=line_color, width=2))

    # レイアウト設定 (サイズを大きくし、余白を削る)
    fig.update_layout(
        width=1000,   # 幅を拡大
        height=600,   # 高さを調整
        xaxis=dict(
            range=[-1, 15], 
            showgrid=False, 
            zeroline=False, 
            visible=False, 
            scaleanchor="y", 
            scaleratio=1
        ),
        yaxis=dict(
            range=[-8, 8], 
            showgrid=False, 
            zeroline=False, 
            visible=False
        ),
        plot_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=60, b=10) # マージンを最小化
    )
    
    return fig
    
# 4. 関数呼び出し側でも df_shot として受け取る
df_team, df_player, df_lineup, df_shot, analysis_period = load_all_data()

# 3. サイドバーのフィルター
st.sidebar.header("検索フィルター")
list_league = list(dict.fromkeys(df_team['League']))
sel_league = st.sidebar.selectbox("リーグ選択", list_league)

teams_in_league = df_team[df_team['League'] == sel_league].copy()
if 'Order' in teams_in_league.columns:
    teams_sorted = teams_in_league.sort_values(by='Order', ascending=True)
else:
    teams_sorted = teams_in_league.sort_values(by='TeamID', ascending=True)

list_teams = ["リーグ全体"] + teams_sorted['Team'].tolist()
sel_team_name = st.sidebar.selectbox("チーム選択", list_teams)

if sel_team_name == "リーグ全体":
    target_team_id = None
else:
    target_team_id = int(teams_sorted[teams_sorted['Team'] == sel_team_name]['TeamID'].iloc[0])

# 4. メインタイトル
st.title(f"🏀 Bリーグ選手評価：{sel_team_name} ")
st.info(f"📅 分析対象期間：{analysis_period}")

with st.expander("💡 この分析ツールの使い方はこちら"):
    st.write("""
    1. 左側のサイドバーでリーグとチームを選択してください。
    2. 各グラフのドットにマウスを合わせると詳細データが表示されます。
    3. ラインナップ分析では、特定の選手を強調して表示できます。
    """)
st.caption(f"Developed by [@konakalab](https://x.com/konakalab) | 📅 データ更新：{analysis_period}")

tab1, tab2, tab3 = st.tabs(["選手分析", "ラインナップ分析","評価方法の概要"])

# --- タブ1: 選手分析 ---
with tab1:
    df_all_p = df_player.copy()
    df_all_p['TotalApps'] = df_all_p['OFFApps'] + df_all_p['DEFApps']
    df_all_p['MarkerSize'] = np.sqrt(df_all_p['TotalApps'] + 1)
    is_league_mode = (target_team_id is None)

    if is_league_mode:
        st.subheader(f"リーグ全体 選手評価分布 ({sel_league})")
        df_all_p['DisplayGroup'] = sel_league
        df_all_p['is_selected'] = True
        df_all_p['Label'] = ""
        color_map = {sel_league: '#636EFA'}
        opacity_val = 0.3
    else:
        st.subheader(f"選手別 評価値分布 ({sel_team_name})")
        df_all_p['is_selected'] = (df_all_p['TeamID'] == target_team_id)
        df_all_p['DisplayGroup'] = df_all_p['is_selected'].map({True: sel_team_name, False: 'その他'})
        df_all_p['Label'] = df_all_p.apply(lambda r: str(int(r['PlayerNo'])) if r['is_selected'] and r['PlayerNo'] != 0 else "", axis=1)
        color_map = {sel_team_name: '#EF553B', 'その他': '#E5ECF6'}
        df_all_p = df_all_p.sort_values('is_selected')
        opacity_val = df_all_p['is_selected'].map({True: 0.15, False: 0.3})

    fig_p = px.scatter(
        df_all_p, x='HensatiOFF', y='HensatiDEF', color='DisplayGroup', size='MarkerSize', text='Label', hover_name='PlayerNameJ',
        hover_data={'HensatiOFF': ':.1f', 'HensatiDEF': ':.1f', 'TotalApps': True, 'DisplayGroup': False, 'MarkerSize': False, 'Label': False},
        color_discrete_map=color_map, labels={'HensatiOFF': '攻撃評価', 'HensatiDEF': '守備評価', 'TotalApps': '合計プレイ数'},
        opacity=opacity_val if is_league_mode else None
    )

    fig_p.update_layout(
        title={'text': f"<b>{sel_team_name}</b> 選手評価分布<br><span style='font-size:12px; color:gray;'>期間: {analysis_period}</span>", 'x': 0.5, 'y': 0.98, 'xanchor': 'center', 'yanchor': 'top'},
        margin=dict(l=20, r=20, t=100, b=100),
        xaxis=dict(range=[-30, 30], title="攻撃評価", gridcolor='lightgray', showspikes=True, spikecolor="gray", spikethickness=1, spikedash="dot", spikemode="across"),
        yaxis=dict(range=[-30, 30], title="守備評価", gridcolor='lightgray', scaleanchor="x", scaleratio=1, showspikes=True, spikecolor="gray", spikethickness=1, spikedash="dot", spikemode="across"),
        height=750, plot_bgcolor='white', hovermode='closest', legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
    )
    if not is_league_mode:
        fig_p.update_traces(textposition='top center', selector=dict(name=sel_team_name))
    fig_p.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_p.add_vline(x=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_p, use_container_width=True)

    # --- ショット分析セクション ---
    if not is_league_mode:
        st.divider()
        st.write(f"## 🏀 {sel_team_name} ショット分析")
        
        # チーム全体のデータを抽出
        team_shots = df_shot[df_shot['TeamID'] == target_team_id]
        
        # 選手選択（チーム全体か個人か）
        team_players = df_all_p[df_all_p['TeamID'] == target_team_id].sort_values('PlayerNo')
        p_options = ["チーム全体"] + [f"{int(r['PlayerNo'])} {r['PlayerNameJ']}" for _, r in team_players.iterrows()]
        sel_p_shot = st.selectbox("表示対象を選択", p_options)

        if sel_p_shot == "チーム全体":
            display_shots = team_shots
            chart_title = f"{sel_team_name} (チーム全体)"
        else:
            p_name_only = sel_p_shot.split(" ", 1)[1]
            selected_player_id = int(team_players[team_players['PlayerNameJ'] == p_name_only]['PlayerID'].iloc[0])
            display_shots = team_shots[team_shots['PlayerID'] == selected_player_id]
            chart_title = p_name_only

        if not display_shots.empty:
            # 1. 統計数値を上に配置（横並びのメトリクス）
            total = len(display_shots)
            made = len(display_shots[display_shots['ShotPoints'] > 0])
            fg_pct = (made / total * 100) if total > 0 else 0
            
            m1, m2, m3 = st.columns(3)
            m1.metric("総シュート試投数", f"{total} 本")
            m2.metric("成功数", f"{made} 本")
            m3.metric("成功率 (FG%)", f"{fg_pct:.1f} %")

            # 2. ショットチャートを下に配置（大きく表示）
            # draw_shot_chart関数の内部でwidth=1100などに設定するとより効果的です
            st.plotly_chart(draw_shot_chart(display_shots, chart_title), use_container_width=True)
            
        else:
            st.warning("表示対象のショット位置データが見つかりません。")
            
        st.divider()
        
    # --- テーブル表示 ---
    st.write(f"### {sel_team_name} 選手データ一覧")
    output_p = df_all_p[df_all_p['is_selected']].copy()
    
    # 総合評価の計算（平均：(攻+守)/2）と貢献量の計算
    output_p['総合評価'] = (output_p['HensatiOFF'] + output_p['HensatiDEF']) / 2
    output_p['貢献量'] = (output_p['HensatiOFF'] + output_p['HensatiDEF']) * output_p['TotalApps'] # 貢献量は和のスケールを維持
    output_p['公式サイト'] = "https://www.bleague.jp/roster_detail/?PlayerID=" + output_p['PlayerID'].astype(str)
    
    # 列順：総合 -> 攻撃 -> 守備
    if is_league_mode:
        team_dict = dict(zip(df_team['TeamID'], df_team['Team']))
        output_p['チーム'] = output_p['TeamID'].map(team_dict)
        output_p = output_p.dropna(subset=['チーム'])
        cols = ['チーム', 'PlayerNo', 'PlayerNameJ', '公式サイト', 'TotalApps', '貢献量', '総合評価', 'HensatiOFF', 'HensatiDEF']
    else:
        cols = ['PlayerNo', 'PlayerNameJ', '公式サイト', 'TotalApps', '貢献量', '総合評価', 'HensatiOFF', 'HensatiDEF']
    
    rename_dict = {'PlayerNo': '背番号', 'PlayerNameJ': '選手名', 'TotalApps': '合計プレイ数', 'HensatiOFF': '攻撃評価', 'HensatiDEF': '守備評価'}
    res_p = output_p[cols].rename(columns=rename_dict).sort_values('合計プレイ数', ascending=False)

    st.dataframe(
        res_p.style.format({'合計プレイ数': '{:d}', '貢献量': '{:,.0f}', '攻撃評価': '{:.1f}', '守備評価': '{:.1f}', '総合評価': '{:.1f}'}), 
        use_container_width=True, hide_index=True,
        column_config={
            "公式サイト": st.column_config.LinkColumn("公式", display_text="↗", width="small"),
            "背番号": st.column_config.NumberColumn(width="small"),
            "選手名": st.column_config.TextColumn(width="medium"),
            "総合評価": st.column_config.NumberColumn(help="(攻撃評価 + 守備評価) / 2")
        }
    )

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
