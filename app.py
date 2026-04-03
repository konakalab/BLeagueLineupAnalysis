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
    
    # --- 修正箇所：CSVからParquetの読み込みに変更 ---
    try:
        # engine='pyarrow' を指定するとより高速・安定します
        df_s = pd.read_parquet('table_shotpos.parquet')
    except Exception as e:
        st.error(f"Parquetファイルの読み込みに失敗しました: {e}")
        # ファイルがない場合のバックアップとして空のDFを作成
        df_s = pd.DataFrame(columns=['ScheduleKey', 'TeamID', 'PlayerID', 'ActionCD1', 'RelativeShotX', 'RelativeShotY', 'ShotPoints'])

    # 前処理（列名の空白削除など）
    for df in [df_t, df_p, df_l, df_s]:
        df.columns = [str(c).strip() for c in df.columns]
        
        # 数値型の列を安全に変換
        num_cols = ['TeamID', 'PlayerID', 'Order', 'PlayerNo', 'Lineup_1', 'Lineup_2', 'Lineup_3', 'Lineup_4', 'Lineup_5', 'OFFApps', 'DEFApps', 'ShotPoints', 'ActionCD1']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # 小数点型の列を丸める
        for col in ['HensatiOFF', 'HensatiDEF', 'RatingOFF', 'RelativeShotX', 'RelativeShotY']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # --- 以下、既存の辞書作成や期間取得ロジック ---
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
    
def draw_shot_chart(player_shots, player_name):
    if player_shots.empty:
        return go.Figure()

    # --- 1. データのコピーとハニカムグリッド集計ロジック ---
    df = player_shots.copy()  # 元データを保護するためにコピー
    
    size = 0.8  # 六角形のサイズ（密度）の調整用
    
    # Y座標をインデックス化
    df['y_int'] = (df['RelativeShotY'] / (size * 1.5)).round().astype(int)
    
    # X座標の計算：Yのインデックスが「奇数」の場合に X を半分（dx/2）ずらす
    dx = size * np.sqrt(3)
    is_odd = (df['y_int'] % 2 != 0)
    
    df['x_bin'] = np.where(
        is_odd,
        (np.floor(df['RelativeShotX'] / dx) + 0.5) * dx,
        np.round(df['RelativeShotX'] / dx) * dx
    )
    df['y_bin'] = df['y_int'] * (size * 1.5)

    # エリアごとに集計
    bin_stats = df.groupby(['x_bin', 'y_bin']).agg(
        attempts=('ShotPoints', 'count'),
        made=('ShotPoints', lambda x: (x > 0).sum())
    ).reset_index()

    bin_stats['fg_pct'] = (bin_stats['made'] / bin_stats['attempts']) * 100
    
    # マーカーサイズ：試投数が多いほど大きく（最小12, 最大22）
    # √本数 を使うことで、本数が多い時の肥大化を抑えつつ差を出す
    bin_stats['msize'] = bin_stats['attempts'].apply(lambda x: min(np.sqrt(x) * 6 + 5, 25))

    # --- 2. 描画 ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bin_stats['x_bin'],
        y=bin_stats['y_bin'],
        mode='markers',
        marker=dict(
            size=bin_stats['msize'],
            color=bin_stats['fg_pct'],
            symbol='hexagon', 
            colorscale='RdBu_r', 
            showscale=True,
            colorbar=dict(title="FG%", ticksuffix="%"),
            line=dict(width=0.5, color='white'), 
            cmid=45 # 45%付近を白（中立）にする
        ),
        text=[f"試投: {int(a)}<br>成功: {int(m)}<br>確率: {p:.1f}%" 
              for a, m, p in zip(bin_stats['attempts'], bin_stats['made'], bin_stats['fg_pct'])],
        hoverinfo='text'
    ))

    # --- 3. コート描画（既存のロジック） ---
    line_color = "#333333"
    # ゴール付近
    fig.add_shape(type="line", x0=1.2, y0=-0.9, x1=1.2, y1=0.9, line=dict(color="black", width=3))
    fig.add_shape(type="circle", x0=1.575-0.225, y0=-0.225, x1=1.575+0.225, y1=0.225, line=dict(color="orange", width=2))
    # 制限区域
    fig.add_shape(type="rect", x0=0, y0=-2.45, x1=5.8, y1=2.45, line=dict(color=line_color, width=1.5), layer="below")
    # 3Pライン
    three_point_path = "M 0 -6.6 L 2.99 -6.6 A 6.75 6.75 0 0 1 2.99 6.6 L 0 6.6"
    fig.add_shape(type="path", path=three_point_path, line=dict(color=line_color, width=2.5), layer="below")
    # コート外枠
    fig.add_shape(type="rect", x0=0, y0=-7.5, x1=14, y1=7.5, line=dict(color=line_color, width=2), layer="below")

    # レイアウト設定
    fig.update_layout(
        title={
            'text': f"🔥 {player_name} ショット効率マップ",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        width=1200, 
        height=850, 
        xaxis=dict(
            range=[-0.5, 14.5], 
            visible=False, 
            fixedrange=True, # ズーム禁止
            scaleanchor="y", 
            scaleratio=1
        ),
        yaxis=dict(
            range=[-7.8, 7.8], 
            visible=False,
            fixedrange=True # ズーム禁止
        ),
        # 余白を最小化
        margin=dict(l=5, r=5, t=60, b=5), 
        plot_bgcolor='white',
        dragmode=False,
        hovermode='closest'
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

    # --- ショット分析セクション ---
    if not is_league_mode:
        st.divider()
        st.write(f"## 🏀 {sel_team_name} ショット分析")
        
        # 選手選択（チーム全体か個人か）
        team_players = df_all_p[df_all_p['TeamID'] == target_team_id].sort_values('PlayerNo')
        p_options = ["チーム全体"] + [f"{int(r['PlayerNo'])} {r['PlayerNameJ']}" for _, r in team_players.iterrows()]
        sel_p_shot = st.selectbox("表示対象を選択", p_options)

        # --- データの抽出ロジック (ScheduleKey対応版) ---
        # 試合識別列の候補に 'ScheduleKey' を追加
        possible_game_cols = ['ScheduleKey', 'ScheduleID', 'GameID', 'Game_ID']
        g_id = next((c for c in possible_game_cols if c in df_shot.columns), None)

        if g_id is None:
            st.error(f"試合識別列が見つかりません。列名を確認してください。")
            st.write("検出された列名:", list(df_shot.columns))
            s_all = pd.DataFrame()
        else:
            # 念のためID列を数値または文字列として統一
            df_shot[g_id] = df_shot[g_id].astype(str)
            
            if sel_p_shot == "チーム全体":
                # チームが関わっている全試合のKeyを特定
                relevant_games = df_shot[df_shot['TeamID'] == target_team_id][g_id].unique()
                # その試合の全ショット（自他含む）を抽出
                s_all = df_shot[df_shot[g_id].isin(relevant_games)].copy()
                chart_title = f"{sel_team_name} (チーム全体)"
            else:
                # 個人の場合
                p_name_only = sel_p_shot.split(" ", 1)[1]
                selected_player_id = int(team_players[team_players['PlayerNameJ'] == p_name_only]['PlayerID'].iloc[0])
                # その選手が出場した試合のKeyを特定
                player_games = df_shot[df_shot['PlayerID'] == selected_player_id][g_id].unique()
                s_all = df_shot[df_shot[g_id].isin(player_games)].copy()
                chart_title = p_name_only

        if not s_all.empty:
            # データのクリーニング
            s_all['ActionCD1'] = pd.to_numeric(s_all['ActionCD1'], errors='coerce').fillna(0).astype(int)
            s_all['TeamID'] = pd.to_numeric(s_all['TeamID'], errors='coerce').fillna(0).astype(int)
            current_team_id = int(target_team_id)

            def aggregate_stats(df_sub, label):
                # ActionCD1の定義に従った集計
                is_3p = df_sub['ActionCD1'].isin([1, 2])
                is_2p = df_sub['ActionCD1'].isin([3, 4, 5, 6])
                is_made = df_sub['ActionCD1'].isin([1, 3, 4])
                
                _3fgm, _3fga = int((is_3p & is_made).sum()), int(is_3p.sum())
                _2fgm, _2fga = int((is_2p & is_made).sum()), int(is_2p.sum())
                fgm, fga = _3fgm + _2fgm, _3fga + _2fga
                
                calc_pct = lambda m, a: (m / a * 100) if a > 0 else 0.0
                
                return {
                    "区分": label, "FGM": fgm, "FGA": fga, "FG%": calc_pct(fgm, fga),
                    "2FGM": _2fgm, "2FGA": _2fga, "2FG%": calc_pct(_2fgm, _2fga),
                    "3FGM": _3fgm, "3FGA": _3fga, "3FG%": calc_pct(_3fgm, _3fga)
                }

            # 集計実行
            df_own = s_all[s_all['TeamID'] == current_team_id]
            df_opp = s_all[s_all['TeamID'] != current_team_id]

            res_df = pd.DataFrame([
                aggregate_stats(df_own, "自チーム"), 
                aggregate_stats(df_opp, "相手チーム")
            ])

            st.write(f"### {chart_title} オンコート時シュート統計")
            st.dataframe(
                res_df.style.format({"FG%": "{:.1f}%", "2FG%": "{:.1f}%", "3FG%": "{:.1f}%"}), 
                use_container_width=True, 
                hide_index=True
            )

            # 自チームのショット位置をプロット
            st.plotly_chart(
                draw_shot_chart(df_own, chart_title), 
                use_container_width=True,
                config={'displayModeBar': False} # ツールバーを非表示
            )
        elif g_id is not None:
            st.warning("該当するショットデータが見つかりませんでした。")
            
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
