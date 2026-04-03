import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 1. ページ全体の基本設定
st.set_page_config(page_title="B-League Analytics Dash", layout="wide")
bg_color = "#FFFFFF"
sum_values = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
x_range = np.array([-30, 30]) # グラフの表示範囲


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

# --- 統計集計用の関数 (再定義) ---
def aggregate_stats(df_sub, label):
    # ActionCD1 の定義に基づいてシュート種別を判定 (Bリーグ等の一般的なデータ構造)
    # 3P成功:1, 3P失敗:2, 2P成功:3,4, 2P失敗:5,6 と仮定
    is_3p = df_sub['ActionCD1'].isin([1, 2])
    is_2p = df_sub['ActionCD1'].isin([3, 4, 5, 6])
    is_made = df_sub['ActionCD1'].isin([1, 3, 4])
    
    _3fgm, _3fga = int((is_3p & is_made).sum()), int(is_3p.sum())
    _2fgm, _2fga = int((is_2p & is_made).sum()), int(is_2p.sum())
    fgm, fga = _3fgm + _2fgm, _3fga + _2fga
    
    calc_pct = lambda m, a: (m / a * 100) if a > 0 else 0.0
    
    return {
        "区分": label,
        "FGM": fgm, "FGA": fga, "FG%": calc_pct(fgm, fga),
        "2FGM": _2fgm, "2FGA": _2fga, "2FG%": calc_pct(_2fgm, _2fga),
        "3FGM": _3fgm, "3FGA": _3fga, "3FG%": calc_pct(_3fgm, _3fga)
    }

def draw_shot_chart(player_shots, player_name):
    if player_shots.empty:
        return go.Figure()

    # --- 1. データのコピーとハニカムグリッド集計ロジック ---
    df = player_shots.copy()
    
    size = 0.8  
    df['y_int'] = (df['RelativeShotY'] / (size * 1.5)).round().astype(int)
    dx = size * np.sqrt(3)
    is_odd = (df['y_int'] % 2 != 0)
    
    df['x_bin'] = np.where(
        is_odd,
        (np.floor(df['RelativeShotX'] / dx) + 0.5) * dx,
        np.round(df['RelativeShotX'] / dx) * dx
    )
    df['y_bin'] = df['y_int'] * (size * 1.5)

    # 【修正箇所：期待値計算のために total_points を集計に追加】
    bin_stats = df.groupby(['x_bin', 'y_bin']).agg(
        attempts=('ShotPoints', 'count'),
        made=('ShotPoints', lambda x: (x > 0).sum()),
        total_points=('ShotPoints', 'sum') # 合計得点
    ).reset_index()

    # 【修正箇所：期待値(pps)と成功率(fg_pct)を計算】
    bin_stats['pps'] = bin_stats['total_points'] / bin_stats['attempts']
    bin_stats['fg_pct'] = (bin_stats['made'] / bin_stats['attempts']) * 100
    
    bin_stats['msize'] = bin_stats['attempts'].apply(lambda x: min(np.sqrt(x) * 6 + 2, 25))

    # --- 2. 描画 ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bin_stats['x_bin'],
        y=bin_stats['y_bin'],
        mode='markers',
        marker=dict(
            size=bin_stats['msize'],
            color=bin_stats['pps'],     # 【修正：色を期待値に変更】
            symbol='hexagon', 
            colorscale='Viridis', 
            showscale=True,
            # --- 範囲の設定を追加 ---
            cmin=0.0,           # 最小値（0点：すべて外れ）
            cmax=1.5,           # 最大値（1.5pt：3Pが50%で入る超高効率）
            # ----------------------
            # 【修正：カラーバーのタイトル・単位・長さを変更】
            colorbar=dict(
                title="期待値 (PPS)", 
                ticksuffix="pt",
                len=0.5,                # 長さを半分に
                lenmode='fraction',
                y=0.5,
                yanchor='middle',
                thickness=20
            ),
            line=dict(width=0.5, color='white'), 
            cmid=1.0 # 【修正：期待値1.0点を中間色（白）に設定】
        ),
        # 【修正：ホバーテキストに期待値を表示】
        text=[f"期待値: {p:.2f} pt<br>試投: {int(a)}<br>成功率: {pct:.1f}%" 
              for p, a, pct in zip(bin_stats['pps'], bin_stats['attempts'], bin_stats['fg_pct'])],
        hoverinfo='text'
    ))

    # --- 3. コート描画（三角関数による精密な円弧再現） ---
    line_color = "#333333"
    hoop_x = 1.575
    three_radius = 6.75
    side_dist_y = 6.6 

    # 円弧と直線の交点における角度を計算 (ラジアン)
    # sin(theta) = 6.6 / 6.75
    angle_at_intersect = np.arcsin(side_dist_y / three_radius)
    
    # 円弧上の点を生成 (20分割して滑らかに)
    angles = np.linspace(-angle_at_intersect, angle_at_intersect, 20)
    arc_points = []
    for a in angles:
        px = hoop_x + three_radius * np.cos(a)
        py = hoop_x + three_radius * np.sin(a) # ここは hoop_y(0) + ...
        # 正確には py = three_radius * np.sin(a)
        arc_points.append(f"L {px:.3f} {three_radius * np.sin(a):.3f}")

    # パスの組み立て
    # 1. 下側のコーナー直線
    # 2. 生成した円弧の点群
    # 3. 上側のコーナー直線
    path_segments = [f"M 0 {-side_dist_y}", f"L {hoop_x + three_radius * np.cos(-angle_at_intersect):.3f} {-side_dist_y}"]
    path_segments.extend(arc_points)
    path_segments.append(f"L 0 {side_dist_y}")
    
    three_point_full_path = " ".join(path_segments)

    fig.add_shape(
        type="path",
        path=three_point_full_path,
        line=dict(color=line_color, width=2.5),
        layer="below"
    )

    # --- 他のパーツはそのまま ---
    # ゴール付近
    fig.add_shape(type="line", x0=1.2, y0=-0.9, x1=1.2, y1=0.9, line=dict(color="black", width=3))
    fig.add_shape(type="circle", x0=hoop_x-0.225, y0=-0.225, x1=hoop_x+0.225, y1=0.225, line=dict(color="orange", width=2))
    # 制限区域
    fig.add_shape(type="rect", x0=0, y0=-2.45, x1=5.8, y1=2.45, line=dict(color=line_color, width=1.5), layer="below")
    # 外枠
    fig.add_shape(type="rect", x0=0, y0=-7.5, x1=14, y1=7.5, line=dict(color=line_color, width=2), layer="below")
    
    fig.update_layout(
        title={
            'text': f"🔥 {player_name} ショット効率マップ",
            'y': 0.98, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=24)
        },
        width=1200, height=640, 
        xaxis=dict(range=[-0.5, 14.5], visible=False, fixedrange=True, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-7.8, 7.8], visible=False, fixedrange=True),
        margin=dict(l=1, r=1, t=50, b=1), 
        plot_bgcolor='white',
        dragmode=False,
        hovermode='closest'
    )
    
    return fig
    
# 4. 関数呼び出し側でも df_shot として受け取る
df_team, df_player, df_lineup, df_shot, analysis_period = load_all_data()

# --- 4. メインタイトル ---
st.title(f"🏀 Bリーグ選手評価：{sel_team_name if 'sel_team_name' in locals() else ''}")
st.info(f"📅 分析対象期間：{analysis_period}")

# --- 3. メインエリアのフィルター (サイドバーから移動) ---
# フィルターを横並びにする
f_col1, f_col2 = st.columns(2)

with f_col1:
    list_league = list(dict.fromkeys(df_team['League']))
    sel_league = st.selectbox("リーグ選択", list_league)

with f_col2:
    teams_in_league = df_team[df_team['League'] == sel_league].copy()
    if 'Order' in teams_in_league.columns:
        teams_sorted = teams_in_league.sort_values(by='Order', ascending=True)
    else:
        teams_sorted = teams_in_league.sort_values(by='TeamID', ascending=True)

    list_teams = ["リーグ全体"] + teams_sorted['Team'].tolist()
    sel_team_name = st.selectbox("チーム選択", list_teams)

# 選択されたIDの確定
if sel_team_name == "リーグ全体":
    target_team_id = None
else:
    target_team_id = int(teams_sorted[teams_sorted['Team'] == sel_team_name]['TeamID'].iloc[0])

# --- ツールチップ・キャプション ---
with st.expander("💡 この分析ツールの使い方はこちら"):
    st.markdown(f"""
    * **リーグ・チーム選択**: 上のドロップダウンメニューから対象を絞り込めます(2026/4 今のところB1のみです．処理が重いので…すいません！)。
    * **詳細データの確認**: 各グラフのドットにマウスを合わせると、具体的な数値（ツールチップ）が表示されます。
        * ラインナップ分析では選択したチームのみツールチップが表示されます．
    * **ショット分析**:
        * チームおよび個人のショット統計が表示されます．通常の個人のフィールドゴールに加え，オンコートの攻撃と守備の集計も表示されます．
    * **ラインナップ分析**: 
        * 特定の選手を選択して強調表示（ハイライト）が可能です。
        * ラインナップのオンコートの攻守の集計も表示できます．
    * **画像の保存**: グラフ右上のカメラアイコンから、分析結果をPNG形式で保存できます。
    """)
    
st.caption(f"Developed by [@konakalab](https://x.com/konakalab) | 📅 データ更新：{analysis_period}")

# --- タブの配置 ---
tab1, tab2, tab3 = st.tabs(["選手分析", "ラインナップ分析", "評価方法の概要"])

# --- タブ1: 選手分析 ---
with tab1:
    df_all_p = df_player.copy()
    df_all_p['TotalApps'] = df_all_p['OFFApps'] + df_all_p['DEFApps']
    df_all_p['MarkerSize'] = np.sqrt(df_all_p['TotalApps'] + 1)
    is_league_mode = (target_team_id is None)

    # 1. 選手評価分布のデータ準備
    if is_league_mode:
        st.subheader(f"リーグ全体 選手評価分布 ({sel_league})")
        df_all_p['DisplayGroup'] = sel_league
        df_all_p['is_selected'] = True
        df_all_p['Label'] = ""
        color_map = {sel_league: '#636EFA'}
        opacity_val = 0.2
    else:
        st.subheader(f"選手別 評価値分布 ({sel_team_name})")
        df_all_p['is_selected'] = (df_all_p['TeamID'] == target_team_id)
        df_all_p['DisplayGroup'] = df_all_p['is_selected'].map({True: sel_team_name, False: 'その他'})
        df_all_p['Label'] = df_all_p.apply(lambda r: str(int(r['PlayerNo'])) if r['is_selected'] and r['PlayerNo'] != 0 else "", axis=1)
        color_map = {sel_team_name: '#EF553B', 'その他': '#E5ECF6'}
        df_all_p = df_all_p.sort_values('is_selected')
        # 選択チームを前面に出すための不透明度設定
        opacity_val = 0.4 

    # 2. 選手評価散布図の作成
    fig_p = px.scatter(
        df_all_p, x='HensatiOFF', y='HensatiDEF', color='DisplayGroup', size='MarkerSize', text='Label', hover_name='PlayerNameJ',
        hover_data={'HensatiOFF': ':.1f', 'HensatiDEF': ':.1f', 'TotalApps': True, 'DisplayGroup': False, 'MarkerSize': False, 'Label': False},
        color_discrete_map=color_map, labels={'HensatiOFF': '攻撃評価', 'HensatiDEF': '守備評価', 'TotalApps': '合計プレイ数'}
    )

    fig_p.update_layout(
        title={'text': f"<b>{sel_team_name}</b> 選手評価分布<br><span style='font-size:12px; color:gray;'>期間: {analysis_period}</span>", 'x': 0.5, 'y': 0.98, 'xanchor': 'center', 'yanchor': 'top'},
        margin=dict(l=20, r=20, t=100, b=100),
        xaxis=dict(range=[-30, 30], title="攻撃評価", gridcolor='lightgray', showspikes=True),
        yaxis=dict(range=[-30, 30], title="守備評価", gridcolor='lightgray', scaleanchor="x", scaleratio=1, showspikes=True),
        height=750, plot_bgcolor='white', hovermode='closest',
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
    )

    for k in sum_values:
        fig_p.add_trace(go.Scattergl(
            x=x_range,
            y=k - x_range, # y = -x + k
            mode='lines',
            line=dict(color='black', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip',
            opacity=0.3
        ))
        
    fig_p.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_p.add_vline(x=0, line_dash="dot", line_color="gray")
    
    # 散布図の表示
    st.plotly_chart(fig_p, use_container_width=True)

    # --- Tab 1 内: ショット分析セクション ---
    if not is_league_mode:
        st.divider()
        st.write(f"## 🏀 {sel_team_name} ショット分析")
        
        # --- 1. 選手選択 ---
        team_players = df_all_p[df_all_p['TeamID'] == target_team_id].sort_values('PlayerNo')
        p_options = ["チーム全体"] + [f"{int(r['PlayerNo'])} {r['PlayerNameJ']}" for _, r in team_players.iterrows()]
        sel_p_shot = st.selectbox("分析対象の選手を選択", p_options)

        # --- 2. データの切り分け ---
        if sel_p_shot != "チーム全体":
            # --- 【選手個人モード】 ---
            analysis_mode = st.radio(
                "図示内容",
                ["① 選手個人のショット", "② オンコート時の自チーム全体", "③ オンコート時の相手チーム"],
                horizontal=True
            )
            
            p_name_only = sel_p_shot.split(" ", 1)[1]
            # 💡 ここで変数を確実に定義
            selected_player_id = int(team_players[team_players['PlayerNameJ'] == p_name_only]['PlayerID'].iloc[0])
            
            # オンコート判定 (新カラム名に対応)
            all_lup_cols = [f'hLup{i}' for i in range(1, 6)] + [f'aLup{i}' for i in range(1, 6)]
            is_on_court = (df_shot[all_lup_cols] == selected_player_id).any(axis=1)
            df_on_court_all = df_shot[is_on_court].copy()

            # 3行スタッツの作成
            stats_list = []
            df_personal = df_shot[df_shot['PlayerID'] == selected_player_id]
            df_own_on = df_on_court_all[df_on_court_all['TeamID'] == target_team_id]
            df_opp_on = df_on_court_all[df_on_court_all['TeamID'] != target_team_id]
            
            stats_list.append(aggregate_stats(df_personal, "1. 選手個人"))
            stats_list.append(aggregate_stats(df_own_on, "2. 自チーム(オンコート)"))
            stats_list.append(aggregate_stats(df_opp_on, "3. 相手チーム(オンコート)"))
            
            st.write(f"### 📊 {p_name_only} オンコート統計まとめ")
            st.dataframe(pd.DataFrame(stats_list).style.format({"FG%": "{:.1f}%", "2FG%": "{:.1f}%", "3FG%": "{:.1f}%"}), use_container_width=True, hide_index=True)

            # チャート表示用データの決定
            if analysis_mode == "① 選手個人のショット":
                df_display = df_personal
                chart_title = f"{p_name_only} (個人)"
                target_cmid = 1.0
            elif analysis_mode == "② オンコート時の自チーム全体":
                df_display = df_own_on
                chart_title = f"{p_name_only} 出場時 (自チーム)"
                target_cmid = 1.0
            else:
                df_display = df_opp_on
                chart_title = f"{p_name_only} 出場時 (相手チーム)"
                target_cmid = 0.9

        else:
            # --- 【チーム全体モード】 ---
            df_display = df_shot[df_shot['TeamID'] == target_team_id].copy()
            chart_title = f"{sel_team_name} (チーム全体)"
            target_cmid = 1.0
            
            # --- 📊 チーム全体の攻撃と守備（被シュート）を並べて集計 ---
            team_stats_list = []
            
            # ① 自チームの攻撃
            team_stats_list.append(aggregate_stats(df_display, "自チーム（攻撃）"))
            
            # ② 相手チームの攻撃（＝自チームの被シュート）
            # 同じ試合(ScheduleKey)における、自チーム以外のショットを抽出
            opp_shots = df_shot[
                (df_shot['ScheduleKey'].isin(df_display['ScheduleKey'])) & 
                (df_shot['TeamID'] != target_team_id)
            ]
            team_stats_list.append(aggregate_stats(opp_shots, "相手チーム（守備）"))
            
            st.write(f"### 📊 {sel_team_name} チーム統計まとめ")
            st.dataframe(
                pd.DataFrame(team_stats_list).style.format({
                    "FG%": "{:.1f}%", 
                    "2FG%": "{:.1f}%", 
                    "3FG%": "{:.1f}%"
                }), 
                use_container_width=True, 
                hide_index=True
            )
            
        # --- 3. ショットチャートの描画 (共通処理) ---
        if not df_display.empty:
            fig_shot = draw_shot_chart(df_display, chart_title)
            
            # 💡 チャート内部にタイトルを追加し、期待値レンジを 0.0 〜 1.5 で固定
            fig_shot.update_layout(
                title={
                    'text': f"<b>{chart_title}</b> <span style='font-size:12px; color:gray;'>期間: {analysis_period}</span>",
                    'y': 0.95,        # グラフ内上部の位置
                    'x': 0.5,         # 中央
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=20)
                },
                plot_bgcolor=bg_color
            )

            fig_shot.update_traces(
                marker=dict(
                    cmin=0.0,    # 期待値最小固定
                    cmax=1.5,    # 期待値最大固定
                    cmid=target_cmid
                )
            )
            
            st.plotly_chart(
                fig_shot, 
                use_container_width=False, 
                config={
                    'displayModeBar': True,           # ツールバー自体は有効にする
                    'modeBarButtonsToRemove': [       # カメラ以外をすべて削除
                        'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 
                        'zoomOut2d', 'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 
                        'hoverCompareCartesian', 'toggleSpikelines'
                    ],
                    'displaylogo': False,             # Plotlyのロゴを隠す
                    'toImageButtonOptions': {         # 保存時のファイル設定
                        'format': 'png',
                        'filename': f'shot_chart_{sel_team_name}',
                        'height': 640,
                        'width': 1200,
                        'scale': 2                    # 2倍の解像度で保存（高画質）
                    }
                }
            )
        else:
            st.warning("表示できるショットデータがありません。")
            
    # 4. 選手データ一覧テーブル
    st.divider()
    st.write(f"### {sel_team_name} 選手データ一覧")
    output_p = df_all_p[df_all_p['is_selected']].copy()
    output_p['総合評価'] = (output_p['HensatiOFF'] + output_p['HensatiDEF']) / 2
    output_p['貢献量'] = (output_p['HensatiOFF'] + output_p['HensatiDEF']) * output_p['TotalApps']
    output_p['公式サイト'] = "https://www.bleague.jp/roster_detail/?PlayerID=" + output_p['PlayerID'].astype(str)
    
    if is_league_mode:
        team_dict = dict(zip(df_team['TeamID'], df_team['Team']))
        output_p['チーム'] = output_p['TeamID'].map(team_dict)
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
        plot_configs = [{"name": "その他", "color": "#E5ECF6", "opacity": 0.15}, {"name": sel_team_name, "color": "#EF553B", "opacity": 0.4}, {"name": "注目選手", "color": "#19D3F3", "opacity": 0.6}]

    fig_l = go.Figure()
    for cfg in plot_configs:
        # データのコピーと丸め処理（先ほどの修正を維持）
        sub = df_plot[df_plot['DisplayGroup'] == cfg["name"]].copy()
        if sub.empty: continue
        
        sub['HensatiOFF'] = sub['HensatiOFF'].astype(float).round(1)
        sub['HensatiDEF'] = sub['HensatiDEF'].astype(float).round(1)

        # --- 【修正箇所】注目データのみツールチップを表示する設定 ---
        if cfg["name"] == "その他":
            # 「その他」は反応させない
            current_hovertemplate = None
            current_hoverinfo = 'skip'
        else:
            # 注目選手や自チームは詳細を表示
            current_hovertemplate = (
                "<b>%{text}</b><br>" +
                "プレイ数: %{customdata}回<br>" +
                "攻: %{x:+.1f} / 守: %{y:+.1f}<extra></extra>"
            )
            current_hoverinfo = 'all'

        fig_l.add_trace(go.Scattergl(
            x=sub['HensatiOFF'], 
            y=sub['HensatiDEF'], 
            mode='markers', 
            name=cfg["name"], 
            text=sub['UnitNames'], 
            customdata=sub['TotalApps_L'],
            marker=dict(
                size=np.sqrt(sub['TotalApps_L'] + 1) * 1.5, 
                color=cfg["color"], 
                opacity=cfg["opacity"], 
                line=dict(width=0.5, color='white') if cfg["name"] != "その他" else None
            ),
            # 設定を適用
            hovertemplate=current_hovertemplate,
            hoverinfo=current_hoverinfo
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

    # --- 1. 全体平均の基準線（0,0の十字） ---
    fig_l.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1, opacity=0.5)
    fig_l.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1, opacity=0.5)
            
    for k in sum_values:
        fig_l.add_trace(go.Scattergl(
            x=x_range,
            y=k - x_range, # y = -x + k
            mode='lines',
            line=dict(color='black', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip',
            opacity=0.3
        ))
        
    # 既存の 0 基準線（十字）
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

    # --- 【修正版】ラインナップ別 統計セクション（リーグ全体対応） ---
    if not output_l.empty:
        st.divider()
        st.write(f"### 📊 {sel_team_name} 特定ラインナップのショット統計")
        
        # 1. プルダウン（プレイ数順）
        lup_sorted = output_l.sort_values('合計プレイ数', ascending=False)
        lup_options = lup_sorted['ラインナップ'].tolist()
        sel_lup_name = st.selectbox("詳細統計を表示するラインナップを選択", lup_options, key="lup_stats_select")
        
        # 2. 選択されたラインナップの情報を取得
        selected_lup_info = df_table[df_table['UnitNames'] == sel_lup_name].iloc[0]
        target_lup_ids = {int(float(pid)) for pid in selected_lup_info['LineupSet']}
        
        # 💡 リーグ全体モードの場合、そのラインナップが所属する本来の TeamID を取得しておく
        actual_team_id = int(selected_lup_info['TeamID'])
        
        # 3. ショットデータの抽出
        h_cols = [f'hLup{i}' for i in range(1, 6)]
        a_cols = [f'aLup{i}' for i in range(1, 6)]
        
        def is_exact_match(row, target_set):
            try:
                h_set = {int(float(row[c])) for c in h_cols if pd.notna(row[c])}
                if h_set == target_set: return True
                a_set = {int(float(row[c])) for c in a_cols if pd.notna(row[c])}
                if a_set == target_set: return True
            except:
                return False
            return False

        # 💡 判定対象：リーグ全体なら全データ、チーム選択中ならそのチーム関連に絞る
        if is_league_mode:
            df_search_base = df_shot.copy()
        else:
            df_search_base = df_shot[
                (df_shot['TeamID'] == target_team_id) | 
                (df_shot['ScheduleKey'].isin(df_shot[df_shot['TeamID'] == target_team_id]['ScheduleKey']))
            ].copy()

        # 判定実行
        is_lup_on_court = df_search_base.apply(lambda r: is_exact_match(r, target_lup_ids), axis=1)
        df_lup_all_shots = df_search_base[is_lup_on_court].copy()
        
        if not df_lup_all_shots.empty:
            # 4. 統計表の作成
            lup_stats_list = []
            
            # 💡 「自チーム」の判定に target_team_id ではなく actual_team_id を使用
            df_lup_own = df_lup_all_shots[df_lup_all_shots['TeamID'] == actual_team_id]
            lup_stats_list.append(aggregate_stats(df_lup_own, "攻撃"))
            
            # 相手チーム（守備時）
            df_lup_opp = df_lup_all_shots[df_lup_all_shots['TeamID'] != actual_team_id]
            lup_stats_list.append(aggregate_stats(df_lup_opp, "守備"))
            
            st.write(f"#### 選択中: {sel_lup_name}")
            st.dataframe(
                pd.DataFrame(lup_stats_list).style.format({
                    "FG%": "{:.1f}%", "2FG%": "{:.1f}%", "3FG%": "{:.1f}%"
                }),
                use_container_width=True, hide_index=True
            )
            # --- 5. ショットチャートの表示（ラジオボタンで切り替え） ---
            st.write(f"#### 🎯 {sel_lup_name} ショットチャート分析")
            
            lup_chart_mode = st.radio(
                "表示内容を選択", 
                ["① ラインナップの攻撃", "② ラインナップの守備"], 
                horizontal=True, 
                key="lup_chart_radio"
            )
            
            # モードに応じてデータを切り替え
            if lup_chart_mode == "① ラインナップの攻撃":
                df_lup_disp = df_lup_own
                lup_chart_title = f"{sel_lup_name} (攻撃)"
                lup_target_cmid = 1.0  
            else:
                df_lup_disp = df_lup_opp
                lup_chart_title = f"{sel_lup_name} (守備)"
                lup_target_cmid = 1.0  
                
            if not df_lup_disp.empty:
                # Tab 1 と同じ共通関数でチャート作成
                fig_lup_shot = draw_shot_chart(df_lup_disp, lup_chart_title)
                
                # ラインナップ分析用にレイアウトを微調整
                fig_lup_shot.update_layout(
                    title={
                        'text': f"<b>{lup_chart_title}</b><br><span style='font-size:12px; color:gray;'>期間: {analysis_period}</span>",
                        'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 
                        'font': dict(size=20)
                    },
                    margin=dict(t=80) # タイトル用の余白
                )

                # 期待値レンジを 0.0 〜 1.5 で固定（Tab 1 と統一）
                fig_lup_shot.update_traces(
                    marker=dict(
                        cmin=0.0, 
                        cmax=1.5, 
                        cmid=lup_target_cmid
                    )
                )
                    
                st.plotly_chart(
                    fig_lup_shot, 
                    use_container_width=False, 
                    config={
                        'displayModeBar': True,           # ツールバー自体は有効にする
                        'modeBarButtonsToRemove': [       # カメラ以外をすべて削除
                            'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 
                            'zoomOut2d', 'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 
                            'hoverCompareCartesian', 'toggleSpikelines'
                        ],
                        'displaylogo': False,             # Plotlyのロゴを隠す
                        'toImageButtonOptions': {         # 保存時のファイル設定
                            'format': 'png',
                            'filename': f'shot_chart_{sel_team_name}',
                            'height': 640,
                            'width': 1200,
                            'scale': 2                    # 2倍の解像度で保存（高画質）
                        }
                    }
                )
            else:
                st.info("表示できるショットデータがありません。")
        else:
            st.warning(f"このラインナップ（ID: {target_lup_ids}）の出場シーンが見つかりませんでした。")
            
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
