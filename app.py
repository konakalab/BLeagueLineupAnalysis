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
    
    # 前処理（数値変換など）
    for df in [df_t, df_p, df_l]:
        df.columns = [str(c).strip() for c in df.columns]
        num_cols = ['TeamID', 'PlayerID', 'Order', 'PlayerNo', 'Lineup_1', 'Lineup_2', 'Lineup_3', 'Lineup_4', 'Lineup_5', 'OFFApps', 'DEFApps']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        for col in ['HensatiOFF', 'HensatiDEF', 'RatingOFF']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).round(1)

    # --- IDから「名前」を引く辞書と、「背番号」を引く辞書を作成 ---
    p_dict = dict(zip(df_p['PlayerID'], df_p['PlayerNameJ']))
    p_no_dict = dict(zip(df_p['PlayerID'], df_p['PlayerNo']))

    # --- ラインナップ内の選手名を背番号順に並び替えて結合する関数 ---
    def get_sorted_unit_names(row):
        p_ids = []
        # ここで 1番目から5番目の PlayerID をリストに格納
        for i in range(1, 6):
            val = row[f'Lineup_{i}']
            if pd.notna(val) and int(val) != 0:
                p_ids.append(int(val))
        
        # (背番号, 名前) のタプルリストを作成
        p_info = []
        for pid in p_ids:
            no = p_no_dict.get(pid, 999) # 背番号がない場合は末尾へ
            name = p_dict.get(pid, "??")
            p_info.append((no, name))
        
        # 背番号(x[0])で昇順ソート
        p_info.sort(key=lambda x: x[0])
        
        # 名前だけを結合（ここが閉じ括弧不足になりやすい箇所です）
        return " / ".join([x[1] for x in p_info])

    # UnitNames の作成に適用
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
        
    return df_t, df_p, df_l, period_str

df_team, df_player, df_lineup, analysis_period = load_all_data()

# 3. サイドバーのフィルター
st.sidebar.header("検索フィルター")

# リーグ選択 (CSVの登場順を維持)
list_league = list(dict.fromkeys(df_team['League']))
sel_league = st.sidebar.selectbox("リーグ選択", list_league)

# --- チーム名の表示順を 'Order' カラムで厳密に制御 ---
teams_in_league = df_team[df_team['League'] == sel_league].copy()

if 'Order' in teams_in_league.columns:
    teams_sorted = teams_in_league.sort_values(by='Order', ascending=True)
else:
    teams_sorted = teams_in_league.sort_values(by='TeamID', ascending=True)

# 選択肢に「リーグ全体」を先頭に追加
list_teams = ["リーグ全体"] + teams_sorted['Team'].tolist()
sel_team_name = st.sidebar.selectbox("チーム選択", list_teams)

# 選択されたチームのIDを取得（リーグ全体の場合は None）
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
    # --- データの準備 ---
    df_all_p = df_player.copy()
    df_all_p['TotalApps'] = df_all_p['OFFApps'] + df_all_p['DEFApps']
    df_all_p['MarkerSize'] = np.sqrt(df_all_p['TotalApps'] + 1)

    # --- モード判定（リーグ全体 or 特定チーム） ---
    is_league_mode = (target_team_id is None)

    if is_league_mode:
        st.subheader(f"リーグ全体 選手評価分布 ({sel_league})")
        df_all_p['DisplayGroup'] = sel_league
        df_all_p['is_selected'] = True
        df_all_p['Label'] = "" # リーグ全体ではラベル非表示（密集回避）
        color_map = {sel_league: '#636EFA'}
        opacity_val = 0.3
    else:
        st.subheader(f"選手別 評価値分布 ({sel_team_name})")
        df_all_p['is_selected'] = (df_all_p['TeamID'] == target_team_id)
        df_all_p['DisplayGroup'] = df_all_p['is_selected'].map({True: sel_team_name, False: 'その他'})
        # 自チームの背番号のみ表示
        df_all_p['Label'] = df_all_p.apply(
            lambda r: str(int(r['PlayerNo'])) if r['is_selected'] and r['PlayerNo'] != 0 else "", axis=1
        )
        color_map = {sel_team_name: '#EF553B', 'その他': '#E5ECF6'}
        # 描画順を制御（自チームを最前面へ）
        df_all_p = df_all_p.sort_values('is_selected')
        opacity_val = df_all_p['is_selected'].map({True: 0.15, False: 0.3})

    # --- グラフ描画 ---
    fig_p = px.scatter(
        df_all_p, x='HensatiOFF', y='HensatiDEF', 
        color='DisplayGroup', 
        size='MarkerSize',
        text='Label', 
        hover_name='PlayerNameJ',
        hover_data={
            'HensatiOFF': ':.1f', 'HensatiDEF': ':.1f', 
            'TotalApps': True, 'DisplayGroup': False, 
            'MarkerSize': False, 'Label': False
        },
        color_discrete_map=color_map,
        labels={'HensatiOFF': '攻撃評価', 'HensatiDEF': '守備評価', 'TotalApps': '合計プレイ数'},
        opacity=opacity_val if is_league_mode else None # モードに応じて不透明度指定を切り替え
    )

    # 既存の凡例・スパイクライン等のレイアウト設定を適用
    fig_p.update_layout(
        title={
            'text': f"<b>{sel_team_name}</b> 選手評価分布<br><span style='font-size:12px; color:gray;'>点サイズ: 合計プレイ数 / ラベル: 背番号</span>",
            'x': 0.5,
            'y': 0.95,          # 【追加】タイトルの垂直位置（0〜1の範囲。0.95は上端に近い位置）
            'xanchor': 'center',
            'yanchor': 'top'    # 【追加】タイトルの上端を基準にする
        },
        # --- マージンの調整 ---
        margin=dict(l=20, r=20, t=100, b=100), # 【修正】t（上部）を 20 から 100 に増やしました
        xaxis=dict(
            range=[-30, 30], title="攻撃評価", gridcolor='lightgray',
            showspikes=True, spikecolor="gray", spikethickness=1, spikedash="dot", spikemode="across"
        ),
        yaxis=dict(
            range=[-30, 30], title="守備評価", gridcolor='lightgray', scaleanchor="x", scaleratio=1,
            showspikes=True, spikecolor="gray", spikethickness=1, spikedash="dot", spikemode="across"
        ),
        height=750,
        margin=dict(l=20, r=20, t=20, b=100),
        plot_bgcolor='white', hovermode='closest',
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
    )
    
    if not is_league_mode:
        fig_p.update_traces(textposition='top center', selector=dict(name=sel_team_name))

    fig_p.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_p.add_vline(x=0, line_dash="dot", line_color="gray")
    
    st.plotly_chart(fig_p, use_container_width=True)

    # --- テーブル表示 ---
    st.write(f"### {sel_team_name} 選手データ一覧")
    
    if is_league_mode:
        # 1. 必要な列を内部変数名のまま抽出
        output_p = df_all_p[['TeamID', 'PlayerNo', 'PlayerNameJ', 'TotalApps', 'HensatiOFF', 'HensatiDEF']].copy()
        
        # チーム名をマッピング
        team_dict = dict(zip(df_team['TeamID'], df_team['Team']))
        output_p['チーム'] = output_p['TeamID'].map(team_dict)
        
        # 2. 貢献量の計算
        output_p['貢献量'] = (output_p['HensatiOFF'] + output_p['HensatiDEF']) * output_p['TotalApps']
        
        # 3. 列の並べ替え（この時点ではまだ元の列名を使用する）
        output_p = output_p[['チーム', 'PlayerNo', 'PlayerNameJ', 'TotalApps', '貢献量', 'HensatiOFF', 'HensatiDEF']]
        
        # 4. 最後に表示用の日本語名に一括置換
        output_p.columns = ['チーム', '背番号', '選手名', '合計プレイ数', '貢献量', '攻撃評価', '守備評価']
        
        # --- ここを修正： '貢献量' から '合計プレイ数' へ ---
        output_p = output_p.sort_values('合計プレイ数', ascending=False)
        
        st.caption("※ リーグ全体の全選手を「合計プレイ数」順に表示しています。貢献量 = (攻撃評価 + 守備評価) × プレイ数")
    else:
        # 特定チームモード
        df_tp = df_all_p[df_all_p['is_selected']].copy()
        
        # 貢献量の計算
        df_tp['貢献量'] = (df_tp['HensatiOFF'] + df_tp['HensatiDEF']) * df_tp['TotalApps']
        
        # 列の選択
        output_p = df_tp[['PlayerNo', 'PlayerNameJ', 'TotalApps', '貢献量', 'HensatiOFF', 'HensatiDEF']]
        
        # 日本語名に置換
        output_p.columns = ['背番号', '選手名', '合計プレイ数', '貢献量', '攻撃評価', '守備評価']
        
        # 貢献量でソート
        output_p = output_p.sort_values('合計プレイ数', ascending=False)
        
    # 共通のデータフレーム表示
    st.dataframe(
        output_p.style.format({
            '背番号': '{:d}',
            '合計プレイ数': '{:d}',
            '貢献量': '{:,.0f}', 
            '攻撃評価': '{:.1f}', 
            '守備評価': '{:.1f}'
        }), 
        use_container_width=True, 
        hide_index=True
    )
    


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
        
        # 判定ロジック
        if cfg["name"] == "その他":
            hover_setting = "skip"
        elif target_p_id is not None:
            hover_setting = "all" if cfg["name"] == "注目選手" else "skip"
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
            # ホバーラベルの外観調整（重なりで見づらくなるのを防ぐ）
            hoverlabel=dict(
                bgcolor="rgba(255, 255, 255, 0.9)", 
                bordercolor="gray",
                font_size=12,
                namelength=0
            ),
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "合計プレイ数: %{customdata}回，" + 
                "攻撃評価: %{x}，" +
                "守備評価: %{y}<extra></extra>"
            ) if hover_setting == "all" else None
        ))

    # --- 3. 選択選手の個人評価値を基準線として追加 ---
        if target_p_id is not None:
            # df_player から選択された選手の評価値を取得
            p_stats = df_player[df_player['PlayerID'] == target_p_id]
            if not p_stats.empty:
                p_off = p_stats['HensatiOFF'].iloc[0]
                p_def = p_stats['HensatiDEF'].iloc[0]
                
                # 攻撃評価の垂直線
                fig_l.add_vline(
                    x=p_off, 
                    line_width=1.5, 
                    line_dash="dash", 
                    line_color="#19D3F3",
                    annotation_text=f" 選手平均(攻): {p_off}",
                    annotation_position="top left"
                )
                # 守備評価の水平線
                fig_l.add_hline(
                    y=p_def, 
                    line_width=1.5, 
                    line_dash="dash", 
                    line_color="#19D3F3",
                    annotation_text=f" 選手平均(守): {p_def}",
                    annotation_position="bottom right"
                )
                
    # --- グラフ全体のレイアウト設定 ---
    fig_l.update_layout(
        hovermode='closest',
        xaxis=dict(range=[-30, 30], title="攻撃評価", gridcolor='lightgray'),
        yaxis=dict(range=[-30, 30], title="守備評価", gridcolor='lightgray', scaleanchor="x", scaleratio=1),
        height=700, 
        margin=dict(l=20, r=20, t=20, b=100), # 下側の余白（b）を少し広げます
        plot_bgcolor='white',

        # --- 【追加】凡例の設定（グラフ下部・中央） ---
        legend=dict(
            orientation="h",   # 水平（Horizontal）に並べる
            yanchor="top",     # 凡例の「上端」を基準点にする
            y=-0.15,           # グラフエリアからの垂直距離（マイナスで下側）
            xanchor="center",  # 凡例の「中心」を基準点にする
            x=0.5              # グラフの横幅の中央（0.5）に配置
        )
    )
    
    # エラーの原因となった行 (fig_l.update_traces(hoveron='points')) は削除しました
    
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

# --- タブ3: 算出方法 (新規追加) ---
with tab3:
    st.header("評価値の算出方法について")
    
    st.markdown("""
    本分析サイトで使用している指標の定義と算出方法は以下の通りです。

    ### 1. 評価値の定義
    グラフの軸となっている **「攻撃評価」「守備評価」** は、リーグ全体の平均を **0**，標準偏差を **10**として算出しています．
    * **選手評価**: 後述する「ラインナップ評価」で，その選手を含むランナップのプレイ数重み付平均です．

    ### 2. ラインナップデータの集計
    * 同時にコートに立っている5人の組み合わせを1つの「ラインナップ」として集計しています。
    * **合計プレイ数**: その5人の組み合わせが合計で何回起用されたか（攻撃/守備の合計）を示します。回数は通常利用されるポゼッション **ではなく**，得点，ボール保持の変更で1回としています．
    * 1回のプレイで，攻撃側は得点すること，守備側は失点しないことが **小さな勝利**であるとみなし，その勝率をEloレーティングに似た評価式で評価しました．攻撃側の勝利には得点の重みをつけています．攻守ともにリーグ平均を **0**，標準偏差を **10** と変換しています．
    * プレイ数が少ないラインナップは、一時的な結果により数値が極端に高く（または低く）出ることがあるため、バブルの大きさ（プレイ数）と合わせて判断することをお勧めします。

    ### 3. データ期間
    * 冒頭に記載の「分析対象期間」に基づき、各試合のボックススコアおよびプレイバイプレイデータから抽出・計算されています。
    """)
    
    st.info("※ 本データは公式統計を元にkonakalabが独自に算出したものであり、B.LEAGUE公式の指標ではありません")
