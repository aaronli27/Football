import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
from model import PoissonMatchPredictor
from ucl_and_daily import CrossLeagueUCLPredictor, DB_PATH

st.set_page_config(page_title="HKJC 足球量化戰情室", layout="wide")
st.title("⚽ HKJC 個人足球量化與 +EV 全玩法戰情室")

def remove_vig_two_way(odd1: float, odd2: float):
    inv1, inv2 = 1.0 / odd1, 1.0 / odd2
    total = inv1 + inv2
    return inv1 / total, inv2 / total

def remove_vig_three_way(o1: float, o2: float, o3: float):
    inv1, inv2, inv3 = 1.0 / o1, 1.0 / o2, 1.0 / o3
    total = inv1 + inv2 + inv3
    return inv1 / total, inv2 / total, inv3 / total

def format_kelly_advice(ev: float, odd: float, bankroll: float, strict_range: bool = True) -> str:
    if ev <= 0.015:
        return "跳過 (無正期望值)"
    if ev > 0.15:
        return "⚠ 分歧過大 (先查傷停/陣容)"
    if strict_range and not (1.50 <= odd <= 3.30):
        return "⚠ 賠率超出風控區間"
    q_k = min(0.03, (ev / (odd - 1.0)) * 0.25)
    return f"★ 推薦買入 ${bankroll * q_k:.0f} ({q_k*100:.1f}%)"

tab1, tab2, tab3 = st.tabs(["📅 未來賽程自動掃描", "🧮 HKJC 全玩法 +EV 計算器", "🏆 歐聯跨聯賽模擬器"])

# --- 分頁 1：自動掃描未來賽程 ---
with tab1:
    st.subheader("即將開賽賽程與模型初盤基準")
    days = st.slider("掃描未來天數", 1, 7, 3)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM matches WHERE status = 'scheduled'", conn)
    conn.close()

    if df.empty:
        st.warning("資料庫目前沒有待開賽賽程，請先執行 `python src/scraper.py`。")
    else:
        df["dt"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = today + timedelta(days=days)
        upcoming = df[(df["dt"] >= today) & (df["dt"] <= end_date)].sort_values(["dt", "league"])

        if upcoming.empty:
            st.info("所選天數內暫無賽程，以下顯示最近期的 10 場待開賽賽事：")
            upcoming = df[df["dt"] >= today].sort_values(["dt", "league"]).head(10)

        ucl_model = CrossLeagueUCLPredictor()
        rows = []
        for _, r in upcoming.iterrows():
            lg, h, a, d_str = r["league"], r["home_team"], r["away_team"], r["date"]
            try:
                if lg == "UEFA-Champions-League":
                    res = ucl_model.predict_ucl_match(h, a)
                    cor_str = "-"
                else:
                    res = ucl_model.league_models[lg].predict_match(h, a)
                    cor_str = f"{res['exp_total_corners']} ({res['exp_home_corners']}-{res['exp_away_corners']})"
                rows.append({
                    "日期": d_str,
                    "聯賽/盃賽": lg,
                    "主隊": h,
                    "客隊": a,
                    "預期入球": f"{res['lambda_home']} - {res['lambda_away']}",
                    "預期角球": cor_str,
                    "主勝 %": f"{res['prob_home']*100:.1f}%",
                    "和局 %": f"{res['prob_draw']*100:.1f}%",
                    "客勝 %": f"{res['prob_away']*100:.1f}%",
                    "大 2.5 %": f"{res['prob_over_25']*100:.1f}%",
                    "首選波膽": res["top_scores"][0][0]
                })
            except Exception:
                continue

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- 分頁 2：HKJC 全玩法互動式 +EV 計算器 ---
with tab2:
    st.subheader("五大聯賽對陣與 HKJC 全玩法賠率評估")
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1.5])
    with col1:
        league = st.selectbox("選擇聯賽", ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"])
        model = PoissonMatchPredictor(league=league, goal_weight=0.4, half_life_days=45.0)
        teams = model.list_teams()
    with col2:
        home_team = st.selectbox("主隊", teams, index=0)
    with col3:
        away_team = st.selectbox("客隊", teams, index=1 if len(teams) > 1 else 0)
    with col4:
        bankroll = st.number_input("總本金 (HKD)", min_value=100, value=10000, step=500)

    pred = model.predict_match(home_team, away_team)
    matrix = pred["matrix"]

    # 頂部核心量化指標卡片
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("預期入球 (主 vs 客)", f"{pred['lambda_home']} vs {pred['lambda_away']}")
    m2.metric("預期總角球 (主 vs 客)", f"{pred['exp_total_corners']} 球 ({pred['exp_home_corners']} vs {pred['exp_away_corners']})")
    m3.metric("最高機率波膽", f"{pred['top_scores'][0][0]} ({pred['top_scores'][0][1]*100:.1f}%)")
    m4.metric("次高機率波膽", f"{pred['top_scores'][1][0]} ({pred['top_scores'][1][1]*100:.1f}%)")

    st.divider()

    # 區塊 A：主客和 (HAD) 與 讓球主客和 (HHAD)
    st.markdown("### 1. 主客和 (HAD) & 讓球主客和 (HHAD)")
    c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1, 1, 1.2, 1, 1, 1])
    h_odd = c1.number_input("主勝 (1)", min_value=1.01, value=2.10, step=0.05)
    d_odd = c2.number_input("和局 (X)", min_value=1.01, value=3.40, step=0.05)
    a_odd = c3.number_input("客勝 (2)", min_value=1.01, value=3.20, step=0.05)
    hhad_line = c4.selectbox("讓球主客和盤口", [-2, -1, 1, 2], index=1, format_func=lambda x: f"主隊 [{x:+d}]")
    hh_odd = c5.number_input("讓球主勝", min_value=1.01, value=3.80, step=0.05)
    hd_odd = c6.number_input("讓球和", min_value=1.01, value=3.60, step=0.05)
    ha_odd = c7.number_input("讓球客勝", min_value=1.01, value=1.68, step=0.05)

    # 區塊 B：亞洲讓球盤 (HDC)
    st.markdown("### 2. 亞洲讓球盤 (HDC - 支援平手/半球、半/一雙盤口)")
    ah1, ah2, ah3 = st.columns(3)
    ah_options = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    asian_line = ah1.selectbox("主隊讓球盤口 (負數=主讓, 正數=主受讓)", ah_options, index=ah_options.index(-0.25))
    h_ah_odd = ah2.number_input("讓球 - 主隊賠率", min_value=1.01, value=1.88, step=0.02)
    a_ah_odd = ah3.number_input("讓球 - 客隊賠率", min_value=1.01, value=1.96, step=0.02)

    # 區塊 C：入球大細 (支援 1.5 / 2.5 / 3.5 / 4.5) 與 角球大細 (支援 8.5 ~ 12.5)
    st.markdown("### 3. 入球大細 (HiLo) & 角球大細 (Corners HiLo)")
    g1, g2, g3, cor1, cor2, cor3 = st.columns(6)
    goal_line = g1.selectbox("入球大細球線", [1.5, 2.5, 3.5, 4.5], index=1)
    over_g_odd = g2.number_input(f"入球大 ({goal_line})", min_value=1.01, value=1.85, step=0.05)
    under_g_odd = g3.number_input(f"入球細 ({goal_line})", min_value=1.01, value=1.90, step=0.05)

    corner_line = cor1.selectbox("角球大細盤口", [7.5, 8.5, 9.5, 10.5, 11.5, 12.5], index=2)
    over_c_odd = cor2.number_input(f"角球大 ({corner_line})", min_value=1.01, value=1.85, step=0.05)
    under_c_odd = cor3.number_input(f"角球細 ({corner_line})", min_value=1.01, value=1.85, step=0.05)

    # --- 彙整所有玩法計算結果 ---
    results = []

    # 1. 主客和 (70% 市場去水 + 30% 模型校準)
    mkt_h, mkt_d, mkt_a = remove_vig_three_way(h_odd, d_odd, a_odd)
    for label, p_mod, p_mkt, odd in [
        ("主客和 - 主勝", pred["prob_home"], mkt_h, h_odd),
        ("主客和 - 和局", pred["prob_draw"], mkt_d, d_odd),
        ("主客和 - 客勝", pred["prob_away"], mkt_a, a_odd),
    ]:
        cal_p = 0.70 * p_mkt + 0.30 * p_mod
        ev = cal_p * odd - 1.0
        results.append({"玩法類別": "主客和 (HAD)", "選項": label, "模型/校準勝率": f"{cal_p*100:.1f}%", "HKJC 賠率": f"{odd:.2f}", "期望值 (EV)": f"{ev*100:+.1f}%", "風控建議": format_kelly_advice(ev, odd, bankroll)})

    # 2. 讓球主客和 (HHAD)
    hhad_probs = PoissonMatchPredictor.evaluate_hhad_probs(matrix, hhad_line)
    mkt_hh, mkt_hd, mkt_ha = remove_vig_three_way(hh_odd, hd_odd, ha_odd)
    for label, key, p_mkt, odd in [
        (f"讓球[{hhad_line:+d}] 主勝", "讓球主勝", mkt_hh, hh_odd),
        (f"讓球[{hhad_line:+d}] 和", "讓球和", mkt_hd, hd_odd),
        (f"讓球[{hhad_line:+d}] 客勝", "讓球客勝", mkt_ha, ha_odd),
    ]:
        cal_p = 0.70 * p_mkt + 0.30 * hhad_probs[key]
        ev = cal_p * odd - 1.0
        results.append({"玩法類別": "讓球主客和 (HHAD)", "選項": label, "模型/校準勝率": f"{cal_p*100:.1f}%", "HKJC 賠率": f"{odd:.2f}", "期望值 (EV)": f"{ev*100:+.1f}%", "風控建議": format_kelly_advice(ev, odd, bankroll)})

    # 3. 亞洲讓球盤 (HDC)
    h_ah_res = PoissonMatchPredictor.evaluate_asian_handicap_ev(matrix, asian_line, h_ah_odd, "home")
    a_ah_res = PoissonMatchPredictor.evaluate_asian_handicap_ev(matrix, asian_line, a_ah_odd, "away")
    for label, res_obj, odd in [
        (f"亞洲讓球 主 ({asian_line:+g})", h_ah_res, h_ah_odd),
        (f"亞洲讓球 客 ({-asian_line:+g})", a_ah_res, a_ah_odd),
    ]:
        ev = res_obj["ev"]
        win_desc = f"全贏 {res_obj['p_win_full']*100:.1f}%" + (f" / 贏半 {res_obj['p_win_half']*100:.1f}%" if res_obj["p_win_half"] > 0 else "")
        results.append({"玩法類別": "亞洲讓球 (HDC)", "選項": label, "模型/校準勝率": win_desc, "HKJC 賠率": f"{odd:.2f}", "期望值 (EV)": f"{ev*100:+.1f}%", "風控建議": format_kelly_advice(ev, odd, bankroll, strict_range=False)})

    # 4. 入球大細 (支援 1.5 / 2.5 / 3.5 / 4.5)
    p_over_g_mod = PoissonMatchPredictor.calc_total_over_prob(matrix, goal_line)
    mkt_og, _ = remove_vig_two_way(over_g_odd, under_g_odd)
    cal_og = 0.65 * mkt_og + 0.35 * p_over_g_mod
    for label, p_val, odd in [
        (f"入球大 ({goal_line})", cal_og, over_g_odd),
        (f"入球細 ({goal_line})", 1.0 - cal_og, under_g_odd),
    ]:
        ev = p_val * odd - 1.0
        results.append({"玩法類別": "入球大細 (HiLo)", "選項": label, "模型/校準勝率": f"{p_val*100:.1f}%", "HKJC 賠率": f"{odd:.2f}", "期望值 (EV)": f"{ev*100:+.1f}%", "風控建議": format_kelly_advice(ev, odd, bankroll, strict_range=False)})

    # 5. 角球大細 (Corners HiLo)
    p_over_c_mod = PoissonMatchPredictor.calc_corner_over_prob(pred["exp_total_corners"], corner_line)
    mkt_oc, _ = remove_vig_two_way(over_c_odd, under_c_odd)
    cal_oc = 0.65 * mkt_oc + 0.35 * p_over_c_mod
    for label, p_val, odd in [
        (f"角球大 ({corner_line})", cal_oc, over_c_odd),
        (f"角球細 ({corner_line})", 1.0 - cal_oc, under_c_odd),
    ]:
        ev = p_val * odd - 1.0
        results.append({"玩法類別": "角球大細 (Corners)", "選項": label, "模型/校準勝率": f"{p_val*100:.1f}%", "HKJC 賠率": f"{odd:.2f}", "期望值 (EV)": f"{ev*100:+.1f}%", "風控建議": format_kelly_advice(ev, odd, bankroll, strict_range=False)})

    st.divider()
    st.markdown("### 📊 全玩法期望值 (EV) 與凱利注碼評估總表")
    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

    # 額外展開區：一次查看所有球線 (1.5 / 2.5 / 3.5 / 4.5) 與所有角球線 (8.5 ~ 12.5) 的純模型勝率參考
    with st.expander("🔍 點擊展開：全盤口參考基準表（1.5 ~ 4.5 球線 & 8.5 ~ 12.5 角球線公平賠率）"):
        ref_col1, ref_col2 = st.columns(2)
        with ref_col1:
            g_lines = [1.5, 2.5, 3.5, 4.5]
            g_ref = []
            for gl in g_lines:
                po = PoissonMatchPredictor.calc_total_over_prob(matrix, gl)
                pu = 1.0 - po
                g_ref.append({
                    "入球盤口": f"{gl} 球",
                    "大球勝率": f"{po*100:.1f}%",
                    "大球公平賠率": f"{(1/po):.2f}" if po > 0.01 else "-",
                    "細球勝率": f"{pu*100:.1f}%",
                    "細球公平賠率": f"{(1/pu):.2f}" if pu > 0.01 else "-",
                })
            st.table(pd.DataFrame(g_ref))
        with ref_col2:
            c_lines = [8.5, 9.5, 10.5, 11.5, 12.5]
            c_ref = []
            for cl in c_lines:
                po = PoissonMatchPredictor.calc_corner_over_prob(pred["exp_total_corners"], cl)
                pu = 1.0 - po
                c_ref.append({
                    "角球盤口": f"{cl} 球",
                    "大角球勝率": f"{po*100:.1f}%",
                    "大角公平賠率": f"{(1/po):.2f}" if po > 0.01 else "-",
                    "細角球勝率": f"{pu*100:.1f}%",
                    "細角公平賠率": f"{(1/pu):.2f}" if pu > 0.01 else "-",
                })
            st.table(pd.DataFrame(c_ref))
    with st.expander("📐 點擊展開：6×6 波膽機率矩陣 (0-0 至 5-5) & 全聯賽戰力評級 (α / β)"):
        mat_col, rank_col = st.columns([1.2, 1])
        with mat_col:
            st.markdown(f"**{home_team} vs {away_team} 波膽百分比矩陣 (%)**")
            mat_df = pd.DataFrame(
                (matrix * 100).round(1),
                index=[f"主 {i} 球" for i in range(6)],
                columns=[f"客 {j} 球" for j in range(6)],
            )
            st.dataframe(mat_df, use_container_width=True)
            st.caption(f"理論公平賠率 (Fair Odds)：主勝 **{pred['fair_odds']['主勝 (1)']}** ｜ 和局 **{pred['fair_odds']['和局 (X)']}** ｜ 客勝 **{pred['fair_odds']['客勝 (2)']}** ｜ 大2.5 **{pred['fair_odds']['入球大 (2.5)']}**")
        with rank_col:
            st.markdown(f"**{league} 球隊實力排行榜 (MLE 求解)**")
            st.dataframe(model.get_team_ratings(), use_container_width=True, hide_index=True, height=280)
 
# --- 分頁 3：歐聯跨聯賽模擬器 ---
with tab3:
    st.subheader("UEFA Champions League 跨聯賽強弱對決模擬")
    ucl = CrossLeagueUCLPredictor()
    all_teams = sorted(ucl.team_to_league.keys())
    uc1, uc2 = st.columns(2)
    u_home = uc1.selectbox("歐聯主隊", all_teams, index=all_teams.index("Real Madrid") if "Real Madrid" in all_teams else 0)
    u_away = uc2.selectbox("歐聯客隊", all_teams, index=all_teams.index("Liverpool") if "Liverpool" in all_teams else 1)

    u_pred = ucl.predict_ucl_match(u_home, u_away)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("預期入球 (主-客)", f"{u_pred['lambda_home']} - {u_pred['lambda_away']}")
    m2.metric("主勝機率", f"{u_pred['prob_home']*100:.1f}%")
    m3.metric("和局機率", f"{u_pred['prob_draw']*100:.1f}%")
    m4.metric("客勝機率", f"{u_pred['prob_away']*100:.1f}%")