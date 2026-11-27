import sqlite3
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson
from model import PoissonMatchPredictor, FORMATION_PROFILES, analyze_in_play_match
from ucl_and_daily import CrossLeagueUCLPredictor, DB_PATH
from odds_fetcher import fetch_espn_live_odds
from residual_ml import predict_home_residual_adjustment
from hkjc_ocr import parse_hkjc_screenshot
from lineup_intel import fetch_match_lineup_and_injuries
from sim_ledger import (
    STATUS_OPTIONS,
    MARKET_TYPES,
    add_sim_bet,
    settle_bet_manual,
    delete_sim_bet,
    auto_settle_pending_bets,
    load_all_sim_bets,
    import_sim_bets_csv,
)

st.set_page_config(page_title="HKJC 足球量化戰情室", layout="wide")
st.title("⚽ HKJC 個人足球量化與 +EV 全玩法戰情室（10 大聯賽 × 6 季 npxG/PPDA 版）")

PROFITABLE_MARKETS = {
    ("NED-Eredivisie", "入球大細 (HiLo)"),
    ("NED-Eredivisie", "主客和 (HAD)"),
    ("POR-Primeira Liga", "主客和 (HAD)"),
    ("POR-Primeira Liga", "入球大細 (HiLo)"),
    ("ENG-Championship", "入球大細 (HiLo)"),
    ("ENG-Championship", "主客和 (HAD)"),
    ("ENG-Premier League", "入球大細 (HiLo)"),
    ("ENG-Premier League", "主客和 (HAD)"),
    ("ESP-La Liga", "主客和 (HAD)"),
    ("GER-Bundesliga", "入球大細 (HiLo)"),
    ("ITA-Serie A", "主客和 (HAD)"),
}

LEAGUE_MOD_WEIGHTS = {
    "ENG-Premier League": 0.35,
    "ENG-Championship":   0.28,
    "ESP-La Liga":        0.30,
    "GER-Bundesliga":     0.20,
    "ITA-Serie A":        0.15,
    "FRA-Ligue 1":        0.22,
    "NED-Eredivisie":     0.28,
    "POR-Primeira Liga":  0.25,
    "JPN-J1 League":      0.26,
    "KOR-K League 1":     0.25,
}


def remove_vig_two_way(odd1: float, odd2: float):
    inv1, inv2 = 1.0 / odd1, 1.0 / odd2
    total = inv1 + inv2
    return inv1 / total, inv2 / total


def remove_vig_three_way(o1: float, o2: float, o3: float):
    inv1, inv2, inv3 = 1.0 / o1, 1.0 / o2, 1.0 / o3
    total = inv1 + inv2 + inv3
    return inv1 / total, inv2 / total, inv3 / total


def format_kelly_advice(
    ev: float,
    odd: float,
    bankroll: float,
    league: str = "",
    market_type: str = "",
    strict_range: bool = True,
) -> str:
    if ev <= 0.015:
        return "跳過 (無正期望值)"
    if ev > 0.15:
        return "⚠ 分歧過大 (先查傷停/陣容)"
    if strict_range and not (1.50 <= odd <= 3.30):
        return "⚠ 賠率超出風控區間"

    q_k = min(0.03, (ev / (odd - 1.0)) * 0.25)
    if league and market_type and (league, market_type) not in PROFITABLE_MARKETS:
        return f"⚡ 觀察級 (非回測主力賽道，上限 ${bankroll * q_k * 0.4:.0f})"
    return f"★ 主力推薦買入 ${bankroll * q_k:.0f} ({q_k*100:.1f}%)"


@st.cache_resource
def load_domestic_model(league: str):
    return PoissonMatchPredictor(league=league, goal_weight=0.4, half_life_days=45.0)


@st.cache_resource
def load_ucl_model():
    return CrossLeagueUCLPredictor()


@st.cache_data(ttl=600)
def get_cached_live_odds(league: str):
    return fetch_espn_live_odds(league)


@st.cache_data(ttl=300)
def get_cached_lineup_intel(league: str, home_team: str, away_team: str):
    return fetch_match_lineup_and_injuries(league, home_team, away_team)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 未來賽程自動掃描",
    "🧮 HKJC 全玩法 +EV 計算器",
    "🏆 歐聯與跨聯賽戰情室",
    "⚡ 即時走地對局分析 (In-Play)",
    "📝 模擬買波與注單追蹤記錄",
])

# ==============================================================================
# Tab 1: 未來賽程自動掃描
# ==============================================================================
with tab1:
    st.subheader("即將開賽賽程與模型初盤基準（覆蓋 10 大聯賽 + 歐聯）")
    days = st.slider("掃描未來天數", 1, 7, 3)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM matches WHERE status = 'scheduled'", conn)
    conn.close()

    if df.empty:
        st.warning("資料庫目前沒有待開賽賽程，請先在終端機執行 `python src/scraper.py`。")
    else:
        df["dt"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = today + timedelta(days=days)
        upcoming = df[(df["dt"] >= today) & (df["dt"] <= end_date)].sort_values(["dt", "league"])

        if upcoming.empty:
            st.info("所選天數內暫無賽程，以下顯示最近期的 10 場待開賽賽事：")
            upcoming = df[df["dt"] >= today].sort_values(["dt", "league"]).head(10)

        ucl_model = load_ucl_model()
        rows = []
        for _, r in upcoming.iterrows():
            lg, h, a, d_str = r["league"], r["home_team"], r["away_team"], r["date"]
            try:
                if lg == "UEFA-Champions-League":
                    res = ucl_model.predict_ucl_match(h, a)
                else:
                    res = ucl_model.league_models[lg].predict_match(h, a)
                cor_str = f"{res['exp_total_corners']} ({res['exp_home_corners']}-{res['exp_away_corners']})"
                ppda_str = f"{res.get('home_ppda', 11.5)} vs {res.get('away_ppda', 11.5)}"
                fo_h = f"{res['fair_odds']['主勝 (1)']}"
                fo_d = f"{res['fair_odds']['和局 (X)']}"
                fo_a = f"{res['fair_odds']['客勝 (2)']}"
                rows.append({
                    "日期": d_str,
                    "聯賽/盃賽": lg,
                    "主隊": h,
                    "客隊": a,
                    "預期運動戰入球 (npxG)": f"{res['lambda_home']} - {res['lambda_away']}",
                    "PPDA 逼搶對比": ppda_str,
                    "預期角球": cor_str,
                    "主勝 % (公平賠率)": f"{res['prob_home']*100:.1f}% ({fo_h})",
                    "和局 % (公平賠率)": f"{res['prob_draw']*100:.1f}% ({fo_d})",
                    "客勝 % (公平賠率)": f"{res['prob_away']*100:.1f}% ({fo_a})",
                    "大 2.5 %": f"{res['prob_over_25']*100:.1f}%",
                    "首選波膽": res["top_scores"][0][0],
                })
            except Exception:
                continue

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ==============================================================================
# Tab 2: 10 大聯賽 HKJC 全玩法 +EV 計算器
# ==============================================================================
with tab2:
    st.subheader("10 大聯賽對陣與 HKJC 全玩法賠率評估（動態 Elo ＋ 負二項/npxG ＋ PPDA 壓迫 ＋ LightGBM）")
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1.5])
    with col1:
        league = st.selectbox("選擇聯賽", list(LEAGUE_MOD_WEIGHTS.keys()))
        model = load_domestic_model(league)
        teams = model.list_teams()
    with col2:
        home_team = st.selectbox("主隊", teams, index=0)
    with col3:
        away_team = st.selectbox("客隊", teams, index=1 if len(teams) > 1 else 0)
    with col4:
        bankroll = st.number_input("總本金 (HKD)", min_value=100, value=10000, step=500)

    intel = get_cached_lineup_intel(league, home_team, away_team)

    st.markdown("#### 戰術、傷停折減與賽程密度參數（已自動對接資料庫休息天數 & ESPN 傷停/正選陣型）")
    t1, t2, t3, t4 = st.columns([1.2, 1.2, 2, 2])
    form_list = list(FORMATION_PROFILES.keys())
    with t1:
        home_rest = st.number_input("主隊休息天數 (自動計算)", min_value=2, max_value=21, value=int(intel["home_rest_days"]), step=1)
    with t2:
        away_rest = st.number_input("客隊休息天數 (自動計算)", min_value=2, max_value=21, value=int(intel["away_rest_days"]), step=1)
    with t3:
        def_hf_idx = form_list.index(intel["home_formation"]) if intel["home_formation"] in form_list else 0
        home_form = st.selectbox("主隊陣型 (賽前1小時自動鎖定)", form_list, index=def_hf_idx)
    with t4:
        def_af_idx = form_list.index(intel["away_formation"]) if intel["away_formation"] in form_list else 0
        away_form = st.selectbox("客隊陣型 (賽前1小時自動鎖定)", form_list, index=def_af_idx)

    if intel["home_injuries"] or intel["away_injuries"] or intel["lineup_confirmed"]:
        with st.expander(f"🏥 點擊展開：傷停名單與官方正選 11 人明細（{intel['summary_note']}）", expanded=True):
            ic1, ic2 = st.columns(2)
            with ic1:
                st.markdown(f"**🏠 {home_team} 傷停名單**（進攻係數 `{intel['h_att_mult']:.2f}` / 防守係數 `{intel['h_def_mult']:.2f}`）")
                st.write("、".join(intel["home_injuries"]) if intel["home_injuries"] else "暫無主要傷停紀錄")
                if intel["home_xi"]:
                    st.caption(f"正選 11 人：{', '.join(intel['home_xi'])}")
            with ic2:
                st.markdown(f"**✈️ {away_team} 傷停名單**（進攻係數 `{intel['a_att_mult']:.2f}` / 防守係數 `{intel['a_def_mult']:.2f}`）")
                st.write("、".join(intel["away_injuries"]) if intel["away_injuries"] else "暫無主要傷停紀錄")
                if intel["away_xi"]:
                    st.caption(f"正選 11 人：{', '.join(intel['away_xi'])}")

    pred = model.predict_match(
        home_team,
        away_team,
        home_rest_days=home_rest,
        away_rest_days=away_rest,
        home_formation=home_form,
        away_formation=away_form,
        injury_adj=intel,
    )
    matrix = pred["matrix"]
    w_mod = LEAGUE_MOD_WEIGHTS.get(league, 0.25)

    press_info = f" ｜ {pred['press_note']}" if pred.get("press_note") else ""
    st.info(
        f"**引擎架構**：{pred['dist_mode']}（本聯賽模型融合權重 **{int(w_mod*100)}%**，已啟用純運動戰 **npxG**） ｜ **雙方動態 Elo**：`{home_team}` **{int(pred['home_elo'])}** vs **{int(pred['away_elo'])}** `{away_team}`  \n"
        f"**傷停與陣容狀態**：{intel['summary_note']}  \n"
        f"**體能與逼搶節奏**：{pred['rest_note']}{press_info}  \n**陣型戰術解析**：{pred['tactical_note']}"
    )

    live_odds_dict = get_cached_live_odds(league)
    match_key = f"{home_team} vs {away_team}"
    auto_data = live_odds_dict.get(match_key, {})

    ocr_col1, ocr_col2 = st.columns([1.3, 1.7])
    with ocr_col1:
        uploaded_img = st.file_uploader(
            "📸 上傳手機 HKJC 足智彩賠率截圖（自動辨識主客和 / 讓球 / 大細 / 角球）",
            type=["png", "jpg", "jpeg", "webp"],
        )
    ocr_data = {}
    if uploaded_img is not None:
        with st.spinner("正在高速識別足智彩截圖賠率..."):
            ocr_data = parse_hkjc_screenshot(uploaded_img)
        with ocr_col2:
            if "error" in ocr_data:
                st.error(ocr_data["error"])
            else:
                recognized_keys = [k for k in ocr_data.keys() if k != "raw_preview"]
                st.success(f"✅ 截圖識別成功！已自動填入 **{len(recognized_keys)}** 項賠率／盤口參數（下方可隨時微調）")
                with st.expander("查看截圖文字辨識原始明細"):
                    st.code(ocr_data.get("raw_preview", ""))

    raw_paste = st.text_input(
        "⚡ 或手動一鍵貼上 HKJC 賠率（依序貼上數字：主勝 和局 客勝 入球大 入球細，例如 `1.85 3.40 3.80 1.80 1.95`）",
        placeholder="1.85 3.40 3.80 1.80 1.95"
    )
    parsed = [float(x) for x in raw_paste.replace(",", " ").split() if x.replace(".", "", 1).isdigit()]

    def_h = parsed[0] if len(parsed) > 0 else ocr_data.get("home_odd", auto_data.get("home_odd") or 2.10)
    def_d = parsed[1] if len(parsed) > 1 else ocr_data.get("draw_odd", auto_data.get("draw_odd") or 3.40)
    def_a = parsed[2] if len(parsed) > 2 else ocr_data.get("away_odd", auto_data.get("away_odd") or 3.20)
    def_o = parsed[3] if len(parsed) > 3 else ocr_data.get("over_g_odd", 1.85)
    def_u = parsed[4] if len(parsed) > 4 else ocr_data.get("under_g_odd", 1.90)

    def_hhad_line = ocr_data.get("hhad_line", -1)
    def_hh = ocr_data.get("hh_odd", 3.80)
    def_hd = ocr_data.get("hd_odd", 3.60)
    def_ha = ocr_data.get("ha_odd", 1.68)

    ah_options = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    ocr_ah_l = ocr_data.get("asian_line", -0.25)
    def_ah_idx = ah_options.index(ocr_ah_l) if ocr_ah_l in ah_options else ah_options.index(-0.25)
    def_h_ah = ocr_data.get("h_ah_odd", 1.88)
    def_a_ah = ocr_data.get("a_ah_odd", 1.96)

    g_options = [1.5, 2.5, 3.5, 4.5]
    ocr_gl = ocr_data.get("goal_line", 2.5)
    def_g_idx = g_options.index(ocr_gl) if ocr_gl in g_options else 1

    c_options = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
    ocr_cl = ocr_data.get("corner_line", 9.5)
    def_c_idx = c_options.index(ocr_cl) if ocr_cl in c_options else 2
    def_oc = ocr_data.get("over_c_odd", 1.85)
    def_uc = ocr_data.get("under_c_odd", 1.85)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "修正後預期運動戰入球 (npxG)",
        f"{pred['lambda_home']} vs {pred['lambda_away']}",
        delta=f"基礎主客: {pred['base_lambda_home']} vs {pred['base_lambda_away']}"
    )
    m2.metric("預期總角球 (主 vs 客)", f"{pred['exp_total_corners']} 球 ({pred['exp_home_corners']} vs {pred['exp_away_corners']})")
    m3.metric("最高機率波膽", f"{pred['top_scores'][0][0]} ({pred['top_scores'][0][1]*100:.1f}%)")
    m4.metric("次高機率波膽", f"{pred['top_scores'][1][0]} ({pred['top_scores'][1][1]*100:.1f}%)")

    st.divider()

    st.markdown("### 1. 主客和 (HAD) & 讓球主客和 (HHAD)")
    c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1, 1, 1.2, 1, 1, 1])
    h_odd = c1.number_input("主勝 (1)", min_value=1.01, value=float(def_h), step=0.05)
    d_odd = c2.number_input("和局 (X)", min_value=1.01, value=float(def_d), step=0.05)
    a_odd = c3.number_input("客勝 (2)", min_value=1.01, value=float(def_a), step=0.05)
    hhad_options = [-2, -1, 1, 2]
    hhad_line = c4.selectbox("讓球主客和盤口", hhad_options, index=hhad_options.index(def_hhad_line), format_func=lambda x: f"主隊 [{x:+d}]")
    hh_odd = c5.number_input("讓球主勝", min_value=1.01, value=float(def_hh), step=0.05)
    hd_odd = c6.number_input("讓球和", min_value=1.01, value=float(def_hd), step=0.05)
    ha_odd = c7.number_input("讓球客勝", min_value=1.01, value=float(def_ha), step=0.05)

    st.markdown("### 2. 亞洲讓球盤 (HDC - 支援平手/半球、半/一雙盤口)")
    ah1, ah2, ah3 = st.columns(3)
    asian_line = ah1.selectbox("主隊讓球盤口 (負數=主讓, 正數=主受讓)", ah_options, index=def_ah_idx)
    h_ah_odd = ah2.number_input("讓球 - 主隊賠率", min_value=1.01, value=float(def_h_ah), step=0.02)
    a_ah_odd = ah3.number_input("讓球 - 客隊賠率", min_value=1.01, value=float(def_a_ah), step=0.02)

    st.markdown("### 3. 入球大細 (HiLo) & 角球大細 (Corners HiLo)")
    g1, g2, g3, cor1, cor2, cor3 = st.columns(6)
    goal_line = g1.selectbox("入球大細球線", g_options, index=def_g_idx)
    over_g_odd = g2.number_input(f"入球大 ({goal_line})", min_value=1.01, value=float(def_o), step=0.05)
    under_g_odd = g3.number_input(f"入球細 ({goal_line})", min_value=1.01, value=float(def_u), step=0.05)

    corner_line = cor1.selectbox("角球大細盤口", c_options, index=def_c_idx)
    over_c_odd = cor2.number_input(f"角球大 ({corner_line})", min_value=1.01, value=float(def_oc), step=0.05)
    under_c_odd = cor3.number_input(f"角球細 ({corner_line})", min_value=1.01, value=float(def_uc), step=0.05)

    mkt_h, mkt_d, mkt_a = remove_vig_three_way(h_odd, d_odd, a_odd)
    vig = (1.0 / h_odd + 1.0 / d_odd + 1.0 / a_odd) - 1.0
    ah_implied_adv = -asian_line + (0.5 / h_ah_odd - 0.5 / a_ah_odd)
    eu_implied_adv = (1.0 / h_odd) - (1.0 / a_odd)

    ml_delta_h = predict_home_residual_adjustment({
        "elo_diff": pred["home_elo"] - pred["away_elo"],
        "xg_lam_diff": pred["lambda_home"] - pred["lambda_away"],
        "sot_efficiency_diff": pred["sot_efficiency_diff"],
        "dc_prob_home": pred["prob_home"],
        "mkt_prob_home": mkt_h,
        "ah_line": asian_line,
        "ah_eu_divergence": ah_implied_adv - (eu_implied_adv * 1.5),
        "market_overround": vig,
        "rest_days_diff": home_rest - away_rest,
    })

    adj_mod_h = float(np.clip(pred["prob_home"] + ml_delta_h, 0.02, 0.95))
    adj_mod_a = float(np.clip(pred["prob_away"] - ml_delta_h * 0.7, 0.02, 0.95))
    adj_mod_d = max(0.02, 1.0 - adj_mod_h - adj_mod_a)
    norm_sum = adj_mod_h + adj_mod_d + adj_mod_a
    adj_mod_h, adj_mod_d, adj_mod_a = adj_mod_h / norm_sum, adj_mod_d / norm_sum, adj_mod_a / norm_sum

    if abs(ml_delta_h) >= 0.005:
        st.caption(f"🤖 **LightGBM 第二層殘差修正已啟動**：主勝模型基準機率微調 **{ml_delta_h*100:+.1f}%**（結合射正轉化率差、亞盤歐賠背離度與休息天數差）")

    results = []

    for label, p_mod, p_mkt, odd in [
        ("主客和 - 主勝", adj_mod_h, mkt_h, h_odd),
        ("主客和 - 和局", adj_mod_d, mkt_d, d_odd),
        ("主客和 - 客勝", adj_mod_a, mkt_a, a_odd),
    ]:
        cal_p = (1.0 - w_mod) * p_mkt + w_mod * p_mod
        ev = cal_p * odd - 1.0
        results.append({
            "玩法類別": "主客和 (HAD)",
            "選項": label,
            "模型/校準勝率": f"{cal_p*100:.1f}%",
            "HKJC 賠率": f"{odd:.2f}",
            "期望值 (EV)": f"{ev*100:+.1f}%",
            "風控建議": format_kelly_advice(ev, odd, bankroll, league, "主客和 (HAD)"),
        })

    hhad_probs = PoissonMatchPredictor.evaluate_hhad_probs(matrix, hhad_line)
    mkt_hh, mkt_hd, mkt_ha = remove_vig_three_way(hh_odd, hd_odd, ha_odd)
    for label, key, p_mkt, odd in [
        (f"讓球[{hhad_line:+d}] 主勝", "讓球主勝", mkt_hh, hh_odd),
        (f"讓球[{hhad_line:+d}] 和", "讓球和", mkt_hd, hd_odd),
        (f"讓球[{hhad_line:+d}] 客勝", "讓球客勝", mkt_ha, ha_odd),
    ]:
        cal_p = (1.0 - w_mod) * p_mkt + w_mod * hhad_probs[key]
        ev = cal_p * odd - 1.0
        results.append({
            "玩法類別": "讓球主客和 (HHAD)",
            "選項": label,
            "模型/校準勝率": f"{cal_p*100:.1f}%",
            "HKJC 賠率": f"{odd:.2f}",
            "期望值 (EV)": f"{ev*100:+.1f}%",
            "風控建議": format_kelly_advice(ev, odd, bankroll, league, "讓球主客和 (HHAD)"),
        })

    h_ah_res = PoissonMatchPredictor.evaluate_asian_handicap_ev(matrix, asian_line, h_ah_odd, "home")
    a_ah_res = PoissonMatchPredictor.evaluate_asian_handicap_ev(matrix, asian_line, a_ah_odd, "away")
    for label, res_obj, odd in [
        (f"亞洲讓球 主 ({asian_line:+g})", h_ah_res, h_ah_odd),
        (f"亞洲讓球 客 ({-asian_line:+g})", a_ah_res, a_ah_odd),
    ]:
        ev = res_obj["ev"]
        win_desc = f"全贏 {res_obj['p_win_full']*100:.1f}%" + (f" / 贏半 {res_obj['p_win_half']*100:.1f}%" if res_obj["p_win_half"] > 0 else "")
        results.append({
            "玩法類別": "亞洲讓球 (HDC)",
            "選項": label,
            "模型/校準勝率": win_desc,
            "HKJC 賠率": f"{odd:.2f}",
            "期望值 (EV)": f"{ev*100:+.1f}%",
            "風控建議": format_kelly_advice(ev, odd, bankroll, league, "亞洲讓球 (HDC)", strict_range=False),
        })

    p_over_g_mod = PoissonMatchPredictor.calc_total_over_prob(matrix, goal_line)
    mkt_og, _ = remove_vig_two_way(over_g_odd, under_g_odd)
    cal_og = 0.70 * mkt_og + 0.30 * p_over_g_mod
    for label, p_val, odd in [
        (f"入球大 ({goal_line})", cal_og, over_g_odd),
        (f"入球細 ({goal_line})", 1.0 - cal_og, under_g_odd),
    ]:
        ev = p_val * odd - 1.0
        results.append({
            "玩法類別": "入球大細 (HiLo)",
            "選項": label,
            "模型/校準勝率": f"{p_val*100:.1f}%",
            "HKJC 賠率": f"{odd:.2f}",
            "期望值 (EV)": f"{ev*100:+.1f}%",
            "風控建議": format_kelly_advice(ev, odd, bankroll, league, "入球大細 (HiLo)", strict_range=False),
        })

    p_over_c_mod = PoissonMatchPredictor.calc_corner_over_prob(pred["exp_total_corners"], corner_line)
    mkt_oc, _ = remove_vig_two_way(over_c_odd, under_c_odd)
    cal_oc = 0.70 * mkt_oc + 0.30 * p_over_c_mod
    for label, p_val, odd in [
        (f"角球大 ({corner_line})", cal_oc, over_c_odd),
        (f"角球細 ({corner_line})", 1.0 - cal_oc, under_c_odd),
    ]:
        ev = p_val * odd - 1.0
        results.append({
            "玩法類別": "角球大細 (Corners)",
            "選項": label,
            "模型/校準勝率": f"{p_val*100:.1f}%",
            "HKJC 賠率": f"{odd:.2f}",
            "期望值 (EV)": f"{ev*100:+.1f}%",
            "風控建議": format_kelly_advice(ev, odd, bankroll, league, "角球大細 (Corners)", strict_range=False),
        })

    st.divider()
    st.markdown("### 📊 全玩法期望值 (EV) 與凱利注碼評估總表")
    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

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

    with st.expander("📐 點擊展開：6×6 波膽機率矩陣 (0-0 至 5-5) & 主客場獨立戰力評級 (Elo / α / β)"):
        mat_col, rank_col = st.columns([1.2, 1])
        with mat_col:
            st.markdown(f"**{home_team} vs {away_team} 波膽百分比矩陣 (%)**")
            mat_df = pd.DataFrame(
                (matrix * 100).round(1),
                index=[f"主 {i} 球" for i in range(6)],
                columns=[f"客 {j} 球" for j in range(6)],
            )
            st.dataframe(mat_df, use_container_width=True)
            st.caption(
                f"理論公平賠率 (Fair Odds)：主勝 **{pred['fair_odds']['主勝 (1)']}** ｜ "
                f"和局 **{pred['fair_odds']['和局 (X)']}** ｜ 客勝 **{pred['fair_odds']['客勝 (2)']}** ｜ "
                f"大2.5 **{pred['fair_odds']['入球大 (2.5)']}**"
            )
        with rank_col:
            st.markdown(f"**{league} 球隊實力排行榜 (動態 Elo + MLE 求解)**")
            st.dataframe(model.get_team_ratings(), use_container_width=True, hide_index=True, height=280)

# ==============================================================================
# Tab 3: 歐聯 (UCL) 與跨國聯賽深度戰情室
# ==============================================================================
with tab3:
    st.subheader("🏆 歐聯 (UCL) 與跨國聯賽深度戰情室（傷停自動抓取 / 中立場 / 晉級盤 / HKJC 全玩法 +EV）")
    ucl = load_ucl_model()
    all_teams = sorted(ucl.team_to_league.keys())

    conn = sqlite3.connect(DB_PATH)
    ucl_sched = pd.read_sql_query(
        "SELECT date, home_team, away_team FROM matches WHERE league = 'UEFA-Champions-League' AND status = 'scheduled' ORDER BY date",
        conn,
    )
    conn.close()

    def_u_h = "Real Madrid" if "Real Madrid" in all_teams else all_teams[0]
    def_u_a = "Liverpool" if "Liverpool" in all_teams else all_teams[1]

    if not ucl_sched.empty:
        ucl_options = ["手動自選對陣"] + [
            f"{r['date']} | {r['home_team']} vs {r['away_team']}" for _, r in ucl_sched.iterrows()
        ]
        sel_ucl_fix = st.selectbox("📅 一鍵載入未來歐聯賽程（或選擇手動自選對陣）", ucl_options, key="ucl_sched_sel")
        if sel_ucl_fix != "手動自選對陣":
            cand_h, cand_a = sel_ucl_fix.split(" | ")[1].split(" vs ")
            if cand_h in all_teams:
                def_u_h = cand_h
            if cand_a in all_teams:
                def_u_a = cand_a

    uc1, uc2, uc3 = st.columns([2, 2, 1.3])
    u_home = uc1.selectbox("主隊 / 名義主隊", all_teams, index=all_teams.index(def_u_h), key="ucl_home")
    u_away = uc2.selectbox("客隊 / 名義客隊", all_teams, index=all_teams.index(def_u_a), key="ucl_away")
    u_bankroll = uc3.number_input("歐聯專項本金 (HKD)", min_value=100, value=10000, step=500, key="ucl_bankroll")

    u_intel = get_cached_lineup_intel("UEFA-Champions-League", u_home, u_away)

    st.markdown("#### 賽制、場地、傷停折減與賽程密度參數")
    sc1, sc2, sc3, sc4 = st.columns([1.6, 1.4, 1.1, 1.1])
    venue_mode = sc1.selectbox(
        "場地與主客場設定",
        ["標準主客場 (有歐戰主場優勢)", "中立場 / 決賽 (消除主場優勢)"],
        key="ucl_venue",
    )
    tie_mode = sc2.selectbox(
        "賽制模式",
        ["單場 / 聯賽階段", "淘汰賽次回合 (計算晉級機率)"],
        key="ucl_tie_mode",
    )
    leg1_h = sc3.number_input("首回合：本場主隊入球", min_value=0, max_value=15, value=0, step=1, disabled=(tie_mode == "單場 / 聯賽階段"), key="ucl_leg1_h")
    leg1_a = sc4.number_input("首回合：本場客隊入球", min_value=0, max_value=15, value=0, step=1, disabled=(tie_mode == "單場 / 聯賽階段"), key="ucl_leg1_a")

    ut1, ut2, ut3, ut4 = st.columns([1.2, 1.2, 2, 2])
    form_list_u = list(FORMATION_PROFILES.keys())
    with ut1:
        u_h_rest = st.number_input("主隊休息天數", min_value=2, max_value=21, value=int(u_intel["home_rest_days"]), step=1, key="ucl_h_rest")
    with ut2:
        u_a_rest = st.number_input("客隊休息天數", min_value=2, max_value=21, value=int(u_intel["away_rest_days"]), step=1, key="ucl_a_rest")
    with ut3:
        uhf_idx = form_list_u.index(u_intel["home_formation"]) if u_intel["home_formation"] in form_list_u else 0
        u_h_form = st.selectbox("主隊陣型 (賽前1小時自動鎖定)", form_list_u, index=uhf_idx, key="ucl_h_form")
    with ut4:
        uaf_idx = form_list_u.index(u_intel["away_formation"]) if u_intel["away_formation"] in form_list_u else 0
        u_a_form = st.selectbox("客隊陣型 (賽前1小時自動鎖定)", form_list_u, index=uaf_idx, key="ucl_a_form")

    if u_intel["home_injuries"] or u_intel["away_injuries"] or u_intel["lineup_confirmed"]:
        with st.expander(f"🏥 點擊展開：歐聯傷停名單與官方正選 11 人明細（{u_intel['summary_note']}）", expanded=True):
            uic1, uic2 = st.columns(2)
            with uic1:
                st.markdown(f"**🏠 {u_home} 傷停名單**（進攻係數 `{u_intel['h_att_mult']:.2f}` / 防守係數 `{u_intel['h_def_mult']:.2f}`）")
                st.write("、".join(u_intel["home_injuries"]) if u_intel["home_injuries"] else "暫無主要傷停紀錄")
                if u_intel["home_xi"]:
                    st.caption(f"正選 11 人：{', '.join(u_intel['home_xi'])}")
            with uic2:
                st.markdown(f"**✈️ {u_away} 傷停名單**（進攻係數 `{u_intel['a_att_mult']:.2f}` / 防守係數 `{u_intel['a_def_mult']:.2f}`）")
                st.write("、".join(u_intel["away_injuries"]) if u_intel["away_injuries"] else "暫無主要傷停紀錄")
                if u_intel["away_xi"]:
                    st.caption(f"正選 11 人：{', '.join(u_intel['away_xi'])}")

    u_pred = ucl.predict_ucl_match(
        u_home,
        u_away,
        neutral_venue=("中立場" in venue_mode),
        home_rest_days=u_h_rest,
        away_rest_days=u_a_rest,
        home_formation=u_h_form,
        away_formation=u_a_form,
        injury_adj=u_intel,
    )
    u_mat = u_pred["matrix"]

    st.info(
        f"**跨聯賽戰力對照**：`{u_home}`（**{u_pred['home_league']}**｜跨聯賽 Elo **{int(u_pred['home_elo'])}**） vs "
        f"`{u_away}`（**{u_pred['away_league']}**｜跨聯賽 Elo **{int(u_pred['away_elo'])}**）  \n"
        f"**傷停與陣容狀態**：{u_intel['summary_note']}  \n"
        f"**體能與逼搶節奏**：{u_pred['rest_note']} ｜ {u_pred['press_note']}  \n"
        f"**陣型戰術解析**：{u_pred['tactical_note']}"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("修正後預期運動戰入球 (npxG)", f"{u_pred['lambda_home']} vs {u_pred['lambda_away']}", delta=f"基礎: {u_pred['base_lambda_home']} vs {u_pred['base_lambda_away']}")
    m2.metric("預期總角球 (主 vs 客)", f"{u_pred['exp_total_corners']} 球 ({u_pred['exp_home_corners']} vs {u_pred['exp_away_corners']})")
    m3.metric("最高機率波膽", f"{u_pred['top_scores'][0][0]} ({u_pred['top_scores'][0][1]*100:.1f}%)")
    m4.metric("次高機率波膽", f"{u_pred['top_scores'][1][0]} ({u_pred['top_scores'][1][1]*100:.1f}%)")

    if tie_mode == "淘汰賽次回合 (計算晉級機率)":
        p_qual_h, p_qual_a = 0.0, 0.0
        max_g = u_mat.shape[0]
        for i in range(max_g):
            for j in range(max_g):
                agg_h = leg1_h + i
                agg_a = leg1_a + j
                if agg_h > agg_a:
                    p_qual_h += u_mat[i, j]
                elif agg_a > agg_h:
                    p_qual_a += u_mat[i, j]
                else:
                    et_h_edge = 0.50 if "中立場" in venue_mode else 0.53
                    p_qual_h += u_mat[i, j] * et_h_edge
                    p_qual_a += u_mat[i, j] * (1.0 - et_h_edge)

        st.markdown("#### 🎟️ 淘汰賽「晉級隊伍 (To Qualify)」機率與公平賠率（含加時及十二碼）")
        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric(f"{u_home} 晉級機率", f"{p_qual_h*100:.1f}%", f"公平賠率 {1/max(0.01, p_qual_h):.2f}")
        qc2.metric(f"{u_away} 晉級機率", f"{p_qual_a*100:.1f}%", f"公平賠率 {1/max(0.01, p_qual_a):.2f}")
        hkjc_qh = qc3.number_input(f"HKJC {u_home} 晉級賠率", min_value=1.01, value=1.85, step=0.05, key="ucl_qh")
        hkjc_qa = qc4.number_input(f"HKJC {u_away} 晉級賠率", min_value=1.01, value=1.90, step=0.05, key="ucl_qa")

        ev_qh = p_qual_h * hkjc_qh - 1.0
        ev_qa = p_qual_a * hkjc_qa - 1.0
        st.caption(
            f"晉級盤期望值評估：**{u_home}** EV = **{ev_qh*100:+.1f}%**（{format_kelly_advice(ev_qh, hkjc_qh, u_bankroll, strict_range=False)}） ｜ "
            f"**{u_away}** EV = **{ev_qa*100:+.1f}%**（{format_kelly_advice(ev_qa, hkjc_qa, u_bankroll, strict_range=False)}）"
        )

    st.divider()

    u_ocr_c1, u_ocr_c2 = st.columns([1.3, 1.7])
    with u_ocr_c1:
        u_img = st.file_uploader(
            "📸 上傳歐聯 HKJC 足智彩賠率截圖（自動辨識主客和 / 讓球 / 大細 / 角球）",
            type=["png", "jpg", "jpeg", "webp"],
            key="ucl_ocr_img",
        )
    u_ocr = {}
    if u_img is not None:
        with st.spinner("正在識別歐聯足智彩截圖賠率..."):
            u_ocr = parse_hkjc_screenshot(u_img)
        with u_ocr_c2:
            if "error" in u_ocr:
                st.error(u_ocr["error"])
            else:
                st.success("✅ 歐聯截圖識別成功！已自動填入下方賠率參數")

    u_live_odds = get_cached_live_odds("UEFA-Champions-League")
    u_auto = u_live_odds.get(f"{u_home} vs {u_away}", {})

    uc_o1, uc_o2, uc_o3, uc_o4, uc_o5, uc_o6, uc_o7 = st.columns([1, 1, 1, 1.2, 1, 1, 1])
    uh_odd = uc_o1.number_input("主勝 (1)", min_value=1.01, value=float(u_ocr.get("home_odd", u_auto.get("home_odd") or 2.05)), step=0.05, key="ucl_h_odd")
    ud_odd = uc_o2.number_input("和局 (X)", min_value=1.01, value=float(u_ocr.get("draw_odd", u_auto.get("draw_odd") or 3.50)), step=0.05, key="ucl_d_odd")
    ua_odd = uc_o3.number_input("客勝 (2)", min_value=1.01, value=float(u_ocr.get("away_odd", u_auto.get("away_odd") or 3.30)), step=0.05, key="ucl_a_odd")
    uhhad_opts = [-2, -1, 1, 2]
    uhhad_l = uc_o4.selectbox("讓球主客和盤口", uhhad_opts, index=uhhad_opts.index(u_ocr.get("hhad_line", -1)), format_func=lambda x: f"主隊 [{x:+d}]", key="ucl_hhad_l")
    uhh_odd = uc_o5.number_input("讓球主勝", min_value=1.01, value=float(u_ocr.get("hh_odd", 3.80)), step=0.05, key="ucl_hh_odd")
    uhd_odd = uc_o6.number_input("讓球和", min_value=1.01, value=float(u_ocr.get("hd_odd", 3.60)), step=0.05, key="ucl_hd_odd")
    uha_odd = uc_o7.number_input("讓球客勝", min_value=1.01, value=float(u_ocr.get("ha_odd", 1.68)), step=0.05, key="ucl_ha_odd")

    uah_c1, uah_c2, uah_c3 = st.columns(3)
    ah_opts_u = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    u_ocr_ahl = u_ocr.get("asian_line", -0.25)
    u_ah_idx = ah_opts_u.index(u_ocr_ahl) if u_ocr_ahl in ah_opts_u else ah_opts_u.index(-0.25)
    u_ah_line = uah_c1.selectbox("歐聯亞洲讓球盤口 (HDC)", ah_opts_u, index=u_ah_idx, key="ucl_ah_line")
    u_h_ah = uah_c2.number_input("讓球主隊賠率", min_value=1.01, value=float(u_ocr.get("h_ah_odd", 1.88)), step=0.02, key="ucl_h_ah")
    u_a_ah = uah_c3.number_input("讓球客隊賠率", min_value=1.01, value=float(u_ocr.get("a_ah_odd", 1.96)), step=0.02, key="ucl_a_ah")

    ug1, ug2, ug3, ucor1, ucor2, ucor3 = st.columns(6)
    g_opts_u = [1.5, 2.5, 3.5, 4.5]
    u_gl = u_ocr.get("goal_line", 2.5)
    ug_line = ug1.selectbox("入球大細球線", g_opts_u, index=g_opts_u.index(u_gl) if u_gl in g_opts_u else 1, key="ucl_g_line")
    uo_odd = ug2.number_input(f"入球大 ({ug_line})", min_value=1.01, value=float(u_ocr.get("over_g_odd", 1.80)), step=0.05, key="ucl_o_odd")
    uu_odd = ug3.number_input(f"入球細 ({ug_line})", min_value=1.01, value=float(u_ocr.get("under_g_odd", 1.95)), step=0.05, key="ucl_u_odd")

    c_opts_u = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
    u_cl = u_ocr.get("corner_line", 9.5)
    uc_line = ucor1.selectbox("角球大細盤口", c_opts_u, index=c_opts_u.index(u_cl) if u_cl in c_opts_u else 2, key="ucl_c_line")
    uoc_odd = ucor2.number_input(f"角球大 ({uc_line})", min_value=1.01, value=float(u_ocr.get("over_c_odd", 1.85)), step=0.05, key="ucl_oc_odd")
    uuc_odd = ucor3.number_input(f"角球細 ({uc_line})", min_value=1.01, value=float(u_ocr.get("under_c_odd", 1.85)), step=0.05, key="ucl_uc_odd")

    umkt_h, umkt_d, umkt_a = remove_vig_three_way(uh_odd, ud_odd, ua_odd)
    u_vig = (1.0 / uh_odd + 1.0 / ud_odd + 1.0 / ua_odd) - 1.0
    u_ah_imp = -u_ah_line + (0.5 / u_h_ah - 0.5 / u_a_ah)
    u_eu_imp = (1.0 / uh_odd) - (1.0 / ua_odd)

    u_ml_delta = predict_home_residual_adjustment({
        "elo_diff": u_pred["home_elo"] - u_pred["away_elo"],
        "xg_lam_diff": u_pred["lambda_home"] - u_pred["lambda_away"],
        "sot_efficiency_diff": u_pred["sot_efficiency_diff"],
        "dc_prob_home": u_pred["prob_home"],
        "mkt_prob_home": umkt_h,
        "ah_line": u_ah_line,
        "ah_eu_divergence": u_ah_imp - (u_eu_imp * 1.5),
        "market_overround": u_vig,
        "rest_days_diff": u_h_rest - u_a_rest,
    })

    u_adj_h = float(np.clip(u_pred["prob_home"] + u_ml_delta, 0.02, 0.95))
    u_adj_a = float(np.clip(u_pred["prob_away"] - u_ml_delta * 0.7, 0.02, 0.95))
    u_adj_d = max(0.02, 1.0 - u_adj_h - u_adj_a)
    u_nsum = u_adj_h + u_adj_d + u_adj_a
    u_adj_h, u_adj_d, u_adj_a = u_adj_h / u_nsum, u_adj_d / u_nsum, u_adj_a / u_nsum

    w_ucl_mod = 0.28
    u_results = []
    for lbl, pm, pk, od in [
        (f"主客和 - 主勝 ({u_home})", u_adj_h, umkt_h, uh_odd),
        ("主客和 - 和局 (Draw)", u_adj_d, umkt_d, ud_odd),
        (f"主客和 - 客勝 ({u_away})", u_adj_a, umkt_a, ua_odd),
    ]:
        cp = (1.0 - w_ucl_mod) * pk + w_ucl_mod * pm
        ev = cp * od - 1.0
        u_results.append({
            "玩法類別": "主客和 (HAD)",
            "選項": lbl,
            "融合勝率": f"{cp*100:.1f}%",
            "HKJC 賠率": f"{od:.2f}",
            "期望值 (EV)": f"{ev*100:+.1f}%",
            "風控建議": format_kelly_advice(ev, od, u_bankroll, strict_range=True),
        })

    uhhad_probs = PoissonMatchPredictor.evaluate_hhad_probs(u_mat, uhhad_l)
    umkt_hh, umkt_hd, umkt_ha = remove_vig_three_way(uhh_odd, uhd_odd, uha_odd)
    for lbl, k_name, pk, od in [
        (f"讓球[{uhhad_l:+d}] 主勝", "讓球主勝", umkt_hh, uhh_odd),
        (f"讓球[{uhhad_l:+d}] 和", "讓球和", umkt_hd, uhd_odd),
        (f"讓球[{uhhad_l:+d}] 客勝", "讓球客勝", umkt_ha, uha_odd),
    ]:
        cp = (1.0 - w_ucl_mod) * pk + w_ucl_mod * uhhad_probs[k_name]
        ev = cp * od - 1.0
        u_results.append({
            "玩法類別": "讓球主客和 (HHAD)",
            "選項": lbl,
            "融合勝率": f"{cp*100:.1f}%",
            "HKJC 賠率": f"{od:.2f}",
            "期望值 (EV)": f"{ev*100:+.1f}%",
            "風控建議": format_kelly_advice(ev, od, u_bankroll, strict_range=True),
        })

    uh_ah_res = PoissonMatchPredictor.evaluate_asian_handicap_ev(u_mat, u_ah_line, u_h_ah, "home")
    ua_ah_res = PoissonMatchPredictor.evaluate_asian_handicap_ev(u_mat, u_ah_line, u_a_ah, "away")
    for lbl, r_obj, od in [
        (f"亞洲讓球 主 ({u_ah_line:+g})", uh_ah_res, u_h_ah),
        (f"亞洲讓球 客 ({-u_ah_line:+g})", ua_ah_res, u_a_ah),
    ]:
        ev = r_obj["ev"]
        w_desc = f"全贏 {r_obj['p_win_full']*100:.1f}%" + (f" / 贏半 {r_obj['p_win_half']*100:.1f}%" if r_obj["p_win_half"] > 0 else "")
        u_results.append({
            "玩法類別": "亞洲讓球 (HDC)",
            "選項": lbl,
            "融合勝率": w_desc,
            "HKJC 賠率": f"{od:.2f}",
            "期望值 (EV)": f"{ev*100:+.1f}%",
            "風控建議": format_kelly_advice(ev, od, u_bankroll, strict_range=False),
        })

    up_over_mod = PoissonMatchPredictor.calc_total_over_prob(u_mat, ug_line)
    umkt_o, _ = remove_vig_two_way(uo_odd, uu_odd)
    ucal_o = (1.0 - w_ucl_mod) * umkt_o + w_ucl_mod * up_over_mod
    for lbl, cp, od in [
        (f"入球大 ({ug_line})", ucal_o, uo_odd),
        (f"入球細 ({ug_line})", 1.0 - ucal_o, uu_odd),
    ]:
        ev = cp * od - 1.0
        u_results.append({
            "玩法類別": "入球大細 (HiLo)",
            "選項": lbl,
            "融合勝率": f"{cp*100:.1f}%",
            "HKJC 賠率": f"{od:.2f}",
            "期望值 (EV)": f"{ev*100:+.1f}%",
            "風控建議": format_kelly_advice(ev, od, u_bankroll, strict_range=False),
        })

    up_cor_mod = PoissonMatchPredictor.calc_corner_over_prob(u_pred["exp_total_corners"], uc_line)
    umkt_oc, _ = remove_vig_two_way(uoc_odd, uuc_odd)
    ucal_oc = (1.0 - w_ucl_mod) * umkt_oc + w_ucl_mod * up_cor_mod
    for lbl, cp, od in [
        (f"角球大 ({uc_line})", ucal_oc, uoc_odd),
        (f"角球細 ({uc_line})", 1.0 - ucal_oc, uuc_odd),
    ]:
        ev = cp * od - 1.0
        u_results.append({
            "玩法類別": "角球大細 (Corners)",
            "選項": lbl,
            "融合勝率": f"{cp*100:.1f}%",
            "HKJC 賠率": f"{od:.2f}",
            "期望值 (EV)": f"{ev*100:+.1f}%",
            "風控建議": format_kelly_advice(ev, od, u_bankroll, strict_range=False),
        })

    st.markdown("#### 📊 歐聯 (UCL) 全玩法期望值與注碼建議總表（90 分鐘常規時間＋補時計算）")
    st.dataframe(pd.DataFrame(u_results), use_container_width=True, hide_index=True)

    with st.expander("📐 點擊展開：歐聯 6×6 波膽 (CRS) 機率矩陣 & 全盤口公平賠率參考表"):
        umat_c1, umat_c2 = st.columns([1.2, 1])
        with umat_c1:
            umat_df = pd.DataFrame(
                (u_mat * 100).round(1),
                index=[f"{u_home} {i}球" for i in range(6)],
                columns=[f"{u_away} {j}球" for j in range(6)],
            )
            st.dataframe(umat_df, use_container_width=True)
        with umat_c2:
            ug_ref = []
            for gl in [1.5, 2.5, 3.5, 4.5]:
                po = PoissonMatchPredictor.calc_total_over_prob(u_mat, gl)
                pu = 1.0 - po
                ug_ref.append({
                    "入球線": f"{gl} 球",
                    "大球勝率 (公平賠率)": f"{po*100:.1f}% ({1/max(0.01, po):.2f})",
                    "細球勝率 (公平賠率)": f"{pu*100:.1f}% ({1/max(0.01, pu):.2f})",
                })
            st.table(pd.DataFrame(ug_ref))

# ==============================================================================
# Tab 4: 即時走地對局量化分析 (In-Play)
# ==============================================================================
with tab4:
    st.subheader("⚡ 即時走地對局量化分析（輸入場上實時統計，推算動能與買球方向）")
    st.caption("適用任何聯賽／盃賽走地賽事：輸入當前時間、比數與技術統計，系統自動計算實時 Live xG 與剩餘時間機率。")

    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    live_min = t_col1.slider("當前進行分鐘數", min_value=1, max_value=89, value=45)
    cur_h_goals = t_col2.number_input("主隊現有入球", min_value=0, max_value=15, value=0, step=1)
    cur_a_goals = t_col3.number_input("客隊現有入球", min_value=0, max_value=15, value=0, step=1)
    h_poss = t_col4.slider("主隊控球率 (%)", min_value=10, max_value=90, value=55)
    st.caption(f"客隊控球率自動對應為：**{100 - h_poss}%**")

    st.divider()

    stat_h, stat_a = st.columns(2)
    with stat_h:
        st.markdown("### 🏠 主隊實時數據")
        sh1, sh2, sh3 = st.columns(3)
        h_shots = sh1.number_input("主隊 射門", min_value=0, max_value=50, value=8, step=1)
        h_sot = sh2.number_input("主隊 射正", min_value=0, max_value=30, value=3, step=1)
        h_cor = sh3.number_input("主隊 角球", min_value=0, max_value=25, value=4, step=1)

        sh4, sh5, sh6 = st.columns(3)
        h_att = sh4.number_input("主隊 進攻", min_value=0, max_value=200, value=52, step=1)
        h_dang = sh5.number_input("主隊 危險進攻", min_value=0, max_value=150, value=28, step=1)
        h_big = sh6.number_input("主隊 重大機會 (Big Chances)", min_value=0, max_value=15, value=2, step=1)

    with stat_a:
        st.markdown("### ✈️ 客隊實時數據")
        sa1, sa2, sa3 = st.columns(3)
        a_shots = sa1.number_input("客隊 射門", min_value=0, max_value=50, value=3, step=1)
        a_sot = sa2.number_input("客隊 射正", min_value=0, max_value=30, value=1, step=1)
        a_cor = sa3.number_input("客隊 角球", min_value=0, max_value=25, value=1, step=1)

        sa4, sa5, sa6 = st.columns(3)
        a_att = sa4.number_input("客隊 進攻", min_value=0, max_value=200, value=35, step=1)
        a_dang = sa5.number_input("客隊 危險進攻", min_value=0, max_value=150, value=12, step=1)
        a_big = sa6.number_input("客隊 重大機會 (Big Chances)", min_value=0, max_value=15, value=0, step=1)

    live_res = analyze_in_play_match(
        minute=live_min,
        h_goals=cur_h_goals,
        a_goals=cur_a_goals,
        h_poss=h_poss,
        h_shots=h_shots,
        a_shots=a_shots,
        h_sot=h_sot,
        a_sot=a_sot,
        h_corners=h_cor,
        a_corners=a_cor,
        h_attacks=h_att,
        a_attacks=a_att,
        h_dangerous=h_dang,
        a_dangerous=a_dang,
        h_big_chances=h_big,
        a_big_chances=a_big,
    )

    st.divider()
    st.markdown("### 🎯 走地量化評估與買球方向建議")

    lm1, lm2, lm3, lm4 = st.columns(4)
    lm1.metric("實時累計 Live xG", f"{live_res['live_xg_h']} vs {live_res['live_xg_a']}", delta=f"現有比數 {cur_h_goals} - {cur_a_goals}")
    lm2.metric("場上動能支配比 (主 vs 客)", f"{live_res['h_dom_pct']}% vs {live_res['a_dom_pct']}%")
    lm3.metric(f"剩餘 {90 - live_min} 分鐘預期再入球", f"{live_res['rem_total_goals']} 球", delta=f"再入1球機率 {live_res['p_rem_over_05']*100:.1f}%")
    lm4.metric("預計全場總角球", f"{live_res['total_exp_corners']} 球", delta=f"剩餘預計 +{live_res['rem_corners']} 球")

    for sig in live_res["signals"]:
        st.success(sig)

    st.markdown("#### 完場賽果與剩餘入球機率分佈（可用作對照 HKJC 走地公平賠率）")
    live_table = pd.DataFrame([
        {
            "走地預測項目": "最終主勝 (1)",
            "模型推算機率": f"{live_res['p_final_h']*100:.1f}%",
            "理論公平賠率 (高於此值具價值)": f"{(1/live_res['p_final_h']):.2f}" if live_res["p_final_h"] > 0.01 else "-",
        },
        {
            "走地預測項目": "最終和局 (X)",
            "模型推算機率": f"{live_res['p_final_d']*100:.1f}%",
            "理論公平賠率 (高於此值具價值)": f"{(1/live_res['p_final_d']):.2f}" if live_res["p_final_d"] > 0.01 else "-",
        },
        {
            "走地預測項目": "最終客勝 (2)",
            "模型推算機率": f"{live_res['p_final_a']*100:.1f}%",
            "理論公平賠率 (高於此值具價值)": f"{(1/live_res['p_final_a']):.2f}" if live_res["p_final_a"] > 0.01 else "-",
        },
        {
            "走地預測項目": f"走地入球大 ({cur_h_goals + cur_a_goals + 0.5} 球線 / 至少再入1球)",
            "模型推算機率": f"{live_res['p_rem_over_05']*100:.1f}%",
            "理論公平賠率 (高於此值具價值)": f"{(1/live_res['p_rem_over_05']):.2f}" if live_res["p_rem_over_05"] > 0.01 else "-",
        },
        {
            "走地預測項目": f"走地入球大 ({cur_h_goals + cur_a_goals + 1.5} 球線 / 至少再入2球)",
            "模型推算機率": f"{live_res['p_rem_over_15']*100:.1f}%",
            "理論公平賠率 (高於此值具價值)": f"{(1/live_res['p_rem_over_15']):.2f}" if live_res["p_rem_over_15"] > 0.01 else "-",
        },
    ])
    st.dataframe(live_table, use_container_width=True, hide_index=True)

# ==============================================================================
# Tab 5: 📝 HKJC 模擬買波與注單追蹤記錄 (Paper Trading & Bet Ledger)
# ==============================================================================
with tab5:
    st.subheader("📝 足智彩 (HKJC) 模擬買波落注台與實戰注單記錄簿")
    st.caption("專用於模擬真實 HKJC 落注紀律、記錄每筆 +EV 出手、自動核對完賽賽果派彩，並追蹤長期 ROI 與資金曲線。")

    # 1. 載入現有模擬注單並計算資金總覽 KPI
    bets_df = load_all_sim_bets()

    top_c1, top_c2, top_c3 = st.columns([1.4, 1.3, 1.3])
    init_bankroll = top_c1.number_input("模擬起始總本金 (HKD)", min_value=500, value=10000, step=1000, key="sim_init_br")

    with top_c2:
        st.write("")
        if st.button("🔄 自動核對完賽賽果並結算", use_container_width=True, key="sim_auto_settle_btn"):
            n_settled = auto_settle_pending_bets()
            if n_settled > 0:
                st.success(f"✅ 已自動核對完賽比數並完成結算 **{n_settled}** 張注單！")
                bets_df = load_all_sim_bets()
            else:
                st.info("暫無新增已完賽可自動結算的待開獎注單（盃賽或角球盤請於下方手動結算）。")

    with top_c3:
        st.write("")
        if not bets_df.empty:
            csv_bytes = bets_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 匯出全部買波記錄 (CSV 備份)",
                data=csv_bytes,
                file_name=f"hkjc_sim_bets_{datetime.now():%Y%m%d}.csv",
                mime="text/csv",
                use_container_width=True,
                key="sim_export_csv",
            )

    settled_df = bets_df[bets_df["status"] != "待開獎 (Pending)"] if not bets_df.empty else pd.DataFrame()
    pending_df = bets_df[bets_df["status"] == "待開獎 (Pending)"] if not bets_df.empty else pd.DataFrame()

    total_pnl = float(settled_df["pnl"].sum()) if not settled_df.empty else 0.0
    total_staked = float(settled_df["stake"].sum()) if not settled_df.empty else 0.0
    pending_stake = float(pending_df["stake"].sum()) if not pending_df.empty else 0.0
    cur_balance = init_bankroll + total_pnl
    roi_pct = (total_pnl / total_staked * 100.0) if total_staked > 0 else 0.0

    won_cnt = int((settled_df["pnl"] > 0).sum()) if not settled_df.empty else 0
    valid_cnt = int((settled_df["pnl"] != 0).sum()) if not settled_df.empty else 0
    win_rate = (won_cnt / valid_cnt * 100.0) if valid_cnt > 0 else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("目前模擬總資金", f"${cur_balance:,.1f}", delta=f"{total_pnl:+,.1f} HKD")
    k2.metric("已結算總投入本金", f"${total_staked:,.0f}", delta=f"待開獎注碼 ${pending_stake:,.0f}")
    k3.metric("累計淨盈虧 (Net P&L)", f"${total_pnl:+,.1f}")
    k4.metric("已結算總回報率 (ROI)", f"{roi_pct:+.2f}%", delta=f"已結算 {len(settled_df)} 注")
    k5.metric("有效命中率 (Win Rate)", f"{win_rate:.1f}%", delta=f"贏 {won_cnt} / 有效 {valid_cnt}")

    st.divider()

    # 2. 新增模擬落注表單（支援從資料庫待開賽賽程一鍵帶入）
    st.markdown("### 🎫 新增模擬買波注單（模擬落注台）")

    conn = sqlite3.connect(DB_PATH)
    sched_all = pd.read_sql_query(
        "SELECT date, league, home_team, away_team FROM matches WHERE status = 'scheduled' ORDER BY date, league",
        conn,
    )
    conn.close()

    entry_mode = st.radio(
        "選擇賽事來源",
        ["從資料庫待開賽賽程挑選（自動載入模型機率與 EV）", "手動輸入自訂賽事（盃賽 / 走地 / 過關）"],
        horizontal=True,
        key="sim_entry_mode",
    )

    if entry_mode == "從資料庫待開賽賽程挑選（自動載入模型機率與 EV）" and not sched_all.empty:
        fix_labels = [
            f"{r['date']} | [{r['league']}] {r['home_team']} vs {r['away_team']}"
            for _, r in sched_all.iterrows()
        ]
        sel_fix = st.selectbox("選擇待開賽場次", fix_labels, key="sim_sched_pick")
        picked_row = sched_all.iloc[fix_labels.index(sel_fix)]
        s_date = str(picked_row["date"])
        s_league = str(picked_row["league"])
        s_home = str(picked_row["home_team"])
        s_away = str(picked_row["away_team"])
    else:
        mc1, mc2, mc3, mc4 = st.columns([1.2, 1.5, 1.8, 1.8])
        s_date = mc1.text_input("比賽日期 (DD/MM/YYYY)", value=datetime.now().strftime("%d/%m/%Y"), key="sim_m_date")
        s_league = mc2.selectbox("聯賽 / 盃賽", list(LEAGUE_MOD_WEIGHTS.keys()) + ["UEFA-Champions-League", "其他盃賽/國際賽"], key="sim_m_lg")
        s_home = mc3.text_input("主隊名稱", value="Arsenal", key="sim_m_home")
        s_away = mc4.text_input("客隊名稱", value="Chelsea", key="sim_m_away")

    bc1, bc2, bc3, bc4 = st.columns([1.5, 1.8, 1.1, 1.1])
    s_mkt = bc1.selectbox("足智彩玩法類別", MARKET_TYPES, key="sim_mkt_type")

    # 根據玩法動態提供常用選項建議
    if s_mkt == "主客和 (HAD)":
        sel_presets = [f"主勝 ({s_home})", "和局 (Draw)", f"客勝 ({s_away})"]
        s_sel = bc2.selectbox("投注選項", sel_presets, key="sim_sel_had")
        s_line = bc3.number_input("盤口球線 (主客和填 0)", value=0.0, step=0.5, disabled=True, key="sim_line_had")
    elif s_mkt == "讓球主客和 (HHAD)":
        s_line = float(bc3.selectbox("主隊讓球盤口", [-2, -1, 1, 2], index=1, key="sim_line_hhad"))
        sel_presets = [f"讓球[{int(s_line):+d}] 主勝 ({s_home})", f"讓球[{int(s_line):+d}] 和", f"讓球[{int(s_line):+d}] 客勝 ({s_away})"]
        s_sel = bc2.selectbox("投注選項", sel_presets, key="sim_sel_hhad")
    elif s_mkt == "亞洲讓球 (HDC)":
        ah_list_s = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        s_line = float(bc3.selectbox("所選球隊讓球盤口", ah_list_s, index=ah_list_s.index(-0.25), key="sim_line_hdc"))
        sel_presets = [f"主隊 {s_home} ({s_line:+g})", f"客隊 {s_away} ({s_line:+g})"]
        s_sel = bc2.selectbox("投注選項", sel_presets, key="sim_sel_hdc")
    elif s_mkt == "入球大細 (HiLo)":
        s_line = float(bc3.selectbox("入球大細球線", [1.5, 2.25, 2.5, 2.75, 3.0, 3.5, 4.5], index=2, key="sim_line_hilo"))
        sel_presets = [f"入球大 ({s_line})", f"入球細 ({s_line})"]
        s_sel = bc2.selectbox("投注選項", sel_presets, key="sim_sel_hilo")
    elif s_mkt == "角球大細 (Corners)":
        s_line = float(bc3.selectbox("角球大細盤口", [7.5, 8.5, 9.5, 10.5, 11.5, 12.5], index=2, key="sim_line_cor"))
        sel_presets = [f"角球大 ({s_line})", f"角球細 ({s_line})"]
        s_sel = bc2.selectbox("投注選項", sel_presets, key="sim_sel_cor")
    else:
        s_line = bc3.number_input("盤口數值 (無則填 0)", value=0.0, step=0.5, key="sim_line_other")
        s_sel = bc2.text_input("具體選項（例如 2-1 / 主隊晉級 / 2串1）", value="2-1", key="sim_sel_other")

    s_odds = bc4.number_input("HKJC 買入賠率", min_value=1.01, value=1.90, step=0.05, key="sim_odds")

    # 自動嘗試計算該場次模型參考勝率與 EV
    auto_mod_p = 0.0
    try:
        ucl_ref = load_ucl_model()
        if s_league == "UEFA-Champions-League" and s_home in ucl_ref.team_to_league and s_away in ucl_ref.team_to_league:
            p_res = ucl_ref.predict_ucl_match(s_home, s_away)
        elif s_league in ucl_ref.league_models:
            p_res = ucl_ref.league_models[s_league].predict_match(s_home, s_away)
        else:
            p_res = None

        if p_res is not None:
            mat_s = p_res["matrix"]
            if s_mkt == "主客和 (HAD)":
                auto_mod_p = p_res["prob_home"] if "主勝" in s_sel else (p_res["prob_draw"] if "和局" in s_sel else p_res["prob_away"])
            elif s_mkt == "讓球主客和 (HHAD)":
                hh_p = PoissonMatchPredictor.evaluate_hhad_probs(mat_s, int(round(s_line)))
                auto_mod_p = hh_p["讓球主勝"] if "主勝" in s_sel else (hh_p["讓球和"] if "和" in s_sel else hh_p["讓球客勝"])
            elif s_mkt == "亞洲讓球 (HDC)":
                side_s = "home" if "主隊" in s_sel else "away"
                eff_line = s_line if side_s == "home" else -s_line
                ah_r = PoissonMatchPredictor.evaluate_asian_handicap_ev(mat_s, eff_line, s_odds, side_s)
                auto_mod_p = ah_r["p_win_full"] + 0.5 * ah_r["p_win_half"]
            elif s_mkt == "入球大細 (HiLo)":
                po = PoissonMatchPredictor.calc_total_over_prob(mat_s, s_line)
                auto_mod_p = po if "大" in s_sel else (1.0 - po)
            elif s_mkt == "角球大細 (Corners)":
                pc = PoissonMatchPredictor.calc_corner_over_prob(p_res["exp_total_corners"], s_line)
                auto_mod_p = pc if "大" in s_sel else (1.0 - pc)
    except Exception:
        auto_mod_p = 0.0

    is_wl = (s_league, s_mkt) in PROFITABLE_MARKETS
    est_ev = (auto_mod_p * s_odds - 1.0) * 100.0 if auto_mod_p > 0 else 0.0
    kelly_rec_amt = 0.0
    if est_ev > 1.5 and s_odds > 1.01:
        q_f = min(0.03, ((est_ev / 100.0) / (s_odds - 1.0)) * 0.25)
        kelly_rec_amt = round(cur_balance * q_f * (1.0 if is_wl else 0.4), -1)

    sc1, sc2, sc3 = st.columns([1.2, 2.3, 1.2])
    default_stake = max(50.0, kelly_rec_amt) if kelly_rec_amt >= 50.0 else 200.0
    s_stake = sc1.number_input("模擬投注金額 (HKD)", min_value=10.0, value=float(default_stake), step=50.0, key="sim_stake")
    s_notes = sc2.text_input("買波筆記 / 理由（例如：荷甲主力大球白名單 +EV 6.2%）", value="", key="sim_notes")

    wl_badge = "🌟 黃金白名單賽道" if is_wl else "👀 觀察級賽道"
    if auto_mod_p > 0:
        st.caption(
            f"🤖 **模型即時評估**：{wl_badge} ｜ 模型基準勝率 **{auto_mod_p*100:.1f}%** ｜ "
            f"預估期望值 **{est_ev:+.1f}%** ｜ 凱利風控建議注碼 **${kelly_rec_amt:.0f}**"
        )
    else:
        st.caption(f"📌 賽道標記：{wl_badge}（手動作戰 / 非模型庫內聯賽，請按本金 1%～3% 紀律控制注碼）")

    with sc3:
        st.write("")
        if st.button("✅ 確認模擬落注", type="primary", use_container_width=True, key="sim_submit_btn"):
            if s_mkt in ("亞洲讓球 (HDC)", "過關 / 混合過關 (All-Up)") and s_stake < 200:
                st.warning("⚠️ 溫馨提示：足智彩亞洲讓球及讓球過關最低注額通常為 $200，已為你照常記錄此模擬注單。")
            add_sim_bet(
                match_date=s_date,
                league=s_league,
                home_team=s_home,
                away_team=s_away,
                market_type=s_mkt,
                selection=s_sel,
                line=s_line,
                odds=s_odds,
                stake=s_stake,
                model_prob=round(auto_mod_p * 100, 1),
                ev_pct=round(est_ev, 2),
                is_whitelist=is_wl,
                notes=s_notes,
            )
            st.success(f"🎉 已成功記錄模擬注單：{s_home} vs {s_away}｜{s_sel} @ {s_odds:.2f}（注碼 ${s_stake:.0f}）")
            st.rerun()

    st.divider()

    # 3. 資金曲線與賽道績效分析
    if not settled_df.empty:
        st.markdown("### 📈 模擬買波累計盈虧資金曲線 (Cumulative P&L)")
        chart_df = settled_df.sort_values("bet_id").copy()
        chart_df["累計淨盈虧 (HKD)"] = chart_df["pnl"].cumsum()
        chart_df["模擬總資金 (HKD)"] = init_bankroll + chart_df["累計淨盈虧 (HKD)"]
        chart_df["注單編號"] = [f"#{i+1}" for i in range(len(chart_df))]
        st.line_chart(chart_df.set_index("注單編號")[["模擬總資金 (HKD)"]])

        with st.expander("📊 點擊展開：各玩法與主力白名單對比績效統計", expanded=False):
            grp_mkt = (
                settled_df.groupby("market_type")
                .agg(
                    注數=("bet_id", "count"),
                    總投入=("stake", "sum"),
                    淨盈虧=("pnl", "sum"),
                )
                .reset_index()
            )
            grp_mkt["ROI (%)"] = (grp_mkt["淨盈虧"] / grp_mkt["總投入"] * 100).round(2)
            grp_mkt.columns = ["足智彩玩法", "已結算注數", "總投入 (HKD)", "淨盈虧 (HKD)", "ROI (%)"]
            st.dataframe(grp_mkt, use_container_width=True, hide_index=True)

    # 4. 完整買波記錄簿與手動結算／刪除工具
    st.markdown("### 📋 模擬買波歷史注單總表")
    if bets_df.empty:
        st.info("目前尚未有任何模擬買波記錄，請在上方「模擬落注台」新增你的第一筆模擬注單！")
    else:
        filter_status = st.selectbox("篩選注單狀態", ["全部注單"] + STATUS_OPTIONS, key="sim_filter_status")
        disp_df = bets_df if filter_status == "全部注單" else bets_df[bets_df["status"] == filter_status]

        show_cols = pd.DataFrame({
            "注單ID": disp_df["bet_id"],
            "落注時間": disp_df["created_at"],
            "比賽日期": disp_df["match_date"],
            "聯賽": disp_df["league"],
            "對陣": disp_df["home_team"] + " vs " + disp_df["away_team"],
            "玩法": disp_df["market_type"],
            "選項": disp_df["selection"],
            "賠率": disp_df["odds"].map(lambda x: f"{x:.2f}"),
            "注碼": disp_df["stake"].map(lambda x: f"${x:,.0f}"),
            "預估EV": disp_df["ev_pct"].map(lambda x: f"{x:+.1f}%"),
            "完場比數": disp_df["result_score"],
            "狀態": disp_df["status"],
            "淨盈虧": disp_df["pnl"].map(lambda x: f"${x:+,.1f}"),
            "備註": disp_df["notes"],
        })
        st.dataframe(show_cols, use_container_width=True, hide_index=True)

        # 手動結算或刪除指定注單
        with st.expander("🛠️ 手動派彩結算 / 修改狀態 / 刪除誤輸入注單", expanded=True):
            act_c1, act_c2, act_c3, act_c4, act_c5 = st.columns([1.2, 1.5, 1.1, 1.1, 1.0])
            bet_ids = bets_df["bet_id"].tolist()
            target_id = act_c1.selectbox("選擇注單 ID", bet_ids, key="sim_edit_id")
            new_st = act_c2.selectbox("設定結算結果", STATUS_OPTIONS, index=1, key="sim_edit_status")
            sc_txt = act_c3.text_input("完場比數 (選填)", placeholder="2-1", key="sim_edit_score")
            with act_c4:
                st.write("")
                if st.button("💾 更新結算結果", use_container_width=True, key="sim_save_settle"):
                    settle_bet_manual(int(target_id), new_st, sc_txt)
                    st.success(f"注單 #{target_id} 已更新為「{new_st}」！")
                    st.rerun()
            with act_c5:
                st.write("")
                if st.button("🗑️ 刪除注單", use_container_width=True, key="sim_del_btn"):
                    delete_sim_bet(int(target_id))
                    st.warning(f"已刪除注單 #{target_id}。")
                    st.rerun()

    # 5. CSV 匯入還原區（確保雲端重啟後可隨時恢復歷史記錄）
    with st.expander("📤 從 CSV 備份檔匯入還原買波記錄"):
        up_csv = st.file_uploader("上傳先前匯出的 `hkjc_sim_bets_*.csv`", type=["csv"], key="sim_import_csv_file")
        if up_csv is not None:
            if st.button("確認匯入 CSV 記錄", key="sim_confirm_import"):
                try:
                    n_imp = import_sim_bets_csv(up_csv)
                    st.success(f"✅ 已成功匯入 **{n_imp}** 筆歷史模擬買波記錄！")
                    st.rerun()
                except Exception as e:
                    st.error(f"匯入失敗：{e}")