import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom
from pathlib import Path

DB_PATH = Path("data/soccer.db")

# 分聯賽校準參數：根據聯賽戰術風格獨立設定「模型融合權重」、「負二項厚尾權重」與「客場進攻保守度」
LEAGUE_CALIBRATION = {
    "ENG-Premier League": {"mod_w_1x2": 0.35, "nb_w": 0.18, "away_damp": 1.00, "ou_mod_w": 0.30},
    "ESP-La Liga":        {"mod_w_1x2": 0.30, "nb_w": 0.12, "away_damp": 0.98, "ou_mod_w": 0.28},
    "GER-Bundesliga":     {"mod_w_1x2": 0.20, "nb_w": 0.48, "away_damp": 0.98, "ou_mod_w": 0.30},
    "ITA-Serie A":        {"mod_w_1x2": 0.15, "nb_w": 0.10, "away_damp": 0.93, "ou_mod_w": 0.25},
    "FRA-Ligue 1":        {"mod_w_1x2": 0.22, "nb_w": 0.15, "away_damp": 0.96, "ou_mod_w": 0.28},
}


def remove_vig_1x2(h_odd, d_odd, a_odd):
    inv_h, inv_d, inv_a = 1.0 / h_odd, 1.0 / d_odd, 1.0 / a_odd
    total = inv_h + inv_d + inv_a
    return inv_h / total, inv_d / total, inv_a / total


def remove_vig_two_way(o1, o2):
    inv1, inv2 = 1.0 / o1, 1.0 / o2
    total = inv1 + inv2
    return inv1 / total, inv2 / total


def compute_brier_and_logloss(probs_list: list, outcomes_list: list) -> tuple:
    probs = np.clip(np.array(probs_list), 1e-6, 1.0 - 1e-6)
    outcomes = np.array(outcomes_list)
    brier = float(np.mean(np.sum((probs - outcomes) ** 2, axis=1)))
    logloss = float(-np.mean(np.sum(outcomes * np.log(probs), axis=1)))
    return brier, logloss


def eval_ah_bet(matrix: np.ndarray, line: float, odd: float, side: str, hg: int, ag: int) -> tuple:
    """亞洲讓球盤 (AH) 期望值與實際盈虧精算（支援半球/一球雙盤口與走水）"""
    max_g = matrix.shape[0]
    ev = 0.0
    for i in range(max_g):
        for j in range(max_g):
            p = matrix[i, j]
            diff = (i - j + line) if side == "home" else (j - i - line)
            if diff >= 0.5:
                ret = odd - 1.0
            elif np.isclose(diff, 0.25):
                ret = (odd - 1.0) * 0.5
            elif np.isclose(diff, 0.0):
                ret = 0.0
            elif np.isclose(diff, -0.25):
                ret = -0.5
            else:
                ret = -1.0
            ev += p * ret

    actual_diff = (hg - ag + line) if side == "home" else (ag - hg - line)
    if actual_diff >= 0.5:
        profit = odd - 1.0
    elif np.isclose(actual_diff, 0.25):
        profit = (odd - 1.0) * 0.5
    elif np.isclose(actual_diff, 0.0):
        profit = 0.0
    elif np.isclose(actual_diff, -0.25):
        profit = -0.5
    else:
        profit = -1.0

    return float(ev), float(profit)


def eval_ou_bet_calibrated(
    matrix: np.ndarray,
    line: float,
    over_odd: float,
    under_odd: float,
    side: str,
    hg: int,
    ag: int,
    ou_mod_w: float = 0.28,
) -> tuple:
    """
    加入「市場去水貝氏先驗 + 混合模型」校準的多球線大小球精算引擎：
    - 先計算純模型在該盤口線下的期望收益率，轉換為模型隱含無抽水勝率
    - 再與莊家雙向去水勝率 (1 - ou_mod_w) 進行貝氏融合，徹底過濾偽正期望值
    """
    max_g = matrix.shape[0]
    p_win_full, p_win_half, p_push, p_loss_half, p_loss_full = 0.0, 0.0, 0.0, 0.0, 0.0

    for i in range(max_g):
        for j in range(max_g):
            p = matrix[i, j]
            tot = i + j
            diff = (tot - line) if side == "over" else (line - tot)
            if diff >= 0.5:
                p_win_full += p
            elif np.isclose(diff, 0.25):
                p_win_half += p
            elif np.isclose(diff, 0.0):
                p_push += p
            elif np.isclose(diff, -0.25):
                p_loss_half += p
            else:
                p_loss_full += p

    # 計算莊家去水隱含勝率（扣除走水機率後的有效勝負比例）
    mkt_o, mkt_u = remove_vig_two_way(over_odd, under_odd)
    mkt_side_prob = mkt_o if side == "over" else mkt_u
    target_odd = over_odd if side == "over" else under_odd

    # 模型有效勝率（折算贏半與輸半）
    eff_non_push = max(1e-5, 1.0 - p_push)
    mod_eff_win = (p_win_full + 0.5 * p_win_half - 0.5 * p_loss_half) / eff_non_push
    mod_eff_win = float(np.clip(mod_eff_win, 0.05, 0.95))

    # 貝氏融合：以市場去水為錨點，疊加模型方向修正
    cal_eff_win = (1.0 - ou_mod_w) * mkt_side_prob + ou_mod_w * mod_eff_win
    cal_ev = eff_non_push * (cal_eff_win * (target_odd - 1.0) - (1.0 - cal_eff_win))

    actual_tot = hg + ag
    actual_diff = (actual_tot - line) if side == "over" else (line - actual_tot)
    if actual_diff >= 0.5:
        profit = target_odd - 1.0
    elif np.isclose(actual_diff, 0.25):
        profit = (target_odd - 1.0) * 0.5
    elif np.isclose(actual_diff, 0.0):
        profit = 0.0
    elif np.isclose(actual_diff, -0.25):
        profit = -0.5
    else:
        profit = -1.0

    return float(cal_ev), float(profit)


def run_pro_backtest(league: str = "ENG-Premier League", warmup_matches: int = 45):
    cal_cfg = LEAGUE_CALIBRATION.get(
        league, {"mod_w_1x2": 0.25, "nb_w": 0.15, "away_damp": 0.98, "ou_mod_w": 0.28}
    )
    mod_w = cal_cfg["mod_w_1x2"]
    nb_w = cal_cfg["nb_w"]
    away_damp = cal_cfg["away_damp"]
    ou_mod_w = cal_cfg["ou_mod_w"]

    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT m.match_id, m.date, m.home_team, m.away_team, m.home_goals, m.away_goals,
           s.home_xg, s.away_xg, s.home_shots, s.away_shots, s.home_sot, s.away_sot,
           o.home_odd, o.draw_odd, o.away_odd, o.open_home_odd, o.open_away_odd,
           o.asian_line, o.home_ah_odd, o.away_ah_odd, o.ou_line, o.over_odd, o.under_odd
    FROM matches m
    LEFT JOIN match_stats s ON m.match_id = s.match_id
    INNER JOIN odds_history o ON m.match_id = o.match_id
    WHERE m.league = ? AND m.status = 'finished'
    """
    df = pd.read_sql_query(query, conn, params=(league,))
    conn.close()

    df["dt"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df = df.sort_values("dt").reset_index(drop=True)

    df["eff_home"] = np.where(df["home_xg"].notna(), 0.4 * df["home_goals"] + 0.6 * df["home_xg"], df["home_goals"])
    df["eff_away"] = np.where(df["away_xg"].notna(), 0.4 * df["away_goals"] + 0.6 * df["away_xg"], df["away_goals"])

    elos = {t: 1500.0 for t in set(df["home_team"]).union(set(df["away_team"]))}
    last_played = {}

    model_probs_1x2, market_probs_1x2, actual_outcomes_1x2 = [], [], []
    bets_1x2, bets_ah, bets_ou = [], [], []
    feature_rows = []
    prior = 4.5

    for idx, row in df.iterrows():
        h, a, dt = row["home_team"], row["away_team"], row["dt"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])

        h_rest = (dt - last_played[h]).days if h in last_played else 7
        a_rest = (dt - last_played[a]).days if a in last_played else 7
        rest_diff = h_rest - a_rest

        if idx >= warmup_matches:
            hist = df.iloc[:idx]
            avg_h = hist["eff_home"].mean()
            avg_a = hist["eff_away"].mean()

            h_games = hist[hist["home_team"] == h]
            a_games = hist[hist["away_team"] == a]

            elo_mult_h = 1.0 + (elos[h] - 1500.0) / 1400.0
            elo_mult_a = 1.0 + (elos[a] - 1500.0) / 1400.0

            h_att = (h_games["eff_home"].sum() + prior * avg_h * elo_mult_h) / ((len(h_games) + prior) * avg_h)
            a_def = (a_games["eff_home"].sum() + prior * avg_h / elo_mult_a) / ((len(a_games) + prior) * avg_h)
            a_att = (a_games["eff_away"].sum() + prior * avg_a * elo_mult_a) / ((len(a_games) + prior) * avg_a)
            h_def = (h_games["eff_away"].sum() + prior * avg_a / elo_mult_h) / ((len(h_games) + prior) * avg_a)

            rest_factor_h = 0.94 if h_rest <= 3 else 1.0
            rest_factor_a = 0.94 if a_rest <= 3 else 1.0

            lam_h = h_att * a_def * avg_h * rest_factor_h
            # 套用分聯賽客場保守度折減（例如意甲強隊作客領先後傾向收縮保勝）
            lam_a = a_att * h_def * avg_a * rest_factor_a * away_damp

            p_mat = np.outer([poisson.pmf(i, lam_h) for i in range(7)], [poisson.pmf(j, lam_a) for j in range(7)])
            r = 1.0 / 0.18
            nb_mat = np.outer(
                [nbinom.pmf(i, r, r / (r + lam_h)) for i in range(7)],
                [nbinom.pmf(j, r, r / (r + lam_a)) for j in range(7)],
            )
            mat = (1.0 - nb_w) * p_mat + nb_w * nb_mat
            mat /= mat.sum()

            p_h_mod = float(np.sum(np.tril(mat, -1)))
            p_d_mod = float(np.trace(mat))
            p_a_mod = float(np.sum(np.triu(mat, 1)))

            mkt_h, mkt_d, mkt_a = remove_vig_1x2(row["home_odd"], row["draw_odd"], row["away_odd"])
            # 套用分聯賽模型融合權重（英超 35%、西甲 30%、德甲 20%、意甲 15%）
            p_h = (1.0 - mod_w) * mkt_h + mod_w * p_h_mod
            p_d = (1.0 - mod_w) * mkt_d + mod_w * p_d_mod
            p_a = (1.0 - mod_w) * mkt_a + mod_w * p_a_mod

            model_probs_1x2.append([p_h, p_d, p_a])
            market_probs_1x2.append([mkt_h, mkt_d, mkt_a])
            actual_outcomes_1x2.append([1.0 if hg > ag else 0.0, 1.0 if hg == ag else 0.0, 1.0 if hg < ag else 0.0])

            vig = (1.0 / row["home_odd"] + 1.0 / row["draw_odd"] + 1.0 / row["away_odd"]) - 1.0
            feature_rows.append({
                "match_id": row["match_id"],
                "elo_diff": round(elos[h] - elos[a], 1),
                "xg_lam_diff": round(lam_h - lam_a, 3),
                "rest_days_diff": rest_diff,
                "market_overround": round(vig, 4),
                "dc_prob_home": round(p_h_mod, 4),
                "mkt_prob_home": round(mkt_h, 4),
                "residual_home": (1.0 if hg > ag else 0.0) - p_h_mod,
            })

            # 1. 主客和 (1X2) 與真實無抽水 CLV
            open_h = row["open_home_odd"] if pd.notna(row["open_home_odd"]) else row["home_odd"]
            open_a = row["open_away_odd"] if pd.notna(row["open_away_odd"]) else row["away_odd"]
            clv_h = (open_h / row["home_odd"]) - 1.0
            clv_a = (open_a / row["away_odd"]) - 1.0

            candidates_1x2 = [
                ("主勝", p_h, open_h, clv_h, hg > ag),
                ("和局", p_d, row["draw_odd"], 0.0, hg == ag),
                ("客勝", p_a, open_a, clv_a, hg < ag),
            ]
            best_1x2, best_ev_1x2 = None, 0.015
            for name, prob, bet_odd, clv_val, won in candidates_1x2:
                if pd.isna(bet_odd) or not (1.55 <= bet_odd <= 3.20):
                    continue
                ev = prob * bet_odd - 1.0
                if 0.015 < ev <= 0.14 and ev > best_ev_1x2:
                    best_ev_1x2 = ev
                    best_1x2 = (name, bet_odd, ev, clv_val, won)

            if best_1x2:
                name, bet_odd, ev, clv_val, won = best_1x2
                bets_1x2.append({
                    "pick": name,
                    "odd": bet_odd,
                    "ev": ev,
                    "clv": clv_val,
                    "won": won,
                    "profit": (bet_odd - 1.0) if won else -1.0,
                })

            # 2. 亞洲讓球盤 (AH)
            if pd.notna(row["asian_line"]) and pd.notna(row["home_ah_odd"]) and pd.notna(row["away_ah_odd"]):
                ah_l = float(row["asian_line"])
                h_ah_o, a_ah_o = float(row["home_ah_odd"]), float(row["away_ah_odd"])
                if 1.65 <= h_ah_o <= 2.25 and 1.65 <= a_ah_o <= 2.25:
                    ev_h_ah, prof_h_ah = eval_ah_bet(mat, ah_l, h_ah_o, "home", hg, ag)
                    ev_a_ah, prof_a_ah = eval_ah_bet(mat, ah_l, a_ah_o, "away", hg, ag)
                    if 0.02 < ev_h_ah <= 0.15 and ev_h_ah >= ev_a_ah:
                        bets_ah.append({"pick": f"主({ah_l:+g})", "ev": ev_h_ah, "won": prof_h_ah > 0, "profit": prof_h_ah})
                    elif 0.02 < ev_a_ah <= 0.15 and ev_a_ah > ev_h_ah:
                        bets_ah.append({"pick": f"客({-ah_l:+g})", "ev": ev_a_ah, "won": prof_a_ah > 0, "profit": prof_a_ah})

            # 3. 市場貝氏校準後的多球線大小球 (O/U)
            if pd.notna(row["over_odd"]) and pd.notna(row["under_odd"]):
                o_odd, u_odd = float(row["over_odd"]), float(row["under_odd"])
                ou_l = float(row["ou_line"]) if pd.notna(row["ou_line"]) else 2.5
                ou_vig = (1.0 / o_odd + 1.0 / u_odd) - 1.0

                # 若為動態非 2.5 球線門檻放寬至 2.0%，若為標準 2.5 球線則要求 >= 2.8% 且莊家抽水低於 6%
                min_ou_ev = 0.020 if not np.isclose(ou_l, 2.5) else 0.028

                if 1.70 <= o_odd <= 2.20 and 1.70 <= u_odd <= 2.20 and ou_vig <= 0.065:
                    ev_o, prof_o = eval_ou_bet_calibrated(mat, ou_l, o_odd, u_odd, "over", hg, ag, ou_mod_w=ou_mod_w)
                    ev_u, prof_u = eval_ou_bet_calibrated(mat, ou_l, o_odd, u_odd, "under", hg, ag, ou_mod_w=ou_mod_w)
                    if min_ou_ev < ev_o <= 0.12 and ev_o >= ev_u:
                        bets_ou.append({"pick": f"大{ou_l:g}", "ev": ev_o, "won": prof_o > 0, "profit": prof_o})
                    elif min_ou_ev < ev_u <= 0.12 and ev_u > ev_o:
                        bets_ou.append({"pick": f"細{ou_l:g}", "ev": ev_u, "won": prof_u > 0, "profit": prof_u})

        dr = (elos[h] + 65.0) - elos[a]
        exp_h = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        act_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        g_diff = abs(hg - ag)
        g_mult = 1.0 if g_diff <= 1 else (1.5 if g_diff == 2 else 1.8)
        delta = 24.0 * g_mult * (act_h - exp_h)
        elos[h] += delta
        elos[a] -= delta
        last_played[h] = dt
        last_played[a] = dt

    brier_mod, ll_mod = compute_brier_and_logloss(model_probs_1x2, actual_outcomes_1x2)
    brier_mkt, _ = compute_brier_and_logloss(market_probs_1x2, actual_outcomes_1x2)

    print(f"\n{'='*72}")
    print(f"【分聯賽校準回測報告】聯賽：{league} (1X2模型權重={int(mod_w*100)}% | 負二項={int(nb_w*100)}%)")
    print(f"1X2 概率校準度：混合模型 Brier = {brier_mod:.4f} (LogLoss={ll_mod:.4f}) vs 市場基準 Brier = {brier_mkt:.4f}")

    def summarize_market(m_name: str, bet_list: list, show_clv: bool = False):
        if not bet_list:
            print(f"  ▸ {m_name:<16}：嚴格風控過濾後未觸發偽正期望值注單")
            return {"league": league, "market": m_name, "bets": 0, "hit_rate": 0.0, "profit": 0.0, "roi": 0.0, "clv": 0.0}
        bdf = pd.DataFrame(bet_list)
        roi = (bdf["profit"].sum() / len(bdf)) * 100
        hr = bdf["won"].mean() * 100
        clv_str = f" | 真實 CLV: {bdf['clv'].mean()*100:+.2f}%" if show_clv else ""
        print(f"  ▸ {m_name:<16}：{len(bdf):>2} 注 | 勝率: {hr:>5.1f}% | 淨盈虧: {bdf['profit'].sum():>+6.2f} 單位 | ROI: {roi:>+6.2f}%{clv_str}")
        return {
            "league": league,
            "market": m_name,
            "bets": len(bdf),
            "hit_rate": round(hr, 1),
            "profit": round(bdf["profit"].sum(), 2),
            "roi": round(roi, 2),
            "clv": round(bdf["clv"].mean() * 100, 2) if show_clv else 0.0,
        }

    s_1x2 = summarize_market("主客和 (1X2)", bets_1x2, show_clv=True)
    s_ah = summarize_market("亞洲讓球 (AH)", bets_ah, show_clv=False)
    s_ou = summarize_market("校準大小球 (O/U)", bets_ou, show_clv=False)
    print(f"{'='*72}")

    return pd.DataFrame(feature_rows), [s_1x2, s_ah, s_ou]


if __name__ == "__main__":
    all_features = []
    all_summaries = []
    for lg in ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]:
        feat_df, sums = run_pro_backtest(league=lg, warmup_matches=45)
        feat_df["league"] = lg
        all_features.append(feat_df)
        all_summaries.extend(sums)

    combined_features = pd.concat(all_features, ignore_index=True)
    combined_features.to_csv("data/five_dim_features.csv", index=False)

    print("\n" + "=" * 72)
    print("【五大聯賽 × 三大玩法：校準後正期望值細分市場總覽】")
    sum_df = pd.DataFrame(all_summaries)
    sum_df = sum_df[sum_df["bets"] > 0].sort_values("roi", ascending=False).reset_index(drop=True)
    print(sum_df.to_string(index=False))
    print("=" * 72)
    print(f"五維量化特徵矩陣已更新至 data/five_dim_features.csv（共 {len(combined_features)} 筆樣本）。")