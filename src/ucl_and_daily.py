import sqlite3
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson
from model import (
    PoissonMatchPredictor,
    get_formation_matchup_adjustment,
    get_rest_fatigue_adjustment,
)

DB_PATH = Path("data/soccer.db")

LEAGUES = [
    "ENG-Premier League",
    "ENG-Championship",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
    "NED-Eredivisie",
    "POR-Primeira Liga",
    "JPN-J1 League",
    "KOR-K League 1",
]

# 跨聯賽歐戰相對實力係數（結合歐足聯近五年系數與 Opta 聯賽強度評級，以英超 1.00 為基準）
LEAGUE_STRENGTH = {
    "ENG-Premier League": 1.00,
    "ESP-La Liga":        0.96,
    "GER-Bundesliga":     0.95,
    "ITA-Serie A":        0.95,
    "FRA-Ligue 1":        0.91,
    "POR-Primeira Liga":  0.86,
    "NED-Eredivisie":     0.85,
    "ENG-Championship":   0.82,
    "JPN-J1 League":      0.78,
    "KOR-K League 1":     0.76,
}


class CrossLeagueUCLPredictor:
    def __init__(self):
        self.league_models = {}
        self.team_to_league = {}
        for lg in LEAGUES:
            try:
                m = PoissonMatchPredictor(league=lg, goal_weight=0.4, half_life_days=45.0)
                self.league_models[lg] = m
                for t in m.list_teams():
                    self.team_to_league[t] = lg
            except Exception as e:
                print(f"[提示] 跳過初始化聯賽 {lg}: {e}")

    def predict_ucl_match(
        self,
        home_team: str,
        away_team: str,
        neutral_venue: bool = False,
        home_rest_days: int = 4,
        away_rest_days: int = 4,
        home_formation: str = "標準 / 未知",
        away_formation: str = "標準 / 未知",
        injury_adj: dict = None,
        max_goals: int = 6,
    ) -> dict:
        if home_team not in self.team_to_league or away_team not in self.team_to_league:
            raise ValueError(f"找不到歐聯參賽隊伍所屬聯賽：{home_team} 或 {away_team}")

        h_lg = self.team_to_league[home_team]
        a_lg = self.team_to_league[away_team]
        h_mod = self.league_models[h_lg]
        a_mod = self.league_models[a_lg]

        h_coeff = LEAGUE_STRENGTH.get(h_lg, 0.90)
        a_coeff = LEAGUE_STRENGTH.get(a_lg, 0.90)

        # 1. 中立場模式：主客場攻守參數取雙向平均，並消除主場優勢係數 gamma
        if neutral_venue:
            raw_h_att = 0.5 * (h_mod.home_alpha[home_team] + h_mod.away_alpha[home_team])
            raw_h_def = 0.5 * (h_mod.home_beta[home_team] + h_mod.away_beta[home_team])
            raw_a_att = 0.5 * (a_mod.home_alpha[away_team] + a_mod.away_alpha[away_team])
            raw_a_def = 0.5 * (a_mod.home_beta[away_team] + a_mod.away_beta[away_team])
            gamma_eff = 1.00
        else:
            raw_h_att = h_mod.home_alpha[home_team]
            raw_h_def = h_mod.home_beta[home_team]
            raw_a_att = a_mod.away_alpha[away_team]
            raw_a_def = a_mod.away_beta[away_team]
            # 歐戰主場氣氛加成略高於本土聯賽 (+2%)
            gamma_eff = ((h_mod.gamma + a_mod.gamma) / 2.0) * 1.02

        h_att = raw_h_att * h_coeff
        h_def = raw_h_def / h_coeff
        a_att = raw_a_att * a_coeff
        a_def = raw_a_def / a_coeff

        base_lam_h = h_att * a_def * gamma_eff
        base_mu_a = a_att * h_def

        # 2. 體能休息天數、陣型對碰與歐聯傷停修正
        rest_adj = get_rest_fatigue_adjustment(home_rest_days, away_rest_days)
        form_adj = get_formation_matchup_adjustment(home_formation, away_formation)
        inj = injury_adj or {"h_att_mult": 1.0, "h_def_mult": 1.0, "a_att_mult": 1.0, "a_def_mult": 1.0}
        h_inj_mult = inj.get("h_att_mult", 1.0) * inj.get("a_def_mult", 1.0)
        a_inj_mult = inj.get("a_att_mult", 1.0) * inj.get("h_def_mult", 1.0)

        lam_home = base_lam_h * rest_adj["h_lambda_mult"] * form_adj["h_lambda_mult"] * h_inj_mult
        mu_away = base_mu_a * rest_adj["a_lambda_mult"] * form_adj["a_lambda_mult"] * a_inj_mult

        # 3. PPDA 高位逼搶對碰節奏微調
        h_ppda = getattr(h_mod, "team_ppda", {}).get(home_team, 11.5)
        a_ppda = getattr(a_mod, "team_ppda", {}).get(away_team, 11.5)
        if h_ppda < 9.5 and a_ppda < 9.5:
            lam_home *= 1.03
            mu_away *= 1.03
            press_note = f"🔥 雙高位逼搶對攻（主 PPDA {h_ppda} vs 客 PPDA {a_ppda}，節奏加快有利大球）"
        elif h_ppda > 13.5 and a_ppda > 13.5:
            lam_home *= 0.96
            mu_away *= 0.96
            press_note = f"🛡️ 雙低位保守陣地戰（主 PPDA {h_ppda} vs 客 PPDA {a_ppda}，節奏偏慢）"
        else:
            press_note = f"⚖️ 壓迫強度均衡（主 PPDA {h_ppda} vs 客 PPDA {a_ppda}）"

        # 4. Dixon-Coles + 負二項厚尾混合波膽矩陣（高入球預期對決自動提高厚尾權重）
        avg_rho = (h_mod.rho + a_mod.rho) / 2.0
        dc_mat = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                p_base = poisson.pmf(i, lam_home) * poisson.pmf(j, mu_away)
                tau = PoissonMatchPredictor._tau(i, j, lam_home, mu_away, avg_rho)
                dc_mat[i, j] = p_base * max(0.0, tau)
        dc_mat /= dc_mat.sum()

        nb_weight = 0.40 if (lam_home + mu_away) > 3.05 else 0.18
        nb_mat = PoissonMatchPredictor._nbinom_matrix(lam_home, mu_away, max_goals=max_goals, dispersion=0.18)
        matrix = (1.0 - nb_weight) * dc_mat + nb_weight * nb_mat
        matrix /= matrix.sum()

        # 5. 跨聯賽預期角球計算
        h_st = h_mod.team_stats[home_team]
        a_st = a_mod.team_stats[away_team]
        avg_hc = 0.5 * (h_mod.avg_home_corners + a_mod.avg_home_corners)
        avg_ac = 0.5 * (h_mod.avg_away_corners + a_mod.avg_away_corners)

        if neutral_venue:
            c_att_h = 0.5 * (h_st["home_cor_att"] + h_st["away_cor_att"])
            c_def_h = 0.5 * (h_st["home_cor_def"] + h_st["away_cor_def"])
            c_att_a = 0.5 * (a_st["home_cor_att"] + a_st["away_cor_att"])
            c_def_a = 0.5 * (a_st["home_cor_def"] + a_st["away_cor_def"])
            base_c = 0.5 * (avg_hc + avg_ac)
            exp_home_corners = c_att_h * c_def_a * base_c * form_adj["cor_mult"]
            exp_away_corners = c_att_a * c_def_h * base_c * form_adj["cor_mult"]
        else:
            exp_home_corners = h_st["home_cor_att"] * a_st["away_cor_def"] * avg_hc * form_adj["cor_mult"]
            exp_away_corners = a_st["away_cor_att"] * h_st["home_cor_def"] * avg_ac * form_adj["cor_mult"]
        exp_total_corners = exp_home_corners + exp_away_corners

        # 6. 跨聯賽標準化 Elo（按聯賽強弱係數折算基準分）
        adj_h_elo = round(h_mod.elo_ratings[home_team] + (h_coeff - 1.0) * 600.0, 0)
        adj_a_elo = round(a_mod.elo_ratings[away_team] + (a_coeff - 1.0) * 600.0, 0)

        prob_home = float(np.sum(np.tril(matrix, -1)))
        prob_draw = float(np.trace(matrix))
        prob_away = float(np.sum(np.triu(matrix, 1)))
        prob_over_25 = PoissonMatchPredictor.calc_total_over_prob(matrix, 2.5)

        scores = [(f"{i}-{j}", float(matrix[i, j])) for i in range(max_goals) for j in range(max_goals)]
        scores.sort(key=lambda x: x[1], reverse=True)

        fair_odds = {
            "主勝 (1)": round(1.0 / prob_home, 2) if prob_home > 0 else None,
            "和局 (X)": round(1.0 / prob_draw, 2) if prob_draw > 0 else None,
            "客勝 (2)": round(1.0 / prob_away, 2) if prob_away > 0 else None,
            "入球大 (2.5)": round(1.0 / prob_over_25, 2) if prob_over_25 > 0 else None,
            "入球細 (2.5)": round(1.0 / (1.0 - prob_over_25), 2) if (1.0 - prob_over_25) > 0 else None,
        }

        return {
            "match": f"{home_team} ({h_lg}) vs {away_team} ({a_lg})",
            "home_league": h_lg,
            "away_league": a_lg,
            "home_elo": adj_h_elo,
            "away_elo": adj_a_elo,
            "home_ppda": h_ppda,
            "away_ppda": a_ppda,
            "press_note": press_note,
            "sot_efficiency_diff": round(h_st["sot_ratio"] - a_st["sot_ratio"], 4),
            "dist_mode": f"跨聯賽 Dixon-Coles ({int((1-nb_weight)*100)}%) + 負二項厚尾 ({int(nb_weight*100)}%)",
            "matrix": matrix,
            "base_lambda_home": round(base_lam_h, 2),
            "base_lambda_away": round(base_mu_a, 2),
            "lambda_home": round(lam_home, 2),
            "lambda_away": round(mu_away, 2),
            "exp_home_corners": round(exp_home_corners, 1),
            "exp_away_corners": round(exp_away_corners, 1),
            "exp_total_corners": round(exp_total_corners, 1),
            "prob_home": prob_home,
            "prob_draw": prob_draw,
            "prob_away": prob_away,
            "prob_over_25": prob_over_25,
            "prob_under_25": 1.0 - prob_over_25,
            "fair_odds": fair_odds,
            "top_scores": scores[:5],
            "rest_note": rest_adj["rest_note"],
            "tactical_note": form_adj["tactical_note"],
        }


if __name__ == "__main__":
    ucl = CrossLeagueUCLPredictor()
    print(f"已成功載入 {len(ucl.league_models)} 個聯賽模型，共 {len(ucl.team_to_league)} 支球隊。")