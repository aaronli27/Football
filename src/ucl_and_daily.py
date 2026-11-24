import sqlite3
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson
from model import PoissonMatchPredictor

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

# 跨聯賽歐戰相對實力係數（以英超 1.00 為基準）
LEAGUE_STRENGTH = {
    "ENG-Premier League": 1.00,
    "ESP-La Liga":        0.97,
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

    def predict_ucl_match(self, home_team: str, away_team: str, max_goals: int = 6) -> dict:
        if home_team not in self.team_to_league or away_team not in self.team_to_league:
            raise ValueError(f"找不到歐聯參賽隊伍所屬聯賽：{home_team} 或 {away_team}")

        h_lg = self.team_to_league[home_team]
        a_lg = self.team_to_league[away_team]
        h_mod = self.league_models[h_lg]
        a_mod = self.league_models[a_lg]

        h_coeff = LEAGUE_STRENGTH.get(h_lg, 0.90)
        a_coeff = LEAGUE_STRENGTH.get(a_lg, 0.90)

        h_att = h_mod.home_alpha[home_team] * h_coeff
        h_def = h_mod.home_beta[home_team] / h_coeff
        a_att = a_mod.away_alpha[away_team] * a_coeff
        a_def = a_mod.away_beta[away_team] / a_coeff

        avg_gamma = (h_mod.gamma + a_mod.gamma) / 2.0
        avg_rho = (h_mod.rho + a_mod.rho) / 2.0

        lam_home = h_att * a_def * avg_gamma
        mu_away = a_att * h_def

        matrix = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                p_base = poisson.pmf(i, lam_home) * poisson.pmf(j, mu_away)
                tau = PoissonMatchPredictor._tau(i, j, lam_home, mu_away, avg_rho)
                matrix[i, j] = p_base * max(0.0, tau)
        matrix /= matrix.sum()

        prob_home = float(np.sum(np.tril(matrix, -1)))
        prob_draw = float(np.trace(matrix))
        prob_away = float(np.sum(np.triu(matrix, 1)))
        prob_over_25 = PoissonMatchPredictor.calc_total_over_prob(matrix, 2.5)

        scores = [(f"{i}-{j}", float(matrix[i, j])) for i in range(max_goals) for j in range(max_goals)]
        scores.sort(key=lambda x: x[1], reverse=True)

        return {
            "match": f"{home_team} ({h_lg}) vs {away_team} ({a_lg})",
            "matrix": matrix,
            "lambda_home": round(lam_home, 2),
            "lambda_away": round(mu_away, 2),
            "prob_home": prob_home,
            "prob_draw": prob_draw,
            "prob_away": prob_away,
            "prob_over_25": prob_over_25,
            "top_scores": scores[:5],
        }


if __name__ == "__main__":
    ucl = CrossLeagueUCLPredictor()
    print(f"已成功載入 {len(ucl.league_models)} 個聯賽模型，共 {len(ucl.team_to_league)} 支球隊。")