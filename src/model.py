import sqlite3
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from pathlib import Path

DB_PATH = Path("data/soccer.db")


class PoissonMatchPredictor:
    def __init__(
        self,
        league: str = "ENG-Premier League",
        goal_weight: float = 0.4,
        half_life_days: float = 45.0,
        shrinkage_prior: float = 1.5,
        lookback_days: int = 365,
    ):
        """
        整合版 Dixon-Coles MLE 核心引擎：
        - 結合真實進球與 Synthetic xG，並套用時間衰減權重 w = exp(-ξ * Δt)
        - 使用 scipy.optimize.minimize (L-BFGS-B) 聯立求解經過對手強度校正與貝氏收縮的進攻強度 α 與防守弱度 β
        - 同時保留角球攻防強度擬合與全玩法盤口精算介面
        """
        self.league = league
        self.goal_weight = goal_weight
        self.xg_weight = 1.0 - goal_weight
        self.xi = np.log(2) / half_life_days
        self.shrinkage_prior = shrinkage_prior
        self.lookback_days = lookback_days

        self.teams = []
        self.alpha = {}
        self.beta = {}
        self.gamma = 1.15
        self.rho = -0.10
        self.team_stats = {}
        self.avg_home_corners = 5.2
        self.avg_away_corners = 4.5
        self._fit()

    @staticmethod
    def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
        if x == 0 and y == 0:
            return 1.0 - (lam * mu * rho)
        elif x == 0 and y == 1:
            return 1.0 + (lam * rho)
        elif x == 1 and y == 0:
            return 1.0 + (mu * rho)
        elif x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    def _fit(self):
        conn = sqlite3.connect(DB_PATH)
        query = """
        SELECT m.date, m.home_team, m.away_team, m.home_goals, m.away_goals,
               s.home_xg, s.away_xg, s.home_corners, s.away_corners
        FROM matches m
        LEFT JOIN match_stats s ON m.match_id = s.match_id
        WHERE m.league = ? AND m.status = 'finished'
        """
        df = pd.read_sql_query(query, conn, params=(self.league,))
        conn.close()

        if df.empty:
            raise ValueError(f"資料庫中找不到聯賽 {self.league} 的已完賽數據，請先執行 scraper.py")

        df["dt"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
        max_dt = df["dt"].max()
        df["days_ago"] = (max_dt - df["dt"]).dt.days.fillna(0).clip(lower=0)
        df = df[df["days_ago"] <= self.lookback_days].copy()
        df["weight"] = np.exp(-self.xi * df["days_ago"])

        df["y_home"] = np.where(
            df["home_xg"].notna(),
            self.goal_weight * df["home_goals"] + self.xg_weight * df["home_xg"],
            df["home_goals"],
        )
        df["y_away"] = np.where(
            df["away_xg"].notna(),
            self.goal_weight * df["away_goals"] + self.xg_weight * df["away_xg"],
            df["away_goals"],
        )

        self.teams = sorted(list(set(df["home_team"]).union(set(df["away_team"]))))
        n_teams = len(self.teams)
        team_idx = {t: i for i, t in enumerate(self.teams)}

        h_idx = df["home_team"].map(team_idx).values
        a_idx = df["away_team"].map(team_idx).values
        y_h = df["y_home"].values
        y_a = df["y_away"].values
        g_h = df["home_goals"].astype(int).values
        g_a = df["away_goals"].astype(int).values
        weights = df["weight"].values

        # 1. Dixon-Coles MLE 聯立求解 α, β, γ, ρ
        def neg_log_likelihood(params):
            log_alpha = params[:n_teams]
            log_beta = params[n_teams : 2 * n_teams]
            log_gamma = params[2 * n_teams]
            rho = params[2 * n_teams + 1]

            alpha = np.exp(log_alpha)
            beta = np.exp(log_beta)
            gamma = np.exp(log_gamma)

            lam = alpha[h_idx] * beta[a_idx] * gamma
            mu = alpha[a_idx] * beta[h_idx]

            ll_poisson = weights * (y_h * np.log(lam + 1e-12) - lam + y_a * np.log(mu + 1e-12) - mu)
            tau_vals = np.array([
                max(1e-6, self._tau(gh, ga, l, m, rho))
                for gh, ga, l, m in zip(g_h, g_a, lam, mu)
            ])
            ll_total = np.sum(ll_poisson + weights * np.log(tau_vals))
            reg_penalty = self.shrinkage_prior * (np.sum(log_alpha ** 2) + np.sum(log_beta ** 2))
            return -ll_total + reg_penalty

        init_params = np.zeros(2 * n_teams + 2)
        init_params[2 * n_teams] = 0.15
        init_params[2 * n_teams + 1] = -0.08
        bounds = [(-2.5, 2.5)] * (2 * n_teams) + [(-0.5, 1.0), (-0.35, 0.35)]
        res = minimize(neg_log_likelihood, init_params, method="L-BFGS-B", bounds=bounds)

        opt_alpha = np.exp(res.x[:n_teams])
        opt_beta = np.exp(res.x[n_teams : 2 * n_teams])
        scale = np.mean(opt_alpha)
        opt_alpha /= scale
        opt_beta *= scale

        self.alpha = {t: float(opt_alpha[i]) for i, t in enumerate(self.teams)}
        self.beta = {t: float(opt_beta[i]) for i, t in enumerate(self.teams)}
        self.gamma = float(np.exp(res.x[2 * n_teams]))
        self.rho = float(res.x[2 * n_teams + 1])

        # 2. 擬合角球攻防強度 & 同步建立相容於歐聯模組的 team_stats 字典
        cor_df = df.dropna(subset=["home_corners", "away_corners"])
        if not cor_df.empty:
            cor_w = cor_df["weight"].sum()
            self.avg_home_corners = (cor_df["home_corners"] * cor_df["weight"]).sum() / cor_w
            self.avg_away_corners = (cor_df["away_corners"] * cor_df["weight"]).sum() / cor_w

        prior_cor = 2.0
        for t in self.teams:
            h_g = df[df["home_team"] == t]
            a_g = df[df["away_team"] == t]
            h_c = h_g.dropna(subset=["home_corners", "away_corners"])
            a_c = a_g.dropna(subset=["home_corners", "away_corners"])
            hcw, acw = h_c["weight"].sum(), a_c["weight"].sum()

            h_cor_att = ((h_c["home_corners"] * h_c["weight"]).sum() + prior_cor * self.avg_home_corners) / ((hcw + prior_cor) * self.avg_home_corners)
            h_cor_def = ((h_c["away_corners"] * h_c["weight"]).sum() + prior_cor * self.avg_away_corners) / ((hcw + prior_cor) * self.avg_away_corners)
            a_cor_att = ((a_c["away_corners"] * a_c["weight"]).sum() + prior_cor * self.avg_away_corners) / ((acw + prior_cor) * self.avg_away_corners)
            a_cor_def = ((a_c["home_corners"] * a_c["weight"]).sum() + prior_cor * self.avg_home_corners) / ((acw + prior_cor) * self.avg_home_corners)

            # 將 MLE 求解的 α 與 β 映射到 team_stats，供 CrossLeagueUCLPredictor 無縫調用
            self.team_stats[t] = {
                "home_attack": self.alpha[t],
                "home_defense": self.beta[t],
                "away_attack": self.alpha[t],
                "away_defense": self.beta[t],
                "home_cor_att": h_cor_att,
                "home_cor_def": h_cor_def,
                "away_cor_att": a_cor_att,
                "away_cor_def": a_cor_def,
                "matches_played": len(h_g) + len(a_g),
            }

    def list_teams(self):
        return self.teams

    def get_team_ratings(self) -> pd.DataFrame:
        """輸出全聯賽經過對手強度校正與貝氏收縮的戰力評級表"""
        rows = []
        for t in self.teams:
            a, b = self.alpha[t], self.beta[t]
            rows.append({
                "球隊": t,
                "進攻強度 (α)": round(a, 3),
                "防守弱度 (β)": round(b, 3),
                "綜合淨戰力 (α/β)": round(a / b, 3),
            })
        return pd.DataFrame(rows).sort_values("綜合淨戰力 (α/β)", ascending=False).reset_index(drop=True)

    def get_score_matrix(self, home_team: str, away_team: str, max_goals: int = 6) -> tuple:
        if home_team not in self.alpha or away_team not in self.alpha:
            raise ValueError(f"找不到球隊：{home_team} 或 {away_team}")

        lam_home = self.alpha[home_team] * self.beta[away_team] * self.gamma
        mu_away = self.alpha[away_team] * self.beta[home_team]

        matrix = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                p_base = poisson.pmf(i, lam_home) * poisson.pmf(j, mu_away)
                matrix[i, j] = p_base * max(0.0, self._tau(i, j, lam_home, mu_away, self.rho))

        matrix /= matrix.sum()
        return matrix, lam_home, mu_away

    def predict_match(self, home_team: str, away_team: str, max_goals: int = 6) -> dict:
        matrix, lam_home, mu_away = self.get_score_matrix(home_team, away_team, max_goals)
        h_stats = self.team_stats[home_team]
        a_stats = self.team_stats[away_team]

        exp_home_corners = h_stats["home_cor_att"] * a_stats["away_cor_def"] * self.avg_home_corners
        exp_away_corners = a_stats["away_cor_att"] * h_stats["home_cor_def"] * self.avg_away_corners
        exp_total_corners = exp_home_corners + exp_away_corners

        prob_home = float(np.sum(np.tril(matrix, -1)))
        prob_draw = float(np.trace(matrix))
        prob_away = float(np.sum(np.triu(matrix, 1)))
        prob_over_25 = self.calc_total_over_prob(matrix, 2.5)

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
            "match": f"{home_team} vs {away_team}",
            "matrix": matrix,
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
        }

    @staticmethod
    def calc_total_over_prob(matrix: np.ndarray, line: float) -> float:
        max_g = matrix.shape[0]
        return float(sum(matrix[i, j] for i in range(max_g) for j in range(max_g) if (i + j) > line))

    @staticmethod
    def calc_corner_over_prob(exp_total_corners: float, line: float) -> float:
        return float(1.0 - poisson.cdf(int(np.floor(line)), exp_total_corners))

    @staticmethod
    def evaluate_asian_handicap_ev(matrix: np.ndarray, line: float, odd: float, side: str = "home") -> dict:
        max_g = matrix.shape[0]
        ev = 0.0
        p_win_full, p_win_half, p_push = 0.0, 0.0, 0.0

        for i in range(max_g):
            for j in range(max_g):
                p = matrix[i, j]
                diff = (i - j + line) if side == "home" else (j - i - line)

                if diff >= 0.5:
                    ret = odd - 1.0
                    p_win_full += p
                elif np.isclose(diff, 0.25):
                    ret = (odd - 1.0) * 0.5
                    p_win_half += p
                elif np.isclose(diff, 0.0):
                    ret = 0.0
                    p_push += p
                elif np.isclose(diff, -0.25):
                    ret = -0.5
                else:
                    ret = -1.0
                ev += p * ret

        return {
            "ev": float(ev),
            "p_win_full": p_win_full,
            "p_win_half": p_win_half,
            "p_push": p_push,
        }

    @staticmethod
    def evaluate_hhad_probs(matrix: np.ndarray, handicap: int) -> dict:
        max_g = matrix.shape[0]
        p_h, p_d, p_a = 0.0, 0.0, 0.0
        for i in range(max_g):
            for j in range(max_g):
                adj_diff = (i + handicap) - j
                if adj_diff > 0:
                    p_h += matrix[i, j]
                elif adj_diff == 0:
                    p_d += matrix[i, j]
                else:
                    p_a += matrix[i, j]
        return {"讓球主勝": p_h, "讓球和": p_d, "讓球客勝": p_a}