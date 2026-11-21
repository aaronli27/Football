import sqlite3
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson, nbinom
from pathlib import Path

DB_PATH = Path("data/soccer.db")

FORMATION_PROFILES = {
    "標準 / 未知": {"att": 1.00, "def": 1.00, "cor": 1.00, "desc": "無額外陣型修正"},
    "4-3-3 (高位逼搶/邊路推進)": {"att": 1.06, "def": 1.03, "cor": 1.08, "desc": "邊鋒拉開寬度，進攻與角球頻率高，身後易留反擊空間"},
    "4-2-3-1 (均衡控球/雙後腰)": {"att": 1.03, "def": 0.98, "cor": 1.04, "desc": "中場保護穩健，攻守最均衡"},
    "3-5-2 (三中衛/中路密集)": {"att": 1.02, "def": 0.96, "cor": 1.02, "desc": "中路人數優勢強，翼衛上下奔跑體能消耗大"},
    "3-4-3 (翼衛壓上/高節奏對攻)": {"att": 1.08, "def": 1.06, "cor": 1.10, "desc": "前場三人組與翼衛全線壓上，大開大合易出大球與角球"},
    "5-3-2 / 5-4-1 (低位死守/防反)": {"att": 0.90, "def": 0.88, "cor": 0.86, "desc": "禁區密集防守壓縮對手射門空間，自身反擊次數少"},
    "4-4-2 (雙前鋒/平行中場)": {"att": 1.01, "def": 1.01, "cor": 1.03, "desc": "兩翼傳中頻率高，中路面對三中場易被控球壓制"},
}


def get_formation_matchup_adjustment(home_form: str, away_form: str) -> dict:
    hp = FORMATION_PROFILES.get(home_form, FORMATION_PROFILES["標準 / 未知"])
    ap = FORMATION_PROFILES.get(away_form, FORMATION_PROFILES["標準 / 未知"])

    h_att_mult = hp["att"]
    a_att_mult = ap["att"]
    cor_mult = (hp["cor"] + ap["cor"]) / 2.0
    notes = []

    if "5-3-2" in away_form and ("4-3-3" in home_form or "4-2-3-1" in home_form):
        h_att_mult *= 0.94
        cor_mult *= 1.08
        notes.append("客隊擺出五後衛低位大巴，主隊圍攻易刷高角球但破門效率略降")
    if "4-4-2" in home_form and ("4-3-3" in away_form or "3-5-2" in away_form):
        a_att_mult *= 1.05
        notes.append("客隊三中場人數占優，對主隊平行 4-4-2 具備中路控球壓制力")
    if "3-4-3" in home_form and "3-4-3" in away_form:
        h_att_mult *= 1.04
        a_att_mult *= 1.04
        notes.append("雙方皆採三中衛翼衛壓上陣型，邊路對攻空間極大，利好入球大與角球大")
    if "3-5-2" in home_form and "4-3-3" in away_form:
        h_att_mult *= 1.03
        a_att_mult *= 1.03
        notes.append("主隊中路人數占優，但客隊雙邊鋒可打擊主隊翼衛身後肋部空檔")

    if not notes:
        notes.append(f"主隊 ({home_form.split()[0]})：{hp['desc']}；客隊 ({away_form.split()[0]})：{ap['desc']}")

    return {
        "h_lambda_mult": h_att_mult * ap["def"],
        "a_lambda_mult": a_att_mult * hp["def"],
        "cor_mult": cor_mult,
        "tactical_note": " ｜ ".join(notes),
    }


def get_rest_fatigue_adjustment(home_rest_days: int, away_rest_days: int) -> dict:
    def calc_factor(days: int):
        if days <= 3:
            return 0.92, 1.07, f"一周雙賽密集疲勞(休{days}天：進攻-8%、防守失球風險+7%)"
        elif days == 4:
            return 0.96, 1.03, f"短休調整(休4天：進攻-4%、防守風險+3%)"
        elif days >= 10:
            return 0.98, 1.00, f"長休復出(休{days}天：比賽節奏略慢-2%)"
        return 1.00, 1.00, f"體能充沛(休{days}天：正常基準)"

    h_att, h_def, h_note = calc_factor(home_rest_days)
    a_att, a_def, a_note = calc_factor(away_rest_days)

    return {
        "h_lambda_mult": h_att * a_def,
        "a_lambda_mult": a_att * h_def,
        "rest_note": f"主隊：{h_note} ｜ 客隊：{a_note}",
    }


class PoissonMatchPredictor:
    def __init__(
        self,
        league: str = "ENG-Premier League",
        goal_weight: float = 0.4,
        half_life_days: float = 45.0,
        shrinkage_prior: float = 1.5,
        lookback_days: int = 365,
    ):
        self.league = league
        self.goal_weight = goal_weight
        self.xg_weight = 1.0 - goal_weight
        self.xi = np.log(2) / half_life_days
        self.shrinkage_prior = shrinkage_prior
        self.lookback_days = lookback_days

        self.teams = []
        self.elo_ratings = {}
        self.home_alpha = {}
        self.home_beta = {}
        self.away_alpha = {}
        self.away_beta = {}
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

    def _compute_dynamic_elo(self, df: pd.DataFrame) -> dict:
        sorted_df = df.sort_values("dt").reset_index(drop=True)
        elos = {t: 1500.0 for t in self.teams}
        k_base = 24.0

        for _, row in sorted_df.iterrows():
            h, a = row["home_team"], row["away_team"]
            hg, ag = row["home_goals"], row["away_goals"]
            if pd.isna(hg) or pd.isna(ag):
                continue

            dr = (elos[h] + 65.0) - elos[a]
            exp_h = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
            actual_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            goal_diff = abs(int(hg) - int(ag))
            g_mult = 1.0 if goal_diff <= 1 else (1.5 if goal_diff == 2 else (1.75 + (goal_diff - 3) / 8.0))

            delta = k_base * g_mult * (actual_h - exp_h)
            elos[h] += delta
            elos[a] -= delta

        return elos

    def _fit(self):
        conn = sqlite3.connect(DB_PATH)
        query = """
        SELECT m.date, m.home_team, m.away_team, m.home_goals, m.away_goals,
               s.home_xg, s.away_xg, s.home_shots, s.away_shots, s.home_sot, s.away_sot,
               s.home_corners, s.away_corners
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

        self.teams = sorted(list(set(df["home_team"]).union(set(df["away_team"]))))
        self.elo_ratings = self._compute_dynamic_elo(df)

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

        n_teams = len(self.teams)
        team_idx = {t: i for i, t in enumerate(self.teams)}

        h_idx = df["home_team"].map(team_idx).values
        a_idx = df["away_team"].map(team_idx).values
        y_h = df["y_home"].values
        y_a = df["y_away"].values
        g_h = df["home_goals"].astype(int).values
        g_a = df["away_goals"].astype(int).values
        weights = df["weight"].values

        elo_prior_log = np.array([(self.elo_ratings[t] - 1500.0) / 700.0 for t in self.teams])

        def neg_log_likelihood(params):
            log_h_a = params[0:n_teams]
            log_h_b = params[n_teams : 2 * n_teams]
            log_a_a = params[2 * n_teams : 3 * n_teams]
            log_a_b = params[3 * n_teams : 4 * n_teams]
            log_gamma = params[4 * n_teams]
            rho = params[4 * n_teams + 1]

            h_a, h_b = np.exp(log_h_a), np.exp(log_h_b)
            a_a, a_b = np.exp(log_a_a), np.exp(log_a_b)
            gamma = np.exp(log_gamma)

            lam = h_a[h_idx] * a_b[a_idx] * gamma
            mu = a_a[a_idx] * h_b[h_idx]

            ll_poisson = weights * (y_h * np.log(lam + 1e-12) - lam + y_a * np.log(mu + 1e-12) - mu)
            tau_vals = np.array([
                max(1e-6, self._tau(gh, ga, l, m, rho))
                for gh, ga, l, m in zip(g_h, g_a, lam, mu)
            ])
            ll_total = np.sum(ll_poisson + weights * np.log(tau_vals))

            reg_elo_anchor = self.shrinkage_prior * (
                np.sum((log_h_a - 0.5 * elo_prior_log) ** 2)
                + np.sum((log_h_b + 0.5 * elo_prior_log) ** 2)
                + np.sum((log_a_a - 0.5 * elo_prior_log) ** 2)
                + np.sum((log_a_b + 0.5 * elo_prior_log) ** 2)
            )
            reg_split = (self.shrinkage_prior * 0.8) * (
                np.sum((log_h_a - log_a_a) ** 2) + np.sum((log_h_b - log_a_b) ** 2)
            )
            return -ll_total + reg_elo_anchor + reg_split

        init_params = np.zeros(4 * n_teams + 2)
        init_params[4 * n_teams] = 0.16
        init_params[4 * n_teams + 1] = -0.08
        bounds = [(-2.5, 2.5)] * (4 * n_teams) + [(-0.5, 1.0), (-0.35, 0.35)]
        res = minimize(neg_log_likelihood, init_params, method="L-BFGS-B", bounds=bounds)

        opt_ha = np.exp(res.x[0:n_teams])
        opt_hb = np.exp(res.x[n_teams : 2 * n_teams])
        opt_aa = np.exp(res.x[2 * n_teams : 3 * n_teams])
        opt_ab = np.exp(res.x[3 * n_teams : 4 * n_teams])

        scale_h = np.mean(opt_ha)
        opt_ha /= scale_h
        opt_ab *= scale_h

        scale_a = np.mean(opt_aa)
        opt_aa /= scale_a
        opt_hb *= scale_a

        self.home_alpha = {t: float(opt_ha[i]) for i, t in enumerate(self.teams)}
        self.home_beta = {t: float(opt_hb[i]) for i, t in enumerate(self.teams)}
        self.away_alpha = {t: float(opt_aa[i]) for i, t in enumerate(self.teams)}
        self.away_beta = {t: float(opt_ab[i]) for i, t in enumerate(self.teams)}
        self.gamma = float(np.exp(res.x[4 * n_teams]))
        self.rho = float(res.x[4 * n_teams + 1])

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

            # 計算該球隊加權射正轉化率 (Shots on Target / Total Shots)
            tot_sot = (h_g["home_sot"].fillna(0) * h_g["weight"]).sum() + (a_g["away_sot"].fillna(0) * a_g["weight"]).sum()
            tot_sh = (h_g["home_shots"].fillna(0) * h_g["weight"]).sum() + (a_g["away_shots"].fillna(0) * a_g["weight"]).sum()
            sot_ratio = float(tot_sot / tot_sh) if tot_sh > 0 else 0.33

            self.team_stats[t] = {
                "elo": round(self.elo_ratings[t], 1),
                "sot_ratio": round(sot_ratio, 4),
                "home_attack": self.home_alpha[t],
                "home_defense": self.home_beta[t],
                "away_attack": self.away_alpha[t],
                "away_defense": self.away_beta[t],
                "home_cor_att": h_cor_att,
                "home_cor_def": h_cor_def,
                "away_cor_att": a_cor_att,
                "away_cor_def": a_cor_def,
                "matches_played": len(h_g) + len(a_g),
            }

    def list_teams(self):
        return self.teams

    def get_team_ratings(self) -> pd.DataFrame:
        rows = []
        for t in self.teams:
            ha, hb = self.home_alpha[t], self.home_beta[t]
            aa, ab = self.away_alpha[t], self.away_beta[t]
            net = ((ha / hb) + (aa / ab)) / 2.0
            rows.append({
                "球隊": t,
                "動態 Elo": round(self.elo_ratings[t], 0),
                "射正率": f"{self.team_stats[t]['sot_ratio']*100:.1f}%",
                "主場攻/防 (α/β)": f"{ha:.2f} / {hb:.2f}",
                "客場攻/防 (α/β)": f"{aa:.2f} / {ab:.2f}",
                "綜合淨戰力": round(net, 3),
            })
        return pd.DataFrame(rows).sort_values("動態 Elo", ascending=False).reset_index(drop=True)

    @staticmethod
    def _nbinom_matrix(lam_h: float, mu_a: float, max_goals: int = 6, dispersion: float = 0.16) -> np.ndarray:
        r = 1.0 / dispersion
        p_h = r / (r + lam_h)
        p_a = r / (r + mu_a)
        h_probs = [nbinom.pmf(i, r, p_h) for i in range(max_goals)]
        a_probs = [nbinom.pmf(j, r, p_a) for j in range(max_goals)]
        mat = np.outer(h_probs, a_probs)
        return mat / mat.sum()

    def get_score_matrix(self, home_team: str, away_team: str, max_goals: int = 6) -> tuple:
        if home_team not in self.home_alpha or away_team not in self.away_alpha:
            raise ValueError(f"找不到球隊：{home_team} 或 {away_team}")

        lam_home = self.home_alpha[home_team] * self.away_beta[away_team] * self.gamma
        mu_away = self.away_alpha[away_team] * self.home_beta[home_team]

        dc_mat = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                p_base = poisson.pmf(i, lam_home) * poisson.pmf(j, mu_away)
                dc_mat[i, j] = p_base * max(0.0, self._tau(i, j, lam_home, mu_away, self.rho))
        dc_mat /= dc_mat.sum()

        nb_weight = 0.45 if (self.league == "GER-Bundesliga" or (lam_home + mu_away) > 3.10) else 0.15
        nb_mat = self._nbinom_matrix(lam_home, mu_away, max_goals=max_goals, dispersion=0.18)
        blended_matrix = (1.0 - nb_weight) * dc_mat + nb_weight * nb_mat
        blended_matrix /= blended_matrix.sum()

        return blended_matrix, lam_home, mu_away

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        home_rest_days: int = 6,
        away_rest_days: int = 6,
        home_formation: str = "標準 / 未知",
        away_formation: str = "標準 / 未知",
        max_goals: int = 6,
    ) -> dict:
        if home_team not in self.home_alpha or away_team not in self.away_alpha:
            raise ValueError(f"找不到球隊：{home_team} 或 {away_team}")

        base_lam_h = self.home_alpha[home_team] * self.away_beta[away_team] * self.gamma
        base_mu_a = self.away_alpha[away_team] * self.home_beta[home_team]

        rest_adj = get_rest_fatigue_adjustment(home_rest_days, away_rest_days)
        form_adj = get_formation_matchup_adjustment(home_formation, away_formation)

        lam_home = base_lam_h * rest_adj["h_lambda_mult"] * form_adj["h_lambda_mult"]
        mu_away = base_mu_a * rest_adj["a_lambda_mult"] * form_adj["a_lambda_mult"]

        dc_mat = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                p_base = poisson.pmf(i, lam_home) * poisson.pmf(j, mu_away)
                dc_mat[i, j] = p_base * max(0.0, self._tau(i, j, lam_home, mu_away, self.rho))
        dc_mat /= dc_mat.sum()

        is_high_var = (self.league == "GER-Bundesliga") or ((lam_home + mu_away) > 3.10)
        nb_weight = 0.45 if is_high_var else 0.15
        nb_mat = self._nbinom_matrix(lam_home, mu_away, max_goals=max_goals, dispersion=0.18)
        matrix = (1.0 - nb_weight) * dc_mat + nb_weight * nb_mat
        matrix /= matrix.sum()

        h_stats = self.team_stats[home_team]
        a_stats = self.team_stats[away_team]
        exp_home_corners = h_stats["home_cor_att"] * a_stats["away_cor_def"] * self.avg_home_corners * form_adj["cor_mult"]
        exp_away_corners = a_stats["away_cor_att"] * h_stats["home_cor_def"] * self.avg_away_corners * form_adj["cor_mult"]
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
            "home_elo": round(self.elo_ratings[home_team], 0),
            "away_elo": round(self.elo_ratings[away_team], 0),
            "sot_efficiency_diff": round(h_stats["sot_ratio"] - a_stats["sot_ratio"], 4),
            "dist_mode": f"Dixon-Coles ({int((1-nb_weight)*100)}%) + 負二項厚尾 ({int(nb_weight*100)}%)",
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
        return {"ev": float(ev), "p_win_full": p_win_full, "p_win_half": p_win_half, "p_push": p_push}

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


def analyze_in_play_match(
    minute: int,
    h_goals: int,
    a_goals: int,
    h_poss: float,
    h_shots: int,
    a_shots: int,
    h_sot: int,
    a_sot: int,
    h_corners: int,
    a_corners: int,
    h_attacks: int,
    a_attacks: int,
    h_dangerous: int,
    a_dangerous: int,
    h_big_chances: int,
    a_big_chances: int,
) -> dict:
    elapsed = max(1, min(90, minute))
    rem_mins = max(1, 90 - elapsed)
    late_boost = 1.15 if elapsed >= 65 else 1.02

    h_other_sot = max(0, h_sot - h_big_chances)
    h_off_shots = max(0, h_shots - h_sot)
    live_xg_h = (h_big_chances * 0.38) + (h_other_sot * 0.16) + (h_off_shots * 0.03) + (h_dangerous * 0.008)

    a_other_sot = max(0, a_sot - a_big_chances)
    a_off_shots = max(0, a_shots - a_sot)
    live_xg_a = (a_big_chances * 0.38) + (a_other_sot * 0.16) + (a_off_shots * 0.03) + (a_dangerous * 0.008)

    a_poss = 100.0 - h_poss
    h_mom = (h_dangerous * 1.5) + (h_shots * 3.0) + (h_sot * 6.0) + (h_corners * 2.5) + (h_big_chances * 12.0) + (h_poss * 0.3)
    a_mom = (a_dangerous * 1.5) + (a_shots * 3.0) + (a_sot * 6.0) + (a_corners * 2.5) + (a_big_chances * 12.0) + (a_poss * 0.3)
    total_mom = max(1e-5, h_mom + a_mom)
    h_dom_pct = (h_mom / total_mom) * 100.0
    a_dom_pct = 100.0 - h_dom_pct

    rate_h = (live_xg_h / elapsed) * late_boost
    rate_a = (live_xg_a / elapsed) * late_boost

    if h_goals <= a_goals and h_dom_pct >= 62.0:
        rate_h *= 1.15
    if a_goals <= h_goals and a_dom_pct >= 62.0:
        rate_a *= 1.15

    rem_lam_h = rate_h * rem_mins
    rem_lam_a = rate_a * rem_mins
    rem_total_goals = rem_lam_h + rem_lam_a

    cor_rate = ((h_corners + a_corners) / elapsed) * (1.12 if elapsed >= 65 else 1.0)
    rem_corners = cor_rate * rem_mins

    mat = np.outer([poisson.pmf(i, rem_lam_h) for i in range(5)], [poisson.pmf(j, rem_lam_a) for j in range(5)])
    p_next_goal_any = 1.0 - float(mat[0, 0])
    p_rem_over_05 = p_next_goal_any
    p_rem_over_15 = 1.0 - float(mat[0, 0] + mat[1, 0] + mat[0, 1])

    p_final_h, p_final_d, p_final_a = 0.0, 0.0, 0.0
    for i in range(5):
        for j in range(5):
            fh, fa = h_goals + i, a_goals + j
            if fh > fa:
                p_final_h += mat[i, j]
            elif fh == fa:
                p_final_d += mat[i, j]
            else:
                p_final_a += mat[i, j]

    signals = []
    if h_dom_pct >= 65.0 and (live_xg_h - live_xg_a) >= 0.45:
        if h_goals <= a_goals:
            signals.append(f"🔥 **強烈建議：走地主隊讓球 / 主隊下一球**（主隊場上壓制力達 {h_dom_pct:.0f}%，Live xG {live_xg_h:.2f} vs {live_xg_a:.2f} 卻未領先，存在明顯『得勢未得分』價值）")
        else:
            signals.append(f"✅ **建議：走地主隊讓球**（主隊全面掌控局面，動能佔比 {h_dom_pct:.0f}%，客隊反撲威脅極低）")
    elif a_dom_pct >= 65.0 and (live_xg_a - live_xg_h) >= 0.45:
        if a_goals <= h_goals:
            signals.append(f"🔥 **強烈建議：走地客隊讓球 / 客隊下一球**（客隊反客為主壓制力達 {a_dom_pct:.0f}%，Live xG {live_xg_a:.2f} vs {live_xg_h:.2f}，具備極高追平或反超動能）")
        else:
            signals.append(f"✅ **建議：走地客隊讓球**（客隊攻勢完全主導比賽，動能佔比 {a_dom_pct:.0f}%）")
    else:
        signals.append("⚖️ **讓球方向：雙方攻守勢均力敵**（場上動能接近，不建議強追讓球方，優先考慮受讓方或轉戰入球/角球盤）")

    if rem_total_goals >= 1.15 and p_rem_over_05 >= 0.68:
        signals.append(f"⚽ **入球大細：偏好多入球（追走地大球）**（剩餘時間預期再入 **{rem_total_goals:.2f}** 球，再出現至少 1 球機率高達 **{p_rem_over_05*100:.1f}%**，至少 2 球機率 **{p_rem_over_15*100:.1f}%**）")
    elif rem_total_goals <= 0.55 and elapsed >= 55:
        signals.append(f"🧊 **入球大細：偏好少入球（走地細球）**（雙方禁區創造力低迷，剩餘時間預期入球僅 **{rem_total_goals:.2f}** 球，無球收場機率高）")
    else:
        signals.append(f"📊 **入球大細：中性節奏**（剩餘時間預期再入 **{rem_total_goals:.2f}** 球，再入一球機率 **{p_rem_over_05*100:.1f}%**，建議等高水再切入）")

    total_exp_corners = h_corners + a_corners + rem_corners
    if (h_dangerous + a_dangerous) / elapsed >= 1.15 and rem_corners >= 3.2:
        signals.append(f"🚩 **角球方向：看好走地角球大**（危險進攻頻率極高，預計全場總角球達 **{total_exp_corners:.1f}** 球，剩餘時間預計再刷 **{rem_corners:.1f}** 個角球）")
    elif rem_corners <= 2.0 and elapsed >= 60:
        signals.append(f"🛑 **角球方向：偏好看好角球細**（邊路推進停滯，預計全場總角球僅 **{total_exp_corners:.1f}** 球）")

    return {
        "live_xg_h": round(live_xg_h, 2),
        "live_xg_a": round(live_xg_a, 2),
        "h_dom_pct": round(h_dom_pct, 1),
        "a_dom_pct": round(a_dom_pct, 1),
        "rem_lam_h": round(rem_lam_h, 2),
        "rem_lam_a": round(rem_lam_a, 2),
        "rem_total_goals": round(rem_total_goals, 2),
        "p_rem_over_05": p_rem_over_05,
        "p_rem_over_15": p_rem_over_15,
        "rem_corners": round(rem_corners, 1),
        "total_exp_corners": round(total_exp_corners, 1),
        "p_final_h": p_final_h,
        "p_final_d": p_final_d,
        "p_final_a": p_final_a,
        "signals": signals,
    }