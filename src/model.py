import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import poisson
from pathlib import Path

DB_PATH = Path("data/soccer.db")

class PoissonMatchPredictor:
    def __init__(self, league: str = "ENG-Premier League", goal_weight: float = 0.7, half_life_days: float = 45.0):
        """
        half_life_days: 時間衰減半衰期（預設 45 天，即 45 天前的比賽權重為今日的一半）
        """
        self.league = league
        self.goal_weight = goal_weight
        self.xg_weight = 1.0 - goal_weight
        self.half_life_days = half_life_days
        self.team_stats = {}
        self.avg_home_goals = 0.0
        self.avg_away_goals = 0.0
        self._fit()

    def _fit(self):
        conn = sqlite3.connect(DB_PATH)
        query = """
        SELECT m.date, m.home_team, m.away_team, m.home_goals, m.away_goals,
               s.home_xg, s.away_xg
        FROM matches m
        LEFT JOIN match_stats s ON m.match_id = s.match_id
        WHERE m.league = ? AND m.status = 'finished'
        """
        df = pd.read_sql_query(query, conn, params=(self.league,))
        conn.close()

        if df.empty:
            raise ValueError(f"資料庫中找不到聯賽 {self.league} 的已完賽數據，請先執行 scraper.py")

        # 計算時間衰減權重（越近期的比賽權重越高）
        df["dt"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
        max_dt = df["dt"].max()
        days_ago = (max_dt - df["dt"]).dt.days.fillna(0).clip(lower=0)
        decay_rate = np.log(2) / self.half_life_days
        df["weight"] = np.exp(-decay_rate * days_ago)

        # 混合真實進球與 Synthetic xG
        df["eff_home"] = np.where(
            df["home_xg"].notna(),
            self.goal_weight * df["home_goals"] + self.xg_weight * df["home_xg"],
            df["home_goals"]
        )
        df["eff_away"] = np.where(
            df["away_xg"].notna(),
            self.goal_weight * df["away_goals"] + self.xg_weight * df["away_xg"],
            df["away_goals"]
        )

        # 加權聯賽場均進球
        total_w = df["weight"].sum()
        self.avg_home_goals = (df["eff_home"] * df["weight"]).sum() / total_w
        self.avg_away_goals = (df["eff_away"] * df["weight"]).sum() / total_w

        teams = set(df["home_team"]).union(set(df["away_team"]))
        prior_weight = 1.5  # 貝葉斯先驗平滑權重

        for team in teams:
            h_g = df[df["home_team"] == team]
            a_g = df[df["away_team"] == team]

            hw = h_g["weight"].sum()
            aw = a_g["weight"].sum()

            h_att = ((h_g["eff_home"] * h_g["weight"]).sum() + prior_weight * self.avg_home_goals) / ((hw + prior_weight) * self.avg_home_goals)
            h_def = ((h_g["eff_away"] * h_g["weight"]).sum() + prior_weight * self.avg_away_goals) / ((hw + prior_weight) * self.avg_away_goals)
            a_att = ((a_g["eff_away"] * a_g["weight"]).sum() + prior_weight * self.avg_away_goals) / ((aw + prior_weight) * self.avg_away_goals)
            a_def = ((a_g["eff_home"] * a_g["weight"]).sum() + prior_weight * self.avg_home_goals) / ((aw + prior_weight) * self.avg_home_goals)

            self.team_stats[team] = {
                "home_attack": h_att,
                "home_defense": h_def,
                "away_attack": a_att,
                "away_defense": a_def,
                "matches_played": len(h_g) + len(a_g)
            }

    def list_teams(self):
        return sorted(self.team_stats.keys())

    def get_score_matrix(self, home_team: str, away_team: str, max_goals: int = 8) -> tuple:
        if home_team not in self.team_stats:
            raise ValueError(f"找不到主隊 '{home_team}'，可用隊名：{self.list_teams()}")
        if away_team not in self.team_stats:
            raise ValueError(f"找不到客隊 '{away_team}'，可用隊名：{self.list_teams()}")

        h_stats = self.team_stats[home_team]
        a_stats = self.team_stats[away_team]

        lambda_home = h_stats["home_attack"] * a_stats["away_defense"] * self.avg_home_goals
        lambda_away = a_stats["away_attack"] * h_stats["home_defense"] * self.avg_away_goals

        h_probs = [poisson.pmf(i, lambda_home) for i in range(max_goals)]
        a_probs = [poisson.pmf(j, lambda_away) for j in range(max_goals)]
        matrix = np.outer(h_probs, a_probs)

        # Dixon-Coles 低比分相關性修正
        rho = -0.10
        matrix[0, 0] *= max(0.0, 1 - lambda_home * lambda_away * rho)
        matrix[0, 1] *= max(0.0, 1 + lambda_home * rho)
        matrix[1, 0] *= max(0.0, 1 + lambda_away * rho)
        matrix[1, 1] *= max(0.0, 1 - rho)
        matrix /= matrix.sum()

        return matrix, lambda_home, lambda_away

    def predict_match(self, home_team: str, away_team: str, max_goals: int = 8) -> dict:
        matrix, lambda_home, lambda_away = self.get_score_matrix(home_team, away_team, max_goals)

        prob_home = float(np.sum(np.tril(matrix, -1)))
        prob_draw = float(np.trace(matrix))
        prob_away = float(np.sum(np.triu(matrix, 1)))

        prob_over_25 = 0.0
        scores = []
        for i in range(max_goals):
            for j in range(max_goals):
                if i + j > 2.5:
                    prob_over_25 += matrix[i, j]
                scores.append((f"{i}-{j}", float(matrix[i, j])))

        scores.sort(key=lambda x: x[1], reverse=True)

        return {
            "match": f"{home_team} vs {away_team}",
            "matrix": matrix,
            "lambda_home": round(lambda_home, 2),
            "lambda_away": round(lambda_away, 2),
            "prob_home": prob_home,
            "prob_draw": prob_draw,
            "prob_away": prob_away,
            "prob_over_25": float(prob_over_25),
            "prob_under_25": float(1.0 - prob_over_25),
            "top_scores": scores[:3]
        }

    @staticmethod
    def evaluate_asian_handicap_ev(matrix: np.ndarray, line: float, odd: float, side: str = "home") -> float:
        """
        精算 HKJC 亞洲讓球盤 (HDC) 期望值（支援整數盤走水、半球盤、以及 -0.25/-0.75 雙盤口贏半輸半）
        line: 主隊讓球盤口（例如主讓半/一填 -0.75，主受讓平/半填 +0.25）
        side: 'home' 代表買主隊，'away' 代表買客隊
        """
        max_g = matrix.shape[0]
        ev = 0.0
        for i in range(max_g):
            for j in range(max_g):
                p = matrix[i, j]
                # 計算讓球後的淨勝球差
                diff = (i - j + line) if side == "home" else (j - i - line)

                if diff >= 0.5:
                    ret = odd - 1.0          # 全贏
                elif np.isclose(diff, 0.25):
                    ret = (odd - 1.0) * 0.5  # 贏半
                elif np.isclose(diff, 0.0):
                    ret = 0.0                # 走水退本
                elif np.isclose(diff, -0.25):
                    ret = -0.5               # 輸半
                else:
                    ret = -1.0               # 全輸
                ev += p * ret
        return float(ev)

    @staticmethod
    def evaluate_hhad_probs(matrix: np.ndarray, handicap: int) -> dict:
        """
        精算 HKJC 讓球主客和 (HHAD) 真實勝率（例如主隊 [-1] 則 handicap = -1）
        """
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

def run_full_hkjc_card(prediction: dict, markets: list, bankroll: float = 10000.0):
    """
    整合主客和 (HAD)、入球大細 (HiLo)、讓球 (HDC) 與讓球主客和 (HHAD) 的一站式評估
    """
    matrix = prediction["matrix"]
    print(f"\n{'='*68}")
    print(f"【HKJC 量化戰情室】{prediction['match']} (時間衰減半衰期: 45天)")
    print(f"預期入球 (xG)：主隊 {prediction['lambda_home']}  vs  客隊 {prediction['lambda_away']}")
    top_cs = ", ".join([f"{s} ({p*100:.1f}%)" for s, p in prediction["top_scores"]])
    print(f"最高機率波膽前三：{top_cs}")
    print(f"{'-'*68}")
    print(f"{'HKJC 玩法選項':<18} | {'模型勝率/期望':<12} | {'HKJC賠率':<8} | {'期望值(EV)':<10} | {'建議注碼'}")
    print(f"{'-'*68}")

    for m in markets:
        label = m["label"]
        odd = m["odd"]
        m_type = m["type"]

        if m_type == "1X2_OU":
            key_map = {
                "主勝": prediction["prob_home"],
                "和局": prediction["prob_draw"],
                "客勝": prediction["prob_away"],
                "入球大(2.5)": prediction["prob_over_25"],
                "入球細(2.5)": prediction["prob_under_25"],
            }
            p = key_map[m["key"]]
            ev = (p * odd) - 1.0
            prob_str = f"{p*100:>6.1f}% 勝率"
        elif m_type == "HDC":
            ev = PoissonMatchPredictor.evaluate_asian_handicap_ev(matrix, m["line"], odd, m["side"])
            prob_str = f"盤口 {m['line']:+g}"
        elif m_type == "HHAD":
            hhad_probs = PoissonMatchPredictor.evaluate_hhad_probs(matrix, m["handicap"])
            p = hhad_probs[m["key"]]
            ev = (p * odd) - 1.0
            prob_str = f"{p*100:>6.1f}% 勝率"
        else:
            continue

        full_kelly = max(0.0, ev / (odd - 1.0)) if odd > 1.0 else 0.0
        quarter_kelly = min(0.03, full_kelly * 0.25)
        stake = round(bankroll * quarter_kelly, 0)

        tag = "★ +EV" if ev > 0.03 else ("微幅+EV" if ev > 0 else "無價值")
        stake_str = f"${stake:.0f} ({quarter_kelly*100:.1f}%)" if ev > 0.02 else "-"
        print(f"{label:<18} | {prob_str:<12} | {odd:>8.2f} | {ev*100:>+7.1f}% ({tag}) | {stake_str}")
    print(f"{'='*68}\n")

if __name__ == "__main__":
    model = PoissonMatchPredictor(league="ENG-Premier League", half_life_days=45.0)
    pred = model.predict_match("Man United", "Everton")

    # 測試同時評估 主客和 + 入球大細 + 亞洲讓球盤(HDC) + 讓球主客和(HHAD)
    hkjc_board = [
        {"type": "1X2_OU", "key": "主勝", "label": "主客和-主勝", "odd": 1.72},
        {"type": "1X2_OU", "key": "和局", "label": "主客和-和局", "odd": 3.65},
        {"type": "1X2_OU", "key": "客勝", "label": "主客和-客勝", "odd": 4.20},
        {"type": "1X2_OU", "key": "入球大(2.5)", "label": "入球大(2.5)", "odd": 1.68},
        {"type": "HDC", "side": "away", "line": -0.75, "label": "讓球-客受讓半/一", "odd": 1.92},
        {"type": "HHAD", "handicap": -1, "key": "讓球和", "label": "讓球主客和[-1]和", "odd": 3.45},
    ]
    run_full_hkjc_card(pred, hkjc_board, bankroll=10000)