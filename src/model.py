import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import poisson
from pathlib import Path

DB_PATH = Path("data/soccer.db")

class PoissonMatchPredictor:
    def __init__(self, league: str = "ENG-Premier League", goal_weight: float = 0.7):
        self.league = league
        self.goal_weight = goal_weight
        self.xg_weight = 1.0 - goal_weight
        self.team_stats = {}
        self.avg_home_goals = 0.0
        self.avg_away_goals = 0.0
        self._fit()

    def _fit(self):
        conn = sqlite3.connect(DB_PATH)
        query = """
        SELECT m.home_team, m.away_team, m.home_goals, m.away_goals,
               s.home_xg, s.away_xg
        FROM matches m
        LEFT JOIN match_stats s ON m.match_id = s.match_id
        WHERE m.league = ? AND m.status = 'finished'
        """
        df = pd.read_sql_query(query, conn, params=(self.league,))
        conn.close()

        if df.empty:
            raise ValueError(f"資料庫中找不到聯賽 {self.league} 的已完賽數據，請先執行 scraper.py")

        # 混合真實進球與 Synthetic xG 作為有效進球指標
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

        self.avg_home_goals = df["eff_home"].mean()
        self.avg_away_goals = df["eff_away"].mean()

        teams = set(df["home_team"]).union(set(df["away_team"]))
        for team in teams:
            home_games = df[df["home_team"] == team]
            away_games = df[df["away_team"] == team]

            # 避免季初場次過少導致極端值，加入輕微貝葉斯平滑 (先驗 2 場聯賽平均)
            h_att = (home_games["eff_home"].sum() + 2 * self.avg_home_goals) / ((len(home_games) + 2) * self.avg_home_goals)
            h_def = (home_games["eff_away"].sum() + 2 * self.avg_away_goals) / ((len(home_games) + 2) * self.avg_away_goals)
            a_att = (away_games["eff_away"].sum() + 2 * self.avg_away_goals) / ((len(away_games) + 2) * self.avg_away_goals)
            a_def = (away_games["eff_home"].sum() + 2 * self.avg_home_goals) / ((len(away_games) + 2) * self.avg_home_goals)

            self.team_stats[team] = {
                "home_attack": h_att,
                "home_defense": h_def,
                "away_attack": a_att,
                "away_defense": a_def,
                "matches_played": len(home_games) + len(away_games)
            }

    def list_teams(self):
        return sorted(self.team_stats.keys())

    def predict_match(self, home_team: str, away_team: str, max_goals: int = 8) -> dict:
        if home_team not in self.team_stats:
            raise ValueError(f"找不到主隊 '{home_team}'，可用隊名：{self.list_teams()}")
        if away_team not in self.team_stats:
            raise ValueError(f"找不到客隊 '{away_team}'，可用隊名：{self.list_teams()}")

        h_stats = self.team_stats[home_team]
        a_stats = self.team_stats[away_team]

        # 計算主客隊預期入球數 (Lambda)
        lambda_home = h_stats["home_attack"] * a_stats["away_defense"] * self.avg_home_goals
        lambda_away = a_stats["away_attack"] * h_stats["home_defense"] * self.avg_away_goals

        # 建立獨立泊松機率矩陣
        h_probs = [poisson.pmf(i, lambda_home) for i in range(max_goals)]
        a_probs = [poisson.pmf(j, lambda_away) for j in range(max_goals)]
        matrix = np.outer(h_probs, a_probs)

        # Dixon-Coles 低比分相關性修正 (修正 0-0, 1-0, 0-1, 1-1 偏差)
        rho = -0.10
        matrix[0, 0] *= max(0.0, 1 - lambda_home * lambda_away * rho)
        matrix[0, 1] *= max(0.0, 1 + lambda_home * rho)
        matrix[1, 0] *= max(0.0, 1 + lambda_away * rho)
        matrix[1, 1] *= max(0.0, 1 - rho)
        matrix /= matrix.sum()  # 重新歸一化確保總機率為 100%

        prob_home = float(np.sum(np.tril(matrix, -1)))
        prob_draw = float(np.trace(matrix))
        prob_away = float(np.sum(np.triu(matrix, 1)))

        # 計算入球大細 2.5 球機率
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
            "lambda_home": round(lambda_home, 2),
            "lambda_away": round(lambda_away, 2),
            "prob_home": prob_home,
            "prob_draw": prob_draw,
            "prob_away": prob_away,
            "prob_over_25": float(prob_over_25),
            "prob_under_25": float(1.0 - prob_over_25),
            "top_scores": scores[:3]
        }

def evaluate_hkjc_ev(prediction: dict, hkjc_odds: dict, bankroll: float = 10000.0):
    """
    對比模型真實機率與 HKJC 足智彩賠率，篩選 +EV 選項並計算四分之一凱利注碼
    hkjc_odds 範例: {"主勝": 1.85, "和局": 3.55, "客勝": 3.70, "入球大(2.5)": 1.72, "入球細(2.5)": 2.00}
    """
    prob_map = {
        "主勝": prediction["prob_home"],
        "和局": prediction["prob_draw"],
        "客勝": prediction["prob_away"],
        "入球大(2.5)": prediction["prob_over_25"],
        "入球細(2.5)": prediction["prob_under_25"],
    }

    print(f"\n{'='*58}")
    print(f"對陣分析：{prediction['match']}")
    print(f"模型預期入球 (xG)：主隊 {prediction['lambda_home']}  vs  客隊 {prediction['lambda_away']}")
    top_cs = ", ".join([f"{s} ({p*100:.1f}%)" for s, p in prediction["top_scores"]])
    print(f"最高機率波膽前三：{top_cs}")
    print(f"{'-'*58}")
    print(f"{'玩法選項':<12} | {'模型勝率':<8} | {'HKJC賠率':<8} | {'期望值(EV)':<10} | {'建議注碼'}")
    print(f"{'-'*58}")

    for market, odd in hkjc_odds.items():
        if market not in prob_map or not odd:
            continue
        p = prob_map[market]
        ev = (p * odd) - 1.0
        # 完整凱利公式: f* = (p * odd - 1) / (odd - 1)，實戰採用 0.25 倍凱利控制波動，上限 3%
        full_kelly = max(0.0, ev / (odd - 1.0)) if odd > 1.0 else 0.0
        quarter_kelly = min(0.03, full_kelly * 0.25)
        stake = round(bankroll * quarter_kelly, 0)

        tag = "★ +EV" if ev > 0.03 else ("微幅+EV" if ev > 0 else "無價值")
        stake_str = f"${stake:.0f} ({quarter_kelly*100:.1f}%)" if ev > 0.02 else "-"
        print(f"{market:<12} | {p*100:>6.1f}% | {odd:>8.2f} | {ev*100:>+7.1f}% ({tag}) | {stake_str}")
    print(f"{'='*58}\n")

if __name__ == "__main__":
    # 測試範例：英超對陣評估
    model = PoissonMatchPredictor(league="ENG-Premier League")
    print("資料庫已載入英超球隊：", model.list_teams())

    # 測試計算一場對陣（可隨時替換為本輪實際對陣與 HKJC 賠率）
    pred = model.predict_match("Man United", "Everton")
    sample_hkjc_odds = {
        "主勝": 1.72,
        "和局": 3.65,
        "客勝": 4.20,
        "入球大(2.5)": 1.68,
        "入球細(2.5)": 2.05
    }
    evaluate_hkjc_ev(pred, sample_hkjc_odds, bankroll=10000)