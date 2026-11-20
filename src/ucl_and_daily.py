import sqlite3
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import poisson
from pathlib import Path
from model import PoissonMatchPredictor

DB_PATH = Path("data/soccer.db")

# 聯賽整體實力校正係數（解決不同聯賽防守強度與節奏差異）
LEAGUE_STRENGTH = {
    "ENG-Premier League": 1.08,
    "ESP-La Liga": 1.04,
    "GER-Bundesliga": 1.03,
    "ITA-Serie A": 1.02,
    "FRA-Ligue 1": 0.96,
    "OTHER-Europe": 0.90,  # 非五大聯賽參賽隊伍預設基準
}

class CrossLeagueUCLPredictor:
    def __init__(self):
        self.league_models = {}
        self.team_to_league = {}
        for lg in ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]:
            try:
                m = PoissonMatchPredictor(league=lg, goal_weight=0.4, half_life_days=45.0)
                self.league_models[lg] = m
                for t in m.list_teams():
                    self.team_to_league[t] = lg
            except Exception:
                pass

    def get_adjusted_strength(self, team: str) -> tuple:
        """回傳 (校正後主場進攻, 校正後主場防守, 校正後客場進攻, 校正後客場防守, 所屬聯賽)"""
        lg = self.team_to_league.get(team)
        if lg and lg in self.league_models:
            stats = self.league_models[lg].team_stats[team]
            coef = LEAGUE_STRENGTH[lg]
            # 強聯賽隊伍進攻力上調、失球率下調（防守更強）
            return (
                stats["home_attack"] * coef,
                stats["home_defense"] / coef,
                stats["away_attack"] * coef,
                stats["away_defense"] / coef,
                lg
            )
        # 若為非五大聯賽歐聯隊伍（如 Sporting CP、PSV 等），套用基準值
        coef = LEAGUE_STRENGTH["OTHER-Europe"]
        return (1.0 * coef, 1.0 / coef, 0.95 * coef, 1.05 / coef, "OTHER-Europe")

    def predict_ucl_match(self, home_team: str, away_team: str, max_goals: int = 8) -> dict:
        h_att, h_def, _, _, h_lg = self.get_adjusted_strength(home_team)
        _, _, a_att, a_def, a_lg = self.get_adjusted_strength(away_team)

        # 歐聯平均單隊入球基準約 1.48（主場）vs 1.22（客場）
        lambda_home = h_att * a_def * 1.48
        lambda_away = a_att * h_def * 1.22

        h_probs = [poisson.pmf(i, lambda_home) for i in range(max_goals)]
        a_probs = [poisson.pmf(j, lambda_away) for j in range(max_goals)]
        matrix = np.outer(h_probs, a_probs)

        # Dixon-Coles 低比分修正
        rho = -0.10
        matrix[0, 0] *= max(0.0, 1 - lambda_home * lambda_away * rho)
        matrix[0, 1] *= max(0.0, 1 + lambda_home * rho)
        matrix[1, 0] *= max(0.0, 1 + lambda_away * rho)
        matrix[1, 1] *= max(0.0, 1 - rho)
        matrix /= matrix.sum()

        prob_home = float(np.sum(np.tril(matrix, -1)))
        prob_draw = float(np.trace(matrix))
        prob_away = float(np.sum(np.triu(matrix, 1)))
        prob_over_25 = sum(matrix[i, j] for i in range(max_goals) for j in range(max_goals) if i + j > 2.5)

        scores = [(f"{i}-{j}", float(matrix[i, j])) for i in range(max_goals) for j in range(max_goals)]
        scores.sort(key=lambda x: x[1], reverse=True)

        return {
            "match": f"{home_team} ({h_lg}) vs {away_team} ({a_lg})",
            "lambda_home": round(lambda_home, 2),
            "lambda_away": round(lambda_away, 2),
            "prob_home": prob_home,
            "prob_draw": prob_draw,
            "prob_away": prob_away,
            "prob_over_25": float(prob_over_25),
            "top_scores": scores[:3],
            "matrix": matrix
        }

def show_upcoming_fixtures(days_ahead: int = 3):
    """自動掃描資料庫中今日至未來 N 天的所有未開賽賽程並輸出模型初盤預測"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM matches WHERE status = 'scheduled'", conn)
    conn.close()

    if df.empty:
        print("資料庫目前沒有未開賽賽程，請確認是否已執行最新 scraper.py。")
        return

    df["dt"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = today + timedelta(days=days_ahead)

    upcoming = df[(df["dt"] >= today) & (df["dt"] <= end_date)].sort_values(["dt", "league"])

    print(f"\n{'='*78}")
    print(f"【自動賽程掃描】{today.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')} 待開賽場次（共 {len(upcoming)} 場）")
    print(f"{'='*78}")

    if upcoming.empty:
        print("未來 3 天內暫無賽程（可能正值國際賽期或休賽日），以下顯示最近期的未來賽程：")
        upcoming = df[df["dt"] >= today].sort_values(["dt", "league"]).head(10)

    ucl_model = CrossLeagueUCLPredictor()

    for _, row in upcoming.iterrows():
        lg = row["league"]
        h, a, d_str = row["home_team"], row["away_team"], row["date"]
        try:
            if lg == "UEFA-Champions-League":
                res = ucl_model.predict_ucl_match(h, a)
            elif lg in ucl_model.league_models:
                res = ucl_model.league_models[lg].predict_match(h, a)
            else:
                continue

            top_cs = res["top_scores"][0][0]
            print(f"[{d_str}] {lg:<21} | {h} vs {a}")
            print(f"   預期入球: {res['lambda_home']} - {res['lambda_away']} | "
                  f"主勝 {res['prob_home']*100:.1f}% / 和 {res['prob_draw']*100:.1f}% / 客勝 {res['prob_away']*100:.1f}% | "
                  f"大2.5: {res['prob_over_25']*100:.1f}% | 首選波膽: {top_cs}")
        except Exception as e:
            print(f"[{d_str}] {lg:<21} | {h} vs {a} (數據不足略過: {e})")
    print(f"{'='*78}\n")

if __name__ == "__main__":
    # 1. 自動掃描今日與未來 3 天的賽程並跑出量化預測
    show_upcoming_fixtures(days_ahead=3)

    # 2. 歐聯跨聯賽焦點對決單場模擬測試
    ucl = CrossLeagueUCLPredictor()
    sample_ucl = ucl.predict_ucl_match("Real Madrid", "Liverpool")
    print("【歐聯跨聯賽模擬測試】", sample_ucl["match"])
    print(f"預期入球：{sample_ucl['lambda_home']} vs {sample_ucl['lambda_away']} | "
          f"主勝 {sample_ucl['prob_home']*100:.1f}% / 和 {sample_ucl['prob_draw']*100:.1f}% / 客勝 {sample_ucl['prob_away']*100:.1f}%")