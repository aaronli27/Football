import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import poisson
from pathlib import Path

DB_PATH = Path("data/soccer.db")

def remove_vig_1x2(h_odd, d_odd, a_odd):
    """去除莊家抽水（Overround），還原市場真實隱含勝率"""
    inv_h, inv_d, inv_a = 1.0 / h_odd, 1.0 / d_odd, 1.0 / a_odd
    total = inv_h + inv_d + inv_a
    return inv_h / total, inv_d / total, inv_a / total

def run_calibrated_backtest(league: str = "ENG-Premier League", warmup_matches: int = 50):
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT m.match_id, m.date, m.home_team, m.away_team, m.home_goals, m.away_goals,
           s.home_xg, s.away_xg, o.home_odd, o.draw_odd, o.away_odd, o.over_odd, o.under_odd
    FROM matches m
    LEFT JOIN match_stats s ON m.match_id = s.match_id
    INNER JOIN odds_history o ON m.match_id = o.match_id
    WHERE m.league = ? AND m.status = 'finished'
    """
    df = pd.read_sql_query(query, conn, params=(league,))
    conn.close()

    df["dt"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df = df.sort_values("dt").reset_index(drop=True)

    # 提高 Synthetic xG 權重至 60%，過濾單場運氣進球
    df["eff_home"] = np.where(df["home_xg"].notna(), 0.4 * df["home_goals"] + 0.6 * df["home_xg"], df["home_goals"])
    df["eff_away"] = np.where(df["away_xg"].notna(), 0.4 * df["away_goals"] + 0.6 * df["away_xg"], df["away_goals"])

    bets = []
    prior = 5.0  # 提高先驗平滑強度，防止小樣本極端值

    for idx in range(warmup_matches, len(df)):
        row = df.iloc[idx]
        hist = df[df["dt"] < row["dt"]]
        if len(hist) < warmup_matches:
            continue

        avg_h = hist["eff_home"].mean()
        avg_a = hist["eff_away"].mean()

        h_games = hist[hist["home_team"] == row["home_team"]]
        a_games = hist[hist["away_team"] == row["away_team"]]

        h_att = (h_games["eff_home"].sum() + prior * avg_h) / ((len(h_games) + prior) * avg_h)
        a_def = (a_games["eff_home"].sum() + prior * avg_h) / ((len(a_games) + prior) * avg_h)
        a_att = (a_games["eff_away"].sum() + prior * avg_a) / ((len(a_games) + prior) * avg_a)
        h_def = (h_games["eff_away"].sum() + prior * avg_a) / ((len(h_games) + prior) * avg_a)

        lam_h = h_att * a_def * avg_h
        lam_a = a_att * h_def * avg_a

        mat = np.outer([poisson.pmf(i, lam_h) for i in range(8)], [poisson.pmf(j, lam_a) for j in range(8)])
        p_home_mod = float(np.sum(np.tril(mat, -1)))
        p_draw_mod = float(np.trace(mat))
        p_away_mod = float(np.sum(np.triu(mat, 1)))
        p_over_mod = sum(mat[i, j] for i in range(8) for j in range(8) if i + j > 2.5)

        # 結合市場去水隱含機率（70% 市場基準 + 30% xG泊松模型）
        mkt_h, mkt_d, mkt_a = remove_vig_1x2(row["home_odd"], row["draw_odd"], row["away_odd"])
        p_home = 0.70 * mkt_h + 0.30 * p_home_mod
        p_draw = 0.70 * mkt_d + 0.30 * p_draw_mod
        p_away = 0.70 * mkt_a + 0.30 * p_away_mod

        candidates = [
            ("主勝", p_home, row["home_odd"], row["home_goals"] > row["away_goals"]),
            ("和局", p_draw, row["draw_odd"], row["home_goals"] == row["away_goals"]),
            ("客勝", p_away, row["away_odd"], row["home_goals"] < row["away_goals"]),
        ]

        if pd.notna(row["over_odd"]) and pd.notna(row["under_odd"]):
            inv_o, inv_u = 1.0 / row["over_odd"], 1.0 / row["under_odd"]
            mkt_o = inv_o / (inv_o + inv_u)
            p_over = 0.70 * mkt_o + 0.30 * p_over_mod
            candidates.append(("大2.5", p_over, row["over_odd"], (row["home_goals"] + row["away_goals"]) > 2.5))
            candidates.append(("細2.5", 1.0 - p_over, row["under_odd"], (row["home_goals"] + row["away_goals"]) < 2.5))

        best_pick = None
        best_ev = 0.015  # 至少 1.5% 正期望值
        for name, prob, odd, won in candidates:
            if pd.isna(odd) or not (1.55 <= odd <= 3.10):
                continue
            ev = prob * odd - 1.0
            # 過濾 EV > 12% 的異常分歧（多半代表模型未掌握主力傷停等場外情報）
            if 0.015 < ev <= 0.12 and ev > best_ev:
                best_ev = ev
                best_pick = (name, prob, odd, ev, won)

        if best_pick:
            name, prob, odd, ev, won = best_pick
            profit = (odd - 1.0) if won else -1.0
            bets.append({"date": row["date"], "match": f"{row['home_team']} vs {row['away_team']}",
                         "pick": name, "odd": odd, "ev": ev, "won": won, "profit": profit})

    res_df = pd.DataFrame(bets)
    print(f"\n{'='*58}")
    print(f"【校準後回測報告】聯賽：{league}")
    if res_df.empty:
        print("在當前風控條件下未觸發下注。")
    else:
        total_bets = len(res_df)
        win_rate = res_df["won"].mean() * 100
        net_profit = res_df["profit"].sum()
        roi = (net_profit / total_bets) * 100
        print(f"總觸發注數：{total_bets} 注 | 命中率：{win_rate:.1f}%")
        print(f"總淨盈虧：{net_profit:+.2f} 單位 | 投資報酬率 (ROI)：{roi:+.2f}%")
    print(f"{'='*58}")

if __name__ == "__main__":
    for lg in ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]:
        run_calibrated_backtest(league=lg, warmup_matches=50)