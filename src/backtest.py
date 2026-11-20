import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import poisson
from pathlib import Path

DB_PATH = Path("data/soccer.db")

def run_walk_forward_backtest(league: str = "ENG-Premier League", ev_threshold: float = 0.05, warmup_matches: int = 30):
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

    df["eff_home"] = np.where(df["home_xg"].notna(), 0.7 * df["home_goals"] + 0.3 * df["home_xg"], df["home_goals"])
    df["eff_away"] = np.where(df["away_xg"].notna(), 0.7 * df["away_goals"] + 0.3 * df["away_xg"], df["away_goals"])

    bets = []
    # 跳過賽季初前 warmup_matches 場冷啟動期，逐場向後回測
    for idx in range(warmup_matches, len(df)):
        row = df.iloc[idx]
        hist = df[df["dt"] < row["dt"]]  # 嚴格只使用開賽前的歷史數據
        if len(hist) < warmup_matches:
            continue

        avg_h = hist["eff_home"].mean()
        avg_a = hist["eff_away"].mean()

        h_games = hist[hist["home_team"] == row["home_team"]]
        a_games = hist[hist["away_team"] == row["away_team"]]

        h_att = (h_games["eff_home"].sum() + 2 * avg_h) / ((len(h_games) + 2) * avg_h)
        a_def = (a_games["eff_home"].sum() + 2 * avg_h) / ((len(a_games) + 2) * avg_h)
        a_att = (a_games["eff_away"].sum() + 2 * avg_a) / ((len(a_games) + 2) * avg_a)
        h_def = (h_games["eff_away"].sum() + 2 * avg_a) / ((len(h_games) + 2) * avg_a)

        lam_h = h_att * a_def * avg_h
        lam_a = a_att * h_def * avg_a

        mat = np.outer([poisson.pmf(i, lam_h) for i in range(8)], [poisson.pmf(j, lam_a) for j in range(8)])
        p_home = float(np.sum(np.tril(mat, -1)))
        p_draw = float(np.trace(mat))
        p_away = float(np.sum(np.triu(mat, 1)))
        p_over = sum(mat[i, j] for i in range(8) for j in range(8) if i + j > 2.5)

        candidates = [
            ("主勝", p_home, row["home_odd"], row["home_goals"] > row["away_goals"]),
            ("和局", p_draw, row["draw_odd"], row["home_goals"] == row["away_goals"]),
            ("客勝", p_away, row["away_odd"], row["home_goals"] < row["away_goals"]),
            ("大2.5", p_over, row["over_odd"], (row["home_goals"] + row["away_goals"]) > 2.5),
            ("細2.5", 1 - p_over, row["under_odd"], (row["home_goals"] + row["away_goals"]) < 2.5),
        ]

        # 每場最多挑選 EV 最高且超過門檻的一個選項進行平注回測 (1 單位)
        best_pick = None
        best_ev = ev_threshold
        for name, prob, odd, won in candidates:
            if pd.isna(odd) or odd <= 1.0:
                continue
            ev = prob * odd - 1.0
            if ev > best_ev:
                best_ev = ev
                best_pick = (name, prob, odd, ev, won)

        if best_pick:
            name, prob, odd, ev, won = best_pick
            profit = (odd - 1.0) if won else -1.0
            bets.append({"date": row["date"], "match": f"{row['home_team']} vs {row['away_team']}",
                         "pick": name, "odd": odd, "ev": ev, "won": won, "profit": profit})

    res_df = pd.DataFrame(bets)
    print(f"\n{'='*55}")
    print(f"【Walk-Forward 歷史回測報告】聯賽：{league} (EV 門檻 > {ev_threshold*100:.0f}%)")
    if res_df.empty:
        print("在當前門檻下未觸發符合條件的下注。")
    else:
        total_bets = len(res_df)
        win_rate = res_df["won"].mean() * 100
        net_profit = res_df["profit"].sum()
        roi = (net_profit / total_bets) * 100
        print(f"總觸發注數：{total_bets} 注 | 命中率：{win_rate:.1f}%")
        print(f"總淨盈虧：{net_profit:+.2f} 單位 | 投資報酬率 (ROI)：{roi:+.2f}%")
    print(f"{'='*55}")

if __name__ == "__main__":
    for lg in ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]:
        run_walk_forward_backtest(league=lg, ev_threshold=0.05, warmup_matches=30)