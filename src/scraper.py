import time
import random
import pandas as pd
from curl_cffi import requests
from io import StringIO
from db import get_connection, init_db

LEAGUES = {
    "E0": "ENG-Premier League",
    "SP1": "ESP-La Liga",
    "D1": "GER-Bundesliga",
    "I1": "ITA-Serie A",
    "F1": "FRA-Ligue 1",
}

def fetch_football_data_csv(season_code: str, league_code: str) -> pd.DataFrame:
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"
    resp = requests.get(url, impersonate="chrome", timeout=20)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    return df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])

def pick_first_valid(row, candidates):
    for col in candidates:
        if col in row and pd.notna(row[col]):
            return float(row[col])
    return None

def calculate_synthetic_xg(shots, sot, corners, is_home: bool):
    """根據射正、非射正射門與角球壓制力計算合成預期入球 (Synthetic xG)"""
    if shots is None or sot is None:
        return None
    non_sot = max(0, shots - sot)
    cor = corners if corners is not None else 0
    # 基礎射門轉化權重 + 角球定位球威脅 + 微幅主場係數
    base_xg = (sot * 0.29) + (non_sot * 0.035) + (cor * 0.025)
    multiplier = 1.04 if is_home else 0.96
    return round(base_xg * multiplier, 2)

def run_phase1_pipeline(season_code: str = "2627"):
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    total_matches = 0
    total_odds = 0
    total_xg = 0

    for code, league_name in LEAGUES.items():
        print(f"正在更新 {league_name} ({code})...")
        try:
            df = fetch_football_data_csv(season_code, code)
        except Exception as e:
            print(f"  [CSV] 抓取失敗：{e}")
            continue

        for _, row in df.iterrows():
            date_str = str(row["Date"]).strip()
            home = str(row["HomeTeam"]).strip()
            away = str(row["AwayTeam"]).strip()
            match_id = f"{date_str}_{home}_{away}"

            home_goals = int(row["FTHG"]) if pd.notna(row.get("FTHG")) else None
            away_goals = int(row["FTAG"]) if pd.notna(row.get("FTAG")) else None
            status = "finished" if home_goals is not None else "scheduled"

            cursor.execute("""
            INSERT INTO matches (match_id, league, season, date, home_team, away_team, home_goals, away_goals, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
                home_goals=excluded.home_goals,
                away_goals=excluded.away_goals,
                status=excluded.status
            """, (match_id, league_name, "2026-2027", date_str, home, away, home_goals, away_goals, status))
            total_matches += 1

            h_shots = int(row["HS"]) if pd.notna(row.get("HS")) else None
            a_shots = int(row["AS"]) if pd.notna(row.get("AS")) else None
            h_sot = int(row["HST"]) if pd.notna(row.get("HST")) else None
            a_sot = int(row["AST"]) if pd.notna(row.get("AST")) else None
            h_cor = int(row["HC"]) if pd.notna(row.get("HC")) else None
            a_cor = int(row["AC"]) if pd.notna(row.get("AC")) else None

            h_xg = calculate_synthetic_xg(h_shots, h_sot, h_cor, is_home=True)
            a_xg = calculate_synthetic_xg(a_shots, a_sot, a_cor, is_home=False)
            if h_xg is not None:
                total_xg += 1

            cursor.execute("""
            INSERT INTO match_stats (match_id, home_xg, away_xg, home_shots, away_shots, home_sot, away_sot, home_corners, away_corners)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
                home_xg=excluded.home_xg,
                away_xg=excluded.away_xg,
                home_shots=excluded.home_shots,
                away_shots=excluded.away_shots,
                home_sot=excluded.home_sot,
                away_sot=excluded.away_sot,
                home_corners=excluded.home_corners,
                away_corners=excluded.away_corners
            """, (match_id, h_xg, a_xg, h_shots, a_shots, h_sot, a_sot, h_cor, a_cor))

            h_odd = pick_first_valid(row, ["PSH", "PSCH", "B365H", "AvgH"])
            d_odd = pick_first_valid(row, ["PSD", "PSCD", "B365D", "AvgD"])
            a_odd = pick_first_valid(row, ["PSA", "PSCA", "B365A", "AvgA"])
            ah_line = pick_first_valid(row, ["AHh", "AHCh"])
            h_ah = pick_first_valid(row, ["PAHH", "PCAHH", "B365AHH", "AvgAHH"])
            a_ah = pick_first_valid(row, ["PAHA", "PCAHA", "B365AHA", "AvgAHA"])
            over_odd = pick_first_valid(row, ["P>2.5", "PC>2.5", "B365>2.5", "Avg>2.5"])
            under_odd = pick_first_valid(row, ["P<2.5", "PC<2.5", "B365<2.5", "Avg<2.5"])

            if h_odd and d_odd and a_odd:
                cursor.execute("""
                INSERT INTO odds_history (
                    match_id, bookmaker, timestamp, home_odd, draw_odd, away_odd,
                    asian_line, home_ah_odd, away_ah_odd, ou_line, over_odd, under_odd
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(match_id, bookmaker) DO UPDATE SET
                    home_odd=excluded.home_odd,
                    draw_odd=excluded.draw_odd,
                    away_odd=excluded.away_odd,
                    asian_line=excluded.asian_line,
                    home_ah_odd=excluded.home_ah_odd,
                    away_ah_odd=excluded.away_ah_odd,
                    over_odd=excluded.over_odd,
                    under_odd=excluded.under_odd
                """, (match_id, "Market_Consensus", date_str, h_odd, d_odd, a_odd,
                      ah_line, h_ah, a_ah, 2.5 if over_odd else None, over_odd, under_odd))
                total_odds += 1

        time.sleep(random.uniform(1.0, 2.0))

    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()
    print(f"\nPhase 1 抓取全部完成！共寫入 {total_matches} 場五大聯賽賽果、{total_xg} 場 Synthetic xG、{total_odds} 場賠率與盤口。")

if __name__ == "__main__":
    run_phase1_pipeline()