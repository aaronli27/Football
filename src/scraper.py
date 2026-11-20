import time
import random
import pandas as pd
import soccerdata as sd
from curl_cffi import requests
from io import StringIO
from db import get_connection, init_db

def fetch_understat(league: str = "ENG-Premier League", season: str = "2025-2026"):
    print(f"嘗試從 Understat 抓取 {league} ({season})...")
    # no_cache=True 避免讀取到剛才失敗留下的空白快取檔
    understat = sd.Understat(leagues=league, seasons=season, no_cache=True)
    schedule = understat.read_schedule().reset_index()
    return schedule

def fetch_football_data_uk(season_code: str = "2627", league_code: str = "E0"):
    """
    穩定備援數據源：Football-Data.co.uk (包含賽果、射門、射正、角球及多家莊家賠率)
    E0 = 英超, SP1 = 西甲, D1 = 德甲, I1 = 意甲, F1 = 法甲
    """
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"
    print(f"啟動穩定數據源抓取：{url}")
    resp = requests.get(url, impersonate="chrome", timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    # 過濾掉沒有日期的空行
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    return df

def run_pipeline():
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    # 1. 先嘗試抓取 Understat xG 數據
    try:
        df_xg = fetch_understat(league="ENG-Premier League", season="2025-2026")
        print(f"Understat 抓取成功！共 {len(df_xg)} 場。")
    except Exception as e:
        print(f"Understat 暫時無法連線或無該季數據 ({type(e).__name__})，自動切換至 CSV 數據源...")
        df_xg = None

    # 2. 抓取 Football-Data.co.uk 賽果、射門統計與賠率 (2627 代表 2026/27 賽季)
    try:
        df_csv = fetch_football_data_uk(season_code="2627", league_code="E0")
        count = 0
        for _, row in df_csv.iterrows():
            date_str = str(row["Date"])
            home = str(row["HomeTeam"])
            away = str(row["AwayTeam"])
            match_id = f"{date_str}_{home}_{away}"

            home_goals = int(row["FTHG"]) if pd.notna(row.get("FTHG")) else None
            away_goals = int(row["FTAG"]) if pd.notna(row.get("FTAG")) else None
            status = "finished" if home_goals is not None else "scheduled"

            # 寫入賽程與賽果
            cursor.execute("""
            INSERT INTO matches (match_id, league, season, date, home_team, away_team, home_goals, away_goals, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
                home_goals=excluded.home_goals,
                away_goals=excluded.away_goals,
                status=excluded.status
            """, (match_id, "ENG-Premier League", "2026-2027", date_str, home, away, home_goals, away_goals, status))

            # 寫入射門與射正數據 (HS, AS, HST, AST)
            if pd.notna(row.get("HS")):
                cursor.execute("""
                INSERT INTO match_stats (match_id, home_shots, away_shots, home_sot, away_sot)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(match_id) DO UPDATE SET
                    home_shots=excluded.home_shots,
                    away_shots=excluded.away_shots,
                    home_sot=excluded.home_sot,
                    away_sot=excluded.away_sot
                """, (match_id, int(row["HS"]), int(row["AS"]), int(row["HST"]), int(row["AST"])))

            # 寫入 Pinnacle 基準賠率 (PSH, PSD, PSA)
            if pd.notna(row.get("PSH")):
                cursor.execute("""
                INSERT INTO odds_history (match_id, bookmaker, timestamp, home_odd, draw_odd, away_odd)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (match_id, "Pinnacle", date_str, float(row["PSH"]), float(row["PSD"]), float(row["PSA"])))

            count += 1

        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        print(f"成功寫入 {count} 場英超賽果、射門統計與 Pinnacle 賠率至 data/soccer.db！")

    except Exception as e:
        print(f"抓取失敗：{e}")
    finally:
        conn.close()

if __name__ == "__main__":
    time.sleep(random.uniform(1, 2))
    run_pipeline()