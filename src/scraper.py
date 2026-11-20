import time
import random
from datetime import datetime, timedelta
import pandas as pd
import requests as std_requests
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

# ESPN 聯賽代碼對照表（涵蓋五大聯賽 + 歐聯）
ESPN_LEAGUES = {
    "eng.1": "ENG-Premier League",
    "esp.1": "ESP-La Liga",
    "ger.1": "GER-Bundesliga",
    "ita.1": "ITA-Serie A",
    "fra.1": "FRA-Ligue 1",
    "uefa.champions": "UEFA-Champions-League",
}

# 隊名標準化映射（將 ESPN 隊名對齊資料庫現有隊名）
TEAM_NAME_MAP = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "Newcastle United": "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "West Ham United": "West Ham",
    "Brighton & Hove Albion": "Brighton",
    "Tottenham Hotspur": "Tottenham",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Ipswich Town": "Ipswich",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "AFC Bournemouth": "Bournemouth",
    "Athletic Club": "Ath Bilbao",
    "Atlético Madrid": "Ath Madrid",
    "Atletico Madrid": "Ath Madrid",
    "Real Sociedad": "Sociedad",
    "Real Betis": "Betis",
    "Celta Vigo": "Celta",
    "Rayo Vallecano": "Vallecano",
    "Deportivo Alavés": "Alaves",
    "RCD Espanyol": "Espanol",
    "Bayern Munich": "Bayern Munich",
    "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Borussia Mönchengladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "VfB Stuttgart": "Stuttgart",
    "Mainz 05": "Mainz",
    "1. FC Köln": "FC Koln",
    "AC Milan": "Milan",
    "Internazionale": "Inter",
    "Inter Milan": "Inter",
    "Hellas Verona": "Verona",
    "Paris Saint-Germain": "Paris SG",
    "AS Monaco": "Monaco",
    "Olympique Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "Stade Rennais": "Rennes",
}

def normalize_team(name: str) -> str:
    extra_map = {
        "FC Barcelona": "Barcelona",
        "Juventus FC": "Juventus",
        "Atalanta BC": "Atalanta",
        "SSC Napoli": "Napoli",
        "AS Roma": "Roma",
        "SS Lazio": "Lazio",
        "Lille OSC": "Lille",
        "Girona FC": "Girona",
        "Bologna FC 1909": "Bologna",
        "AS Monaco FC": "Monaco",
    }
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    if name in extra_map:
        return extra_map[name]
    cleaned = name.replace(" FC", "").replace(" CF", "").strip()
    return TEAM_NAME_MAP.get(cleaned, extra_map.get(cleaned, cleaned))

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
    if shots is None or sot is None:
        return None
    non_sot = max(0, shots - sot)
    cor = corners if corners is not None else 0
    base_xg = (sot * 0.29) + (non_sot * 0.035) + (cor * 0.025)
    multiplier = 1.04 if is_home else 0.96
    return round(base_xg * multiplier, 2)

def fetch_upcoming_schedules_espn(cursor, days_ahead: int = 7) -> int:
    """從 ESPN 公開賽程介面逐日抓取今日至未來 N 天的五大聯賽與歐聯賽程表"""
    print(f"正在抓取今日至未來 {days_ahead} 天賽程表...")
    scheduled_count = 0
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    for espn_code, league_name in ESPN_LEAGUES.items():
        league_events = 0
        for offset in range(days_ahead + 1):
            target_dt = datetime.now() + timedelta(days=offset)
            date_param = target_dt.strftime("%Y%m%d")
            url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_code}/scoreboard?dates={date_param}"
            try:
                resp = std_requests.get(url, headers=headers, timeout=10)
                if resp.status_code != 200:
                    continue
                events = resp.json().get("events", [])
            except Exception:
                continue

            for ev in events:
                raw_dt = ev.get("date", "")
                if len(raw_dt) < 16:
                    continue
                utc_dt = datetime.strptime(raw_dt[:16], "%Y-%m-%dT%H:%M")
                hk_dt = utc_dt + timedelta(hours=8)
                dt_str = hk_dt.strftime("%d/%m/%Y")

                comps = ev.get("competitions", [{}])[0].get("competitors", [])
                if len(comps) < 2:
                    continue

                home_obj = next((c for c in comps if c.get("homeAway") == "home"), comps[0])
                away_obj = next((c for c in comps if c.get("homeAway") == "away"), comps[1])

                home = normalize_team(home_obj["team"]["displayName"])
                away = normalize_team(away_obj["team"]["displayName"])

                state = ev.get("status", {}).get("type", {}).get("state", "pre")
                if state == "post":
                    status = "finished"
                    home_goals = int(home_obj.get("score", 0))
                    away_goals = int(away_obj.get("score", 0))
                else:
                    status = "scheduled"
                    home_goals, away_goals = None, None

                suffix = "_UCL" if espn_code == "uefa.champions" else ""
                match_id = f"{dt_str}_{home}_{away}{suffix}"

                cursor.execute("""
                INSERT INTO matches (match_id, league, season, date, home_team, away_team, home_goals, away_goals, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(match_id) DO UPDATE SET
                    home_goals=COALESCE(excluded.home_goals, matches.home_goals),
                    away_goals=COALESCE(excluded.away_goals, matches.away_goals),
                    status=excluded.status
                """, (match_id, league_name, "2026-2027", dt_str, home, away, home_goals, away_goals, status))
                if status == "scheduled":
                    scheduled_count += 1
                    league_events += 1
        print(f"  [{league_name}] 取得 {league_events} 場待開賽賽程")

    return scheduled_count

def run_pipeline(season_code: str = "2627"):
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    total_matches = 0
    total_odds = 0

    # 1. 更新五大聯賽歷史賽果、統計與賠率
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

        time.sleep(random.uniform(0.5, 1.2))

    # 2. 自動抓取今日至未來 7 天的五大聯賽與歐聯賽程表
    upcoming_count = fetch_upcoming_schedules_espn(cursor, days_ahead=7)

    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()
    print(f"\n更新完成！歷史賽果共 {total_matches} 場（含 {total_odds} 場賠率），新增/更新未來 7 天賽程共 {upcoming_count} 場（含歐聯）。")

if __name__ == "__main__":
    run_pipeline()