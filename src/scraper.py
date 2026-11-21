import sqlite3
from datetime import datetime
from io import StringIO
from pathlib import Path
import pandas as pd
from curl_cffi import requests
from db import init_db, get_connection

LEAGUE_CODES = {
    "ENG-Premier League": "E0",
    "ESP-La Liga": "SP1",
    "GER-Bundesliga": "D1",
    "ITA-Serie A": "I1",
    "FRA-Ligue 1": "F1",
}

ESPN_LEAGUES = {
    "ENG-Premier League": "eng.1",
    "ESP-La Liga": "esp.1",
    "GER-Bundesliga": "ger.1",
    "ITA-Serie A": "ita.1",
    "FRA-Ligue 1": "fra.1",
    "UEFA-Champions-League": "uefa.champions",
}

TEAM_NAME_MAP = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "AFC Bournemouth": "Bournemouth",
    "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Atlético Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao",
    "Real Betis": "Betis",
    "Real Sociedad": "Sociedad",
    "Celta Vigo": "Celta",
    "Rayo Vallecano": "Vallecano",
    "Deportivo Alavés": "Alaves",
    "RCD Espanyol": "Espanol",
    "FC Bayern München": "Bayern Munich",
    "Bayern München": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "Bayer Leverkusen": "Leverkusen",
    "RB Leipzig": "RB Leipzig",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Borussia Mönchengladbach": "M'gladbach",
    "VfB Stuttgart": "Stuttgart",
    "SC Freiburg": "Freiburg",
    "1. FSV Mainz 05": "Mainz",
    "FC Augsburg": "Augsburg",
    "TSG Hoffenheim": "Hoffenheim",
    "SV Werder Bremen": "Werder Bremen",
    "VfL Wolfsburg": "Wolfsburg",
    "1. FC Union Berlin": "Union Berlin",
    "FC St. Pauli": "St Pauli",
    "1. FC Heidenheim 1846": "Heidenheim",
    "Internazionale": "Inter",
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "SS Lazio": "Lazio",
    "Hellas Verona": "Verona",
    "Parma Calcio 1913": "Parma",
    "Como 1907": "Como",
    "Paris Saint-Germain": "Paris SG",
    "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "AS Monaco": "Monaco",
    "LOSC Lille": "Lille",
    "OGC Nice": "Nice",
    "Stade Rennais": "Rennes",
    "RC Lens": "Lens",
    "Stade Brestois 29": "Brest",
    "RC Strasbourg Alsace": "Strasbourg",
    "FC Nantes": "Nantes",
    "Toulouse FC": "Toulouse",
    "AJ Auxerre": "Auxerre",
    "Angers SCO": "Angers",
    "Le Havre AC": "Le Havre",
    "AS Saint-Étienne": "St Etienne",
    "Montpellier HSC": "Montpellier",
    "Stade de Reims": "Reims",
}


def normalize_team(name: str) -> str:
    clean = str(name).strip()
    return TEAM_NAME_MAP.get(clean, clean)


def pick_first_valid(row: pd.Series, cols: list, allow_zero_or_neg: bool = False):
    for c in cols:
        if c in row and pd.notna(row[c]):
            try:
                val = float(row[c])
                if allow_zero_or_neg or val > 0:
                    return val
            except (ValueError, TypeError):
                continue
    return None


def compute_synthetic_xg(shots, sot, corners):
    s = float(shots) if pd.notna(shots) else 10.0
    st = float(sot) if pd.notna(sot) else 3.5
    c = float(corners) if pd.notna(corners) else 4.5
    non_sot = max(0.0, s - st)
    return round(st * 0.29 + non_sot * 0.035 + c * 0.025, 3)


def scrape_historical_results(season_codes: list = ["2526", "2627"]):
    conn = get_connection()
    cursor = conn.cursor()
    total_matches, total_odds = 0, 0

    for season in season_codes:
        for league_name, code in LEAGUE_CODES.items():
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            try:
                resp = requests.get(url, impersonate="chrome", timeout=20)
                if resp.status_code != 200:
                    continue
                df = pd.read_csv(StringIO(resp.text)).dropna(subset=["Date", "HomeTeam", "AwayTeam"])
            except Exception as e:
                print(f"[跳過] {league_name} ({season}) 下載失敗：{e}")
                continue

            for _, row in df.iterrows():
                if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
                    continue

                raw_date = str(row["Date"]).strip()
                try:
                    dt_obj = datetime.strptime(raw_date, "%d/%m/%Y")
                except ValueError:
                    dt_obj = datetime.strptime(raw_date, "%d/%m/%y")
                date_str = dt_obj.strftime("%d/%m/%Y")

                h = normalize_team(row["HomeTeam"])
                a = normalize_team(row["AwayTeam"])
                hg, ag = int(row["FTHG"]), int(row["FTAG"])
                match_id = f"{date_str}_{h}_{a}"

                cursor.execute("""
                INSERT INTO matches (match_id, league, season, date, home_team, away_team, home_goals, away_goals, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'finished')
                ON CONFLICT(match_id) DO UPDATE SET
                    home_goals=excluded.home_goals,
                    away_goals=excluded.away_goals,
                    status='finished'
                """, (match_id, league_name, season, date_str, h, a, hg, ag))
                total_matches += 1

                hs = int(row["HS"]) if pd.notna(row.get("HS")) else None
                as_ = int(row["AS"]) if pd.notna(row.get("AS")) else None
                hst = int(row["HST"]) if pd.notna(row.get("HST")) else None
                ast = int(row["AST"]) if pd.notna(row.get("AST")) else None
                hc = int(row["HC"]) if pd.notna(row.get("HC")) else None
                ac = int(row["AC"]) if pd.notna(row.get("AC")) else None

                h_xg = compute_synthetic_xg(hs, hst, hc)
                a_xg = compute_synthetic_xg(as_, ast, ac)

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
                """, (match_id, h_xg, a_xg, hs, as_, hst, ast, hc, ac))

                # 1. 臨場收盤賠率 (Closing Odds)
                h_odd = pick_first_valid(row, ["PSCH", "B365CH", "PSH", "B365H", "AvgCH", "AvgH"])
                d_odd = pick_first_valid(row, ["PSCD", "B365CD", "PSD", "B365D", "AvgCD", "AvgD"])
                a_odd = pick_first_valid(row, ["PSCA", "B365CA", "PSA", "B365A", "AvgCA", "AvgA"])

                # 2. 初盤賠率 (Opening Odds)
                open_h_odd = pick_first_valid(row, ["PSH", "B365H", "AvgH"])
                open_a_odd = pick_first_valid(row, ["PSA", "B365A", "AvgA"])

                # 3. 亞洲讓球盤口與賠率
                ah_line = pick_first_valid(row, ["AHCh", "AHh"], allow_zero_or_neg=True)
                h_ah = pick_first_valid(row, ["PCAHH", "B365CAHH", "PAHH", "B365AHH", "AvgCAHH", "AvgAHH"])
                a_ah = pick_first_valid(row, ["PCAHA", "B365CAHA", "PAHA", "B365AHA", "AvgCAHA", "AvgAHA"])

                # 4. 動態大小球盤口 (優先抓各場實際開出的動態盤口線，若無則抓標準 2.5 球線)
                dyn_ou_line = pick_first_valid(row, ["OUCh", "OUh", "TotalCh", "Totalh"])
                if dyn_ou_line is not None:
                    ou_line = dyn_ou_line
                    over_odd = pick_first_valid(row, ["PCO", "B365CO", "PO", "B365O", "PC>2.5", "B365C>2.5", "AvgC>2.5"])
                    under_odd = pick_first_valid(row, ["PCU", "B365CU", "PU", "B365U", "PC<2.5", "B365C<2.5", "AvgC<2.5"])
                else:
                    over_odd = pick_first_valid(row, ["PC>2.5", "B365C>2.5", "P>2.5", "B365>2.5", "AvgC>2.5", "Avg>2.5"])
                    under_odd = pick_first_valid(row, ["PC<2.5", "B365C<2.5", "P<2.5", "B365<2.5", "AvgC<2.5", "Avg<2.5"])
                    ou_line = 2.5 if (over_odd and under_odd) else None

                if h_odd and d_odd and a_odd:
                    cursor.execute("""
                    INSERT INTO odds_history (
                        match_id, bookmaker, timestamp, home_odd, draw_odd, away_odd,
                        asian_line, home_ah_odd, away_ah_odd, ou_line, over_odd, under_odd,
                        open_home_odd, open_away_odd
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(match_id, bookmaker) DO UPDATE SET
                        home_odd=excluded.home_odd,
                        draw_odd=excluded.draw_odd,
                        away_odd=excluded.away_odd,
                        asian_line=excluded.asian_line,
                        home_ah_odd=excluded.home_ah_odd,
                        away_ah_odd=excluded.away_ah_odd,
                        ou_line=excluded.ou_line,
                        over_odd=excluded.over_odd,
                        under_odd=excluded.under_odd,
                        open_home_odd=excluded.open_home_odd,
                        open_away_odd=excluded.open_away_odd
                    """, (match_id, "Market_Consensus", date_str, h_odd, d_odd, a_odd,
                          ah_line, h_ah, a_ah, ou_line, over_odd, under_odd,
                          open_h_odd, open_a_odd))
                    total_odds += 1

    conn.commit()
    conn.close()
    print(f"歷史賽果與動態盤口同步完成：共處理 {total_matches} 場賽事、{total_odds} 筆賠率紀錄。")


def scrape_espn_upcoming_fixtures():
    conn = get_connection()
    cursor = conn.cursor()
    scheduled_count = 0

    for league_name, code in ESPN_LEAGUES.items():
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard"
        try:
            resp = requests.get(url, impersonate="chrome", timeout=15)
            events = resp.json().get("events", [])
        except Exception as e:
            print(f"[警告] 無法取得 {league_name} 未來賽程：{e}")
            continue

        for ev in events:
            status_state = ev.get("status", {}).get("type", {}).get("state", "")
            if status_state != "pre":
                continue
            comp = ev.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue

            raw_h = next((c["team"]["displayName"] for c in competitors if c.get("homeAway") == "home"), "")
            raw_a = next((c["team"]["displayName"] for c in competitors if c.get("homeAway") == "away"), "")
            h, a = normalize_team(raw_h), normalize_team(raw_a)

            iso_date = ev.get("date", "")[:10]
            try:
                date_str = datetime.strptime(iso_date, "%Y-%m-%d").strftime("%d/%m/%Y")
            except ValueError:
                continue

            match_id = f"{date_str}_{h}_{a}"
            cursor.execute("""
            INSERT INTO matches (match_id, league, season, date, home_team, away_team, status)
            VALUES (?, ?, '2627', ?, ?, ?, 'scheduled')
            ON CONFLICT(match_id) DO NOTHING
            """, (match_id, league_name, date_str, h, a))
            scheduled_count += 1

    conn.commit()
    conn.close()
    print(f"未來賽程掃描完成：新增/確認 {scheduled_count} 場待開賽賽事。")


if __name__ == "__main__":
    init_db()
    scrape_historical_results(["2526", "2627"])
    scrape_espn_upcoming_fixtures()