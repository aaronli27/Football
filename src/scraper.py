import io
import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd
from curl_cffi import requests

DB_PATH = Path("data/soccer.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# 歐洲八大聯賽（football-data.co.uk 賽季分檔格式）
EURO_LEAGUES = {
    "ENG-Premier League": "E0",
    "ENG-Championship":   "E1",
    "ESP-La Liga":        "SP1",
    "GER-Bundesliga":     "D1",
    "ITA-Serie A":        "I1",
    "FRA-Ligue 1":        "F1",
    "NED-Eredivisie":     "N1",
    "POR-Primeira Liga":  "P1",
}

# 亞洲日韓頂級聯賽（football-data.co.uk/new/ 全歷史單檔格式）
EXTRA_LEAGUES = {
    "JPN-J1 League": "JPN",
    "KOR-K League 1": "KOR",
}

# 覆蓋近 6 個現代足球賽季（2020/21 至 2025/26，完整包含 Pinnacle 收盤與亞盤數據）
SEASONS = ["2021", "2122", "2223", "2324", "2425", "2526"]

# ESPN 賽程與傷停陣容 API 代碼對照表（覆蓋全部 10 大聯賽 + 歐聯）
ESPN_LEAGUES = {
    "ENG-Premier League":    "eng.1",
    "ENG-Championship":      "eng.2",
    "ESP-La Liga":           "esp.1",
    "GER-Bundesliga":        "ger.1",
    "ITA-Serie A":           "ita.1",
    "FRA-Ligue 1":           "fra.1",
    "NED-Eredivisie":        "ned.1",
    "POR-Primeira Liga":     "por.1",
    "JPN-J1 League":         "jpn.1",
    "KOR-K League 1":        "kor.1",
    "UEFA-Champions-League": "uefa.champions",
}

TEAM_NAME_MAP = {
    # 英超 & 英冠
    "Man United": "Manchester United", "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest", "Newcastle": "Newcastle United",
    "Wolves": "Wolverhampton Wanderers", "West Ham": "West Ham United",
    "Tottenham": "Tottenham Hotspur", "Brighton": "Brighton & Hove Albion",
    "Leicester": "Leicester City", "Ipswich": "Ipswich Town",
    "Leeds": "Leeds United", "Sheffield United": "Sheffield United",
    "Sheffield Weds": "Sheffield Wednesday", "West Brom": "West Bromwich Albion",
    "QPR": "Queens Park Rangers", "Blackburn": "Blackburn Rovers",
    "Preston": "Preston North End", "Coventry": "Coventry City",
    "Norwich": "Norwich City", "Middlesbrough": "Middlesbrough",
    "Hull": "Hull City", "Stoke": "Stoke City", "Swansea": "Swansea City",
    "Cardiff": "Cardiff City", "Bristol City": "Bristol City",
    "Luton": "Luton Town", "Burnley": "Burnley", "Sunderland": "Sunderland",
    "Fulham": "Fulham", "Bournemouth": "Bournemouth", "Brentford": "Brentford",
    "Crystal Palace": "Crystal Palace", "Everton": "Everton",
    "Southampton": "Southampton", "Watford": "Watford",
    # 西甲
    "Ath Madrid": "Atletico Madrid", "Ath Bilbao": "Athletic Club",
    "Betis": "Real Betis", "Sociedad": "Real Sociedad",
    "Celta": "Celta Vigo", "Vallecano": "Rayo Vallecano",
    "Alaves": "Deportivo Alaves", "Espanol": "Espanyol",
    "Valladolid": "Real Valladolid", "Granada": "Granada",
    "Cadiz": "Cadiz", "Almeria": "Almeria", "Elche": "Elche", "Levante": "Levante",
    # 德甲
    "Ein Frankfurt": "Eintracht Frankfurt", "Leverkusen": "Bayer Leverkusen",
    "Dortmund": "Borussia Dortmund", "M'gladbach": "Borussia Monchengladbach",
    "RB Leipzig": "RB Leipzig", "Stuttgart": "VfB Stuttgart",
    "Mainz": "Mainz 05", "Freiburg": "SC Freiburg",
    "Augsburg": "FC Augsburg", "Werder Bremen": "Werder Bremen",
    "Union Berlin": "1. FC Union Berlin", "Bochum": "VfL Bochum",
    "Heidenheim": "1. FC Heidenheim", "St Pauli": "FC St. Pauli",
    "Holstein Kiel": "Holstein Kiel", "Bayern Munich": "Bayern Munich",
    "FC Koln": "1. FC Koln", "Hertha": "Hertha Berlin", "Schalke 04": "Schalke 04",
    "Darmstadt": "SV Darmstadt 98", "Bielefeld": "Arminia Bielefeld",
    "Greuther Furth": "SpVgg Greuther Furth",
    # 意甲
    "Inter": "Internazionale", "AC Milan": "AC Milan",
    "Verona": "Hellas Verona", "Parma": "Parma Calcio 1913",
    "Como": "Como 1907", "Spezia": "Spezia", "Salernitana": "Salernitana",
    "Sampdoria": "Sampdoria", "Cremonese": "Cremonese", "Venezia": "Venezia",
    "Benevento": "Benevento", "Crotone": "Crotone",
    # 法甲
    "Paris SG": "Paris Saint-Germain", "St Etienne": "Saint-Etienne",
    "Lyon": "Olympique Lyonnais", "Marseille": "Olympique de Marseille",
    "Clermont": "Clermont Foot", "Ajaccio": "AC Ajaccio",
    "Troyes": "ESTAC Troyes", "Auxerre": "AJ Auxerre",
    "Bordeaux": "Girondins de Bordeaux", "Metz": "FC Metz",
    "Lorient": "FC Lorient", "Angers": "Angers SCO",
    # 荷甲 & 葡超
    "PSV Eindhoven": "PSV", "Feyenoord": "Feyenoord", "Ajax": "Ajax",
    "AZ Alkmaar": "AZ", "Twente": "FC Twente", "Utrecht": "FC Utrecht",
    "Sp Lisbon": "Sporting CP", "Sporting Lisbon": "Sporting CP",
    "FC Porto": "Porto", "Benfica": "Benfica", "Sp Braga": "Braga",
    "Guimaraes": "Vitoria de Guimaraes",
    # 日職 J1 & 韓職 K1
    "Vissel Kobe": "Vissel Kobe", "Sanfrecce Hiroshima": "Sanfrecce Hiroshima",
    "Machida Zelvia": "Machida Zelvia", "Gamba Osaka": "Gamba Osaka",
    "Kashima Antlers": "Kashima Antlers", "Urawa Reds": "Urawa Red Diamonds",
    "Kawasaki Frontale": "Kawasaki Frontale", "Yokohama F. Marinos": "Yokohama F. Marinos",
    "FC Tokyo": "FC Tokyo", "Cerezo Osaka": "Cerezo Osaka",
    "Nagoya Grampus": "Nagoya Grampus", "Kashiwa Reysol": "Kashiwa Reysol",
    "Kyoto Sanga": "Kyoto Sanga", "Avispa Fukuoka": "Avispa Fukuoka",
    "Tokyo Verdy": "Tokyo Verdy", "Albirex Niigata": "Albirex Niigata",
    "Shonan Bellmare": "Shonan Bellmare", "Jubilo Iwata": "Jubilo Iwata",
    "Ulsan HD": "Ulsan HD FC", "Ulsan Hyundai": "Ulsan HD FC",
    "Gimcheon Sangmu": "Gimcheon Sangmu", "Gangwon FC": "Gangwon FC",
    "Pohang Steelers": "Pohang Steelers", "FC Seoul": "FC Seoul",
    "Suwon FC": "Suwon FC", "Jeju United": "Jeju United",
    "Daejeon Hana Citizen": "Daejeon Hana Citizen", "Daejeon": "Daejeon Hana Citizen",
    "Gwangju FC": "Gwangju FC", "Jeonbuk Motors": "Jeonbuk Hyundai Motors",
    "Daegu FC": "Daegu FC", "Incheon United": "Incheon United",
}


def normalize_team(name: str) -> str:
    clean = str(name).strip()
    return TEAM_NAME_MAP.get(clean, clean)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS matches (
        match_id TEXT PRIMARY KEY,
        league TEXT,
        season TEXT,
        date TEXT,
        home_team TEXT,
        away_team TEXT,
        home_goals INTEGER,
        away_goals INTEGER,
        status TEXT
    );
    CREATE TABLE IF NOT EXISTS match_stats (
        match_id TEXT PRIMARY KEY,
        home_xg REAL,
        away_xg REAL,
        home_shots INTEGER,
        away_shots INTEGER,
        home_sot INTEGER,
        away_sot INTEGER,
        home_corners INTEGER,
        away_corners INTEGER
    );
    CREATE TABLE IF NOT EXISTS odds_history (
        match_id TEXT PRIMARY KEY,
        bookmaker TEXT DEFAULT 'Pinnacle/B365',
        home_odd REAL,
        draw_odd REAL,
        away_odd REAL,
        open_home_odd REAL,
        open_draw_odd REAL,
        open_away_odd REAL,
        asian_line REAL,
        home_ah_odd REAL,
        away_ah_odd REAL,
        ou_line REAL,
        over_odd REAL,
        under_odd REAL,
        timestamp TEXT
    );
    """)
    conn.commit()
    conn.close()


def _upsert_odds_row(cur, data_dict: dict):
    """動態對齊現有 odds_history 欄位並自動填補所有 NOT NULL 欄位，避免任何約束衝突"""
    cur.execute("PRAGMA table_info(odds_history)")
    cols_info = cur.fetchall()
    full_row = dict(data_dict)
    full_row.setdefault("bookmaker", "Pinnacle/B365")
    full_row.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    valid_cols = []
    vals = []
    for _, col_name, col_type, notnull, dflt_val, pk in cols_info:
        if col_name in full_row and full_row[col_name] is not None:
            valid_cols.append(col_name)
            vals.append(full_row[col_name])
        elif notnull == 1 and pk == 0 and dflt_val is None:
            valid_cols.append(col_name)
            t_upper = str(col_type).upper()
            vals.append(0.0 if ("REAL" in t_upper or "INT" in t_upper) else "Pinnacle/B365")

    cols_sql = ", ".join(valid_cols)
    placeholders = ", ".join(["?"] * len(valid_cols))
    cur.execute(f"INSERT OR REPLACE INTO odds_history ({cols_sql}) VALUES ({placeholders})", vals)


def _safe_float(val):
    try:
        v = float(val)
        return None if pd.isna(v) else v
    except Exception:
        return None


def _safe_int(val):
    try:
        v = float(val)
        return None if pd.isna(v) else int(v)
    except Exception:
        return None


def fetch_euro_historical():
    """抓取歐洲八大聯賽近 6 個賽季（2020/21 - 2025/26）歷史賽果與賠率"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    total_inserted = 0

    for league_name, code in EURO_LEAGUES.items():
        for season in SEASONS:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            try:
                resp = requests.get(url, impersonate="chrome", timeout=20)
                if resp.status_code != 200:
                    continue
                df = pd.read_csv(io.StringIO(resp.text))
                df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])

                for _, row in df.iterrows():
                    d_str = str(row["Date"]).strip()
                    if len(d_str.split("/")[-1]) == 2:
                        parts = d_str.split("/")
                        d_str = f"{parts[0]}/{parts[1]}/20{parts[2]}"

                    h_team = normalize_team(row["HomeTeam"])
                    a_team = normalize_team(row["AwayTeam"])
                    hg, ag = int(row["FTHG"]), int(row["FTAG"])
                    mid = f"{league_name}_{d_str}_{h_team}_vs_{a_team}".replace(" ", "_")

                    cur.execute(
                        "INSERT OR REPLACE INTO matches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (mid, league_name, season, d_str, h_team, a_team, hg, ag, "finished"),
                    )

                    hs, as_ = _safe_int(row.get("HS")), _safe_int(row.get("AS"))
                    hst, ast = _safe_int(row.get("HST")), _safe_int(row.get("AST"))
                    hc, ac = _safe_int(row.get("HC")), _safe_int(row.get("AC"))

                    if hs is not None and hst is not None and as_ is not None and ast is not None:
                        h_xg = round(hst * 0.29 + max(0, hs - hst) * 0.035 + (hc or 0) * 0.015, 3)
                        a_xg = round(ast * 0.29 + max(0, as_ - ast) * 0.035 + (ac or 0) * 0.015, 3)
                    else:
                        h_xg, a_xg = None, None

                    cur.execute(
                        "INSERT OR REPLACE INTO match_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (mid, h_xg, a_xg, hs, as_, hst, ast, hc, ac),
                    )

                    h_odd = _safe_float(row.get("PSCH")) or _safe_float(row.get("B365CH")) or _safe_float(row.get("B365H"))
                    d_odd = _safe_float(row.get("PSCD")) or _safe_float(row.get("B365CD")) or _safe_float(row.get("B365D"))
                    a_odd = _safe_float(row.get("PSCA")) or _safe_float(row.get("B365CA")) or _safe_float(row.get("B365A"))
                    open_h = _safe_float(row.get("PSH")) or _safe_float(row.get("B365H"))
                    open_d = _safe_float(row.get("PSD")) or _safe_float(row.get("B365D"))
                    open_a = _safe_float(row.get("PSA")) or _safe_float(row.get("B365A"))

                    ah_line = _safe_float(row.get("AHCh")) or _safe_float(row.get("AHh"))
                    h_ah = _safe_float(row.get("PCAHH")) or _safe_float(row.get("B365CAHH")) or _safe_float(row.get("B365AHH"))
                    a_ah = _safe_float(row.get("PCAHA")) or _safe_float(row.get("B365CAHA")) or _safe_float(row.get("B365AHA"))

                    over_o = _safe_float(row.get("PC>2.5")) or _safe_float(row.get("B365C>2.5")) or _safe_float(row.get("B365>2.5"))
                    under_o = _safe_float(row.get("PC<2.5")) or _safe_float(row.get("B365C<2.5")) or _safe_float(row.get("B365<2.5"))

                    if h_odd and d_odd and a_odd:
                        _upsert_odds_row(cur, {
                            "match_id": mid,
                            "bookmaker": "Pinnacle/B365",
                            "home_odd": h_odd,
                            "draw_odd": d_odd,
                            "away_odd": a_odd,
                            "open_home_odd": open_h,
                            "open_draw_odd": open_d,
                            "open_away_odd": open_a,
                            "asian_line": ah_line,
                            "home_ah_odd": h_ah,
                            "away_ah_odd": a_ah,
                            "ou_line": 2.5,
                            "over_odd": over_o,
                            "under_odd": under_o,
                        })
                    total_inserted += 1
            except Exception as e:
                print(f"[警告] 抓取 {league_name} ({season}) 失敗: {e}")

    conn.commit()
    conn.close()
    print(f"[歐洲八大聯賽] 已同步近 6 季共 {total_inserted} 場歷史賽果與賠率。")


def fetch_extra_asian_historical():
    """抓取日職 J1 與韓職 K1 近 6 個賽季（約 2100 天）歷史賽果與賠率"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    total_inserted = 0
    cutoff_dt = datetime.now() - pd.Timedelta(days=2100)

    for league_name, code in EXTRA_LEAGUES.items():
        url = f"https://www.football-data.co.uk/new/{code}.csv"
        try:
            resp = requests.get(url, impersonate="chrome", timeout=25)
            if resp.status_code != 200:
                continue
            df = pd.read_csv(io.StringIO(resp.text))
            df = df.dropna(subset=["Date", "Home", "Away", "HG", "AG"])

            for _, row in df.iterrows():
                d_str = str(row["Date"]).strip()
                if len(d_str.split("/")[-1]) == 2:
                    parts = d_str.split("/")
                    d_str = f"{parts[0]}/{parts[1]}/20{parts[2]}"

                dt_obj = pd.to_datetime(d_str, format="%d/%m/%Y", errors="coerce")
                if pd.isna(dt_obj) or dt_obj < cutoff_dt:
                    continue

                h_team = normalize_team(row["Home"])
                a_team = normalize_team(row["Away"])
                hg, ag = int(row["HG"]), int(row["AG"])
                season_str = str(row.get("Season", dt_obj.year))
                mid = f"{league_name}_{d_str}_{h_team}_vs_{a_team}".replace(" ", "_")

                cur.execute(
                    "INSERT OR REPLACE INTO matches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (mid, league_name, season_str, d_str, h_team, a_team, hg, ag, "finished"),
                )

                h_odd = _safe_float(row.get("PSCH")) or _safe_float(row.get("MaxCH")) or _safe_float(row.get("AvgCH"))
                d_odd = _safe_float(row.get("PSCD")) or _safe_float(row.get("MaxCD")) or _safe_float(row.get("AvgCD"))
                a_odd = _safe_float(row.get("PSCA")) or _safe_float(row.get("MaxCA")) or _safe_float(row.get("AvgCA"))
                open_h = _safe_float(row.get("PSH")) or _safe_float(row.get("AvgH"))
                open_d = _safe_float(row.get("PSD")) or _safe_float(row.get("AvgD"))
                open_a = _safe_float(row.get("PSA")) or _safe_float(row.get("AvgA"))

                if h_odd and d_odd and a_odd:
                    _upsert_odds_row(cur, {
                        "match_id": mid,
                        "bookmaker": "Pinnacle/Avg",
                        "home_odd": h_odd,
                        "draw_odd": d_odd,
                        "away_odd": a_odd,
                        "open_home_odd": open_h,
                        "open_draw_odd": open_d,
                        "open_away_odd": open_a,
                        "ou_line": 2.5,
                    })
                total_inserted += 1
        except Exception as e:
            print(f"[警告] 抓取 {league_name} 失敗: {e}")

    conn.commit()
    conn.close()
    print(f"[亞洲日韓聯賽] 已同步近 6 季共 {total_inserted} 場日職 J1 / 韓職 K1 賽果與賠率。")


def fetch_espn_upcoming():
    """從 ESPN Scoreboard API 抓取 10 大聯賽 + 歐聯未來 7 天待開賽賽程"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM matches WHERE status = 'scheduled'")
    count = 0

    sql = (
        "INSERT OR REPLACE INTO matches "
        "(match_id, league, season, date, home_team, away_team, home_goals, away_goals, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )

    for league_name, code in ESPN_LEAGUES.items():
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard"
        try:
            resp = requests.get(url, impersonate="chrome", timeout=15).json()
            for ev in resp.get("events", []):
                state = ev.get("status", {}).get("type", {}).get("state", "")
                if state != "pre":
                    continue
                comps = ev.get("competitions", [{}])[0].get("competitors", [])
                if len(comps) < 2:
                    continue
                raw_h = next((c["team"]["displayName"] for c in comps if c.get("homeAway") == "home"), "")
                raw_a = next((c["team"]["displayName"] for c in comps if c.get("homeAway") == "away"), "")
                h_team, a_team = normalize_team(raw_h), normalize_team(raw_a)
                dt_iso = ev.get("date", "")[:10]
                dt_fmt = datetime.strptime(dt_iso, "%Y-%m-%d").strftime("%d/%m/%Y")
                mid = f"SCHED_{league_name}_{dt_fmt}_{h_team}_vs_{a_team}".replace(" ", "_")

                cur.execute(sql, (mid, league_name, "2526", dt_fmt, h_team, a_team, None, None, "scheduled"))
                count += 1
        except Exception as e:
            print(f"[警告] 抓取 {league_name} 未來賽程失敗: {e}")

    conn.commit()
    conn.close()
    print(f"[未來賽程] 已從 ESPN 同步 {count} 場待開賽賽事（覆蓋 10 大聯賽 + 歐聯）。")


if __name__ == "__main__":
    init_db()
    fetch_euro_historical()
    fetch_extra_asian_historical()
    fetch_espn_upcoming()