import requests
from scraper import normalize_team

ESPN_LEAGUE_MAP = {
    "ENG-Premier League": "eng.1",
    "ESP-La Liga": "esp.1",
    "GER-Bundesliga": "ger.1",
    "ITA-Serie A": "ita.1",
    "FRA-Ligue 1": "fra.1",
    "UEFA-Champions-League": "uefa.champions",
}

def fetch_espn_live_odds(league: str) -> dict:
    """
    從 ESPN 公開賽程介面抓取賽前主客和與大小球賠率，並轉為十進制賠率。
    回傳以 '標準化主隊 vs 標準化客隊' 為鍵值的字典。
    """
    code = ESPN_LEAGUE_MAP.get(league)
    if not code:
        return {}
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        events = resp.json().get("events", [])
    except Exception:
        return {}

    odds_map = {}
    for ev in events:
        comp = ev.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue
        raw_h = next((c["team"]["displayName"] for c in competitors if c.get("homeAway") == "home"), "")
        raw_a = next((c["team"]["displayName"] for c in competitors if c.get("homeAway") == "away"), "")
        home, away = normalize_team(raw_h), normalize_team(raw_a)

        odds_list = comp.get("odds") or []
        if not odds_list or not isinstance(odds_list[0], dict):
            continue
        primary = odds_list[0]

        def ml_to_dec(ml):
            if ml is None:
                return None
            ml = float(ml)
            return round(1 + (ml / 100.0), 2) if ml > 0 else round(1 + (100.0 / abs(ml)), 2)

        h_ml = (primary.get("homeTeamOdds") or {}).get("moneyLine")
        a_ml = (primary.get("awayTeamOdds") or {}).get("moneyLine")
        d_ml = (primary.get("drawOdds") or {}).get("moneyLine")
        ou_line = primary.get("overUnder")

        odds_map[f"{home} vs {away}"] = {
            "home_odd": ml_to_dec(h_ml),
            "draw_odd": ml_to_dec(d_ml),
            "away_odd": ml_to_dec(a_ml),
            "ou_line": float(ou_line) if ou_line else 2.5,
        }
    return odds_map