import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd
from curl_cffi import requests
from scraper import ESPN_LEAGUES, normalize_team

DB_PATH = Path("data/soccer.db")


def get_auto_rest_days(league: str, home_team: str, away_team: str) -> tuple:
    """從 SQLite 資料庫自動計算兩隊距離上一場正式比賽的真實休息天數"""
    if not DB_PATH.exists():
        return 6, 6

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT date, home_team, away_team, status FROM matches WHERE league = ?",
        conn,
        params=(league,),
    )
    conn.close()

    if df.empty:
        return 6, 6

    df["dt"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # 尋找本場待開賽日期（若無則預設今天）
    sched = df[
        (df["status"] == "scheduled")
        & (df["home_team"] == home_team)
        & (df["away_team"] == away_team)
        & (df["dt"] >= today)
    ].sort_values("dt")
    target_dt = sched.iloc[0]["dt"] if not sched.empty else today

    finished = df[(df["status"] == "finished") & (df["dt"] < target_dt)]

    def calc_team_rest(team: str) -> int:
        t_games = finished[(finished["home_team"] == team) | (finished["away_team"] == team)].sort_values("dt")
        if t_games.empty:
            return 6
        last_dt = t_games.iloc[-1]["dt"]
        days = int((target_dt - last_dt).days)
        return max(2, min(21, days))

    return calc_team_rest(home_team), calc_team_rest(away_team)


def _map_espn_formation(raw_form: str) -> str:
    """將 ESPN 陣型字串 (如 '4-3-3', '3-4-2-1') 對應至系統 FORMATION_PROFILES 鍵值"""
    f = str(raw_form).strip()
    if f.startswith("4-3-3"):
        return "4-3-3 (高位逼搶/邊路推進)"
    if f.startswith("4-2-3-1") or f.startswith("4-1-4-1"):
        return "4-2-3-1 (均衡控球/雙後腰)"
    if f.startswith("3-5-2") or f.startswith("3-1-4-2"):
        return "3-5-2 (三中衛/中路密集)"
    if f.startswith("3-4-3") or f.startswith("3-4-2-1"):
        return "3-4-3 (翼衛壓上/高節奏對攻)"
    if f.startswith("5-3-2") or f.startswith("5-4-1") or f.startswith("4-5-1"):
        return "5-3-2 / 5-4-1 (低位死守/防反)"
    if f.startswith("4-4-2") or f.startswith("4-4-1-1"):
        return "4-4-2 (雙前鋒/平行中場)"
    return "標準 / 未知"


def fetch_match_lineup_and_injuries(league: str, home_team: str, away_team: str) -> dict:
    """
    透過 ESPN Scoreboard + Summary Endpoint 自動抓取：
    1. 兩隊當輪傷停名單 (Injuries / Suspensions) 並依位置計算進攻/防守折減係數
    2. 賽前 1 小時官方正選 11 人名單 (Starting XI) 與實際陣型
    3. 資料庫真實休息天數
    """
    h_rest, a_rest = get_auto_rest_days(league, home_team, away_team)
    result = {
        "found_event": False,
        "home_rest_days": h_rest,
        "away_rest_days": a_rest,
        "home_formation": "標準 / 未知",
        "away_formation": "標準 / 未知",
        "lineup_confirmed": False,
        "home_xi": [],
        "away_xi": [],
        "home_injuries": [],
        "away_injuries": [],
        "h_att_mult": 1.0,
        "h_def_mult": 1.0,
        "a_att_mult": 1.0,
        "a_def_mult": 1.0,
        "summary_note": f"自動計算休息天數：主隊休 {h_rest} 天 vs 客隊休 {a_rest} 天（暫無 ESPN 傷停明細）",
    }

    code = ESPN_LEAGUES.get(league)
    if not code:
        return result

    try:
        sb_url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard"
        sb_resp = requests.get(sb_url, impersonate="chrome", timeout=12).json()
        event_id = None
        for ev in sb_resp.get("events", []):
            comps = ev.get("competitions", [{}])[0].get("competitors", [])
            if len(comps) < 2:
                continue
            raw_h = next((c["team"]["displayName"] for c in comps if c.get("homeAway") == "home"), "")
            raw_a = next((c["team"]["displayName"] for c in comps if c.get("homeAway") == "away"), "")
            if normalize_team(raw_h) == home_team and normalize_team(raw_a) == away_team:
                event_id = ev.get("id")
                break

        if not event_id:
            return result

        result["found_event"] = True
        sum_url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/summary?event={event_id}"
        sum_data = requests.get(sum_url, impersonate="chrome", timeout=12).json()

        # 1. 解析官方正選 11 人與實際陣型 (rosters)
        for roster_block in sum_data.get("rosters", []):
            side = roster_block.get("homeAway", "")
            raw_form = roster_block.get("formation", "")
            starters = [
                p.get("athlete", {}).get("displayName", "")
                for p in roster_block.get("roster", [])
                if p.get("starter") is True
            ]
            if side == "home":
                if raw_form:
                    result["home_formation"] = _map_espn_formation(raw_form)
                if len(starters) >= 11:
                    result["home_xi"] = starters[:11]
                    result["lineup_confirmed"] = True
            elif side == "away":
                if raw_form:
                    result["away_formation"] = _map_espn_formation(raw_form)
                if len(starters) >= 11:
                    result["away_xi"] = starters[:11]
                    result["lineup_confirmed"] = True

        # 2. 解析傷停名單 (injuries) 並依前鋒/中場/後衛計算動態 λ 折減
        def eval_injury_impact(inj_list: list):
            att_penalty = 0.0
            def_penalty = 0.0
            formatted = []
            for item in inj_list:
                athlete = item.get("athlete", {})
                name = athlete.get("displayName", "未知球員")
                pos = athlete.get("position", {}).get("abbreviation", "")
                status = item.get("status", "Out")
                is_out = any(k in str(status).lower() for k in ["out", "suspend", "injur"])
                weight = 1.0 if is_out else 0.45  # 確定缺陣權重 1.0，上陣成疑 0.45

                if pos in ["F", "FW", "ST", "CF", "LW", "RW"]:
                    att_penalty += 0.045 * weight
                elif pos in ["M", "MF", "CM", "CAM", "CDM", "LM", "RM"]:
                    att_penalty += 0.030 * weight
                    def_penalty += 0.020 * weight
                elif pos in ["D", "DF", "CB", "LB", "RB", "G", "GK"]:
                    def_penalty += 0.035 * weight
                else:
                    att_penalty += 0.020 * weight

                formatted.append(f"{name} ({pos or '主力'} - {status})")

            # 進攻折減上限 -15% (最低乘數 0.85)，防守失球風險上限 +12%
            att_mult = round(max(0.85, 1.0 - att_penalty), 3)
            def_mult = round(min(1.12, 1.0 + def_penalty), 3)
            return att_mult, def_mult, formatted

        for inj_team in sum_data.get("injuries", []):
            t_name = normalize_team(inj_team.get("team", {}).get("displayName", ""))
            injs = inj_team.get("injuries", [])
            if t_name == home_team:
                ha, hd, h_fmt = eval_injury_impact(injs)
                result["h_att_mult"], result["h_def_mult"], result["home_injuries"] = ha, hd, h_fmt
            elif t_name == away_team:
                aa, ad, a_fmt = eval_injury_impact(injs)
                result["a_att_mult"], result["a_def_mult"], result["away_injuries"] = aa, ad, a_fmt

        h_drop = round((1.0 - result["h_att_mult"]) * 100, 1)
        a_drop = round((1.0 - result["a_att_mult"]) * 100, 1)
        xi_tag = "✅ 已鎖定賽前官方正選 11 人" if result["lineup_confirmed"] else "⏳ 尚未公布賽前正選（已套用最新傷停折減）"
        result["summary_note"] = (
            f"{xi_tag} ｜ 傷停進攻折減：主隊 **-{h_drop}%**（缺陣 {len(result['home_injuries'])} 人）vs "
            f"客隊 **-{a_drop}%**（缺陣 {len(result['away_injuries'])} 人）"
        )
        return result

    except Exception:
        return result