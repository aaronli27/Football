import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from curl_cffi import requests

DB_PATH = Path("data/soccer.db")

ESPN_LEAGUE_CODES = {
    "ENG-Premier League": "eng.1",
    "ENG-Championship":   "eng.2",
    "ESP-La Liga":        "esp.1",
    "GER-Bundesliga":     "ger.1",
    "ITA-Serie A":        "ita.1",
    "FRA-Ligue 1":        "fra.1",
    "NED-Eredivisie":     "ned.1",
    "POR-Primeira Liga":  "por.1",
    "JPN-J1 League":      "jpn.1",
}


def upgrade_db_schema():
    """在現有 match_stats 資料表安全新增 home_npxg, away_npxg, home_ppda, away_ppda 欄位"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(match_stats)")
    existing_cols = {row[1] for row in cur.fetchall()}

    new_cols = {
        "home_npxg": "REAL",
        "away_npxg": "REAL",
        "home_ppda": "REAL",
        "away_ppda": "REAL",
    }
    for col, dtype in new_cols.items():
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE match_stats ADD COLUMN {col} {dtype}")
            print(f"[資料庫升級] 已新增欄位: match_stats.{col}")

    conn.commit()
    conn.close()


def _parse_espn_npxg_and_ppda(summary_json: dict, base_h_xg: float, base_a_xg: float) -> tuple:
    """從 ESPN summary JSON 提取十二碼次數（計算 npxG）與傳球/防守動作（計算 PPDA）"""
    teams = summary_json.get("boxscore", {}).get("teams", [])
    if len(teams) < 2:
        return None, None, None, None

    h_id, a_id = None, None
    h_stats, a_stats = {}, {}

    for t_obj in teams:
        side = t_obj.get("homeAway", "")
        tid = str(t_obj.get("team", {}).get("id", ""))
        smap = {}
        for st in t_obj.get("statistics", []):
            try:
                smap[st.get("name", "")] = float(st.get("displayValue", 0))
            except Exception:
                pass
        if side == "home":
            h_id, h_stats = tid, smap
        elif side == "away":
            a_id, a_stats = tid, smap

    # 1. 計算雙方十二碼次數並扣除 0.76 xG/球 得到真實 npxG
    h_pens, a_pens = 0, 0
    for ev in summary_json.get("keyEvents", []):
        ev_type = str(ev.get("type", {}).get("text", "")).lower()
        ev_text = str(ev.get("text", "")).lower()
        if "penalty" in ev_type or "penalty" in ev_text:
            # 排除十二碼大戰（只計算 90 分鐘常規時間內判罰）
            period = ev.get("period", {}).get("number", 1)
            if period <= 2:
                team_tid = str(ev.get("team", {}).get("id", ""))
                if team_tid == h_id:
                    h_pens += 1
                elif team_tid == a_id:
                    a_pens += 1

    h_npxg = round(max(0.0, (base_h_xg or 0.0) - 0.76 * h_pens), 3) if base_h_xg is not None else None
    a_npxg = round(max(0.0, (base_a_xg or 0.0) - 0.76 * a_pens), 3) if base_a_xg is not None else None

    # 2. 計算 PPDA = 對手總傳球數 / (本方鏟球 + 攔截 + 犯規)
    h_passes = h_stats.get("totalPasses") or h_stats.get("accuratePasses")
    a_passes = a_stats.get("totalPasses") or a_stats.get("accuratePasses")

    h_def_actions = (
        h_stats.get("totalTackles", h_stats.get("effectiveTackles", 0))
        + h_stats.get("interceptions", 0)
        + h_stats.get("foulsCommitted", 0)
    )
    a_def_actions = (
        a_stats.get("totalTackles", a_stats.get("effectiveTackles", 0))
        + a_stats.get("interceptions", 0)
        + a_stats.get("foulsCommitted", 0)
    )

    h_ppda = round(min(35.0, max(3.0, a_passes / h_def_actions)), 2) if (a_passes and h_def_actions > 0) else None
    a_ppda = round(min(35.0, max(3.0, h_passes / a_def_actions)), 2) if (h_passes and a_def_actions > 0) else None

    return h_npxg, a_npxg, h_ppda, a_ppda


def fetch_fotmob_next_data_xg(fotmob_match_url: str) -> dict:
    """免簽名備援：直接解析 FotMob 網頁內嵌 __NEXT_DATA__ 提取官方 Non-penalty xG"""
    try:
        resp = requests.get(fotmob_match_url, impersonate="chrome", timeout=15)
        if resp.status_code != 200:
            return {}
        m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', resp.text)
        if not m:
            return {}
        payload = json.loads(m.group(1))
        stats_all = (
            payload.get("props", {})
            .get("pageProps", {})
            .get("content", {})
            .get("stats", {})
            .get("Periods", {})
            .get("All", {})
            .get("stats", [])
        )
        out = {}
        for grp in stats_all:
            for item in grp.get("stats", []):
                key = str(item.get("key", "")).lower()
                if key in ("expected_goals", "non_penalty_xg", "big_chance"):
                    out[key] = item.get("stats")
        return out
    except Exception:
        return {}


def _fetch_espn_date_events(code: str, d_str: str) -> list:
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard?dates={d_str}"
    out = []
    try:
        resp = requests.get(url, impersonate="chrome", timeout=12)
        if resp.status_code != 200:
            return out
        for ev in resp.json().get("events", []):
            if ev.get("status", {}).get("type", {}).get("state") != "post":
                continue
            comps = ev.get("competitions", [{}])[0].get("competitors", [])
            if len(comps) < 2:
                continue
            hc = next((c for c in comps if c.get("homeAway") == "home"), {})
            ac = next((c for c in comps if c.get("homeAway") == "away"), {})
            out.append({
                "ev_id": ev.get("id"),
                "ev_dt": datetime.strptime(ev.get("date", "")[:10], "%Y-%m-%d"),
                "home": hc.get("team", {}).get("displayName", ""),
                "away": ac.get("team", {}).get("displayName", ""),
                "hg": int(hc.get("score", -1)),
                "ag": int(ac.get("score", -1)),
            })
    except Exception:
        pass
    return out


def backfill_npxg_and_ppda(days_back: int = 400):
    """為近一年（預設 400 天，覆蓋實時模型 365 天回溯窗口）的所有完賽場次補齊 npxG 與 PPDA"""
    upgrade_db_schema()
    cutoff_dt = datetime.now() - timedelta(days=days_back)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT m.match_id, m.league, m.date, m.home_team, m.away_team, m.home_goals, m.away_goals,
               s.home_xg, s.away_xg
        FROM matches m
        JOIN match_stats s ON m.match_id = s.match_id
        WHERE m.status = 'finished'
          AND (s.home_ppda IS NULL OR s.home_npxg IS NULL)
    """)
    rows = cur.fetchall()
    conn.close()

    by_league = {lg: [] for lg in ESPN_LEAGUE_CODES}
    for mid, lg, d_str, ht, at, hg, ag, hxg, axg in rows:
        if lg not in by_league:
            continue
        try:
            dt = datetime.strptime(d_str, "%d/%m/%Y")
            if dt >= cutoff_dt:
                by_league[lg].append({
                    "match_id": mid, "date": dt, "home_team": ht, "away_team": at,
                    "hg": int(hg), "ag": int(ag), "h_xg": hxg, "a_xg": axg,
                })
        except Exception:
            continue

    total_updated = 0
    for lg_name, targets in by_league.items():
        if not targets:
            continue
        code = ESPN_LEAGUE_CODES[lg_name]
        q_dates = sorted({(r["date"] + timedelta(days=d)).strftime("%Y%m%d") for r in targets for d in (0, -1)})
        print(f"\n[npxG & PPDA 計算引擎] 正在掃描 {lg_name} ({len(targets)} 場)...")

        events_map = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(_fetch_espn_date_events, code, d) for d in q_dates]
            for f in as_completed(futs):
                for ev in f.result():
                    events_map[ev["ev_id"]] = ev

        paired = []
        unmatched = list(targets)
        for ev in events_map.values():
            for rec in unmatched:
                if rec["hg"] != ev["hg"] or rec["ag"] != ev["ag"]:
                    continue
                if abs((rec["date"] - ev["ev_dt"]).days) > 1:
                    continue
                sim_h = SequenceMatcher(None, rec["home_team"].lower(), ev["home"].lower()).ratio()
                sim_a = SequenceMatcher(None, rec["away_team"].lower(), ev["away"].lower()).ratio()
                if sim_h >= 0.50 or rec["home_team"].lower() in ev["home"].lower():
                    if sim_a >= 0.50 or rec["away_team"].lower() in ev["away"].lower():
                        paired.append((rec, ev["ev_id"]))
                        unmatched.remove(rec)
                        break

        def _worker(rec_item, eid):
            url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/summary?event={eid}"
            try:
                r = requests.get(url, impersonate="chrome", timeout=12)
                if r.status_code == 200:
                    hnx, anx, hpp, app = _parse_espn_npxg_and_ppda(r.json(), rec_item["h_xg"], rec_item["a_xg"])
                    return (hnx, anx, hpp, app, rec_item["match_id"])
            except Exception:
                pass
            return None

        updates = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(_worker, rec, eid) for rec, eid in paired]
            for f in as_completed(futs):
                res = f.result()
                if res and res[2] is not None:
                    updates.append(res)

        if updates:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.executemany(
                "UPDATE match_stats SET home_npxg = ?, away_npxg = ?, home_ppda = ?, away_ppda = ? WHERE match_id = ?",
                updates,
            )
            conn.commit()
            conn.close()
            total_updated += len(updates)
            print(f"  ✓ {lg_name} 已寫入 {len(updates)} 場真實 npxG 與 PPDA 壓迫指數。")

    print(f"\n[全部完成] 共更新 {total_updated} 場比賽的 npxG 與 PPDA 指標！")


if __name__ == "__main__":
    backfill_npxg_and_ppda(days_back=400)