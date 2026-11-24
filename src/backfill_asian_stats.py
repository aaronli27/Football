import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from curl_cffi import requests

DB_PATH = Path("data/soccer.db")

ESPN_ASIAN_CODES = {
    "JPN-J1 League": "jpn.1",
    "KOR-K League 1": "kor.1",
}

ASIAN_TEAM_ALIASES = {
    "urawa reds": "urawa red diamonds",
    "urawa": "urawa red diamonds",
    "yokohama f.marinos": "yokohama f. marinos",
    "yokohama marinos": "yokohama f. marinos",
    "kashima": "kashima antlers",
    "kawasaki": "kawasaki frontale",
    "gamba": "gamba osaka",
    "cerezo": "cerezo osaka",
    "vissel": "vissel kobe",
    "sanfrecce": "sanfrecce hiroshima",
    "nagoya": "nagoya grampus",
    "kashiwa": "kashiwa reysol",
    "kyoto": "kyoto sanga",
    "avispa": "avispa fukuoka",
    "albirex": "albirex niigata",
    "shonan": "shonan bellmare",
    "jubilo": "jubilo iwata",
    "machida": "machida zelvia",
    "consadole sapporo": "hokkaido consadole sapporo",
    "sapporo": "hokkaido consadole sapporo",
    "ulsan hyundai": "ulsan hd fc",
    "ulsan hd": "ulsan hd fc",
    "ulsan": "ulsan hd fc",
    "jeonbuk motors": "jeonbuk hyundai motors",
    "jeonbuk": "jeonbuk hyundai motors",
    "pohang": "pohang steelers",
    "gimcheon": "gimcheon sangmu",
    "gangwon": "gangwon fc",
    "gwangju": "gwangju fc",
    "daejeon": "daejeon hana citizen",
    "daejeon citizen": "daejeon hana citizen",
    "jeju": "jeju united",
    "jeju sk": "jeju united",
    "daegu": "daegu fc",
    "incheon": "incheon united",
    "suwon": "suwon fc",
    "suwon bluewings": "suwon samsung bluewings",
    "seoul": "fc seoul",
}


def _clean_name(name: str) -> str:
    s = str(name).lower().strip()
    return ASIAN_TEAM_ALIASES.get(s, s)


def _similar(a: str, b: str) -> float:
    ca, cb = _clean_name(a), _clean_name(b)
    if ca == cb or ca in cb or cb in ca:
        return 1.0
    return SequenceMatcher(None, ca, cb).ratio()


def load_missing_asian_matches() -> dict:
    """讀取資料庫中尚未有射正統計的日職 J1 與韓職 K1 完賽紀錄"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT m.match_id, m.league, m.date, m.home_team, m.away_team, m.home_goals, m.away_goals
        FROM matches m
        LEFT JOIN match_stats s ON m.match_id = s.match_id
        WHERE m.league IN ('JPN-J1 League', 'KOR-K League 1')
          AND m.status = 'finished'
          AND (s.home_sot IS NULL OR s.away_sot IS NULL)
    """)
    rows = cur.fetchall()
    conn.close()

    by_league = {"JPN-J1 League": [], "KOR-K League 1": []}
    for mid, lg, d_str, ht, at, hg, ag in rows:
        try:
            dt = datetime.strptime(d_str, "%d/%m/%Y")
            by_league[lg].append({
                "match_id": mid,
                "date": dt,
                "year": dt.year,
                "home_team": ht,
                "away_team": at,
                "home_goals": int(hg),
                "away_goals": int(ag),
            })
        except Exception:
            continue
    return by_league


def parse_espn_boxscore(summary_json: dict) -> tuple:
    """從 ESPN summary JSON 提取主客隊總射門、射正與角球"""
    hs = as_ = hst = ast = hc = ac = None
    teams = summary_json.get("boxscore", {}).get("teams", [])
    for t_obj in teams:
        side = t_obj.get("homeAway", "")
        stat_map = {}
        for st in t_obj.get("statistics", []):
            name = st.get("name", "")
            val = st.get("displayValue", "")
            try:
                stat_map[name] = int(float(val))
            except Exception:
                pass

        if side == "home":
            hs = stat_map.get("totalShots")
            hst = stat_map.get("shotsOnTarget")
            hc = stat_map.get("wonCorners")
        elif side == "away":
            as_ = stat_map.get("totalShots")
            ast = stat_map.get("shotsOnTarget")
            ac = stat_map.get("wonCorners")

    return hs, as_, hst, ast, hc, ac


def _fetch_single_event_stat(code: str, ev_id: str):
    sum_url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/summary?event={ev_id}"
    try:
        resp = requests.get(sum_url, impersonate="chrome", timeout=12)
        if resp.status_code == 200:
            return parse_espn_boxscore(resp.json())
    except Exception:
        pass
    return (None, None, None, None, None, None)


def _fetch_scoreboard_date(code: str, d_str: str) -> list:
    sb_url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard?dates={d_str}"
    results = []
    try:
        resp = requests.get(sb_url, impersonate="chrome", timeout=12)
        if resp.status_code != 200:
            return results
        for ev in resp.json().get("events", []):
            if ev.get("status", {}).get("type", {}).get("state") != "post":
                continue
            ev_id = ev.get("id")
            ev_dt = datetime.strptime(ev.get("date", "")[:10], "%Y-%m-%d")
            comps = ev.get("competitions", [{}])[0].get("competitors", [])
            if len(comps) < 2:
                continue
            h_comp = next((c for c in comps if c.get("homeAway") == "home"), {})
            a_comp = next((c for c in comps if c.get("homeAway") == "away"), {})
            try:
                ehg = int(h_comp.get("score", -1))
                eag = int(a_comp.get("score", -1))
            except Exception:
                continue
            results.append({
                "ev_id": ev_id,
                "ev_dt": ev_dt,
                "home": h_comp.get("team", {}).get("displayName", ""),
                "away": a_comp.get("team", {}).get("displayName", ""),
                "hg": ehg,
                "ag": eag,
            })
    except Exception:
        pass
    return results


def probe_valid_years(code: str, targets: list) -> set:
    """快速抽樣探測各年份 ESPN 是否有收錄射正統計，自動跳過無數據年份"""
    years = sorted({r["year"] for r in targets})
    valid_years = set()
    for yr in years:
        yr_recs = [r for r in targets if r["year"] == yr]
        mid_rec = yr_recs[len(yr_recs) // 2]
        has_stats = False
        for delta in (0, -1):
            d_str = (mid_rec["date"] + timedelta(days=delta)).strftime("%Y%m%d")
            evs = _fetch_scoreboard_date(code, d_str)
            if evs:
                _, _, hst, ast, _, _ = _fetch_single_event_stat(code, evs[0]["ev_id"])
                if hst is not None and ast is not None:
                    has_stats = True
                    break
        if has_stats:
            valid_years.add(yr)
            print(f"  ✓ {yr} 年份確認有射正與角球統計，納入回補")
        else:
            print(f"  ✗ {yr} 年份 ESPN 未收錄射正欄位，自動跳過")
    return valid_years


def backfill_via_espn_fast(missing_by_league: dict) -> int:
    total_saved = 0

    for lg_name, targets in missing_by_league.items():
        if not targets:
            continue
        code = ESPN_ASIAN_CODES[lg_name]
        print(f"\n[ESPN 並發回補] 正在探測 {lg_name} ({code}) 各年份數據覆蓋率...")
        valid_years = probe_valid_years(code, targets)
        if not valid_years:
            print(f"[結論] {lg_name} 在 ESPN 均無射正技術統計，無需耗時掃描。")
            continue

        unmatched = [r for r in targets if r["year"] in valid_years]
        query_dates = sorted({
            (r["date"] + timedelta(days=d)).strftime("%Y%m%d")
            for r in unmatched for d in (0, -1)
        })
        print(f"  開始並發抓取 {lg_name} 共 {len(unmatched)} 場（{len(query_dates)} 個比賽日）...")

        # 步驟 1：8 執行緒並發拉取所有比賽日賽果索引
        all_espn_events = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_fetch_scoreboard_date, code, d): d for d in query_dates}
            done_cnt = 0
            for fut in as_completed(futures):
                done_cnt += 1
                for ev in fut.result():
                    all_espn_events[ev["ev_id"]] = ev
                if done_cnt % 40 == 0 or done_cnt == len(query_dates):
                    print(f"  ...已掃描 {done_cnt}/{len(query_dates)} 個比賽日（累積 {len(all_espn_events)} 場賽果）")

        # 步驟 2：在記憶體瞬間完成場次配對
        paired_tasks = []
        for ev in all_espn_events.values():
            for rec in unmatched:
                if rec["home_goals"] != ev["hg"] or rec["away_goals"] != ev["ag"]:
                    continue
                if abs((rec["date"] - ev["ev_dt"]).days) > 1:
                    continue
                if _similar(rec["home_team"], ev["home"]) >= 0.50 and _similar(rec["away_team"], ev["away"]) >= 0.50:
                    paired_tasks.append((rec["match_id"], ev["ev_id"]))
                    unmatched.remove(rec)
                    break

        print(f"  配對成功 {len(paired_tasks)} 場，開始並發下載射正與角球明細...")

        # 步驟 3：8 執行緒並發下載技術統計並批次寫入 SQLite
        rows_to_insert = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            fut_map = {pool.submit(_fetch_single_event_stat, code, eid): mid for mid, eid in paired_tasks}
            done_s = 0
            for fut in as_completed(fut_map):
                done_s += 1
                mid = fut_map[fut]
                hs, as_, hst, ast, hc, ac = fut.result()
                if hst is not None and ast is not None:
                    hs_val = hs if hs is not None else hst
                    as_val = as_ if as_ is not None else ast
                    h_xg = round(hst * 0.29 + max(0, hs_val - hst) * 0.035 + (hc or 0) * 0.015, 3)
                    a_xg = round(ast * 0.29 + max(0, as_val - ast) * 0.035 + (ac or 0) * 0.015, 3)
                    rows_to_insert.append((mid, h_xg, a_xg, hs_val, as_val, hst, ast, hc, ac))
                if done_s % 50 == 0 or done_s == len(paired_tasks):
                    print(f"  ...技術統計進度 {done_s}/{len(paired_tasks)}（有效含射正：{len(rows_to_insert)} 場）")

        if rows_to_insert:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.executemany("INSERT OR REPLACE INTO match_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows_to_insert)
            conn.commit()
            conn.close()
            total_saved += len(rows_to_insert)

    return total_saved


if __name__ == "__main__":
    missing = load_missing_asian_matches()
    total_missing = sum(len(v) for v in missing.values())
    print(f"資料庫共偵測到 {total_missing} 場日韓聯賽（J1 / K1）缺少射正與角球統計。")

    if total_missing > 0:
        n_ok = backfill_via_espn_fast(missing)
        print(f"\n[全部完成] ESPN 並發回補共成功補齊 {n_ok} 場技術統計！")