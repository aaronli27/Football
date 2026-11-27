import difflib
import time
from curl_cffi import requests

HKJC_BASE_PAGE = "https://bet.hkjc.com/ch/football/allodds"
HKJC_POST_URL = "https://info.cld.hkjc.com/graphql/base/"

HKJC_HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "zh-HK,zh;q=0.9,en;q=0.8",
    "Content-Type": "application/json",
    "Origin": "https://bet.hkjc.com",
    "Referer": "https://bet.hkjc.com/",
}

EXACT_MATCH_ODDS_QUERY = (
    "query matchList($startIndex:Int$endIndex:Int$startDate:String$endDate:String"
    "$matchIds:[String]$tournIds:[String]$fbOddsTypes:[FBOddsType]!$fbOddsTypesM:[FBOddsType]!"
    "$inplayOnly:Boolean$featuredMatchesOnly:Boolean$frontEndIds:[String]"
    "$earlySettlementOnly:Boolean$showAllMatch:Boolean){"
    "matches(startIndex:$startIndex endIndex:$endIndex startDate:$startDate endDate:$endDate "
    "matchIds:$matchIds tournIds:$tournIds fbOddsTypes:$fbOddsTypesM inplayOnly:$inplayOnly "
    "featuredMatchesOnly:$featuredMatchesOnly frontEndIds:$frontEndIds "
    "earlySettlementOnly:$earlySettlementOnly showAllMatch:$showAllMatch){"
    "id frontEndId matchDate kickOffTime status updateAt sequence esIndicatorEnabled "
    "homeTeam{id name_en name_ch}awayTeam{id name_en name_ch}"
    "tournament{id frontEndId nameProfileId isInteractiveServiceAvailable code name_en name_ch}"
    "isInteractiveServiceAvailable inplayDelay venue{code name_en name_ch}"
    "tvChannels{code name_en name_ch}liveEvents{id code}featureStartTime featureMatchSequence "
    "poolInfo{normalPools inplayPools sellingPools ntsInfo entInfo definedPools "
    "ngsInfo{str name_en name_ch instNo}agsInfo{str name_en name_ch}}"
    "runningResult{homeScore awayScore corner homeCorner awayCorner}"
    "runningResultExtra{homeScore awayScore corner homeCorner awayCorner}"
    "adminOperation{remark{typ}}"
    "foPools(fbOddsTypes:$fbOddsTypes){"
    "id status oddsType instNo inplay name_ch name_en updateAt expectedSuspendDateTime "
    "lines{lineId status condition main combinations{"
    "combId str status offerEarlySettlement currentOdds selections{selId str name_ch name_en}"
    "}}}}}"
)

# HKJC 前端官方白名單完整 47 彩池枚舉清單
WHITELISTED_ALL_ODDS_TYPES = [
    "HAD", "EHA", "SGA", "CHP", "TQL", "FHA", "HHA", "HDC", "EDC",
    "HIL", "EHL", "FHL", "CHL", "ECH", "FCH", "CRS", "ECS", "FCS",
    "FTS", "TTG", "ETG", "OOE", "FGS", "HFT", "MSP", "NTS", "ENT",
    "FHH", "FHC", "CHD", "ECD", "EHH", "AGS", "LGS", "NGS", "HLH",
    "HLA", "FLH", "FLA", "ELH", "ELA", "CHH", "CHA", "CFH", "CFA",
    "CEH", "CEA",
]

TEAM_ALIASES = {
    "man city": "manchester city",
    "man utd": "manchester united",
    "manchester utd": "manchester united",
    "spurs": "tottenham",
    "tottenham hotspur": "tottenham",
    "newcastle utd": "newcastle",
    "nottm forest": "nott'm forest",
    "nottingham forest": "nott'm forest",
    "wolves": "wolverhampton",
    "wolverhampton wanderers": "wolverhampton",
    "brighton & hove albion": "brighton",
    "west ham united": "west ham",
    "sheff utd": "sheffield united",
    "sheff wed": "sheffield weds",
    "athletic bilbao": "ath bilbao",
    "atletico madrid": "ath madrid",
    "real sociedad": "sociedad",
    "real betis": "betis",
    "celta vigo": "celta",
    "bayer leverkusen": "leverkusen",
    "borussia dortmund": "dortmund",
    "eintracht frankfurt": "ein frankfurt",
    "borussia monchengladbach": "m'gladbach",
    "inter milan": "inter",
    "ac milan": "milan",
    "as roma": "roma",
    "psg": "paris sg",
    "paris saint-germain": "paris sg",
    "sporting lisbon": "sp lisbon",
    "sporting cp": "sp lisbon",
}

ACTIVE_CODES_USED = []
_MEM_CACHE = {"ts": 0.0, "data": []}


def _norm_team_name(name: str) -> str:
    s = name.lower().strip()
    return TEAM_ALIASES.get(s, s)


def _parse_condition_line(cond: str) -> float:
    if not cond:
        return 0.0
    s = str(cond).strip().replace("[", "").replace("]", "")
    if "/" in s:
        parts = s.split("/")
        try:
            return round((float(parts[0]) + float(parts[1])) / 2.0, 2)
        except Exception:
            return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def _post_gql(session: requests.Session, odds_types: list, start_idx=1, end_idx=60, match_ids=None) -> dict:
    variables = {
        "fbOddsTypes": odds_types,
        "fbOddsTypesM": odds_types,
        "featuredMatchesOnly": False,
        "startDate": None,
        "endDate": None,
        "tournIds": None,
        "matchIds": match_ids,
        "startIndex": start_idx,
        "endIndex": end_idx,
        "frontEndIds": None,
        "earlySettlementOnly": False,
        "showAllMatch": False,
        "inplayOnly": False,
    }
    payload = {
        "operationName": "matchList",
        "variables": variables,
        "query": EXACT_MATCH_ODDS_QUERY,
    }
    resp = session.post(HKJC_POST_URL, headers=HKJC_HEADERS, json=payload, timeout=18)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:160]}")
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(data["errors"][0].get("message", "GraphQL Error"))
    return data


def _parse_matches_payload(raw_matches: list) -> list:
    parsed_list = []
    for m in raw_matches:
        mid = m.get("frontEndId") or m.get("id")
        h_obj = m.get("homeTeam") or {}
        a_obj = m.get("awayTeam") or {}
        t_obj = m.get("tournament") or {}

        flat_odds = {}
        pools_detail = {}

        for pool in m.get("foPools") or []:
            p_status = str(pool.get("status", "")).upper()
            if p_status in ("CLOSED", "SUSPENDED", "VOID", "REFUND"):
                continue

            otype = str(pool.get("oddsType", "")).upper()
            lines = pool.get("lines") or []
            if not lines:
                continue

            main_line = next((ln for ln in lines if ln.get("main")), lines[0])
            cond_raw = main_line.get("condition", "")
            cond_val = _parse_condition_line(cond_raw)

            combos_map = {}
            for c in main_line.get("combinations") or []:
                c_status = str(c.get("status", "")).upper()
                if c_status in ("CLOSED", "SUSPENDED", "VOID"):
                    continue

                c_str = str(c.get("str") or "").upper().strip()
                if not c_str and c.get("selections"):
                    c_str = str(c["selections"][0].get("str") or "").upper().strip()

                try:
                    odd_val = float(c.get("currentOdds") or 0)
                except Exception:
                    odd_val = 0.0

                if odd_val > 1.0 and c_str:
                    combos_map[c_str] = odd_val

            if not combos_map:
                continue

            pools_detail[otype] = {"condition": cond_raw, "line": cond_val, "odds": combos_map}

            if otype == "HAD":
                for k in ("H", "1", "HOME"):
                    if k in combos_map:
                        flat_odds["home_odd"] = combos_map[k]
                for k in ("D", "X", "DRAW"):
                    if k in combos_map:
                        flat_odds["draw_odd"] = combos_map[k]
                for k in ("A", "2", "AWAY"):
                    if k in combos_map:
                        flat_odds["away_odd"] = combos_map[k]

            elif otype == "HHA":
                flat_odds["hhad_line"] = int(round(cond_val))
                for k in ("H", "1", "HOME"):
                    if k in combos_map:
                        flat_odds["hh_odd"] = combos_map[k]
                for k in ("D", "X", "DRAW"):
                    if k in combos_map:
                        flat_odds["hd_odd"] = combos_map[k]
                for k in ("A", "2", "AWAY"):
                    if k in combos_map:
                        flat_odds["ha_odd"] = combos_map[k]

            elif otype in ("HDC", "EDC") and "asian_line" not in flat_odds:
                flat_odds["asian_line"] = cond_val
                for k in ("H", "1", "HOME"):
                    if k in combos_map:
                        flat_odds["h_ah_odd"] = combos_map[k]
                for k in ("A", "2", "AWAY"):
                    if k in combos_map:
                        flat_odds["a_ah_odd"] = combos_map[k]

            elif otype == "HIL":
                flat_odds["goal_line"] = cond_val
                for k in ("H", "O", "HIGH", "OVER"):
                    if k in combos_map:
                        flat_odds["over_g_odd"] = combos_map[k]
                for k in ("L", "U", "LOW", "UNDER"):
                    if k in combos_map:
                        flat_odds["under_g_odd"] = combos_map[k]

            elif otype == "CHL":
                flat_odds["corner_line"] = cond_val
                for k in ("H", "O", "HIGH", "OVER"):
                    if k in combos_map:
                        flat_odds["over_c_odd"] = combos_map[k]
                for k in ("L", "U", "LOW", "UNDER"):
                    if k in combos_map:
                        flat_odds["under_c_odd"] = combos_map[k]

        parsed_list.append({
            "internal_id": str(m.get("id") or ""),
            "hkjc_id": mid,
            "match_date": m.get("matchDate"),
            "kickoff": m.get("kickOffTime"),
            "league_en": t_obj.get("name_en", ""),
            "league_ch": t_obj.get("name_ch", ""),
            "home_en": h_obj.get("name_en", ""),
            "away_en": a_obj.get("name_en", ""),
            "home_ch": h_obj.get("name_ch", ""),
            "away_ch": a_obj.get("name_ch", ""),
            "flat_odds": flat_odds,
            "pools": pools_detail,
        })
    return parsed_list


def fetch_hkjc_live_matches(max_matches: int = 60, force_refresh: bool = False) -> list:
    """直連 HKJC GraphQL 接口，使用官方 47 彩池白名單一次拉取全玩法即時賠率"""
    global ACTIVE_CODES_USED
    now = time.time()
    if not force_refresh and _MEM_CACHE["data"] and (now - _MEM_CACHE["ts"] < 180):
        return _MEM_CACHE["data"]

    session = requests.Session(impersonate="chrome")
    try:
        session.get(HKJC_BASE_PAGE, timeout=10)
    except Exception:
        pass

    raw_matches = []
    # 通道 A：直接用官方 47 彩池白名單陣列分頁拉取（每頁 30 場避免單次封包過大）
    try:
        for start in range(1, max_matches + 1, 30):
            end = min(start + 29, max_matches)
            if start > 1:
                time.sleep(0.8)
            res = _post_gql(session, WHITELISTED_ALL_ODDS_TYPES, start_idx=start, end_idx=end, match_ids=None)
            chunk = res.get("data", {}).get("matches") or []
            raw_matches.extend(chunk)
            if len(chunk) < 30:
                break
    except Exception:
        # 通道 B：先以 HAD 取得賽程內部 ID，再以官方 matchIds 陣列模式批次拉取 47 彩池
        raw_matches = []
        had_res = _post_gql(session, ["HAD"], start_idx=1, end_idx=max_matches, match_ids=None)
        base_matches = had_res.get("data", {}).get("matches") or []
        all_ids = [str(m["id"]) for m in base_matches if m.get("id")]

        for i in range(0, len(all_ids), 15):
            time.sleep(0.9)
            id_batch = all_ids[i : i + 15]
            try:
                full_res = _post_gql(
                    session,
                    WHITELISTED_ALL_ODDS_TYPES,
                    start_idx=None,
                    end_idx=None,
                    match_ids=id_batch,
                )
                raw_matches.extend(full_res.get("data", {}).get("matches") or [])
            except Exception:
                pass

        if not raw_matches:
            raw_matches = base_matches

    parsed_list = _parse_matches_payload(raw_matches)

    found_pools = set()
    for item in parsed_list:
        for p in ("HAD", "HHA", "HDC", "HIL", "CHL"):
            if p in item["pools"]:
                found_pools.add(p)
    ACTIVE_CODES_USED = [p for p in ("HAD", "HHA", "HDC", "HIL", "CHL") if p in found_pools]

    if parsed_list:
        _MEM_CACHE["ts"] = time.time()
        _MEM_CACHE["data"] = parsed_list
    return parsed_list


def find_hkjc_match_odds(home_team: str, away_team: str, hkjc_matches: list = None) -> dict:
    """根據主客隊名稱自動模糊匹配馬會即時賠率字典"""
    if hkjc_matches is None:
        hkjc_matches = fetch_hkjc_live_matches()

    target_h = _norm_team_name(home_team)
    target_a = _norm_team_name(away_team)

    best_item = None
    best_score = 0.0

    for item in hkjc_matches:
        jh = _norm_team_name(item["home_en"])
        ja = _norm_team_name(item["away_en"])

        sh = difflib.SequenceMatcher(None, target_h, jh).ratio()
        sa = difflib.SequenceMatcher(None, target_a, ja).ratio()
        if target_h in jh or jh in target_h:
            sh = max(sh, 0.88)
        if target_a in ja or ja in target_a:
            sa = max(sa, 0.88)

        score = 0.5 * (sh + sa)
        if sh >= 0.72 and sa >= 0.72 and score > best_score:
            best_score = score
            best_item = item

    if best_item:
        res = dict(best_item["flat_odds"])
        res["hkjc_id"] = best_item["hkjc_id"]
        res["hkjc_label"] = f"{best_item['home_ch']} vs {best_item['away_ch']} ({best_item['hkjc_id']})"
        return res
    return {}


if __name__ == "__main__":
    items = fetch_hkjc_live_matches(max_matches=60, force_refresh=True)
    print(f"✅ 成功直連 HKJC 官方 GraphQL！解鎖目標彩池: {ACTIVE_CODES_USED}，共抓到 {len(items)} 場賽事：")
    for it in items[:10]:
        print(
            f"  [{it['hkjc_id']}] {it['league_ch']} | "
            f"{it['home_ch']} ({it['home_en']}) vs {it['away_ch']} ({it['away_en']}) "
            f"-> {it['flat_odds']}"
        )