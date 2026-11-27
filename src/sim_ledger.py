import base64
import io
import json
import os
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None

try:
    from ucl_and_daily import DB_PATH as MATCHES_DB_PATH
except Exception:
    try:
        from src.ucl_and_daily import DB_PATH as MATCHES_DB_PATH
    except Exception:
        MATCHES_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "soccer.db"

# 使用獨立的 sim_ledger.db 儲存模擬注單，避免每日更新 soccer.db 時被覆蓋
LEDGER_DB_PATH = Path(MATCHES_DB_PATH).resolve().parent / "sim_ledger.db"
GITHUB_CSV_PATH = "data/sim_bets.csv"
GITHUB_DATA_BRANCH = "sim-ledger-data"
DEFAULT_REPO = "aaronli27/Football"

STATUS_OPTIONS = [
    "待開獎 (Pending)",
    "全贏 (Won)",
    "贏半 (Half Won)",
    "走盤/退本 (Push)",
    "輸半 (Half Lost)",
    "全輸 (Lost)",
]

MARKET_TYPES = [
    "主客和 (HAD)",
    "讓球主客和 (HHAD)",
    "亞洲讓球 (HDC)",
    "入球大細 (HiLo)",
    "角球大細 (Corners)",
    "波膽 (CRS)",
    "晉級隊伍 (To Qualify)",
    "過關 / 混合過關 (All-Up)",
]

_CLOUD_HYDRATED = False


def _get_github_config() -> tuple:
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPO", DEFAULT_REPO)
    if st is not None:
        try:
            if "GITHUB_TOKEN" in st.secrets:
                token = str(st.secrets["GITHUB_TOKEN"]).strip()
            if "GITHUB_REPO" in st.secrets:
                repo = str(st.secrets["GITHUB_REPO"]).strip()
        except Exception:
            pass
    return token, repo


def _gh_request(url: str, token: str, method: str = "GET", payload: dict = None) -> tuple:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "HKJC-Sim-Ledger",
    }
    data_bytes = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data_bytes, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}


def _ensure_data_branch(token: str, repo: str) -> str:
    """確保存在獨立的 sim-ledger-data 分支，避免每次落注推送到 main 觸發 Streamlit Cloud 重啟"""
    ref_url = f"https://api.github.com/repos/{repo}/git/ref/heads/{GITHUB_DATA_BRANCH}"
    st_code, _ = _gh_request(ref_url, token)
    if st_code == 200:
        return GITHUB_DATA_BRANCH

    main_url = f"https://api.github.com/repos/{repo}/git/ref/heads/main"
    m_code, m_data = _gh_request(main_url, token)
    sha = m_data.get("object", {}).get("sha")
    if m_code == 200 and sha:
        create_url = f"https://api.github.com/repos/{repo}/git/refs"
        c_code, _ = _gh_request(
            create_url,
            token,
            method="POST",
            payload={"ref": f"refs/heads/{GITHUB_DATA_BRANCH}", "sha": sha},
        )
        if c_code in (200, 201):
            return GITHUB_DATA_BRANCH
    return "main"


def _github_get_file(token: str, repo: str, path: str, branch: str) -> tuple:
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    st_code, data = _gh_request(url, token)
    if st_code == 200 and data.get("content"):
        try:
            content = base64.b64decode(data["content"]).decode("utf-8-sig")
            return content, data.get("sha")
        except Exception:
            return None, None
    return None, None


def _github_put_file(token: str, repo: str, path: str, csv_text: str, message: str) -> bool:
    branch = _ensure_data_branch(token, repo)
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    encoded = base64.b64encode(csv_text.encode("utf-8-sig")).decode("ascii")
    for _ in range(2):
        _, sha = _github_get_file(token, repo, path, branch)
        payload = {"message": message, "content": encoded, "branch": branch}
        if sha:
            payload["sha"] = sha
        st_code, _ = _gh_request(url, token, method="PUT", payload=payload)
        if st_code in (200, 201):
            return True
        if st_code == 409:
            continue
        return False
    return False


def _sync_db_to_github(action_note: str = "update sim bets"):
    token, repo = _get_github_config()
    if not token:
        return
    try:
        conn = sqlite3.connect(LEDGER_DB_PATH)
        df = pd.read_sql_query("SELECT * FROM sim_bets ORDER BY bet_id ASC", conn)
        conn.close()
        csv_text = df.to_csv(index=False)
        _github_put_file(token, repo, GITHUB_CSV_PATH, csv_text, f"chore(sim): {action_note}")
    except Exception:
        pass


def _hydrate_db_from_github_if_needed():
    global _CLOUD_HYDRATED
    if _CLOUD_HYDRATED:
        return

    token, repo = _get_github_config()
    if not token:
        return

    conn = sqlite3.connect(LEDGER_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM sim_bets")
    local_count = cur.fetchone()[0]
    conn.close()
    if local_count > 0:
        _CLOUD_HYDRATED = True
        return

    # 優先從獨立資料分支讀取，若無則相容舊版 main 分支
    csv_text, _ = _github_get_file(token, repo, GITHUB_CSV_PATH, GITHUB_DATA_BRANCH)
    if not csv_text:
        csv_text, _ = _github_get_file(token, repo, GITHUB_CSV_PATH, "main")

    if not csv_text or not csv_text.strip():
        _CLOUD_HYDRATED = True
        return

    try:
        df = pd.read_csv(io.StringIO(csv_text), keep_default_na=False)
        if df.empty:
            _CLOUD_HYDRATED = True
            return
        conn = sqlite3.connect(LEDGER_DB_PATH)
        cur = conn.cursor()
        for _, r in df.iterrows():
            status = str(r.get("status") or "待開獎 (Pending)")
            stake = float(r["stake"])
            odds = float(r["odds"])
            raw_pnl = str(r.get("pnl", "")).strip()
            pnl = float(raw_pnl) if raw_pnl != "" else calc_hkjc_pnl(status, stake, odds)
            cur.execute(
                """
                INSERT INTO sim_bets (
                    created_at, match_date, league, home_team, away_team,
                    market_type, selection, line, odds, stake,
                    model_prob, ev_pct, is_whitelist, status, result_score, pnl, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(r.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M")),
                    str(r["match_date"]),
                    str(r["league"]),
                    str(r["home_team"]),
                    str(r["away_team"]),
                    str(r["market_type"]),
                    str(r["selection"]),
                    float(r.get("line") or 0.0),
                    odds,
                    stake,
                    float(r.get("model_prob") or 0.0),
                    float(r.get("ev_pct") or 0.0),
                    int(float(r.get("is_whitelist") or 0)),
                    status,
                    str(r.get("result_score") or ""),
                    pnl,
                    str(r.get("notes") or ""),
                ),
            )
        conn.commit()
        conn.close()
        _CLOUD_HYDRATED = True
    except Exception:
        pass


def init_sim_db():
    LEDGER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(LEDGER_DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sim_bets (
            bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            match_date TEXT NOT NULL,
            league TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            market_type TEXT NOT NULL,
            selection TEXT NOT NULL,
            line REAL DEFAULT 0.0,
            odds REAL NOT NULL,
            stake REAL NOT NULL,
            model_prob REAL DEFAULT 0.0,
            ev_pct REAL DEFAULT 0.0,
            is_whitelist INTEGER DEFAULT 0,
            status TEXT DEFAULT '待開獎 (Pending)',
            result_score TEXT DEFAULT '',
            pnl REAL DEFAULT 0.0,
            notes TEXT DEFAULT ''
        )
    """)
    conn.commit()
    conn.close()
    _hydrate_db_from_github_if_needed()


def calc_hkjc_pnl(status: str, stake: float, odds: float) -> float:
    if status == "全贏 (Won)":
        return round(stake * (odds - 1.0), 2)
    elif status == "贏半 (Half Won)":
        return round(stake * (odds - 1.0) * 0.5, 2)
    elif status == "走盤/退本 (Push)":
        return 0.0
    elif status == "輸半 (Half Lost)":
        return round(-stake * 0.5, 2)
    elif status == "全輸 (Lost)":
        return round(-stake, 2)
    return 0.0


def add_sim_bet(
    match_date: str,
    league: str,
    home_team: str,
    away_team: str,
    market_type: str,
    selection: str,
    line: float,
    odds: float,
    stake: float,
    model_prob: float = 0.0,
    ev_pct: float = 0.0,
    is_whitelist: bool = False,
    notes: str = "",
):
    init_sim_db()
    conn = sqlite3.connect(LEDGER_DB_PATH)
    cur = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    cur.execute(
        """
        INSERT INTO sim_bets (
            created_at, match_date, league, home_team, away_team,
            market_type, selection, line, odds, stake,
            model_prob, ev_pct, is_whitelist, status, result_score, pnl, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '待開獎 (Pending)', '', 0.0, ?)
        """,
        (
            created_at,
            match_date,
            league,
            home_team,
            away_team,
            market_type,
            selection,
            float(line),
            float(odds),
            float(stake),
            float(model_prob),
            float(ev_pct),
            1 if is_whitelist else 0,
            notes,
        ),
    )
    conn.commit()
    conn.close()
    _sync_db_to_github(f"add bet {home_team} vs {away_team}")


def settle_bet_manual(bet_id: int, new_status: str, result_score: str = ""):
    init_sim_db()
    conn = sqlite3.connect(LEDGER_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT stake, odds FROM sim_bets WHERE bet_id = ?", (bet_id,))
    row = cur.fetchone()
    if row:
        stake, odds = float(row[0]), float(row[1])
        pnl = calc_hkjc_pnl(new_status, stake, odds)
        cur.execute(
            "UPDATE sim_bets SET status = ?, result_score = ?, pnl = ? WHERE bet_id = ?",
            (new_status, result_score, pnl, bet_id),
        )
        conn.commit()
    conn.close()
    _sync_db_to_github(f"settle bet #{bet_id} -> {new_status}")


def delete_sim_bet(bet_id: int):
    init_sim_db()
    conn = sqlite3.connect(LEDGER_DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM sim_bets WHERE bet_id = ?", (bet_id,))
    conn.commit()
    conn.close()
    _sync_db_to_github(f"delete bet #{bet_id}")


def _evaluate_outcome_from_score(market_type: str, selection: str, line: float, hg: int, ag: int) -> str:
    if market_type == "主客和 (HAD)":
        if "主勝" in selection:
            return "全贏 (Won)" if hg > ag else "全輸 (Lost)"
        elif "和局" in selection:
            return "全贏 (Won)" if hg == ag else "全輸 (Lost)"
        elif "客勝" in selection:
            return "全贏 (Won)" if ag > hg else "全輸 (Lost)"

    elif market_type == "讓球主客和 (HHAD)":
        adj_h = hg + int(round(line))
        if "主勝" in selection:
            return "全贏 (Won)" if adj_h > ag else "全輸 (Lost)"
        elif "和" in selection:
            return "全贏 (Won)" if adj_h == ag else "全輸 (Lost)"
        elif "客勝" in selection:
            return "全贏 (Won)" if ag > adj_h else "全輸 (Lost)"

    elif market_type == "亞洲讓球 (HDC)":
        side = "home" if "主" in selection else "away"
        diff = (hg + line - ag) if side == "home" else (ag + line - hg)
        if diff >= 0.5:
            return "全贏 (Won)"
        elif np.isclose(diff, 0.25):
            return "贏半 (Half Won)"
        elif np.isclose(diff, 0.0):
            return "走盤/退本 (Push)"
        elif np.isclose(diff, -0.25):
            return "輸半 (Half Lost)"
        else:
            return "全輸 (Lost)"

    elif market_type == "入球大細 (HiLo)":
        tot = hg + ag
        diff = (tot - line) if "大" in selection else (line - tot)
        if diff >= 0.5:
            return "全贏 (Won)"
        elif np.isclose(diff, 0.25):
            return "贏半 (Half Won)"
        elif np.isclose(diff, 0.0):
            return "走盤/退本 (Push)"
        elif np.isclose(diff, -0.25):
            return "輸半 (Half Lost)"
        else:
            return "全輸 (Lost)"

    elif market_type == "波膽 (CRS)":
        actual_sc = f"{hg}-{ag}"
        clean_sel = selection.replace(":", "-").strip()
        return "全贏 (Won)" if actual_sc == clean_sel else "全輸 (Lost)"

    return "待開獎 (Pending)"


def auto_settle_pending_bets() -> int:
    init_sim_db()
    conn_l = sqlite3.connect(LEDGER_DB_PATH)
    cur_l = conn_l.cursor()
    cur_l.execute(
        "SELECT bet_id, match_date, league, home_team, away_team, market_type, selection, line, odds, stake "
        "FROM sim_bets WHERE status = '待開獎 (Pending)'"
    )
    pending = cur_l.fetchall()
    if not pending:
        conn_l.close()
        return 0

    conn_m = sqlite3.connect(MATCHES_DB_PATH)
    cur_m = conn_m.cursor()
    settled_count = 0

    for bet_id, m_date, lg, ht, at, mkt, sel, line, odds, stake in pending:
        cur_m.execute(
            """
            SELECT home_goals, away_goals, date FROM matches
            WHERE status = 'finished' AND home_team = ? AND away_team = ?
            ORDER BY rowid DESC LIMIT 3
            """,
            (ht, at),
        )
        candidates = cur_m.fetchall()
        matched_score = None
        for hg, ag, f_date in candidates:
            if m_date == f_date:
                matched_score = (int(hg), int(ag))
                break
            try:
                d1 = datetime.strptime(m_date, "%d/%m/%Y")
                d2 = datetime.strptime(f_date, "%d/%m/%Y")
                if abs((d1 - d2).days) <= 1:
                    matched_score = (int(hg), int(ag))
                    break
            except Exception:
                continue

        if matched_score is not None:
            hg, ag = matched_score
            outcome = _evaluate_outcome_from_score(mkt, sel, float(line), hg, ag)
            if outcome != "待開獎 (Pending)":
                pnl = calc_hkjc_pnl(outcome, float(stake), float(odds))
                cur_l.execute(
                    "UPDATE sim_bets SET status = ?, result_score = ?, pnl = ? WHERE bet_id = ?",
                    (outcome, f"{hg}-{ag}", pnl, bet_id),
                )
                settled_count += 1

    conn_m.close()
    conn_l.commit()
    conn_l.close()
    if settled_count > 0:
        _sync_db_to_github(f"auto-settle {settled_count} bets")
    return settled_count


def load_all_sim_bets() -> pd.DataFrame:
    init_sim_db()
    conn = sqlite3.connect(LEDGER_DB_PATH)
    df = pd.read_sql_query("SELECT * FROM sim_bets ORDER BY bet_id DESC", conn)
    conn.close()
    return df


def import_sim_bets_csv(uploaded_file) -> int:
    init_sim_db()
    df = pd.read_csv(uploaded_file, keep_default_na=False)
    required = {"match_date", "league", "home_team", "away_team", "market_type", "selection", "odds", "stake"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV 缺少必要欄位：{required - set(df.columns)}")

    conn = sqlite3.connect(LEDGER_DB_PATH)
    cur = conn.cursor()
    inserted = 0
    for _, r in df.iterrows():
        status = str(r.get("status") or "待開獎 (Pending)")
        stake = float(r["stake"])
        odds = float(r["odds"])
        raw_pnl = str(r.get("pnl", "")).strip()
        pnl = float(raw_pnl) if raw_pnl != "" else calc_hkjc_pnl(status, stake, odds)
        cur.execute(
            """
            INSERT INTO sim_bets (
                created_at, match_date, league, home_team, away_team,
                market_type, selection, line, odds, stake,
                model_prob, ev_pct, is_whitelist, status, result_score, pnl, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(r.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M")),
                str(r["match_date"]),
                str(r["league"]),
                str(r["home_team"]),
                str(r["away_team"]),
                str(r["market_type"]),
                str(r["selection"]),
                float(r.get("line") or 0.0),
                odds,
                stake,
                float(r.get("model_prob") or 0.0),
                float(r.get("ev_pct") or 0.0),
                int(float(r.get("is_whitelist") or 0)),
                status,
                str(r.get("result_score") or ""),
                pnl,
                str(r.get("notes") or ""),
            ),
        )
        inserted += 1
    conn.commit()
    conn.close()
    _sync_db_to_github(f"import {inserted} bets from csv")
    return inserted