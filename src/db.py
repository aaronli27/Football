import sqlite3
from pathlib import Path

DB_PATH = Path("data/soccer.db")

def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # 1. 賽程與賽果表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        match_id TEXT PRIMARY KEY,
        league TEXT NOT NULL,
        season TEXT NOT NULL,
        date TEXT NOT NULL,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        home_goals INTEGER,
        away_goals INTEGER,
        status TEXT DEFAULT 'scheduled'
    )
    """)

    # 2. 進階數據表 (xG、射門、射正、角球)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS match_stats (
        match_id TEXT PRIMARY KEY,
        home_xg REAL,
        away_xg REAL,
        home_shots INTEGER,
        away_shots INTEGER,
        home_sot INTEGER,
        away_sot INTEGER,
        home_corners INTEGER,
        away_corners INTEGER,
        FOREIGN KEY (match_id) REFERENCES matches(match_id)
    )
    """)

    # 自動補齊舊表可能缺少的角球欄位
    existing_cols = [row[1] for row in cursor.execute("PRAGMA table_info(match_stats)").fetchall()]
    if "home_corners" not in existing_cols:
        cursor.execute("ALTER TABLE match_stats ADD COLUMN home_corners INTEGER")
    if "away_corners" not in existing_cols:
        cursor.execute("ALTER TABLE match_stats ADD COLUMN away_corners INTEGER")

    # 3. 賠率歷史表 (加入唯一約束防止每日排程重複寫入，並支援初盤與收盤賠率)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS odds_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id TEXT NOT NULL,
        bookmaker TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        home_odd REAL,
        draw_odd REAL,
        away_odd REAL,
        asian_line REAL,
        home_ah_odd REAL,
        away_ah_odd REAL,
        ou_line REAL,
        over_odd REAL,
        under_odd REAL,
        open_home_odd REAL,
        open_away_odd REAL,
        UNIQUE(match_id, bookmaker),
        FOREIGN KEY (match_id) REFERENCES matches(match_id)
    )
    """)
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_odds_match_bookie ON odds_history(match_id, bookmaker)")

    # 自動補齊舊表可能缺少的初盤賠率欄位 (用於計算初盤->臨場水位斜率與真實 CLV)
    odds_cols = [row[1] for row in cursor.execute("PRAGMA table_info(odds_history)").fetchall()]
    if "open_home_odd" not in odds_cols:
        cursor.execute("ALTER TABLE odds_history ADD COLUMN open_home_odd REAL")
    if "open_away_odd" not in odds_cols:
        cursor.execute("ALTER TABLE odds_history ADD COLUMN open_away_odd REAL")

    # 4. 每日滾動球隊戰力表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS team_ratings (
        date TEXT NOT NULL,
        team TEXT NOT NULL,
        attack_strength REAL,
        defense_strength REAL,
        elo REAL,
        PRIMARY KEY (date, team)
    )
    """)

    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()
    print(f"資料庫結構升級完成：{DB_PATH.resolve()}")

if __name__ == "__main__":
    init_db()