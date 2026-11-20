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

    # 2. 進階數據表 (xG、射門等)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS match_stats (
        match_id TEXT PRIMARY KEY,
        home_xg REAL,
        away_xg REAL,
        home_shots INTEGER,
        away_shots INTEGER,
        home_sot INTEGER,
        away_sot INTEGER,
        FOREIGN KEY (match_id) REFERENCES matches(match_id)
    )
    """)

    # 3. 賠率歷史表 (HKJC / Pinnacle 初盤與臨場盤)
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
        FOREIGN KEY (match_id) REFERENCES matches(match_id)
    )
    """)

    # 4. 每日滾動球隊戰力表 (避免回測時未來函數洩漏)
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
    # 將 WAL 暫存檔合併回主 .db 檔，確保 Git 提交乾淨
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()
    print(f"資料庫初始化完成：{DB_PATH.resolve()}")

if __name__ == "__main__":
    init_db()