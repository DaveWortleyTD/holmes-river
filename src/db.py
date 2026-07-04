import sqlite3
from typing import Optional


def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            spypoint_id      TEXT UNIQUE NOT NULL,
            taken_at         TEXT NOT NULL,
            image_url        TEXT,
            local_path       TEXT,
            level            REAL,
            confidence       TEXT,
            notes            TEXT,
            raw_json         TEXT,
            processed_at     TEXT,
            corrected_level  REAL
        )
    """)
    # Safe migrations for columns added after initial schema
    for col in ("corrected_level REAL", "local_level REAL"):
        try:
            conn.execute(f"ALTER TABLE photos ADD COLUMN {col}")
        except Exception:
            pass
    conn.commit()
    return conn


def photo_exists(conn: sqlite3.Connection, spypoint_id: str) -> bool:
    row = conn.execute(
        "SELECT id FROM photos WHERE spypoint_id = ?", (spypoint_id,)
    ).fetchone()
    return row is not None


def insert_photo(
    conn: sqlite3.Connection,
    spypoint_id: str,
    taken_at: str,
    image_url: str,
    local_path: str,
    level: Optional[float],
    confidence: str,
    notes: str,
    raw_json: str,
    processed_at: str,
) -> None:
    # INSERT OR IGNORE ensures a new row is created if it doesn't exist yet.
    # The subsequent UPDATE only touches gauge-reading fields — corrected_level
    # is deliberately excluded so manual corrections are never overwritten.
    conn.execute(
        "INSERT OR IGNORE INTO photos (spypoint_id, taken_at) VALUES (?, ?)",
        (spypoint_id, taken_at),
    )
    conn.execute(
        """
        UPDATE photos SET
            taken_at=?, image_url=?, local_path=?,
            level=?, confidence=?, notes=?, raw_json=?, processed_at=?
        WHERE spypoint_id=?
        """,
        (taken_at, image_url, local_path,
         level, confidence, notes, raw_json, processed_at,
         spypoint_id),
    )
    conn.commit()


def save_local_prediction(conn: sqlite3.Connection, spypoint_id: str, local_level: Optional[float]) -> None:
    conn.execute(
        "UPDATE photos SET local_level = ? WHERE spypoint_id = ?",
        (local_level, spypoint_id),
    )
    conn.commit()


def save_correction(conn: sqlite3.Connection, spypoint_id: str, corrected_level: Optional[float]) -> None:
    conn.execute(
        "UPDATE photos SET corrected_level = ? WHERE spypoint_id = ?",
        (corrected_level, spypoint_id),
    )
    conn.commit()


def get_all_photos(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM photos ORDER BY taken_at DESC"
    ).fetchall()


def get_few_shot_examples(conn: sqlite3.Connection, n: int = 6) -> list[sqlite3.Row]:
    """Return up to n corrected photos spread across the value range."""
    rows = conn.execute(
        """
        SELECT * FROM photos
        WHERE corrected_level IS NOT NULL AND local_path IS NOT NULL
        ORDER BY corrected_level ASC
        """
    ).fetchall()
    if len(rows) <= n:
        return list(rows)
    # Pick n evenly-spaced examples across the level range
    step = (len(rows) - 1) / (n - 1)
    return [rows[round(i * step)] for i in range(n)]
