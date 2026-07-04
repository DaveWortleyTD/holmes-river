"""Export gauge readings from readings.db to data.json for GitHub Pages.

Corrected (human-labelled) values take precedence over segment_level.
Only rows with at least one reading are included.

Usage:
    python export_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.config import settings
from src.db import init_db


def main() -> None:
    conn = init_db(settings.db_path)
    rows = conn.execute("""
        SELECT taken_at,
               COALESCE(corrected_level, segment_level) AS level
        FROM photos
        WHERE corrected_level IS NOT NULL OR segment_level IS NOT NULL
        ORDER BY taken_at ASC
    """).fetchall()

    data = [
        {"ts": r["taken_at"], "level": r["level"]}
        for r in rows
        if r["level"] is not None
    ]

    out = Path("data.json")
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Exported {len(data)} readings → {out}")


if __name__ == "__main__":
    main()
