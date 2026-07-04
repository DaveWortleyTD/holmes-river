"""Fetch daily rainfall for McBride (CMOS) from the UNBC weewx station.

The station publishes NOAA-format monthly climatological summaries at
NOAA/NOAA-YYYY-MM.txt with one row per day; the RAIN column is daily total mm.
This script fetches every month from the first river reading to now and writes
rain.json: [{"date": "2026-07-01", "mm": 4.2}, ...]

Never fails the CI run: on any error it warns and leaves the existing
rain.json untouched.

Usage:
    python fetch_rain.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

from src.config import settings
from src.db import init_db

STATION_URL = "https://cyclone.unbc.ca/weather/cmos-mcbride/NOAA/NOAA-{year}-{month:02d}.txt"


def parse_noaa_month(text: str, year: int, month: int) -> list[dict]:
    """Extract {date, mm} rows from a weewx NOAA monthly summary."""
    days = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 9 or not fields[0].isdigit():
            continue
        day = int(fields[0])
        if not 1 <= day <= 31:
            continue
        try:
            mm = float(fields[8])
        except ValueError:
            continue
        days.append({"date": f"{year}-{month:02d}-{day:02d}", "mm": mm})
    return days


def month_range(start: datetime, end: datetime) -> list[tuple[int, int]]:
    months, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def main() -> None:
    conn = init_db(settings.db_path)
    row = conn.execute("SELECT MIN(taken_at) AS first FROM photos").fetchone()
    now = datetime.now(timezone.utc)
    first = datetime.fromisoformat(row["first"]) if row and row["first"] else now

    rain: list[dict] = []
    for year, month in month_range(first, now):
        url = STATION_URL.format(year=year, month=month)
        resp = httpx.get(url, timeout=30)
        if resp.status_code == 404:
            print(f"  {url} not found, skipping")
            continue
        resp.raise_for_status()
        rain.extend(parse_noaa_month(resp.text, year, month))

    if not rain:
        raise RuntimeError("no rainfall rows parsed")

    Path("rain.json").write_text(json.dumps(rain, separators=(",", ":")))
    print(f"Exported {len(rain)} daily rainfall totals → rain.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"WARNING: rainfall fetch failed, keeping existing rain.json: {exc}",
              file=sys.stderr)
        sys.exit(0)
