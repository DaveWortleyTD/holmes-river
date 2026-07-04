"""
Extract timestamps from cached photos and update the database.

SpyPoint photo IDs are MongoDB ObjectIDs — the first 4 bytes (8 hex chars)
encode a Unix timestamp, so no EXIF or OCR is needed.
EXIF is used as a fallback for any IDs that don't follow this format.

Usage:
    python extract_dates.py
    python extract_dates.py --dry-run   # print what would change without writing
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.config import settings
from src.db import init_db
from src.spypoint import _timestamp_from_object_id as timestamp_from_object_id


def timestamp_from_exif(local_path: str) -> datetime | None:
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        with Image.open(local_path) as img:
            exif = img._getexif()
        if not exif:
            return None
        for tag_id, value in exif.items():
            if TAGS.get(tag_id) == "DateTimeOriginal":
                return datetime.strptime(value, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conn = init_db(settings.db_path)
    rows = conn.execute("SELECT spypoint_id, taken_at, local_path FROM photos").fetchall()

    updated = skipped = failed = 0

    for row in rows:
        photo_id   = row["spypoint_id"]
        taken_at   = (row["taken_at"] or "").strip()
        local_path = row["local_path"]

        ts = timestamp_from_object_id(photo_id)
        source = "object-id"

        if ts is None and local_path and Path(local_path).exists():
            ts = timestamp_from_exif(local_path)
            source = "exif"

        if ts is None:
            print(f"  SKIP  {photo_id}  (could not determine timestamp)")
            failed += 1
            continue

        ts_iso = ts.isoformat()
        changed = ts_iso != taken_at

        print(f"  {'DRY ' if args.dry_run else ''}{'UPDATE' if changed else 'ok    '}  "
              f"{photo_id}  {ts_iso}  [{source}]")

        if changed and not args.dry_run:
            conn.execute("UPDATE photos SET taken_at = ? WHERE spypoint_id = ?",
                         (ts_iso, photo_id))
            updated += 1
        else:
            skipped += 1

    if not args.dry_run:
        conn.commit()

    print(f"\n{'[dry-run] ' if args.dry_run else ''}updated={updated}  unchanged={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
