"""
Fetch all new photos from SpyPoint, save them to photos/, run the gauge reader
on each one that isn't already in the database, and store the results.

Usage:
    python fetch_latest.py [--camera CAMERA_ID] [--list-cameras] [--no-gauge]
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

from src.config import settings
from src.db import init_db, photo_exists, insert_photo, get_few_shot_examples
from src.gauge import read_gauge, FewShotExample
from src.spypoint import SpypointClient, SpypointError

PHOTOS_DIR = Path("photos")


def process_photo(conn, photo, no_gauge: bool):
    local_path = PHOTOS_DIR / f"{photo.id}.jpg"

    if not local_path.exists():
        print(f"  Downloading → {local_path}...")
        resp = httpx.get(photo.url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        local_path.write_bytes(resp.content)
        print(f"    Saved {len(resp.content):,} bytes")

    if photo_exists(conn, photo.id):
        print(f"  Already in database — skipping gauge read.")
        return

    if no_gauge:
        insert_photo(
            conn,
            spypoint_id=photo.id,
            taken_at=photo.taken_at.isoformat(),
            image_url=photo.url,
            local_path=str(local_path),
            level=None, confidence="", notes="", raw_json="",
            processed_at=datetime.now(timezone.utc).isoformat(),
        )
        return

    example_rows = get_few_shot_examples(conn, n=6)
    examples = [FewShotExample(local_path=r["local_path"], corrected_level=r["corrected_level"])
                for r in example_rows]

    try:
        reading = read_gauge(str(local_path), examples=examples or None)
        print(f"    Level: {reading.level}  Confidence: {reading.confidence}")
    except Exception as e:
        print(f"    Gauge reading failed: {e}", file=sys.stderr)
        reading = None

    insert_photo(
        conn,
        spypoint_id=photo.id,
        taken_at=photo.taken_at.isoformat(),
        image_url=photo.url,
        local_path=str(local_path),
        level=reading.level if reading else None,
        confidence=reading.confidence if reading else "error",
        notes=reading.notes if reading else "",
        raw_json=reading.raw_json if reading else "",
        processed_at=datetime.now(timezone.utc).isoformat(),
    )


def main():
    parser = argparse.ArgumentParser(description="Download and gauge-read all new SpyPoint photos")
    parser.add_argument("--camera", default=settings.spypoint_camera_id or None)
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--no-gauge", action="store_true",
                        help="Skip gauge reading — just download and store the images")
    args = parser.parse_args()

    PHOTOS_DIR.mkdir(exist_ok=True)
    conn = init_db(settings.db_path)
    client = SpypointClient(settings.spypoint_username, settings.spypoint_password)

    try:
        print("Logging in to SpyPoint...")
        client.login()

        cameras = client.get_cameras()
        for cam in cameras:
            print(f"  {cam.id}  {cam.name}")

        if args.list_cameras:
            return

        camera_id = args.camera or (cameras[0].id if cameras else None)
        if not camera_id:
            print("No cameras found.", file=sys.stderr)
            sys.exit(1)

        print(f"\nFetching all photos for camera {camera_id}...")
        photos = client.get_all_photos(camera_id)
        if not photos:
            print("No photos found.", file=sys.stderr)
            sys.exit(1)

    except SpypointError as e:
        print(f"SpyPoint error: {e}", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        sys.exit(1)

    new_photos = [p for p in photos if not photo_exists(conn, p.id)]
    print(f"Found {len(photos)} total photos, {len(new_photos)} not yet in database.\n")

    for i, photo in enumerate(new_photos, 1):
        print(f"[{i}/{len(new_photos)}] {photo.id}  {photo.taken_at}")
        process_photo(conn, photo, no_gauge=args.no_gauge)

    # Keep latest.jpg pointing at the most recent photo
    if photos:
        latest_path = PHOTOS_DIR / f"{photos[0].id}.jpg"
        if latest_path.exists():
            Path("latest.jpg").write_bytes(latest_path.read_bytes())

    print(f"\nDone. Database: {settings.db_path}")


if __name__ == "__main__":
    main()
