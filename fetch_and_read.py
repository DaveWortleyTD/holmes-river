"""Fetch new SpyPoint photos, run segmentation on each, discard the JPEG.

For CI/automated use: photos are downloaded to a temp file, segmented, then
deleted — only the timestamp + gauge level is stored in the database.

Usage (locally or in CI):
    python fetch_and_read.py

Required env vars (via .env or GitHub Secrets):
    SPYPOINT_USERNAME, SPYPOINT_PASSWORD
Optional:
    SPYPOINT_CAMERA_ID  (defaults to first camera)
    DB_PATH             (defaults to ./readings.db)
"""
from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

from src.config import settings
from src.db import init_db, photo_exists, insert_photo, save_segment_prediction
from src.segmentation_gauge import load_config, load_calibration, read_segment
from src.spypoint import SpypointClient, SpypointError


def main() -> None:
    conn = init_db(settings.db_path)
    config = load_config()
    calibration = load_calibration()

    client = SpypointClient(settings.spypoint_username, settings.spypoint_password)
    try:
        print("Logging in to SpyPoint...")
        client.login()
        cameras = client.get_cameras()
        camera_id = settings.spypoint_camera_id or (cameras[0].id if cameras else None)
        if not camera_id:
            print("No cameras found.", file=sys.stderr)
            sys.exit(1)
        print(f"Fetching photo list for camera {camera_id}...")
        photos = client.get_all_photos(camera_id)
    except SpypointError as e:
        print(f"SpyPoint error: {e}", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        sys.exit(1)

    unprocessed = [
        p for p in photos
        if not conn.execute(
            "SELECT 1 FROM photos WHERE spypoint_id=? AND segment_level IS NOT NULL",
            (p.id,),
        ).fetchone()
    ]
    print(f"{len(photos)} total photos, {len(unprocessed)} need segmentation.\n")

    for i, photo in enumerate(unprocessed, 1):
        print(f"[{i}/{len(unprocessed)}] {photo.id}  {photo.taken_at.isoformat()}", end="  ", flush=True)

        if not photo_exists(conn, photo.id):
            insert_photo(
                conn,
                spypoint_id=photo.id,
                taken_at=photo.taken_at.isoformat(),
                image_url=photo.url,
                local_path=None,
                level=None, confidence="", notes="", raw_json="",
                processed_at=datetime.now(timezone.utc).isoformat(),
            )

        tmp_path = None
        try:
            resp = httpx.get(photo.url, timeout=30, follow_redirects=True)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(resp.content)
                tmp_path = f.name
            result = read_segment(tmp_path, config, calibration)
        except Exception as e:
            print(f"FAILED: {e}")
            save_segment_prediction(conn, photo.id, None)
            continue
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

        save_segment_prediction(conn, photo.id, result.level)
        level_str = f"{result.level:.1f}" if result.level is not None else "None"
        note = f"; {result.notes}" if result.notes else ""
        print(f"→ {level_str}  ({result.method}{note})")

    print(f"\nDone. {len(unprocessed)} photos processed. DB: {settings.db_path}")


if __name__ == "__main__":
    main()
