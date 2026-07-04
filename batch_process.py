"""
Download photos from SpyPoint and run the gauge reader on each.

Usage:
    python batch_process.py                  # fetch from SpyPoint + read
    python batch_process.py --local-only     # re-read photos already in photos/
    python batch_process.py --reprocess      # re-read everything (ignore DB cache)
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

from src.config import settings
from src.db import (init_db, photo_exists, insert_photo, get_few_shot_examples,
                    save_local_prediction, save_segment_prediction)
from src.gauge import read_gauge, FewShotExample
from src.spypoint import SpypointClient, SpypointError

PHOTOS_DIR = Path("photos")


def main():
    parser = argparse.ArgumentParser(description="Batch gauge-read SpyPoint photos")
    parser.add_argument("--camera", default=settings.spypoint_camera_id or None)
    parser.add_argument("--reprocess", action="store_true",
                        help="Re-run gauge reading on photos already in the database")
    parser.add_argument("--local-only", action="store_true",
                        help="Skip SpyPoint — just process images already in the photos/ folder")
    parser.add_argument("--local-model", action="store_true",
                        help="Use the locally-trained model instead of Claude")
    parser.add_argument("--segment-model", action="store_true",
                        help="Use the segmentation pipeline instead of Claude (stacks with --local-model)")
    args = parser.parse_args()

    seg_config = seg_calib = None
    if args.segment_model:
        from src.segmentation_gauge import read_segment, load_config, load_calibration
        try:
            seg_config, seg_calib = load_config(), load_calibration()
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            sys.exit(1)

    PHOTOS_DIR.mkdir(exist_ok=True)
    conn = init_db(settings.db_path)

    # Load few-shot examples from any previously corrected photos
    example_rows = get_few_shot_examples(conn, n=6)
    examples = [FewShotExample(local_path=r["local_path"], corrected_level=r["corrected_level"])
                for r in example_rows]
    if examples:
        print(f"Using {len(examples)} labelled examples for few-shot calibration")

    # --- Build the list of (spypoint_id, local_path, taken_at, image_url) to process ---

    if args.local_only:
        # Use whatever .jpg files are already in photos/ — derive ID from filename
        local_files = sorted(PHOTOS_DIR.glob("*.jpg"))
        print(f"Found {len(local_files)} local images in {PHOTOS_DIR}/\n")
        work_items = [
            {"id": f.stem, "local_path": f, "taken_at": None, "image_url": None}
            for f in local_files
        ]
    else:
        client = SpypointClient(settings.spypoint_username, settings.spypoint_password)
        try:
            print("Logging in to SpyPoint...")
            client.login()
            cameras = client.get_cameras()
            if not cameras:
                print("No cameras found.", file=sys.stderr)
                sys.exit(1)
            camera_id = args.camera or cameras[0].id
            print(f"Camera: {camera_id}")
            print("Fetching photo list from SpyPoint...")
            photos = client.get_all_photos(camera_id)
            print(f"Found {len(photos)} photos on SpyPoint\n")
        except SpypointError as e:
            print(f"SpyPoint error: {e}", file=sys.stderr)
            sys.exit(1)

        work_items = [
            {"id": p.id, "local_path": PHOTOS_DIR / f"{p.id}.jpg",
             "taken_at": p.taken_at, "image_url": p.url}
            for p in photos
        ]

    # --- Process each item ---

    done = skipped = failed = 0
    total = len(work_items)

    for i, item in enumerate(work_items, 1):
        spypoint_id = item["id"]
        local_path  = Path(item["local_path"])
        taken_at    = item["taken_at"]
        image_url   = item["image_url"] or ""
        prefix      = f"[{i}/{total}] {spypoint_id}"

        if not args.reprocess and photo_exists(conn, spypoint_id):
            print(f"{prefix}  — already processed, skipping")
            skipped += 1
            continue

        # Download if we don't have the file locally
        if not local_path.exists():
            if not image_url:
                print(f"{prefix}  — no URL and file missing, skipping")
                failed += 1
                continue
            try:
                resp = httpx.get(image_url, timeout=30, follow_redirects=True)
                resp.raise_for_status()
                local_path.write_bytes(resp.content)
            except Exception as e:
                print(f"{prefix}  — download failed: {e}")
                failed += 1
                continue

        taken_at_str = taken_at.isoformat() if hasattr(taken_at, "isoformat") else (taken_at or "")

        if args.local_model or args.segment_model:
            # Model predictions live in their own columns — never touch Sonnet's
            # level column, and only create the row if it doesn't exist yet.
            if not photo_exists(conn, spypoint_id):
                insert_photo(conn, spypoint_id=spypoint_id, taken_at=taken_at_str,
                             image_url=image_url, local_path=str(local_path),
                             level=None, confidence="", notes="", raw_json="",
                             processed_at=datetime.now(timezone.utc).isoformat())
            if args.local_model:
                try:
                    from src.local_model import predict as _predict
                    local_level = _predict(str(local_path))
                    print(f"{prefix}  → local_level={local_level}")
                except Exception as e:
                    print(f"{prefix}  — local model failed: {e}")
                    local_level = None
                save_local_prediction(conn, spypoint_id, local_level)
            if args.segment_model:
                reading = read_segment(str(local_path), seg_config, seg_calib)
                note = f"  [{reading.notes}]" if reading.notes else ""
                print(f"{prefix}  → segment_level={reading.level} ({reading.method}){note}")
                save_segment_prediction(conn, spypoint_id, reading.level)
        else:
            # Claude / Sonnet path
            try:
                reading = read_gauge(str(local_path), examples=examples or None)
                print(f"{prefix}  → level={reading.level}  confidence={reading.confidence}")
            except Exception as e:
                print(f"{prefix}  — gauge reading failed: {e}")
                reading_level, reading_confidence, reading_notes, reading_raw = None, "error", str(e), ""
            else:
                reading_level      = reading.level
                reading_confidence = reading.confidence
                reading_notes      = reading.notes
                reading_raw        = reading.raw_json
            insert_photo(
                conn,
                spypoint_id=spypoint_id,
                taken_at=taken_at_str,
                image_url=image_url,
                local_path=str(local_path),
                level=reading_level,
                confidence=reading_confidence,
                notes=reading_notes,
                raw_json=reading_raw,
                processed_at=datetime.now(timezone.utc).isoformat(),
            )

        done += 1
        if not (args.local_model or args.segment_model):
            time.sleep(0.5)  # rate-limit courtesy only needed for the Claude API

    print(f"\nDone. processed={done}  skipped={skipped}  failed={failed}")
    print(f"Database: {settings.db_path}")
    print("Run 'python serve.py' to view results in a browser.")


if __name__ == "__main__":
    main()
