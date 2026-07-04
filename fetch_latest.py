"""
Fetch the latest photo from SpyPoint and save it locally.

Usage:
    python fetch_latest.py [--camera CAMERA_ID] [--out PATH]
"""

import argparse
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

from src.config import settings
from src.spypoint import SpypointClient, SpypointError


def main():
    parser = argparse.ArgumentParser(description="Download latest SpyPoint camera photo")
    parser.add_argument("--camera", default=settings.spypoint_camera_id or None)
    parser.add_argument("--out", default="latest.jpg")
    parser.add_argument("--list-cameras", action="store_true")
    args = parser.parse_args()

    client = SpypointClient(settings.spypoint_username, settings.spypoint_password)

    try:
        print("Logging in to SpyPoint...")
        client.login()
        print("  OK")

        print("Fetching cameras...")
        cameras = client.get_cameras()
        for cam in cameras:
            print(f"  {cam.id}  {cam.name}")

        if args.list_cameras:
            return

        camera_id = args.camera or (cameras[0].id if cameras else None)
        if not camera_id:
            print("No cameras found on this account.", file=sys.stderr)
            sys.exit(1)

        print(f"\nFetching latest photo for camera {camera_id}...")
        photo = client.get_latest_photo(camera_id)
        if not photo:
            print("No photos found for this camera.", file=sys.stderr)
            sys.exit(1)

        print(f"  Photo ID : {photo.id}")
        print(f"  Taken at : {photo.taken_at}")
        print(f"  URL      : {photo.url}")

        out_path = Path(args.out)
        print(f"\nDownloading to {out_path}...")
        response = httpx.get(photo.url, timeout=30, follow_redirects=True)
        response.raise_for_status()
        out_path.write_bytes(response.content)
        print(f"  Saved {len(response.content):,} bytes → {out_path}")

    except SpypointError as e:
        print(f"SpyPoint error: {e}", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
