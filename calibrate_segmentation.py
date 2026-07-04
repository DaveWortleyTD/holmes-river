"""One-time calibration for the segmentation gauge reader.

Detects the painted tick marks on a low-water photo (the more marks visible,
the better), assigns them levels counting down from the topmost mark (14 by
default), and writes seg_calibration.json. Also saves calibration_annotated.jpg
so the mark/level assignment can be verified by eye before trusting it.

Usage:
    python calibrate_segmentation.py                 # auto-pick lowest-water photo
    python calibrate_segmentation.py --photo photos/X.jpg --top-level 14
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np

from src.segmentation_gauge import (
    CALIBRATION_PATH,
    brightness_waterline,
    detect_mark_rows,
    extract_waterline,
    get_masks,
    is_ir_photo,
    load_config,
    segment_image,
)


def pick_calibration_photo(db_path: str = "readings.db") -> str:
    """Lowest-water daytime photo = most marks visible above the waterline."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT local_path, corrected_level FROM photos
        WHERE corrected_level IS NOT NULL AND local_path IS NOT NULL
        ORDER BY corrected_level ASC
        """
    ).fetchall()
    for r in rows:
        p = Path(r["local_path"])
        if not p.exists():
            continue
        bgr = cv2.imread(str(p))
        if bgr is not None and not is_ir_photo(bgr):
            print(f"Auto-picked {p} (corrected_level={r['corrected_level']})")
            return str(p)
    raise SystemExit("No labelled daytime photo found; pass --photo explicitly")


def assign_levels(mark_ys: list[int], top_level: int) -> list[dict]:
    """Assign levels top-down, stepping >1 level across unusually large gaps
    (a missed mark must not shift every level below it)."""
    marks = [{"level": top_level, "pixel_y": mark_ys[0]}]
    gaps = np.diff(mark_ys)
    typical = float(np.median(gaps))
    level = top_level
    for i, gap in enumerate(gaps):
        step = max(1, round(gap / typical))
        level -= step
        marks.append({"level": level, "pixel_y": mark_ys[i + 1]})
    return marks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--photo", help="calibration photo (default: lowest-water day photo)")
    ap.add_argument("--top-level", type=int, default=14, help="level of the topmost mark")
    ap.add_argument("--out", default=str(CALIBRATION_PATH))
    args = ap.parse_args()

    config = load_config()
    photo = args.photo or pick_calibration_photo()
    bgr = cv2.imread(photo)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Waterline bounds the mark search; use the same logic as read_segment
    waterline = None
    if not is_ir_photo(bgr):
        water_mask, wall_mask = get_masks(segment_image(photo), config)
        if water_mask.mean() + wall_mask.mean() >= 0.5:
            wl = extract_waterline(water_mask)
            if wl is not None:
                waterline = wl + config["crop_y_start"]
    if waterline is None:
        waterline = brightness_waterline(gray, config)
    if waterline is None:
        raise SystemExit("Could not find a waterline in the calibration photo")

    mark_ys = detect_mark_rows(gray, config, max_y=waterline - 4)
    if len(mark_ys) < 3:
        raise SystemExit(f"Only {len(mark_ys)} marks detected — pick a lower-water photo")

    marks = assign_levels(mark_ys, args.top_level)
    slope, intercept = np.polyfit([m["pixel_y"] for m in marks], [m["level"] for m in marks], 1)

    print(f"Waterline at y={waterline}")
    for m in marks:
        print(f"  level {m['level']:>2} -> y={m['pixel_y']}")
    print(f"Linear fit: level = {slope:.4f} * y + {intercept:.2f}")

    # Annotated image for visual verification
    x0, x1 = config["crop_x_start"], config["crop_x_end"]
    vis = bgr.copy()
    cv2.rectangle(vis, (x0, config["crop_y_start"]), (x1, config["crop_y_end"]), (0, 200, 0), 1)
    cv2.line(vis, (x0 - 15, waterline), (x1 + 15, waterline), (0, 255, 255), 1)
    for m in marks:
        cv2.line(vis, (x0, m["pixel_y"]), (x1, m["pixel_y"]), (0, 0, 255), 1)
        cv2.putText(vis, str(m["level"]), (x1 + 4, m["pixel_y"] + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imwrite("calibration_annotated.jpg", vis)
    print("Wrote calibration_annotated.jpg — verify each red line sits on a painted mark")

    Path(args.out).write_text(json.dumps({
        "marks": marks,
        "regression": {"slope": slope, "intercept": intercept},
        "calibration_photo": photo,
        "waterline_y": waterline,
        "crop": config,
    }, indent=2) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
