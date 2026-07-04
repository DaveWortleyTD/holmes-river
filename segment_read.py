"""Read the gauge from one or more photos using the segmentation pipeline.

Usage:
    python segment_read.py photos/X.jpg [photos/Y.jpg ...] [--debug]

--debug writes debug/<name>.jpg with the crop box (green), calibrated marks
(blue), detected marks (red), and waterline (yellow) drawn on the photo.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.segmentation_gauge import load_calibration, load_config, read_segment


def draw_debug(image_path: str, reading, config: dict, calibration: dict, out_dir: Path) -> Path:
    bgr = cv2.imread(image_path)
    x0, x1 = config["crop_x_start"], config["crop_x_end"]
    cv2.rectangle(bgr, (x0, config["crop_y_start"]), (x1, config["crop_y_end"]), (0, 200, 0), 1)
    for m in calibration["marks"]:
        y = m["pixel_y"]
        cv2.line(bgr, (x0 - 8, y), (x0, y), (255, 128, 0), 1)
        cv2.putText(bgr, str(m["level"]), (x0 - 24, y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 128, 0), 1)
    for y in reading.mark_ys:
        cv2.line(bgr, (x0, y), (x1, y), (0, 0, 255), 1)
    if reading.waterline_y is not None:
        y = reading.waterline_y
        cv2.line(bgr, (x0 - 30, y), (x1 + 30, y), (0, 255, 255), 2)
    label = f"{reading.level if reading.level is not None else '??'} ({reading.method})"
    cv2.putText(bgr, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{Path(image_path).stem}.jpg"
    cv2.imwrite(str(out), bgr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--debug", action="store_true", help="write annotated debug images")
    ap.add_argument("--debug-dir", default="debug")
    args = ap.parse_args()

    config = load_config()
    calibration = load_calibration()
    for path in args.photos:
        r = read_segment(path, config, calibration)
        print(f"{path}: level={r.level} method={r.method} waterline_y={r.waterline_y} "
              f"marks={r.mark_ys}{'  [' + r.notes + ']' if r.notes else ''}")
        if args.debug:
            out = draw_debug(path, r, config, calibration, Path(args.debug_dir))
            print(f"  debug -> {out}")


if __name__ == "__main__":
    main()
