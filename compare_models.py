"""Compare gauge-reading methods against the human-corrected labels.

Reports MAE / max error / coverage for the segmentation pipeline and the
local EfficientNet model on every photo with a corrected_level, split into
day and night (IR) photos.

Usage:
    python compare_models.py
"""
from __future__ import annotations

import cv2
import numpy as np

from src.config import settings
from src.db import init_db
from src.segmentation_gauge import is_ir_photo


def _stats(pairs: list[tuple[float, float]]) -> str:
    if not pairs:
        return "n=0"
    errs = np.array([pred - truth for pred, truth in pairs])
    return (f"n={len(errs)}  MAE={np.abs(errs).mean():.2f}  bias={errs.mean():+.2f}  "
            f"max|err|={np.abs(errs).max():.2f}  within±0.5={np.mean(np.abs(errs) <= 0.5):.0%}")


def main() -> None:
    conn = init_db(settings.db_path)
    rows = conn.execute(
        """
        SELECT local_path, corrected_level, segment_level, local_level FROM photos
        WHERE corrected_level IS NOT NULL AND local_path IS NOT NULL
        """
    ).fetchall()

    groups: dict[str, dict[str, list]] = {
        "day": {"segment": [], "local": [], "seg_missing": 0},
        "night": {"segment": [], "local": [], "seg_missing": 0},
    }
    for r in rows:
        bgr = cv2.imread(r["local_path"])
        if bgr is None:
            continue
        g = groups["night" if is_ir_photo(bgr) else "day"]
        if r["segment_level"] is not None:
            g["segment"].append((r["segment_level"], r["corrected_level"]))
        else:
            g["seg_missing"] += 1
        if r["local_level"] is not None:
            g["local"].append((r["local_level"], r["corrected_level"]))

    for name, g in groups.items():
        print(f"--- {name} photos ---")
        print(f"  segment : {_stats(g['segment'])}  (no reading: {g['seg_missing']})")
        print(f"  local   : {_stats(g['local'])}")
    all_seg = groups["day"]["segment"] + groups["night"]["segment"]
    all_loc = groups["day"]["local"] + groups["night"]["local"]
    print("--- all labelled photos ---")
    print(f"  segment : {_stats(all_seg)}")
    print(f"  local   : {_stats(all_loc)}")

    worst = sorted(all_seg, key=lambda p: -abs(p[0] - p[1]))[:5]
    if worst:
        print("worst segment errors (pred vs truth):",
              ", ".join(f"{p:.1f}/{t:.1f}" for p, t in worst))


if __name__ == "__main__":
    main()
