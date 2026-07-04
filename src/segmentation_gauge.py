"""Geometric gauge reading: water/wall segmentation + painted-mark detection.

Pipeline per photo:
  1. Segment the full frame with SegFormer (ADE20K) and crop the class map to
     the gauge column. The waterline is the topmost sustained water row —
     unless it disagrees badly with where the wall region ends (stained wall
     and turbulent floodwater both segment as neither class), in which case
     painted-mark continuity decides which signal to trust.
  2. Night IR photos (SegFormer classifies the river as "floor") fall back to
     a row-brightness detector: the flash-lit wall is bright, the wet stain
     band above the water is the darkest thing in the column, and the water
     below returns some IR from ripples — so the waterline is the profile
     minimum after the wall drop-off.
  3. Painted tick marks are found by row-darkness profiling against a local
     background blur, matched to the calibrated mark table (plus virtual marks
     extrapolated below it) to correct for small camera shifts, filtered by
     shift consistency to reject dark streaks in floodwater, and used to clamp
     waterlines that drift below the lowest visible mark.
  4. The waterline row converts to a level by piecewise-linear interpolation
     between calibrated marks — mark spacing shrinks from ~13px to ~10px down
     the gauge (perspective), so a single linear fit is not accurate enough.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
ADE_WALL = 0
ADE_WATER_CLASSES = (21, 26)  # water, sea

CONFIG_PATH = Path("seg_config.json")
CALIBRATION_PATH = Path("seg_calibration.json")

DEFAULT_CONFIG = {
    "crop_x_start": 258,
    "crop_x_end": 302,
    "crop_y_start": 0,
    "crop_y_end": 385,  # excludes the SpyPoint banner
}

_processor = None
_model = None


@dataclass
class SegmentReading:
    level: Optional[float]
    waterline_y: Optional[int]  # full-image row of the waterline
    mark_ys: list[int] = field(default_factory=list)  # detected marks (full-image rows)
    method: str = "failed"  # "segformer" | "brightness" | "failed"
    notes: str = ""


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    p = Path(path)
    if p.exists():
        return {**DEFAULT_CONFIG, **json.loads(p.read_text())}
    return dict(DEFAULT_CONFIG)


def load_calibration(path: Path | str = CALIBRATION_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run calibrate_segmentation.py first")
    return json.loads(p.read_text())


def _load_segformer():
    global _processor, _model
    if _model is None:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        _processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
        _model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = _model.to(device)
    return _processor, _model


def segment_image(image_path: str | Path) -> np.ndarray:
    """Return an (H, W) ADE20K class map for the full image."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    processor, model = _load_segformer()
    device = next(model.parameters()).device
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=img.size[::-1], mode="bilinear", align_corners=False
    )
    return upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)


def get_masks(seg_map: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (water_mask, wall_mask) cropped to the gauge column."""
    crop = seg_map[
        config["crop_y_start"] : config["crop_y_end"],
        config["crop_x_start"] : config["crop_x_end"],
    ]
    water = np.isin(crop, ADE_WATER_CLASSES)
    wall = crop == ADE_WALL
    return water, wall


def is_ir_photo(bgr: np.ndarray) -> bool:
    """Night IR frames are stored as grayscale JPEGs (all channels equal)."""
    return np.abs(bgr[:, :, 0].astype(int) - bgr[:, :, 1].astype(int)).mean() < 1.5


def extract_waterline(water_mask: np.ndarray, min_coverage: float = 0.3, min_run: int = 5) -> Optional[int]:
    """Topmost row (crop coords) with sustained water coverage.

    Requires min_coverage across the column and near-continuous water for
    min_run rows below, so isolated misclassified pixels can't trigger.
    """
    coverage = water_mask.mean(axis=1)
    for y in range(len(coverage) - min_run):
        if coverage[y] >= min_coverage and coverage[y : y + min_run].mean() >= min_coverage * 0.8:
            return y
    return None


def wall_bottom_row(wall_mask: np.ndarray) -> Optional[int]:
    """Row just below the longest run of wall-dominant rows (crop coords)."""
    wall_rows = wall_mask.mean(axis=1) >= 0.5
    bottom, run_start, best_len = None, None, 0
    for y, hot in enumerate(np.append(wall_rows, False)):
        if hot and run_start is None:
            run_start = y
        elif not hot and run_start is not None:
            if y - run_start > best_len:
                best_len, bottom = y - run_start, y
            run_start = None
    return bottom


def brightness_waterline(gray: np.ndarray, config: dict, min_run: int = 8) -> Optional[int]:
    """Fallback waterline: first sustained brightness drop down the column.

    Works on night IR frames where the flash-lit wall is bright and water dark.
    The top 15 rows are skipped (frame edge is often dark at night).
    """
    col = gray[
        config["crop_y_start"] : config["crop_y_end"],
        config["crop_x_start"] : config["crop_x_end"],
    ].astype(float).mean(axis=1)
    smooth = np.convolve(col, np.ones(5) / 5, mode="same")
    half = len(smooth) // 2
    wall_level = np.percentile(smooth[15:half], 80)
    dark_level = np.percentile(smooth[half:], 20)
    if wall_level - dark_level < 25:  # no clear wall/water contrast
        return None
    thresh = dark_level + 0.5 * (wall_level - dark_level)
    below = smooth < thresh
    drop = None
    for y in range(15, len(below) - min_run):
        if below[y : y + min_run].all():
            drop = y
            break
    if drop is None:
        return None
    # The wet stain band just above the water is the darkest region at night
    # (water gives some IR return from ripples/foam, the soaked wall gives
    # none). The waterline is the profile minimum after the drop — but only
    # trust it if the profile genuinely rises again below (otherwise there is
    # no stain band and the drop itself is the waterline).
    window = smooth[drop : min(drop + 35, len(smooth))]
    min_y = drop + int(np.argmin(window))
    if smooth[min_y : min(drop + 35, len(smooth))].max() - smooth[min_y] >= 6:
        return min_y + config["crop_y_start"]
    return drop + config["crop_y_start"]


def detect_mark_rows(
    gray: np.ndarray,
    config: dict,
    max_y: Optional[int] = None,
    min_row_frac: float = 0.35,
    max_mark_height: int = 6,
) -> list[int]:
    """Detect painted tick marks in the gauge column; returns full-image rows.

    The marks are ~2-3px-tall dark bars spanning most of the column, so instead
    of edge/line fitting we score each row by how many pixels are markedly
    darker than the local wall background (a tall vertical box blur), then take
    the centers of contiguous high-score row runs. Runs taller than
    max_mark_height are rejected (stain edges, shadows). The darkness threshold
    scales with the local background so marks in the shadowed lower wall are
    still found.
    """
    y0 = config["crop_y_start"]
    y1 = config["crop_y_end"] if max_y is None else min(config["crop_y_end"], max_y)
    strip = gray[y0:y1, config["crop_x_start"] : config["crop_x_end"]].astype(int)
    if strip.shape[0] < 20:
        return []
    background = cv2.blur(strip.astype(np.uint8), (1, 15)).astype(int)
    dark_delta = np.clip(0.22 * background, 12, 30)
    dark = (background - strip) > dark_delta
    score = dark.mean(axis=1)

    marks, run_start = [], None
    for y, hot in enumerate(np.append(score >= min_row_frac, False)):
        if hot and run_start is None:
            run_start = y
        elif not hot and run_start is not None:
            if y - run_start <= max_mark_height:
                center = run_start + int(round(np.average(
                    np.arange(run_start, y), weights=score[run_start:y]
                ))) - run_start
                marks.append(center + y0)
            run_start = None
    return marks


def extended_marks(calib_marks: list[dict], level_min: float = 3.0) -> list[dict]:
    """Calibration marks plus virtual marks extrapolated below the lowest one
    (the stain band hides the bottom marks from calibration). Used for mark
    matching and waterline bounds only — level interpolation extrapolates the
    real table itself."""
    marks = sorted(calib_marks, key=lambda m: m["pixel_y"])
    spacing = marks[-1]["pixel_y"] - marks[-2]["pixel_y"]
    out = list(marks)
    level, y = marks[-1]["level"], marks[-1]["pixel_y"]
    while level - 1 >= level_min:
        level, y = level - 1, y + spacing
        out.append({"level": level, "pixel_y": y, "virtual": True})
    return out


def match_shift(detected_ys: list[int], calib_marks: list[dict], tol: float = 6.0) -> float:
    """Median vertical offset between detected marks and the calibration table.

    Compensates for small camera shifts. Returns 0.0 with <2 confident matches.
    """
    diffs = []
    for dy in detected_ys:
        nearest = min(calib_marks, key=lambda m: abs(m["pixel_y"] - dy))
        diff = dy - nearest["pixel_y"]
        if abs(diff) <= tol:
            diffs.append(diff)
    if len(diffs) < 2:
        return 0.0
    return float(np.median(diffs))


def matched_marks(detected_ys: list[int], ext_marks: list[dict], shift: float, tol: float = 2.5) -> list[int]:
    """Detected marks that sit within tol px of a (shift-corrected) table mark.

    The tight tolerance around the global shift rejects dark streaks in
    floodwater that coincidentally align with one table position."""
    return [
        dy for dy in detected_ys
        if any(abs(dy - (m["pixel_y"] + shift)) <= tol for m in ext_marks)
    ]


def level_from_waterline(waterline_y: float, marks: list[dict]) -> float:
    """Piecewise-linear interpolation of level at a pixel row; extrapolates
    beyond the top/bottom mark using the adjacent segment's slope."""
    pairs = sorted(((m["pixel_y"], m["level"]) for m in marks))
    ys = np.array([p[0] for p in pairs], dtype=float)
    lv = np.array([p[1] for p in pairs], dtype=float)
    if waterline_y <= ys[0]:
        slope = (lv[1] - lv[0]) / (ys[1] - ys[0])
        return float(lv[0] + slope * (waterline_y - ys[0]))
    if waterline_y >= ys[-1]:
        slope = (lv[-1] - lv[-2]) / (ys[-1] - ys[-2])
        return float(lv[-1] + slope * (waterline_y - ys[-1]))
    return float(np.interp(waterline_y, ys, lv))


def _arbitrate_waterline(
    water_mask: np.ndarray,
    wall_mask: np.ndarray,
    gray: np.ndarray,
    config: dict,
    calibration: dict,
    notes: list[str],
    max_gap: int = 15,
) -> Optional[int]:
    """Resolve the waterline (crop coords) when the two mask signals disagree.

    Stained concrete and turbulent floodwater are both frequently classified
    as neither wall nor water, opening a gap between the wall-region bottom
    and the water-region top. The painted marks disambiguate: they continue
    at their calibrated positions through stained wall, but not through water.
    """
    water_top = extract_waterline(water_mask)
    wall_bottom = wall_bottom_row(wall_mask)
    if water_top is None or wall_bottom is None:
        return water_top if water_top is not None else wall_bottom
    if water_top - wall_bottom <= max_gap:
        return water_top

    y0 = config["crop_y_start"]
    ext = extended_marks(calibration["marks"], config.get("level_min", 3.0))
    detected = detect_mark_rows(gray, config, max_y=water_top + y0 - 4)
    shift = match_shift(detected, ext)
    good = matched_marks(detected, ext, shift)
    middle = [m for m in good if m - y0 > wall_bottom]
    if len(middle) >= 2:
        notes.append(f"stained wall band {wall_bottom}-{water_top} kept as wall ({len(middle)} marks)")
        return water_top
    notes.append(f"band {wall_bottom}-{water_top} has no marks, treating as water")
    return wall_bottom


def read_segment(
    image_path: str | Path, config: Optional[dict] = None, calibration: Optional[dict] = None
) -> SegmentReading:
    config = config or load_config()
    calibration = calibration or load_calibration()
    try:
        return _read(image_path, config, calibration)
    except Exception as exc:  # never let one bad photo kill a batch
        return SegmentReading(None, None, [], "failed", f"{type(exc).__name__}: {exc}")


def _read(image_path: str | Path, config: dict, calibration: dict) -> SegmentReading:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"cannot read image {image_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    notes: list[str] = []
    waterline: Optional[int] = None
    method = "brightness"

    if not is_ir_photo(bgr):
        water_mask, wall_mask = get_masks(segment_image(image_path), config)
        coverage = water_mask.mean() + wall_mask.mean()
        if coverage >= 0.5:
            wl = _arbitrate_waterline(water_mask, wall_mask, gray, config, calibration, notes)
            if wl is not None:
                waterline = wl + config["crop_y_start"]
                method = "segformer"
        if waterline is None:
            notes.append(f"segformer coverage {coverage:.2f}, using brightness fallback")
    else:
        notes.append("night IR photo")

    if waterline is None:
        waterline = brightness_waterline(gray, config)
        method = "brightness"
        if waterline is None:
            return SegmentReading(None, None, [], "failed", "no waterline found; ".join(notes))

    mark_ys = detect_mark_rows(gray, config, max_y=waterline - 4)
    ext = extended_marks(calibration["marks"], config.get("level_min", 3.0))
    shift = match_shift(mark_ys, ext)
    good = matched_marks(mark_ys, ext, shift)
    if good:
        # The waterline can't sit far below the lowest visible mark — the marks
        # in between would be visible too. Wave troughs and sheet water on the
        # wall otherwise drag the segmentation waterline down.
        spacing = ext[-1]["pixel_y"] - ext[-2]["pixel_y"]
        limit = max(good) + int(2.5 * spacing)
        if waterline > limit:
            notes.append(f"waterline {waterline} clamped to {limit} (lowest mark {max(good)})")
            waterline = limit
    shifted = [{"level": m["level"], "pixel_y": m["pixel_y"] + shift} for m in calibration["marks"]]
    level = level_from_waterline(waterline, shifted)

    if not (config.get("level_min", 3.0) <= level <= config.get("level_max", 15.0)):
        notes.append(f"level {level:.2f} outside plausible range")
        return SegmentReading(None, waterline, mark_ys, method, "; ".join(notes))

    if abs(shift) > 0.5:
        notes.append(f"camera shift {shift:+.1f}px")
    return SegmentReading(round(level, 1), waterline, mark_ys, method, "; ".join(notes))
