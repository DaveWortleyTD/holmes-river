import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anthropic
import httpx

from src.config import settings

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are a river gauge reader. You will be shown trail camera images of a gauge staff on a river. "
    "Your task: find the number on the gauge staff where the CURRENT water surface intersects it. "
    "The current water surface is the top of the flowing water right now — not foam, not splash, "
    "not the dark wet stain marks left by previous high water above the current surface. "
    "Read the printed number closest to the current water surface and interpolate to one decimal place "
    "(e.g. halfway between 8 and 9 = 8.5). Return the gauge number exactly as printed on the staff. "
    "Do not convert to metres or any other unit. "
    'Always respond with valid JSON only: '
    '{"level": <float or null>, "confidence": "high|medium|low", "notes": "<one sentence>"} '
    "If the gauge is not visible or the image is too dark, set level to null."
)

SYSTEM_PROMPT_WITH_REFERENCE = SYSTEM_PROMPT + (
    " A reference image of the full gauge staff is provided first so you can see all the markings "
    "and numbers clearly. Use it to calibrate your reading of the actual camera image."
)


@dataclass
class GaugeReading:
    level: Optional[float]
    confidence: str
    notes: str
    raw_json: str


@dataclass
class FewShotExample:
    local_path: str
    corrected_level: float


def read_gauge(
    image_source: str,
    reference_image: str | None = None,
    examples: list[FewShotExample] | None = None,
) -> GaugeReading:
    """Read river gauge level from an image URL or local file path."""
    ref_source = reference_image or settings.gauge_reference_image or None
    ref_data, ref_media_type = _load_image(ref_source) if ref_source else (None, None)

    image_data, media_type = _load_image(image_source)

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    system_text = SYSTEM_PROMPT_WITH_REFERENCE if ref_data else SYSTEM_PROMPT

    # Build the messages list.
    # Few-shot examples come first as alternating user/assistant turns so Claude
    # sees real image → correct answer pairs before the actual query.
    messages = []

    if ref_data:
        # Prepend a single user message with the reference image (no assistant reply needed)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Reference image — full gauge staff for calibration:"},
                {"type": "image", "source": {"type": "base64", "media_type": ref_media_type, "data": ref_data}},
            ],
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. I can see the full gauge staff and its markings. I'll use this as a reference.",
        })

    for ex in (examples or []):
        ex_data, ex_media_type = _load_image(ex.local_path)
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": ex_media_type, "data": ex_data}},
                {"type": "text", "text": "What is the current river level?"},
            ],
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps({"level": ex.corrected_level, "confidence": "high", "notes": "Calibration example."}),
        })

    # Final query
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
            {"type": "text", "text": "What is the current river level?"},
        ],
    })

    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
        messages=messages,
    )

    raw = response.content[0].text
    data = _parse_json(raw)

    return GaugeReading(
        level=data.get("level"),
        confidence=data.get("confidence", "low"),
        notes=data.get("notes", ""),
        raw_json=raw,
    )


def _load_image(source: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for a URL or local file."""
    if source.startswith("http://") or source.startswith("https://"):
        resp = httpx.get(source, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        raw_bytes = resp.content
        media_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    else:
        path = Path(source)
        raw_bytes = path.read_bytes()
        media_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "image/jpeg")

    return base64.standard_b64encode(raw_bytes).decode("utf-8"), media_type


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}
