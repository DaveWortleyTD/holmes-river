"""
Inference with the locally-trained EfficientNet gauge model.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = Path("model.pth")

TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model = None  # cached after first load


def _load_model(device: torch.device):
    global _model
    if _model is None:
        m = models.efficientnet_b0(weights=None)
        in_features = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, 1))
        m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        m.to(device)
        m.eval()
        _model = m
    return _model


def predict(image_source: str) -> Optional[float]:
    """Return predicted gauge level from a local file path or URL."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained model found at {MODEL_PATH}. Run train_model.py first.")

    if image_source.startswith("http://") or image_source.startswith("https://"):
        import httpx, io
        resp = httpx.get(image_source, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    else:
        img = Image.open(image_source).convert("RGB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(device)

    tensor = TRANSFORMS(img).unsqueeze(0).to(device)
    with torch.no_grad():
        level = model(tensor).squeeze().item()

    return round(level, 2)
