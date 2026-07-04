"""
Train a small EfficientNet-B0 regression model on manually-corrected gauge readings.

Usage:
    python train_model.py
    python train_model.py --epochs 80 --batch-size 8

The trained model is saved to model.pth and can be used with:
    python read_gauge.py --local-model latest.jpg
    python batch_process.py --local-only --reprocess --local-model
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

from src.config import settings
from src.db import init_db

MODEL_PATH = Path("model.pth")


# ── Dataset ──────────────────────────────────────────────────────────────────

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class GaugeDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["local_path"]).convert("RGB")
        return self.transform(img), torch.tensor(float(row["corrected_level"]), dtype=torch.float32)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Replace the classifier with a single regression output
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1),
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train(epochs: int, batch_size: int, lr: float, val_split: float):
    conn = init_db(settings.db_path)
    rows = conn.execute(
        "SELECT local_path, corrected_level FROM photos "
        "WHERE corrected_level IS NOT NULL AND local_path IS NOT NULL"
    ).fetchall()
    conn.close()

    if len(rows) < 4:
        print(f"Only {len(rows)} labelled examples — need at least 4. Label more in the viewer first.")
        return

    print(f"Loaded {len(rows)} labelled examples")

    # Split into train / val
    n_val = max(1, int(len(rows) * val_split))
    n_train = len(rows) - n_val
    shuffled = list(rows)
    random.shuffle(shuffled)
    train_rows, val_rows = shuffled[n_val:], shuffled[:n_val]

    train_ds = GaugeDataset(train_rows, TRAIN_TRANSFORMS)
    val_ds   = GaugeDataset(val_rows,   VAL_TRANSFORMS)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  train={n_train}  val={n_val}\n")

    model = build_model().to(device)
    criterion = nn.MSELoss()

    # Phase 1: train only the new head (frozen backbone)
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr * 10)
    _run_epochs(model, train_loader, val_loader, criterion, optimizer, device,
                epochs=min(10, epochs), label="warm-up")

    # Phase 2: fine-tune the whole network at a lower lr
    for param in model.features.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    best_val = _run_epochs(model, train_loader, val_loader, criterion, optimizer, device,
                           epochs=epochs, label="fine-tune", scheduler=scheduler,
                           save_path=MODEL_PATH)

    print(f"\nBest val MAE: {best_val:.3f}")
    print(f"Model saved to {MODEL_PATH}")


def _run_epochs(model, train_loader, val_loader, criterion, optimizer, device,
                epochs, label, scheduler=None, save_path=None):
    best_val_mae = float("inf")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs).squeeze(1)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs).squeeze(1)
                val_mae += (preds - targets).abs().sum().item()
        val_mae /= len(val_loader.dataset)

        if scheduler:
            scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            if save_path:
                torch.save(model.state_dict(), save_path)
                marker = " ✓ saved"
            else:
                marker = " ✓"
        else:
            marker = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{label}] epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  val_mae={val_mae:.3f}{marker}")

    return best_val_mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--val-split",  type=float, default=0.15)
    args = parser.parse_args()

    train(args.epochs, args.batch_size, args.lr, args.val_split)
