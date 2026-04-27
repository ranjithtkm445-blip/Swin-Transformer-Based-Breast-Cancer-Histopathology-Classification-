# =============================================================================
# Project  : BreakHis Vision Transformer - Breast Cancer Classification
# Step     : 4 - Training
# Purpose  : Train ViT-small and ResNet-18 on BreakHis subset.
#            AdamW optimizer, cosine LR scheduler, early stopping,
#            class-weighted loss, saves best model by val F1.
#            Logs per-epoch metrics to CSV.
# Author   : Ranjith Kumar
# =============================================================================

import sys
import json
import time
import csv
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)

# =============================================================================
# IMPORT FROM PREVIOUS STEPS
# =============================================================================
sys.path.append(str(Path(__file__).parent))
from step1_dataset import (
    train_loader, val_loader,
    train_df, val_df,
    OUTPUT_DIR
)
from step3_model import get_model

with open(OUTPUT_DIR / "config.json") as f:
    config = json.load(f)

# =============================================================================
# CONFIGURATION
# =============================================================================
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS   = 30           # 30 epochs for 350-image subset
LR           = 1e-5         # small LR for fine-tuning pretrained ViT
WEIGHT_DECAY = 1e-4         # L2 regularization
PATIENCE     = 15           # early stopping patience
MODEL_NAME   = "vit"        # "vit" or "resnet18"

print("=" * 60)
print(f"  STEP 4 — TRAINING ({MODEL_NAME.upper()})")
print("=" * 60)
print(f"[INFO] Device      : {DEVICE}")
print(f"[INFO] Epochs      : {NUM_EPOCHS}")
print(f"[INFO] LR          : {LR}")
print(f"[INFO] Patience    : {PATIENCE}")
print(f"[INFO] Train size  : {config['train_size']}")
print(f"[INFO] Val size    : {config['val_size']}")

# =============================================================================
# METRICS HELPER
# =============================================================================
def compute_metrics(y_true, y_pred):
    """Compute accuracy, F1, precision, recall."""
    return {
        "accuracy" : round(accuracy_score(y_true, y_pred) * 100, 2),
        "f1"       : round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall"   : round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
    }

# =============================================================================
# ONE EPOCH
# =============================================================================
def run_epoch(model, loader, criterion, optimizer=None, phase="train"):
    """Run one epoch of training or validation."""
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    all_preds   = []
    all_labels  = []

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping — prevents exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics

# =============================================================================
# TRAINING LOOP
# =============================================================================
def train(model_name=MODEL_NAME):
    # Load model
    model = get_model(model_name, num_classes=config["num_classes"]).to(DEVICE)

    # Class-weighted loss — further handles benign/malignant imbalance
    # Weight = inverse of class frequency in training set
    benign_w    = config["malignant_count"] / config["benign_count"]  # >1 → upweight benign
    class_weight = torch.tensor([benign_w, 1.0]).to(DEVICE)
    criterion    = nn.CrossEntropyLoss(weight=class_weight)

    print(f"[INFO] Class weights: Benign={benign_w:.3f} | Malignant=1.000")

    # Partial freeze — freeze early blocks, unfreeze last 4 blocks + head
    # Early blocks = low-level features (edges, textures) — keep frozen
    # Last 4 blocks = high-level semantic features — fine-tune for histopathology
    if model_name == "vit":
        # Freeze everything first
        for param in model.vit.parameters():
            param.requires_grad = False
        # Unfreeze last 2 Swin stages (layers[-2] and layers[-1])
        for layer in model.vit.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        # Unfreeze final norm layer
        if hasattr(model.vit, "norm"):
            for param in model.vit.norm.parameters():
                param.requires_grad = True
        # Unfreeze classification head
        for param in model.classifier.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Swin last 2 stages + head unfrozen. Trainable params: {trainable:,}")

    # Optimizer — only pass trainable parameters
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Paths
    model_dir      = OUTPUT_DIR / "models"
    best_model_path = model_dir / f"best_{model_name}.pt"
    log_path       = OUTPUT_DIR / "results" / f"training_log_{model_name}.csv"

    # CSV log header
    log_fields = [
        "epoch", "train_loss", "train_acc", "train_f1",
        "val_loss", "val_acc", "val_f1", "val_precision", "val_recall", "lr"
    ]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_val_f1     = 0.0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  Training {model_name.upper()} for {NUM_EPOCHS} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_m = run_epoch(model, train_loader, criterion, optimizer, "train")
        val_loss,   val_m   = run_epoch(model, val_loader,   criterion, None,      "val")
        scheduler.step()

        elapsed    = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} ({elapsed:.1f}s) | "
            f"Train loss={train_loss:.4f} acc={train_m['accuracy']}% f1={train_m['f1']}% | "
            f"Val loss={val_loss:.4f} acc={val_m['accuracy']}% f1={val_m['f1']}% | "
            f"LR={current_lr:.2e}"
        )

        # Log to CSV
        row = {
            "epoch"      : epoch,
            "train_loss" : round(train_loss, 4),
            "train_acc"  : train_m["accuracy"],
            "train_f1"   : train_m["f1"],
            "val_loss"   : round(val_loss, 4),
            "val_acc"    : val_m["accuracy"],
            "val_f1"     : val_m["f1"],
            "val_precision": val_m["precision"],
            "val_recall" : val_m["recall"],
            "lr"         : round(current_lr, 8)
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow(row)

        # Save best model by val F1 (more robust than accuracy for imbalanced data)
        if val_m["f1"] > best_val_f1:
            best_val_f1      = val_m["f1"]
            patience_counter = 0
            torch.save({
                "epoch"            : epoch,
                "model_name"       : model_name,
                "model_state_dict" : model.state_dict(),
                "val_acc"          : val_m["accuracy"],
                "val_f1"           : val_m["f1"],
                "val_precision"    : val_m["precision"],
                "val_recall"       : val_m["recall"],
                "config"           : config,
            }, best_model_path)
            print(f"  ✓ Best model saved — val_f1={best_val_f1}%")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\n[INFO] Early stopping at epoch {epoch}")
                break

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best val F1  : {best_val_f1}%")
    print(f"  Model saved  : {best_model_path}")
    print(f"  Training log : {log_path}")

    return model, best_model_path

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":

    # Train ViT
    vit_model, vit_path = train("vit")

    # Uncomment to also train ResNet-18 baseline
    # resnet_model, resnet_path = train("resnet18")

    print(f"\n[DONE] Step 4 complete. Proceed to step5_evaluate.py")