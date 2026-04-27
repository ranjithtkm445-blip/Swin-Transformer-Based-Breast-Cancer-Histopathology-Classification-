# =============================================================================
# Project  : BreakHis Vision Transformer - Breast Cancer Classification
# Step     : 5 - Evaluation + Attention Heatmaps
# Purpose  : Load best trained ViT, evaluate on test set, generate
#            attention heatmap overlays for sample images.
#            Saves metrics JSON, confusion matrix PNG, heatmap PNGs.
# Author   : Ranjith Kumar
# =============================================================================

import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# =============================================================================
# IMPORT FROM PREVIOUS STEPS
# =============================================================================
sys.path.append(str(Path(__file__).parent))
from step1_dataset import (
    test_loader, test_df, val_transform,
    OUTPUT_DIR, CLASS_NAMES
)
from step3_model import get_model

with open(OUTPUT_DIR / "config.json") as f:
    config = json.load(f)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATTN_DIR    = OUTPUT_DIR / "attention_maps"
RESULTS_DIR = OUTPUT_DIR / "results"
ATTN_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("  STEP 5 — EVALUATION + ATTENTION HEATMAPS")
print("=" * 60)
print(f"[INFO] Device    : {DEVICE}")
print(f"[INFO] Test size : {config['test_size']}")

# =============================================================================
# LOAD BEST MODEL
# =============================================================================
def load_model(model_name="vit"):
    """Load best checkpoint saved during training."""
    ckpt_path = OUTPUT_DIR / "models" / f"best_{model_name}.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"[ERROR] Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model      = get_model(model_name, num_classes=config["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"[INFO] Loaded {model_name} from epoch {checkpoint['epoch']}")
    print(f"  Val acc : {checkpoint['val_acc']}%")
    print(f"  Val F1  : {checkpoint['val_f1']}%")
    return model

# =============================================================================
# FULL TEST SET EVALUATION
# =============================================================================
def evaluate(model, loader, model_name="vit"):
    """Run inference on full test set and compute all metrics."""
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs  = F.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Metrics
    acc  = accuracy_score(all_labels, all_preds) * 100
    f1   = f1_score(all_labels, all_preds, zero_division=0) * 100
    prec = precision_score(all_labels, all_preds, zero_division=0) * 100
    rec  = recall_score(all_labels, all_preds, zero_division=0) * 100
    cm   = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*60}")
    print(f"  {model_name.upper()} — TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.2f}%")
    print(f"  F1 Score  : {f1:.2f}%")
    print(f"  Precision : {prec:.2f}%")
    print(f"  Recall    : {rec:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"  {'':10} {'Pred Benign':>12} {'Pred Malignant':>15}")
    print(f"  {'True Benign':10} {cm[0][0]:>12} {cm[0][1]:>15}")
    print(f"  {'True Malig.':10} {cm[1][0]:>12} {cm[1][1]:>15}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=CLASS_NAMES)}")

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="Predicted", ylabel="True",
        title=f"{model_name.upper()} — Confusion Matrix\nAcc={acc:.1f}% F1={f1:.1f}%"
    )
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)
    plt.tight_layout()
    cm_path = RESULTS_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved: {cm_path}")

    # Save metrics JSON
    results = {
        "model"           : model_name,
        "test_accuracy"   : round(acc, 2),
        "test_f1"         : round(f1, 2),
        "test_precision"  : round(prec, 2),
        "test_recall"     : round(rec, 2),
        "confusion_matrix": cm.tolist()
    }
    results_path = RESULTS_DIR / f"test_results_{model_name}.json"
    with open(results_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"[INFO] Results saved      : {results_path}")

    return results, np.array(all_probs)

# =============================================================================
# ATTENTION HEATMAP GENERATION
# =============================================================================
def generate_heatmap(model, img_path, true_label):
    """
    Run one image through ViT, extract attention map, overlay on image.
    Returns: overlay PIL image, pred_label str, confidence float
    """
    orig_img = Image.open(img_path).convert("RGB")
    tensor   = val_transform(orig_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, attn_map = model.get_attention_map(tensor)
        probs = F.softmax(logits, dim=1)

    pred_idx   = logits.argmax(dim=1).item()
    confidence = probs[0, pred_idx].item() * 100
    pred_label = CLASS_NAMES[pred_idx]

    # Upsample 14x14 attention map → 224x224
    attn = attn_map[0].cpu().numpy()           # (14, 14)
    attn_tensor = torch.tensor(attn).unsqueeze(0).unsqueeze(0)
    attn_up = F.interpolate(
        attn_tensor, size=(224, 224),
        mode="bilinear", align_corners=False
    ).squeeze().numpy()                        # (224, 224)

    # Apply jet colormap
    colormap    = cm.get_cmap("jet")
    heatmap_rgb = (colormap(attn_up)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_rgb)

    # Blend original + heatmap
    orig_224 = orig_img.resize((224, 224))
    overlay  = Image.blend(orig_224, heatmap_img, alpha=0.45)

    return orig_224, heatmap_img, overlay, pred_label, confidence

# =============================================================================
# SAVE SAMPLE HEATMAPS
# =============================================================================
def save_sample_heatmaps(model, n_per_class=3):
    """
    Generate and save attention heatmap panels for n_per_class images
    from each class in the test set.
    Each panel: Original | Attention Map | Overlay
    """
    print(f"\n[INFO] Generating attention heatmaps ...")

    benign_df    = test_df[test_df.label == 0].head(n_per_class)
    malignant_df = test_df[test_df.label == 1].head(n_per_class)
    sample_df    = pd.concat([benign_df, malignant_df]).reset_index(drop=True)

    saved_paths = []

    for i, row in sample_df.iterrows():
        orig, heatmap, overlay, pred, conf = generate_heatmap(
            model, row["path"], row["label"]
        )
        true_name = CLASS_NAMES[row["label"]]
        correct   = "✓" if pred == true_name else "✗"

        # 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(
            f"True: {true_name} ({row['subtype']})  |  "
            f"Pred: {pred} ({conf:.1f}%) {correct}",
            fontsize=12,
            color="green" if pred == true_name else "red"
        )

        axes[0].imshow(orig);    axes[0].set_title("Original",      fontsize=10)
        axes[1].imshow(heatmap); axes[1].set_title("Attention Map",  fontsize=10)
        axes[2].imshow(overlay); axes[2].set_title("Overlay",        fontsize=10)
        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        save_name = f"heatmap_{i:02d}_{true_name.lower()}_{row['subtype']}.png"
        save_path = ATTN_DIR / save_name
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()

        saved_paths.append(str(save_path))
        status = "CORRECT" if pred == true_name else "WRONG"
        print(f"  [{i+1}] True={true_name} | Pred={pred} ({conf:.1f}%) | {status} → {save_name}")

    print(f"\n[INFO] {len(saved_paths)} heatmaps saved to: {ATTN_DIR}")
    return saved_paths

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    import pandas as pd

    # Load best ViT
    vit_model = load_model("vit")

    # Evaluate on test set
    vit_results, vit_probs = evaluate(vit_model, test_loader, "vit")

    # Generate attention heatmaps
    heatmap_paths = save_sample_heatmaps(vit_model, n_per_class=3)

    # Update config with test results
    with open(OUTPUT_DIR / "config.json") as f:
        config = json.load(f)
    config["test_accuracy"] = vit_results["test_accuracy"]
    config["test_f1"]       = vit_results["test_f1"]
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Test Accuracy : {vit_results['test_accuracy']}%")
    print(f"  Test F1       : {vit_results['test_f1']}%")
    print(f"  Test Precision: {vit_results['test_precision']}%")
    print(f"  Test Recall   : {vit_results['test_recall']}%")
    print(f"  Heatmaps      : {ATTN_DIR}")
    print(f"\n[DONE] Step 5 complete. Proceed to step6_inference.py")