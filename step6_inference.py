# =============================================================================
# Project  : BreakHis Vision Transformer - Breast Cancer Classification
# Step     : 6 - Preloaded Inference
# Purpose  : Pick 1 image per subtype (8 total: 4 benign + 4 malignant),
#            run inference, save predictions + attention heatmaps.
#            Output goes into hf_space\sample_images\ for HF Spaces demo.
#            Users can click these on the app without uploading anything.
# Author   : Ranjith Kumar
# =============================================================================

import sys
import json
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image

# =============================================================================
# IMPORT FROM PREVIOUS STEPS
# =============================================================================
sys.path.append(str(Path(__file__).parent))
from step1_dataset import (
    df as full_df,
    val_transform,
    OUTPUT_DIR, CLASS_NAMES
)
from step3_model import get_model

with open(OUTPUT_DIR / "config.json") as f:
    config = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HF Space folder — all inference outputs go here
HF_SPACE_DIR    = Path(r"D:\BREAST CANCER\hf_space")
SAMPLE_IMG_DIR  = HF_SPACE_DIR / "sample_images"
MODEL_ART_DIR   = HF_SPACE_DIR / "model_artifacts"

for d in [HF_SPACE_DIR, SAMPLE_IMG_DIR, MODEL_ART_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  STEP 6 — PRELOADED INFERENCE")
print("=" * 60)
print(f"[INFO] HF Space dir : {HF_SPACE_DIR}")
print(f"[INFO] Device       : {DEVICE}")

# =============================================================================
# LOAD BEST MODEL
# =============================================================================
ckpt_path  = OUTPUT_DIR / "models" / "best_vit.pt"
checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model      = get_model("vit", num_classes=config["num_classes"], pretrained=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()
print(f"[INFO] Model loaded from epoch {checkpoint['epoch']} "
      f"(val_f1={checkpoint['val_f1']}%)")

# =============================================================================
# PICK 1 IMAGE PER SUBTYPE (8 total)
# =============================================================================
# Benign subtypes    : A, F, PT, TA
# Malignant subtypes : DC, LC, MC, PC
# One representative image per subtype → covers all 8 tumor types in the demo

BENIGN_SUBTYPES    = ["A", "F", "PT", "TA"]
MALIGNANT_SUBTYPES = ["DC", "LC", "MC", "PC"]

SUBTYPE_FULL_NAMES = {
    "A"  : "Adenosis",
    "F"  : "Fibroadenoma",
    "PT" : "Phyllodes Tumor",
    "TA" : "Tubular Adenoma",
    "DC" : "Ductal Carcinoma",
    "LC" : "Lobular Carcinoma",
    "MC" : "Mucinous Carcinoma",
    "PC" : "Papillary Carcinoma",
}

sample_records = []

for subtype in BENIGN_SUBTYPES:
    subset = full_df[(full_df.subtype == subtype) & (full_df.label == 0)]
    if len(subset) > 0:
        row = subset.sample(n=1, random_state=42).iloc[0]
        sample_records.append(row.to_dict())

for subtype in MALIGNANT_SUBTYPES:
    subset = full_df[(full_df.subtype == subtype) & (full_df.label == 1)]
    if len(subset) > 0:
        row = subset.sample(n=1, random_state=42).iloc[0]
        sample_records.append(row.to_dict())

print(f"\n[INFO] Selected {len(sample_records)} sample images (1 per subtype)")

# =============================================================================
# INFERENCE + HEATMAP GENERATION
# =============================================================================
def run_inference(img_path):
    """
    Load image → preprocess → ViT forward → attention map.
    Returns logits, probs, attention map, pred class, confidence.
    """
    orig_img = Image.open(img_path).convert("RGB")
    tensor   = val_transform(orig_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, attn_map = model.get_attention_map(tensor)
        probs = F.softmax(logits, dim=1)

    pred_idx   = logits.argmax(dim=1).item()
    confidence = probs[0, pred_idx].item() * 100
    pred_label = CLASS_NAMES[pred_idx]

    # Upsample attention 14x14 → 224x224
    attn = attn_map[0].cpu().numpy()
    attn_t  = torch.tensor(attn).unsqueeze(0).unsqueeze(0)
    attn_up = F.interpolate(
        attn_t, size=(224, 224),
        mode="bilinear", align_corners=False
    ).squeeze().numpy()

    # Jet colormap heatmap
    colormap    = matplotlib.colormaps["jet"]
    heatmap_rgb = (colormap(attn_up)[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb)

    # Overlay
    orig_224 = orig_img.resize((224, 224))
    overlay  = Image.blend(orig_224, heatmap_pil, alpha=0.45)

    return {
        "original"  : orig_224,
        "heatmap"   : heatmap_pil,
        "overlay"   : overlay,
        "pred_label": pred_label,
        "pred_idx"  : pred_idx,
        "confidence": round(confidence, 2),
        "prob_benign"   : round(probs[0, 0].item() * 100, 2),
        "prob_malignant": round(probs[0, 1].item() * 100, 2),
    }

# =============================================================================
# PROCESS ALL SAMPLES + SAVE
# =============================================================================
sample_results = []

print(f"\n{'='*60}")
print(f"  Running inference on all 8 samples")
print(f"{'='*60}")

for i, record in enumerate(sample_records):
    subtype     = record["subtype"]
    true_label  = record["label"]
    true_class  = CLASS_NAMES[true_label]
    full_name   = SUBTYPE_FULL_NAMES.get(subtype, subtype)
    img_path    = record["path"]

    result = run_inference(img_path)
    correct = (result["pred_idx"] == true_label)
    status  = "CORRECT" if correct else "WRONG"

    print(f"  [{i+1}] {true_class:10} ({full_name:<20}) | "
          f"Pred={result['pred_label']} ({result['confidence']}%) | {status}")

    # ── Save original image to sample_images\ ────────────────────────────────
    orig_filename    = f"sample_{i+1:02d}_{true_class.lower()}_{subtype}.png"
    overlay_filename = f"overlay_{i+1:02d}_{true_class.lower()}_{subtype}.png"

    result["original"].save(SAMPLE_IMG_DIR / orig_filename)
    result["overlay"].save( SAMPLE_IMG_DIR / overlay_filename)

    # ── Save 3-panel heatmap figure ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f"True: {true_class} — {full_name}  |  "
        f"Pred: {result['pred_label']} ({result['confidence']}%)",
        fontsize=11,
        color="green" if correct else "red"
    )
    axes[0].imshow(result["original"]); axes[0].set_title("Original",     fontsize=10)
    axes[1].imshow(result["heatmap"]);  axes[1].set_title("Attention Map", fontsize=10)
    axes[2].imshow(result["overlay"]);  axes[2].set_title("Overlay",       fontsize=10)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    panel_path = SAMPLE_IMG_DIR / f"panel_{i+1:02d}_{true_class.lower()}_{subtype}.png"
    plt.savefig(panel_path, dpi=120, bbox_inches="tight")
    plt.close()

    # ── Record result for JSON ────────────────────────────────────────────────
    sample_results.append({
        "index"          : i + 1,
        "filename"       : orig_filename,
        "overlay_file"   : overlay_filename,
        "true_class"     : true_class,
        "true_label"     : true_label,
        "subtype"        : subtype,
        "subtype_full"   : full_name,
        "pred_class"     : result["pred_label"],
        "pred_idx"       : result["pred_idx"],
        "confidence"     : result["confidence"],
        "prob_benign"    : result["prob_benign"],
        "prob_malignant" : result["prob_malignant"],
        "correct"        : correct,
    })

# =============================================================================
# SAVE SAMPLE RESULTS JSON
# =============================================================================
results_json_path = HF_SPACE_DIR / "sample_results.json"
with open(results_json_path, "w") as f:
    json.dump(sample_results, f, indent=2)
print(f"\n[INFO] Sample results saved : {results_json_path}")

# =============================================================================
# COPY MODEL CHECKPOINT TO HF SPACE
# =============================================================================
src_model = OUTPUT_DIR / "models" / "best_vit.pt"
dst_model = MODEL_ART_DIR / "best_vit.pt"
shutil.copy2(src_model, dst_model)
print(f"[INFO] Model checkpoint copied: {dst_model}")

# Copy config
src_config = OUTPUT_DIR / "config.json"
dst_config = MODEL_ART_DIR / "config.json"
shutil.copy2(src_config, dst_config)
print(f"[INFO] Config copied          : {dst_config}")

# =============================================================================
# SUMMARY
# =============================================================================
correct_count = sum(1 for r in sample_results if r["correct"])
print(f"\n{'='*60}")
print(f"  PRELOADED INFERENCE SUMMARY")
print(f"{'='*60}")
print(f"  Total samples : {len(sample_results)}")
print(f"  Correct       : {correct_count}/{len(sample_results)}")
print(f"  Sample images : {SAMPLE_IMG_DIR}")
print(f"  Results JSON  : {results_json_path}")
print(f"  Model artifact: {dst_model}")
print(f"\n  HF Space folder structure:")
print(f"  hf_space\\")
print(f"  ├── sample_images\\   ({len(list(SAMPLE_IMG_DIR.iterdir()))} files)")
print(f"  ├── model_artifacts\\ (best_vit.pt + config.json)")
print(f"  └── sample_results.json")
print(f"\n[DONE] Step 6 complete. Proceed to step7_app.py")