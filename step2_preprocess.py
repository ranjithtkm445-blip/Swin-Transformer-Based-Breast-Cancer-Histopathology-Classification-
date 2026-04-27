# =============================================================================
# Project  : BreakHis Vision Transformer - Breast Cancer Classification
# Step     : 2 - Image Preprocessing
# Purpose  : Apply Macenko stain normalization to H&E histopathology images,
#            verify transforms, save sample preprocessed images for inspection.
#            Transforms are already defined in step1_dataset.py.
#            This step validates and visualizes the preprocessing pipeline.
# Dataset  : D:\BREAST CANCER\archive\
# Author   : Ranjith Kumar
# =============================================================================

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed on Windows
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms

# =============================================================================
# IMPORT FROM STEP 1
# =============================================================================
sys.path.append(str(Path(__file__).parent))
from step1_dataset import (
    train_df, val_df, test_df,
    train_transform, val_transform,
    train_loader, val_loader,
    OUTPUT_DIR, IMAGENET_MEAN, IMAGENET_STD,
    CLASS_NAMES
)

print("=" * 60)
print("  STEP 2 — IMAGE PREPROCESSING")
print("=" * 60)

# =============================================================================
# MACENKO STAIN NORMALIZATION
# =============================================================================
# H&E stained slides have color variation across different labs/slides.
# Macenko normalization standardizes the stain appearance.
# Reference: Macenko et al., 2009

class MacenkoNormalizer:
    """
    Macenko stain normalization for H&E histopathology images.
    Normalizes the stain color distribution to a reference image.
    Works on PIL images — applied before tensor conversion.
    """
    def __init__(self):
        # Standard H&E reference stain matrix (from literature)
        self.HERef = np.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581]
        ])
        # Reference max stain concentrations
        self.maxCRef = np.array([1.9705, 1.0308])

    def __call__(self, img: Image.Image) -> Image.Image:
        """Normalize a PIL image using Macenko method."""
        try:
            img_np = np.array(img).astype(np.float32)

            # Convert RGB to OD (optical density)
            img_np = np.maximum(img_np, 1)  # avoid log(0)
            OD = -np.log(img_np / 255.0 + 1e-6)

            # Remove pixels with low OD (background)
            ODhat = OD[~np.any(OD < 0.15, axis=2)]

            if len(ODhat) < 10:
                return img  # not enough tissue — return original

            # SVD on OD
            _, _, V = np.linalg.svd(ODhat, full_matrices=False)
            V = V[:2, :].T  # first two singular vectors

            # Project OD onto plane spanned by V
            That = ODhat @ V

            # Find angle of each pixel
            phi = np.arctan2(That[:, 1], That[:, 0])
            minPhi = np.percentile(phi, 1)
            maxPhi = np.percentile(phi, 99)

            # Stain vectors
            v1 = V @ np.array([np.cos(minPhi), np.sin(minPhi)])
            v2 = V @ np.array([np.cos(maxPhi), np.sin(maxPhi)])

            # Make H always the first stain (larger OD in first channel)
            if v1[0] < v2[0]:
                HE = np.array([v2, v1]).T
            else:
                HE = np.array([v1, v2]).T

            # Normalize HE columns
            HE = HE / (np.linalg.norm(HE, axis=0) + 1e-6)

            # Get concentrations
            OD_flat = OD.reshape(-1, 3)
            C, _ = np.linalg.lstsq(HE, OD_flat.T, rcond=None)[:2]

            # Normalize concentrations
            maxC = np.percentile(C, 99, axis=1)
            C = C / (maxC[:, None] + 1e-6) * self.maxCRef[:, None]

            # Reconstruct normalized image
            OD_norm = self.HERef @ C
            img_norm = np.exp(-OD_norm.T.reshape(img_np.shape)) * 255
            img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)

            return Image.fromarray(img_norm)

        except Exception:
            # If normalization fails for any image, return original
            return img


# =============================================================================
# FULL PREPROCESSING PIPELINE WITH STAIN NORMALIZATION
# =============================================================================
stain_normalizer = MacenkoNormalizer()

# Updated transforms with stain normalization inserted before tensor conversion
train_transform_full = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: stain_normalizer(img)),  # Macenko
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform_full = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: stain_normalizer(img)),  # Macenko
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

print(f"\n[INFO] Preprocessing pipeline:")
print(f"  Train : Resize → Macenko Stain Norm → Flip → Rotate → ColorJitter → Normalize")
print(f"  Val   : Resize → Macenko Stain Norm → Normalize")

# =============================================================================
# DENORMALIZE HELPER (for visualization)
# =============================================================================
def denormalize(tensor):
    """Convert normalized tensor back to viewable image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = tensor * std + mean
    img  = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()

# =============================================================================
# VERIFY PREPROCESSING ON SAMPLE IMAGES
# =============================================================================
print(f"\n[INFO] Verifying preprocessing on sample images ...")

# Pick 4 samples — 2 benign, 2 malignant
samples = []
for label, name in [(0, "Benign"), (1, "Malignant")]:
    subset = train_df[train_df.label == label].head(2)
    for _, row in subset.iterrows():
        samples.append((row["path"], row["label"], row["subtype"]))

fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))
fig.suptitle("Preprocessing Verification — Original | Stain Normalized | Augmented",
             fontsize=13, y=1.01)

for i, (img_path, label, subtype) in enumerate(samples):
    pil_img = Image.open(img_path).convert("RGB")

    # Original resized
    orig = pil_img.resize((224, 224))

    # Stain normalized
    normalized = stain_normalizer(orig)

    # Full augmented tensor → denormalize for display
    aug_tensor = train_transform_full(pil_img)
    aug_display = denormalize(aug_tensor)

    axes[i, 0].imshow(orig)
    axes[i, 0].set_title(f"Original\n{CLASS_NAMES[label]} ({subtype})", fontsize=10)
    axes[i, 0].axis("off")

    axes[i, 1].imshow(normalized)
    axes[i, 1].set_title("Stain Normalized\n(Macenko)", fontsize=10)
    axes[i, 1].axis("off")

    axes[i, 2].imshow(aug_display)
    axes[i, 2].set_title("Augmented + Normalized\n(train transform)", fontsize=10)
    axes[i, 2].axis("off")

plt.tight_layout()
save_path = OUTPUT_DIR / "results" / "preprocessing_samples.png"
plt.savefig(save_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"[INFO] Sample preprocessing visualization saved to: {save_path}")

# =============================================================================
# VERIFY PIXEL STATISTICS AFTER NORMALIZATION
# =============================================================================
print(f"\n[INFO] Checking pixel statistics on val batch ...")

# Apply full val transform to a few images
pixel_means, pixel_stds = [], []
for _, row in val_df.head(10).iterrows():
    pil_img = Image.open(row["path"]).convert("RGB")
    tensor  = val_transform_full(pil_img)
    pixel_means.append(tensor.mean(dim=[1, 2]).numpy())
    pixel_stds.append(tensor.std(dim=[1, 2]).numpy())

mean_arr = np.array(pixel_means).mean(axis=0)
std_arr  = np.array(pixel_stds).mean(axis=0)

print(f"  Channel means (R, G, B) : {mean_arr.round(3)}")
print(f"  Channel stds  (R, G, B) : {std_arr.round(3)}")
print(f"  (Expected near 0.0 mean and ~1.0 std after ImageNet normalization)")

# =============================================================================
# SAVE UPDATED TRANSFORMS TO CONFIG
# =============================================================================
config_path = OUTPUT_DIR / "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

config["stain_normalization"] = "Macenko"
config["train_augmentation"]  = ["RandomHorizontalFlip", "RandomVerticalFlip",
                                  "RandomRotation(15)", "ColorJitter"]
config["val_augmentation"]    = ["None"]

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"\n[INFO] Config updated with preprocessing info: {config_path}")

# =============================================================================
# EXPORT NORMALIZER + TRANSFORMS FOR SUBSEQUENT STEPS
# =============================================================================
# Other steps import these directly from this file

print(f"\n[DONE] Step 2 complete.")
print(f"[INFO] Inspect: {OUTPUT_DIR / 'results' / 'preprocessing_samples.png'}")
print(f"[INFO] Proceed to step3_model.py")

# =============================================================================
# SUMMARY
# =============================================================================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  PREPROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"  Input size       : 700 x 460 px (raw BreakHis)")
    print(f"  Output size      : 224 x 224 px (ViT input)")
    print(f"  Stain norm       : Macenko H&E normalization")
    print(f"  Train augment    : HFlip + VFlip + Rotation(15) + ColorJitter")
    print(f"  Val/Test augment : None (resize + normalize only)")
    print(f"  Normalization    : ImageNet mean/std")
    print(f"  Output saved to  : {OUTPUT_DIR / 'results' / 'preprocessing_samples.png'}")