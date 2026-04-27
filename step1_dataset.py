# =============================================================================
# Project  : BreakHis Vision Transformer - Breast Cancer Classification
# Step     : 1 - Dataset Loading
# Purpose  : Scan dataset, parse filenames, perform patient-level
#            train/val/test split, handle class imbalance with
#            WeightedRandomSampler, save splits as CSV files.
# Dataset  : D:\BREAST CANCER\archive\
# Author   : Ranjith Kumar
# =============================================================================

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import torch

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_ROOT     = Path(r"D:\BREAST CANCER\archive\BreaKHis_v1\BreaKHis_v1\histology_slides\breast")
OUTPUT_DIR    = Path(r"D:\BREAST CANCER\outputs")
MAGNIFICATION = "40X"       # Options: 40X | 100X | 200X | 400X
IMG_SIZE      = 224
BATCH_SIZE    = 32
NUM_WORKERS   = 0           # Keep 0 for Windows
RANDOM_SEED   = 42
VAL_SIZE      = 0.15        # 15% of patients for validation
TEST_SIZE     = 0.15        # 15% of patients for test
CLASS_NAMES        = ["Benign", "Malignant"]
SUBSET_MODE        = True        # True = quick test (50 train / 20 val / 20 test)
                                 # False = full dataset for real training
# Only use these 3 malignant subtypes — DC, LC, MC
# DC=Ductal Carcinoma (864 imgs), LC=Lobular Carcinoma (156), MC=Mucinous Carcinoma (205)
MALIGNANT_SUBTYPES = ["DC", "LC", "MC"]

# =============================================================================
# AUTO-CREATE OUTPUT FOLDERS
# =============================================================================
for folder in ["splits", "models", "results", "attention_maps", "sample_images"]:
    (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)
print(f"[INFO] Output folders created at: {OUTPUT_DIR}")

# =============================================================================
# FILENAME PARSER
# =============================================================================
# Filename format : SOB_M_PC-14-9146-40-001.png
#   parts[0] = SOB   (procedure)
#   parts[1] = B/M   (Benign / Malignant)
#   parts[2] = PC-14-9146-40-001  (subtype-patientID-mag-imgnum)
#
# Folder structure:
#   archive\benign\SOB\adenosis\SOB_B_A_14-22549\40X\SOB_B_A-14-22549-40-001.png
#   archive\malignant\SOB\ductal_carcinoma\SOB_M_DC_14-2523\40X\SOB_M_DC-14-2523-40-001.png

def parse_filename(filepath: Path):
    """
    Returns a dict with label, patient_id, subtype, magnification.
    Returns None if filename does not match expected format.
    """
    name  = filepath.stem          # e.g. SOB_M_PC-14-9146-40-001
    parts = name.split("_")

    if len(parts) < 3:
        return None

    # Label
    label_char = parts[1].strip()
    label = 0 if label_char == "B" else 1

    # Subtype: first segment before hyphen in parts[2]
    subtype_raw = parts[2]                     # e.g. PC-14-9146-40-001
    subtype     = subtype_raw.split("-")[0]    # e.g. PC

    # Patient ID: extract digits between subtype and magnification
    # Pattern: SUBTYPE-part1-part2-MAGNIFICATION-IMGNUM
    # e.g.     PC     -14  -9146 -40            -001
    match = re.search(
        r"[A-Za-z]+-(\d+[A-Za-z]?-\d+[A-Za-z]?)-\d+[A-Za-z]*-\d+",
        subtype_raw
    )
    if match:
        patient_id = match.group(1)
    else:
        # Fallback: use the patient folder name (grandparent of mag folder)
        patient_id = filepath.parent.parent.name

    return {
        "path"         : str(filepath),
        "label"        : label,
        "class"        : "Benign" if label == 0 else "Malignant",
        "subtype"      : subtype,
        "patient_id"   : patient_id,
        "magnification": filepath.parent.name,
        "filename"     : filepath.name
    }

# =============================================================================
# SCAN DATASET
# =============================================================================
print(f"\n[INFO] Scanning dataset for magnification = {MAGNIFICATION} ...")
print(f"[INFO] Root: {DATA_ROOT}\n")

all_records = []
skipped     = 0

for class_dir in ["benign", "malignant"]:
    class_path = DATA_ROOT / class_dir / "SOB"

    if not class_path.exists():
        print(f"[WARN] Folder not found: {class_path}")
        continue

    for subtype_dir in sorted(class_path.iterdir()):
        if not subtype_dir.is_dir():
            continue

        for patient_dir in sorted(subtype_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            mag_dir = patient_dir / MAGNIFICATION
            if not mag_dir.exists():
                continue

            for img_path in sorted(mag_dir.glob("*.png")):
                record = parse_filename(img_path)
                if record:
                    all_records.append(record)
                else:
                    skipped += 1

df = pd.DataFrame(all_records)

# Filter malignant to only DC, LC, MC subtypes
# Keep all benign + only selected malignant subtypes
benign_df    = df[df.label == 0]
malignant_df = df[(df.label == 1) & (df.subtype.isin(MALIGNANT_SUBTYPES))]
df = pd.concat([benign_df, malignant_df]).reset_index(drop=True)

print(f"[INFO] Total images found : {len(df)}")
print(f"[INFO] Skipped (parse err): {skipped}")
print(f"[INFO] Benign             : {(df.label == 0).sum()}")
print(f"[INFO] Malignant          : {(df.label == 1).sum()} (subtypes: {MALIGNANT_SUBTYPES})")
print(f"[INFO] Unique patients    : {df.patient_id.nunique()}")
print(f"\n[INFO] Images per subtype:")
print(df.groupby(["class", "subtype"]).size().to_string())

# =============================================================================
# PATIENT-LEVEL SPLIT
# =============================================================================
# CRITICAL: split by PATIENT not by image.
# If the same patient appears in train + val, accuracy is artificially inflated.

print(f"\n[INFO] Performing patient-level split ...")

patient_df = df.groupby("patient_id").agg(
    label    = ("label",  "first"),
    subtype  = ("subtype","first"),
    n_images = ("path",   "count")
).reset_index()

patients        = patient_df["patient_id"].values
patient_labels  = patient_df["label"].values

# Stratify by subtype to ensure DC, LC, MC all appear in train/val/test
# This fixes the problem of LC patients clustering in one split
patient_subtypes = patient_df["subtype"].values

# Train vs (val + test) — stratify by subtype
train_patients, temp_patients, _, temp_subtypes = train_test_split(
    patients, patient_subtypes,
    test_size    = (VAL_SIZE + TEST_SIZE),
    stratify     = patient_subtypes,
    random_state = RANDOM_SEED
)

# Val vs test — stratify by label only (too few patients per subtype to stratify by subtype)
temp_labels = patient_df[patient_df.patient_id.isin(temp_patients)]["label"].values
val_patients, test_patients = train_test_split(
    temp_patients,
    test_size    = 0.5,
    stratify     = temp_labels,
    random_state = RANDOM_SEED
)

train_df = df[df.patient_id.isin(train_patients)].reset_index(drop=True)
val_df   = df[df.patient_id.isin(val_patients)].reset_index(drop=True)
test_df  = df[df.patient_id.isin(test_patients)].reset_index(drop=True)

print(f"\n[INFO] Split results:")
print(f"  Train : {len(train_df):>5} images | {train_df.patient_id.nunique():>3} patients "
      f"| Benign={(train_df.label==0).sum()} | Malignant={(train_df.label==1).sum()}")
print(f"  Val   : {len(val_df):>5} images | {val_df.patient_id.nunique():>3} patients "
      f"| Benign={(val_df.label==0).sum()} | Malignant={(val_df.label==1).sum()}")
print(f"  Test  : {len(test_df):>5} images | {test_df.patient_id.nunique():>3} patients "
      f"| Benign={(test_df.label==0).sum()} | Malignant={(test_df.label==1).sum()}")

# Verify zero patient overlap between splits
assert len(set(train_patients) & set(val_patients))  == 0, "LEAK: train/val patient overlap!"
assert len(set(train_patients) & set(test_patients)) == 0, "LEAK: train/test patient overlap!"
assert len(set(val_patients)   & set(test_patients)) == 0, "LEAK: val/test patient overlap!"
print(f"[INFO] No patient overlap confirmed.")

# Save splits as CSV
train_df.to_csv(OUTPUT_DIR / "splits" / "train.csv", index=False)
val_df.to_csv(  OUTPUT_DIR / "splits" / "val.csv",   index=False)
test_df.to_csv( OUTPUT_DIR / "splits" / "test.csv",  index=False)
print(f"[INFO] Splits saved to {OUTPUT_DIR / 'splits'}")

# =============================================================================
# DATASET CLASS
# =============================================================================
class BreakHisDataset(Dataset):
    """
    PyTorch Dataset for BreakHis.
    Loads PNG images and returns (image_tensor, label).
    """
    def __init__(self, dataframe, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])

# =============================================================================
# TRANSFORMS
# =============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Train: augmentation to reduce overfitting on small dataset
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Val / Test: no augmentation
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# =============================================================================
# SUBSET MODE (for quick pipeline testing)
# =============================================================================
if SUBSET_MODE:
    # Proportional split — 100 images total
    # Benign:Malignant ratio ~1:2.3 maintained across all splits

    # Train: 245 images — Benign=76, Malignant=169
    train_benign    = train_df[train_df.label==0].sample(n=76,  random_state=RANDOM_SEED)
    train_malignant = train_df[train_df.label==1].sample(n=169, random_state=RANDOM_SEED)
    train_df = pd.concat([train_benign, train_malignant]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Val: 62 images — Benign=19, Malignant=43
    val_benign    = val_df[val_df.label==0].sample(n=19, random_state=RANDOM_SEED)
    val_malignant = val_df[val_df.label==1].sample(n=43, random_state=RANDOM_SEED)
    val_df = pd.concat([val_benign, val_malignant]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Test: 43 images — Benign=13, Malignant=30
    test_benign    = test_df[test_df.label==0].sample(n=13, random_state=RANDOM_SEED)
    test_malignant = test_df[test_df.label==1].sample(n=30, random_state=RANDOM_SEED)
    test_df = pd.concat([test_benign, test_malignant]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\n[INFO] SUBSET MODE ON — proportional split:")
    print(f"  Train : {len(train_df)} images | Benign={(train_df.label==0).sum()} | Malignant={(train_df.label==1).sum()}")
    print(f"  Val   : {len(val_df)} images | Benign={(val_df.label==0).sum()} | Malignant={(val_df.label==1).sum()}")
    print(f"  Test  : {len(test_df)} images | Benign={(test_df.label==0).sum()} | Malignant={(test_df.label==1).sum()}")
    print(f"  Total : {len(train_df)+len(val_df)+len(test_df)} images")
    print(f"[INFO] Set SUBSET_MODE=False for full training")

# =============================================================================
# WEIGHTED RANDOM SAMPLER
# =============================================================================
# Handles class imbalance: ~2480 benign vs ~5429 malignant
benign_count    = int((train_df.label == 0).sum())
malignant_count = int((train_df.label == 1).sum())
class_weights   = [1.0 / benign_count, 1.0 / malignant_count]
sample_weights  = [class_weights[label] for label in train_df["label"].values]

sampler = WeightedRandomSampler(
    weights     = sample_weights,
    num_samples = len(sample_weights),
    replacement = True
)

print(f"\n[INFO] Class imbalance handling:")
print(f"  Benign    : {benign_count}  → sample weight = {class_weights[0]:.6f}")
print(f"  Malignant : {malignant_count} → sample weight = {class_weights[1]:.6f}")

# =============================================================================
# DATALOADERS
# =============================================================================
train_dataset = BreakHisDataset(train_df, transform=train_transform)
val_dataset   = BreakHisDataset(val_df,   transform=val_transform)
test_dataset  = BreakHisDataset(test_df,  transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size  = BATCH_SIZE,
    sampler     = sampler,
    num_workers = NUM_WORKERS,
    pin_memory  = True
)
val_loader = DataLoader(
    val_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = NUM_WORKERS,
    pin_memory  = True
)
test_loader = DataLoader(
    test_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = NUM_WORKERS,
    pin_memory  = True
)

print(f"\n[INFO] DataLoaders ready:")
print(f"  Train batches : {len(train_loader)}")
print(f"  Val batches   : {len(val_loader)}")
print(f"  Test batches  : {len(test_loader)}")

# =============================================================================
# SAVE CONFIG (shared across all steps)
# =============================================================================
config = {
    "data_root"      : str(DATA_ROOT),
    "output_dir"     : str(OUTPUT_DIR),
    "magnification"  : MAGNIFICATION,
    "img_size"       : IMG_SIZE,
    "batch_size"     : BATCH_SIZE,
    "num_workers"    : NUM_WORKERS,
    "random_seed"    : RANDOM_SEED,
    "num_classes"    : 2,
    "class_names"    : CLASS_NAMES,
    "imagenet_mean"  : IMAGENET_MEAN,
    "imagenet_std"   : IMAGENET_STD,
    "train_size"     : len(train_df),
    "val_size"       : len(val_df),
    "test_size"      : len(test_df),
    "benign_count"      : benign_count,
    "malignant_count"   : malignant_count,
    "malignant_subtypes": MALIGNANT_SUBTYPES,
}

config_path = OUTPUT_DIR / "config.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"[INFO] Config saved to: {config_path}")

# =============================================================================
# SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    print(f"\n[CHECK] Running sanity check ...")
    images, labels = next(iter(train_loader))
    print(f"  Batch image shape : {images.shape}")        # (32, 3, 224, 224)
    print(f"  Batch labels      : {labels[:8].tolist()}")
    print(f"  Image dtype       : {images.dtype}")
    print(f"  Pixel min/max     : {images.min():.3f} / {images.max():.3f}")
    print(f"\n[DONE] Step 1 complete. Run step2_preprocess.py next.")