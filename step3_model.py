# =============================================================================
# Project  : BreakHis Vision Transformer - Breast Cancer Classification
# Step     : 3 - Model Definition
# Purpose  : Define ViT-small with attention extraction hook for heatmaps.
#            Also define ResNet-18 baseline for comparison.
#            Verify both models with a dummy forward pass.
# Author   : Ranjith Kumar
# =============================================================================

import sys
import json
import torch
import torch.nn as nn
import timm
import torchvision.models as models
from pathlib import Path

# =============================================================================
# IMPORT CONFIG FROM STEP 1
# =============================================================================
sys.path.append(str(Path(__file__).parent))
from step1_dataset import OUTPUT_DIR, CLASS_NAMES

with open(OUTPUT_DIR / "config.json") as f:
    config = json.load(f)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = config["num_classes"]   # 2
IMG_SIZE    = config["img_size"]      # 224

print("=" * 60)
print("  STEP 3 — MODEL DEFINITION")
print("=" * 60)
print(f"[INFO] Device      : {DEVICE}")
print(f"[INFO] Num classes : {NUM_CLASSES} {CLASS_NAMES}")
print(f"[INFO] Image size  : {IMG_SIZE}x{IMG_SIZE}")

# =============================================================================
# VISION TRANSFORMER (ViT-small)
# =============================================================================
class BreakHisViT(nn.Module):
    """
    Swin Transformer (Swin-Tiny) for BreakHis binary classification.

    Architecture:
      - Backbone : swin_tiny_patch4_window7_224 pretrained on ImageNet-1k (via timm)
      - Head     : LayerNorm → Dropout → Linear(768→128) → GELU → Linear(128→2)
      - Attention: GradCAM-style attention via feature hook on last Swin stage
                   Used to generate spatial attention maps → upsampled to 224x224

    Why Swin-Tiny over ViT-small:
      - Uses shifted WINDOW attention — local tissue regions first, then global
      - Hierarchical feature extraction — like how pathologists read slides
      - Much better on small datasets (350 images) vs ViT global attention
      - State of art on BreakHis in recent histopathology papers
      - embed_dim = 768 at final stage
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super().__init__()

        # Load pretrained Swin-Tiny backbone
        self.vit = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained  = pretrained,
            num_classes = 0,          # remove default classification head
            drop_rate   = dropout
        )
        embed_dim = self.vit.num_features  # 768 for swin_tiny final stage

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

        # Storage for attention feature maps — filled by hook during forward pass
        self.attention_weights = None

        # Register hook on last Swin stage to capture spatial features
        self._register_attention_hook()

    def _register_attention_hook(self):
        """
        Hook into the last Swin stage to capture spatial feature maps.
        Swin uses window attention internally — we capture the output
        of the last stage before global average pooling.
        Shape captured: (B, H, W, C) → converted to spatial attention map.
        """
        def hook_fn(module, input, output):
            # output shape: (B, H, W, C) for last Swin stage
            # Convert to attention-like map by averaging across channels
            if isinstance(output, torch.Tensor):
                feat = output.detach().cpu()          # (B, H, W, C)
                # Average across channel dim → (B, H, W)
                attn = feat.abs().mean(dim=-1)
                self.attention_weights = attn
            
        # Hook on the last layer of the last Swin stage
        last_stage = self.vit.layers[-1]
        last_stage.register_forward_hook(hook_fn)

    def forward(self, x):
        """Standard forward pass — returns logits."""
        features = self.vit(x)               # (B, embed_dim)
        logits   = self.classifier(features)  # (B, num_classes)
        return logits

    def get_attention_map(self, x):
        """
        Run forward pass and extract spatial attention map.

        Returns:
          logits   : (B, num_classes)
          attn_map : (B, 7, 7) — normalized spatial map from last Swin stage
                     upsampled to 224x224 in evaluation/app scripts
        """
        logits = self.forward(x)

        if self.attention_weights is None:
            raise RuntimeError("[ERROR] Attention weights not captured. Check hook.")

        attn = self.attention_weights  # (B, H, W)

        # Normalize to [0, 1] per image
        B = attn.shape[0]
        attn_flat = attn.reshape(B, -1)
        attn_flat = attn_flat - attn_flat.min(dim=1, keepdim=True)[0]
        attn_flat = attn_flat / (attn_flat.max(dim=1, keepdim=True)[0] + 1e-8)
        attn_map  = attn_flat.reshape(attn.shape)  # (B, H, W)
        return logits, attn_map


# =============================================================================
# RESNET-18 BASELINE
# =============================================================================
class BreakHisResNet18(nn.Module):
    """
    ResNet-18 baseline for comparison with ViT.
    Same head structure for fair comparison.
    Used to show: accuracy difference + training time difference.
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super().__init__()

        # Load pretrained ResNet-18
        weights   = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone  = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features  # 512

        # Remove default FC head
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Custom classification head (same structure as ViT for fair comparison)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)         # (B, 512)
        logits   = self.classifier(features) # (B, num_classes)
        return logits


# =============================================================================
# MODEL FACTORY
# =============================================================================
def get_model(model_name="vit", num_classes=NUM_CLASSES, pretrained=True):
    """
    Args:
        model_name : "vit" or "resnet18"
        num_classes: 2 for binary classification
        pretrained : use ImageNet pretrained weights
    Returns:
        model instance (not yet moved to device)
    """
    if model_name == "vit":
        model = BreakHisViT(
            num_classes = num_classes,
            pretrained  = pretrained,
            dropout     = 0.3
        )
        print(f"[INFO] BreakHisSwin (Swin-Tiny) loaded (pretrained={pretrained})")

    elif model_name == "resnet18":
        model = BreakHisResNet18(
            num_classes = num_classes,
            pretrained  = pretrained,
            dropout     = 0.3
        )
        print(f"[INFO] BreakHisResNet18 loaded (pretrained={pretrained})")

    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'vit' or 'resnet18'.")

    return model


# =============================================================================
# VERIFY BOTH MODELS WITH DUMMY FORWARD PASS
# =============================================================================
if __name__ == "__main__":

    # ── ViT verification ──────────────────────────────────────────────────────
    print(f"\n[CHECK] Testing ViT-small ...")
    vit = get_model("vit").to(DEVICE)
    dummy = torch.randn(2, 3, 224, 224).to(DEVICE)

    with torch.no_grad():
        # Standard forward
        logits = vit(dummy)
        print(f"  Logits shape        : {logits.shape}")        # (2, 2)

        # Forward with attention map
        logits, attn_map = vit.get_attention_map(dummy)
        print(f"  Attention map shape : {attn_map.shape}")      # (2, 14, 14)
        print(f"  Attn map min/max    : {attn_map.min():.3f} / {attn_map.max():.3f}")

    # Count parameters
    vit_params     = sum(p.numel() for p in vit.parameters())
    vit_trainable  = sum(p.numel() for p in vit.parameters() if p.requires_grad)
    print(f"  Total params        : {vit_params:,}")
    print(f"  Trainable params    : {vit_trainable:,}")

    # ── ResNet-18 verification ────────────────────────────────────────────────
    print(f"\n[CHECK] Testing ResNet-18 ...")
    resnet = get_model("resnet18").to(DEVICE)

    with torch.no_grad():
        logits = resnet(dummy)
        print(f"  Logits shape        : {logits.shape}")        # (2, 2)

    res_params    = sum(p.numel() for p in resnet.parameters())
    res_trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    print(f"  Total params        : {res_params:,}")
    print(f"  Trainable params    : {res_trainable:,}")

    # ── Comparison ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Model':<15} {'Total Params':>15} {'Trainable':>15}")
    print(f"  {'-'*45}")
    print(f"  {'ViT-small':<15} {vit_params:>15,} {vit_trainable:>15,}")
    print(f"  {'ResNet-18':<15} {res_params:>15,} {res_trainable:>15,}")

    # ── Update config ─────────────────────────────────────────────────────────
    with open(OUTPUT_DIR / "config.json") as f:
        config = json.load(f)

    config["vit_params"]     = vit_params
    config["resnet18_params"] = res_params
    config["device"]         = str(DEVICE)

    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[INFO] Config updated with model info.")
    print(f"[DONE] Step 3 complete. Proceed to step4_train.py")